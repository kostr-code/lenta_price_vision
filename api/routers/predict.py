"""api/routers/predict.py — POST /predict/video, /predict/image, /predict/path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from api.pipeline_bridge import process_single_crop, process_video_file, save_run_files
from api.run_store import RunResult, new_run_id, save_run
from pipeline.video import rotate_frame

router = APIRouter()

_UPLOAD_FILE = File(...)
_FORM_MODE = Form("vlm")
_FORM_ENABLE_OCR = Form(None)
_FORM_ENABLE_QR = Form(None)
_FORM_SAVE_CROPS = Form(False)
_FORM_MAX_FRAMES = Form(0)
_FORM_SAMPLE_FPS = Form(None)
_FORM_YOLO_WEIGHTS = Form(None)


def _parse_mode(mode: str, enable_ocr: bool | None) -> dict[str, bool]:
    """Map mode string + explicit flags to use_vlm / use_ocr booleans."""
    use_vlm = mode not in ("cpu_safe", "ocr_only")
    use_ocr = True if enable_ocr is None else enable_ocr
    return {"use_vlm": use_vlm, "use_ocr": use_ocr}


def _make_response(
    run_id: str,
    rows: list[dict[str, str]],
    files: dict,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "ok",
        "rows": rows,
        "row_count": len(rows),
        "download": f"/download/{run_id}/output.csv",
        "debug_download": f"/download/{run_id}/debug.json",
    }


@router.post("/predict/image")
async def predict_image(
    file: UploadFile = _UPLOAD_FILE,
    mode: str = _FORM_MODE,
    enable_ocr: bool | None = _FORM_ENABLE_OCR,
    enable_qr: bool | None = _FORM_ENABLE_QR,
    save_crops: bool = _FORM_SAVE_CROPS,
) -> dict[str, Any]:
    log = structlog.get_logger("predict")
    run_id = new_run_id()
    log.info("image.start", run_id=run_id, mode=mode, filename=file.filename)

    flags = _parse_mode(mode, enable_ocr)

    # Decode image
    payload = await file.read()
    arr = np.frombuffer(payload, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Cannot decode image file")

    # Standard rotation (robot camera orientation)
    img = rotate_frame(img)

    # Try Stage 1 detection first
    rows: list[dict] = []
    try:
        from pipeline.detector import detect_price_tags
        from api.ml_app import settings as app_settings

        if app_settings.weights_stage1 and Path(app_settings.weights_stage1).exists():
            detections = detect_price_tags(img, conf=0.25)
            from pipeline.video import cut_crop_bbox
            for det in detections:
                crop = cut_crop_bbox(img, det.x1, det.y1, det.x2, det.y2)
                if crop is not None and crop.size > 0:
                    row = process_single_crop(crop, **flags)
                    row["x_min"] = str(det.x1)
                    row["y_min"] = str(det.y1)
                    row["x_max"] = str(det.x2)
                    row["y_max"] = str(det.y2)
                    rows.append(row)
    except Exception as exc:
        log.warning("image.detect_failed", error=str(exc))

    # Fallback: treat whole image as a crop
    if not rows:
        row = process_single_crop(img, **flags)
        rows.append(row)

    from api.ml_app import settings as app_settings
    files = save_run_files(run_id, rows, app_settings.runs_dir)
    result = RunResult(run_id=run_id, status="ok", rows=rows, files=files)
    save_run(result)

    log.info("image.done", run_id=run_id, rows=len(rows))
    return _make_response(run_id, rows, files)


@router.post("/predict/video")
async def predict_video(
    file: UploadFile = _UPLOAD_FILE,
    mode: str = _FORM_MODE,
    sample_fps: float | None = _FORM_SAMPLE_FPS,
    max_frames: int = _FORM_MAX_FRAMES,
    yolo_weights: str | None = _FORM_YOLO_WEIGHTS,
    enable_ocr: bool | None = _FORM_ENABLE_OCR,
    enable_qr: bool | None = _FORM_ENABLE_QR,
    save_crops: bool = _FORM_SAVE_CROPS,
) -> dict[str, Any]:
    log = structlog.get_logger("predict")
    run_id = new_run_id()
    log.info("video.start", run_id=run_id, mode=mode, filename=file.filename)

    from api.ml_app import settings as app_settings

    flags = _parse_mode(mode, enable_ocr)

    # Save upload to runs_dir
    run_dir = Path(app_settings.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "input.mp4").suffix or ".mp4"
    video_path = run_dir / f"input{suffix}"
    payload = await file.read()
    video_path.write_bytes(payload)
    log.info("video.saved", path=str(video_path), size_mb=len(payload) / 1e6)

    # Run pipeline
    try:
        rows = process_video_file(
            video_path=str(video_path),
            run_id=run_id,
            runs_dir=app_settings.runs_dir,
            quality_threshold=app_settings.quality_threshold,
            track_top_k=app_settings.track_top_k,
            **flags,
        )
    except Exception as exc:
        log.error("video.failed", run_id=run_id, error=str(exc))
        result = RunResult(run_id=run_id, status="error", rows=[], error=str(exc))
        save_run(result)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    files = save_run_files(run_id, rows, app_settings.runs_dir)
    result = RunResult(run_id=run_id, status="ok", rows=rows, files=files)
    save_run(result)

    log.info("video.done", run_id=run_id, rows=len(rows))
    return _make_response(run_id, rows, files)


class PathPredictRequest(BaseModel):
    video_path: str
    mode: str = "vlm"
    enable_ocr: bool | None = None
    quality_threshold: float | None = None
    max_frames: int = Field(default=0, ge=0)


@router.post("/predict/path")
async def predict_path(request: PathPredictRequest) -> dict[str, Any]:
    """Process a video by server-side path (no upload). Useful for testing."""
    log = structlog.get_logger("predict")
    run_id = new_run_id()
    log.info("path.start", run_id=run_id, path=request.video_path)

    if not Path(request.video_path).exists():
        raise HTTPException(status_code=422, detail=f"Video not found: {request.video_path}")

    from api.ml_app import settings as app_settings

    flags = _parse_mode(request.mode, request.enable_ocr)
    qt = request.quality_threshold or app_settings.quality_threshold

    try:
        rows = process_video_file(
            video_path=request.video_path,
            run_id=run_id,
            runs_dir=app_settings.runs_dir,
            quality_threshold=qt,
            track_top_k=app_settings.track_top_k,
            **flags,
        )
    except Exception as exc:
        log.error("path.failed", run_id=run_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    files = save_run_files(run_id, rows, app_settings.runs_dir)
    result = RunResult(run_id=run_id, status="ok", rows=rows, files=files)
    save_run(result)

    log.info("path.done", run_id=run_id, rows=len(rows))
    return _make_response(run_id, rows, files)
