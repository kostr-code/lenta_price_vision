from __future__ import annotations

import os
import shutil
from pathlib import Path
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .pipeline import (
    DEFAULT_DATA_DIR,
    PipelineConfig,
    RetailShelfPipeline,
    discover_labeled_sequences,
    run_public_evaluation,
    summarize_csv,
)
from .schema import OUTPUT_COLUMNS
from .training import build_yolo_dataset, train_yolo_detector

PACKAGE_DIR = Path(__file__).resolve().parent
WORK_DIR = Path(os.getenv("ML_WORK_DIR", str(PACKAGE_DIR / "runs")))
UPLOAD_FILE_PARAM = File(...)
MODE_FORM_PARAM = Form("cpu_safe")
SAMPLE_FPS_FORM_PARAM = Form(None)
MAX_FRAMES_FORM_PARAM = Form(0)
YOLO_WEIGHTS_FORM_PARAM = Form(None)
ENABLE_OCR_FORM_PARAM = Form(None)
ENABLE_QR_FORM_PARAM = Form(None)
SAVE_CROPS_FORM_PARAM = Form(False)
YOLO_CONF_FORM_PARAM = Form(None)
DETECTOR_IMGSZ_FORM_PARAM = Form(None)
DETECTOR_IOU_FORM_PARAM = Form(None)
DETECTOR_DEVICE_FORM_PARAM = Form(None)
TRACKING_BACKEND_FORM_PARAM = Form(None)
TRACKER_CONFIG_FORM_PARAM = Form(None)
DEFER_OCR_FORM_PARAM = Form(None)
TOP_K_CROPS_FORM_PARAM = Form(None)
TILED_YOLO_FORM_PARAM = Form(None)
TILE_SIZE_FORM_PARAM = Form(None)
TILE_STRIDE_FORM_PARAM = Form(None)
MAX_TILES_FORM_PARAM = Form(None)
CROP_PHASH_DEDUP_FORM_PARAM = Form(None)
DERIVE_QR_FIELDS_FORM_PARAM = Form(None)
RAIL_ROI_FORM_PARAM = Form(None)
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
app = FastAPI(title="Price Tag Audit ML Service")


@app.get("/health")
def health() -> dict[str, str | list[str]]:
    return {"status": "ok", "service": "ml", "csv_columns": OUTPUT_COLUMNS}


@app.get("/schema")
def schema() -> dict[str, list[str]]:
    return {"columns": OUTPUT_COLUMNS}


@app.get("/datasets")
def datasets(data_dir: str | None = None) -> dict[str, object]:
    root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    sequences = discover_labeled_sequences(root)
    return {
        "data_dir": str(root),
        "sequences": [
            {
                "name": item.name,
                "video_path": str(item.video_path),
                "csv_path": str(item.csv_path),
            }
            for item in sequences
        ],
    }


class PathPredictionRequest(BaseModel):
    video_path: str
    output_dir: str | None = None
    mode: str = "cpu_safe"
    sample_fps: float | None = Field(default=None, gt=0)
    max_frames: int = Field(default=0, ge=0)
    yolo_weights: str | None = None
    yolo_conf: float | None = Field(default=None, gt=0, le=1)
    detector_imgsz: int | None = Field(default=None, ge=320)
    detector_iou: float | None = Field(default=None, gt=0, le=1)
    detector_device: str | None = None
    tracking_backend: str | None = None
    tracker_config: str | None = None
    defer_ocr: bool | None = None
    top_k_crops_per_track: int | None = Field(default=None, ge=1)
    enable_ocr: bool | None = None
    enable_qr: bool | None = None
    tiled_yolo: bool | None = None
    tile_size: int | None = Field(default=None, ge=64)
    tile_stride: int | None = Field(default=None, ge=32)
    max_tiles_per_frame: int | None = Field(default=None, ge=0)
    rail_roi_enabled: bool | None = None
    crop_phash_dedup: bool | None = None
    derive_qr_fields_when_missing: bool | None = None
    save_crops: bool = False


class EvaluationRequest(BaseModel):
    data_dir: str = str(DEFAULT_DATA_DIR)
    output_dir: str | None = None
    mode: str = "cpu_safe"
    sample_fps: float | None = Field(default=1.0, gt=0)
    max_frames: int = Field(default=0, ge=0)
    yolo_weights: str | None = None
    yolo_conf: float | None = Field(default=None, gt=0, le=1)
    detector_imgsz: int | None = Field(default=None, ge=320)
    detector_iou: float | None = Field(default=None, gt=0, le=1)
    detector_device: str | None = None
    tracking_backend: str | None = None
    tracker_config: str | None = None
    defer_ocr: bool | None = None
    top_k_crops_per_track: int | None = Field(default=None, ge=1)
    tiled_yolo: bool | None = None
    tile_size: int | None = Field(default=None, ge=64)
    tile_stride: int | None = Field(default=None, ge=32)
    max_tiles_per_frame: int | None = Field(default=None, ge=0)
    rail_roi_enabled: bool | None = None


class YoloDatasetRequest(BaseModel):
    data_dir: str = str(DEFAULT_DATA_DIR)
    output_dir: str
    val_fraction: float = Field(default=0.2, ge=0.0, le=0.8)
    seed: int = 42
    tiled: bool = False
    tile_size: int = Field(default=640, ge=64)
    tile_stride: int = Field(default=512, ge=32)
    min_box_visibility: float = Field(default=0.25, ge=0.01, le=1.0)
    centered_tiles_per_box: int = Field(default=3, ge=0, le=12)
    background_tiles_per_frame: int = Field(default=0, ge=0, le=32)
    propagate_frames: int = Field(default=0, ge=0, le=30)
    template_match_threshold: float = Field(default=0.42, ge=0.0, le=1.0)
    template_search_pad: int = Field(default=80, ge=0, le=512)
    template_min_std: float = Field(default=12.0, ge=0.0, le=255.0)
    template_backward_iou: float = Field(default=0.45, ge=0.0, le=1.0)
    template_motion_tolerance: float = Field(default=55.0, ge=0.0, le=512.0)
    template_edge_margin: int = Field(default=3, ge=0, le=32)
    template_min_edge_density: float = Field(default=0.035, ge=0.0, le=1.0)
    template_min_peak_margin: float = Field(default=0.04, ge=0.0, le=1.0)
    propagate_val: bool = False
    hash_split: bool = True


class YoloTrainRequest(BaseModel):
    data_yaml: str
    model: str = "yolo11n.pt"
    epochs: int = Field(default=150, ge=1)
    imgsz: int = Field(default=1280, ge=320)
    batch: int = Field(default=4, ge=1)
    device: str = "cpu"
    project: str | None = None
    name: str = "price_tag_yolo"


@app.post("/predict/path")
def predict_path(request: PathPredictionRequest) -> dict[str, object]:
    video_path = Path(request.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
    output_dir = Path(request.output_dir) if request.output_dir else new_run_dir(video_path.stem)
    result = run_pipeline(
        video_path=video_path,
        output_dir=output_dir,
        mode=request.mode,
        sample_fps=request.sample_fps,
        max_frames=request.max_frames,
        yolo_weights=request.yolo_weights,
        yolo_conf=request.yolo_conf,
        detector_imgsz=request.detector_imgsz,
        detector_iou=request.detector_iou,
        detector_device=request.detector_device,
        tracking_backend=request.tracking_backend,
        tracker_config=request.tracker_config,
        defer_ocr=request.defer_ocr,
        top_k_crops_per_track=request.top_k_crops_per_track,
        enable_ocr=request.enable_ocr,
        enable_qr=request.enable_qr,
        tiled_yolo=request.tiled_yolo,
        tile_size=request.tile_size,
        tile_stride=request.tile_stride,
        max_tiles_per_frame=request.max_tiles_per_frame,
        rail_roi_enabled=request.rail_roi_enabled,
        crop_phash_dedup=request.crop_phash_dedup,
        derive_qr_fields_when_missing=request.derive_qr_fields_when_missing,
        save_crops=request.save_crops,
    )
    return result


@app.post("/predict/video")
async def predict_video(
    file: UploadFile = UPLOAD_FILE_PARAM,
    mode: str = MODE_FORM_PARAM,
    sample_fps: float | None = SAMPLE_FPS_FORM_PARAM,
    max_frames: int = MAX_FRAMES_FORM_PARAM,
    yolo_weights: str | None = YOLO_WEIGHTS_FORM_PARAM,
    yolo_conf: float | None = YOLO_CONF_FORM_PARAM,
    detector_imgsz: int | None = DETECTOR_IMGSZ_FORM_PARAM,
    detector_iou: float | None = DETECTOR_IOU_FORM_PARAM,
    detector_device: str | None = DETECTOR_DEVICE_FORM_PARAM,
    tracking_backend: str | None = TRACKING_BACKEND_FORM_PARAM,
    tracker_config: str | None = TRACKER_CONFIG_FORM_PARAM,
    defer_ocr: bool | None = DEFER_OCR_FORM_PARAM,
    top_k_crops_per_track: int | None = TOP_K_CROPS_FORM_PARAM,
    enable_ocr: bool | None = ENABLE_OCR_FORM_PARAM,
    enable_qr: bool | None = ENABLE_QR_FORM_PARAM,
    tiled_yolo: bool | None = TILED_YOLO_FORM_PARAM,
    tile_size: int | None = TILE_SIZE_FORM_PARAM,
    tile_stride: int | None = TILE_STRIDE_FORM_PARAM,
    max_tiles_per_frame: int | None = MAX_TILES_FORM_PARAM,
    rail_roi_enabled: bool | None = RAIL_ROI_FORM_PARAM,
    crop_phash_dedup: bool | None = CROP_PHASH_DEDUP_FORM_PARAM,
    derive_qr_fields_when_missing: bool | None = DERIVE_QR_FIELDS_FORM_PARAM,
    save_crops: bool = SAVE_CROPS_FORM_PARAM,
) -> dict[str, object]:
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Upload an .mp4 video")
    run_dir = new_run_dir(Path(file.filename).stem)
    upload_path = run_dir / "input.mp4"
    run_dir.mkdir(parents=True, exist_ok=True)
    with upload_path.open("wb") as stream:
        shutil.copyfileobj(file.file, stream)
    return run_pipeline(
        video_path=upload_path,
        output_dir=run_dir,
        mode=mode,
        sample_fps=sample_fps,
        max_frames=max_frames,
        yolo_weights=yolo_weights,
        yolo_conf=yolo_conf,
        detector_imgsz=detector_imgsz,
        detector_iou=detector_iou,
        detector_device=detector_device,
        tracking_backend=tracking_backend,
        tracker_config=tracker_config,
        defer_ocr=defer_ocr,
        top_k_crops_per_track=top_k_crops_per_track,
        enable_ocr=enable_ocr,
        enable_qr=enable_qr,
        tiled_yolo=tiled_yolo,
        tile_size=tile_size,
        tile_stride=tile_stride,
        max_tiles_per_frame=max_tiles_per_frame,
        rail_roi_enabled=rail_roi_enabled,
        crop_phash_dedup=crop_phash_dedup,
        derive_qr_fields_when_missing=derive_qr_fields_when_missing,
        save_crops=save_crops,
    )


@app.post("/predict/image")
async def predict_image(
    file: UploadFile = UPLOAD_FILE_PARAM,
    mode: str = MODE_FORM_PARAM,
    yolo_weights: str | None = YOLO_WEIGHTS_FORM_PARAM,
    enable_ocr: bool | None = ENABLE_OCR_FORM_PARAM,
    enable_qr: bool | None = ENABLE_QR_FORM_PARAM,
    save_crops: bool = SAVE_CROPS_FORM_PARAM,
) -> dict[str, object]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Upload an image file")
    suffix = Path(file.filename).suffix.lower()
    content_type = (file.content_type or "").lower()
    if not is_supported_image_upload(suffix, content_type):
        raise HTTPException(
            status_code=400,
            detail="Upload an image: .jpg, .jpeg, .png, .bmp, .webp, .tif, .tiff",
        )

    run_dir = new_run_dir(Path(file.filename).stem)
    upload_path = run_dir / f"input{suffix or '.jpg'}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with upload_path.open("wb") as stream:
        shutil.copyfileobj(file.file, stream)

    return run_image_pipeline(
        image_path=upload_path,
        output_dir=run_dir,
        mode=mode,
        yolo_weights=yolo_weights,
        enable_ocr=enable_ocr,
        enable_qr=enable_qr,
        save_crops=save_crops,
    )


@app.post("/evaluate/public")
def evaluate_public(request: EvaluationRequest) -> dict[str, object]:
    output_dir = Path(request.output_dir) if request.output_dir else new_run_dir("eval_public")
    config = PipelineConfig.from_mode(
        request.mode,
        sample_fps=request.sample_fps,
        max_frames=request.max_frames,
        yolo_weights=request.yolo_weights,
        yolo_conf=request.yolo_conf,
        detector_imgsz=request.detector_imgsz,
        detector_iou=request.detector_iou,
        detector_device=request.detector_device,
        tracking_backend=request.tracking_backend,
        tracker_config=request.tracker_config,
        defer_ocr=request.defer_ocr,
        top_k_crops_per_track=request.top_k_crops_per_track,
        tiled_yolo=request.tiled_yolo,
        tile_size=request.tile_size,
        tile_stride=request.tile_stride,
        max_tiles_per_frame=request.max_tiles_per_frame,
        rail_roi_enabled=request.rail_roi_enabled,
    )
    return run_public_evaluation(Path(request.data_dir), output_dir, config)


@app.post("/dataset/yolo")
def create_yolo_dataset(request: YoloDatasetRequest) -> dict[str, object]:
    result = build_yolo_dataset(
        data_dir=Path(request.data_dir),
        output_dir=Path(request.output_dir),
        val_fraction=request.val_fraction,
        seed=request.seed,
        tiled=request.tiled,
        tile_size=request.tile_size,
        tile_stride=request.tile_stride,
        min_box_visibility=request.min_box_visibility,
        centered_tiles_per_box=request.centered_tiles_per_box,
        background_tiles_per_frame=request.background_tiles_per_frame,
        propagate_frames=request.propagate_frames,
        template_match_threshold=request.template_match_threshold,
        template_search_pad=request.template_search_pad,
        template_min_std=request.template_min_std,
        template_backward_iou=request.template_backward_iou,
        template_motion_tolerance=request.template_motion_tolerance,
        template_edge_margin=request.template_edge_margin,
        template_min_edge_density=request.template_min_edge_density,
        template_min_peak_margin=request.template_min_peak_margin,
        propagate_val=request.propagate_val,
        hash_split=request.hash_split,
    )
    return result.__dict__


@app.post("/train/yolo")
def train_yolo(request: YoloTrainRequest) -> dict[str, object]:
    try:
        result = train_yolo_detector(
            data_yaml=Path(request.data_yaml),
            model=request.model,
            epochs=request.epochs,
            imgsz=request.imgsz,
            batch=request.batch,
            device=request.device,
            project=request.project,
            name=request.name,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "started_or_completed", "result": str(result)}


@app.get("/download/{run_id}/{filename}")
def download(run_id: str, filename: str) -> FileResponse:
    path = (WORK_DIR / run_id / filename).resolve()
    if not is_relative_to(path, WORK_DIR.resolve()) or not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


def run_pipeline(
    video_path: Path,
    output_dir: Path,
    mode: str,
    sample_fps: float | None,
    max_frames: int,
    yolo_weights: str | None,
    yolo_conf: float | None,
    detector_imgsz: int | None,
    detector_iou: float | None,
    detector_device: str | None,
    tracking_backend: str | None,
    tracker_config: str | None,
    defer_ocr: bool | None,
    top_k_crops_per_track: int | None,
    enable_ocr: bool | None,
    enable_qr: bool | None,
    tiled_yolo: bool | None,
    tile_size: int | None,
    tile_stride: int | None,
    max_tiles_per_frame: int | None,
    rail_roi_enabled: bool | None,
    crop_phash_dedup: bool | None,
    derive_qr_fields_when_missing: bool | None,
    save_crops: bool,
) -> dict[str, object]:
    config = PipelineConfig.from_mode(
        mode,
        sample_fps=sample_fps,
        max_frames=max_frames,
        yolo_weights=yolo_weights,
        yolo_conf=yolo_conf,
        detector_imgsz=detector_imgsz,
        detector_iou=detector_iou,
        detector_device=detector_device,
        tracking_backend=tracking_backend,
        tracker_config=tracker_config,
        defer_ocr=defer_ocr,
        top_k_crops_per_track=top_k_crops_per_track,
        enable_ocr=enable_ocr,
        enable_qr=enable_qr,
        tiled_yolo=tiled_yolo,
        tile_size=tile_size,
        tile_stride=tile_stride,
        max_tiles_per_frame=max_tiles_per_frame,
        rail_roi_enabled=rail_roi_enabled,
        crop_phash_dedup=crop_phash_dedup,
        derive_qr_fields_when_missing=derive_qr_fields_when_missing,
        save_crops=save_crops,
    )
    try:
        result = RetailShelfPipeline(config).run_video(video_path, output_dir)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response: dict[str, object] = result.__dict__.copy()
    response["csv_summary"] = summarize_csv(Path(result.output_csv))
    response["download"] = download_hint(Path(result.output_csv))
    if result.debug_json:
        response["debug_download"] = download_hint(Path(result.debug_json))
    if result.debug_tracks_json:
        response["debug_tracks_download"] = download_hint(Path(result.debug_tracks_json))
    if result.debug_detections_json:
        response["debug_detections_download"] = download_hint(Path(result.debug_detections_json))
    return response


def run_image_pipeline(
    image_path: Path,
    output_dir: Path,
    mode: str,
    yolo_weights: str | None,
    enable_ocr: bool | None,
    enable_qr: bool | None,
    save_crops: bool,
) -> dict[str, object]:
    config = PipelineConfig.from_mode(
        mode,
        yolo_weights=yolo_weights,
        enable_ocr=enable_ocr,
        enable_qr=enable_qr,
        save_crops=save_crops,
    )
    try:
        result = RetailShelfPipeline(config).run_image(image_path, output_dir)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response: dict[str, object] = result.__dict__.copy()
    response["csv_summary"] = summarize_csv(Path(result.output_csv))
    response["download"] = download_hint(Path(result.output_csv))
    if result.debug_json:
        response["debug_download"] = download_hint(Path(result.debug_json))
    return response


def is_supported_image_upload(suffix: str, content_type: str) -> bool:
    return suffix in SUPPORTED_IMAGE_SUFFIXES or content_type.startswith("image/")


def new_run_dir(prefix: str) -> Path:
    safe_prefix = "".join(char if char.isalnum() or char in "._-" else "_" for char in prefix)
    return WORK_DIR / f"{safe_prefix}_{uuid4().hex[:10]}"


def download_hint(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(WORK_DIR.resolve())
    except ValueError:
        return str(path)
    parts = relative.parts
    if len(parts) < 2:
        return str(path)
    return f"/download/{parts[0]}/{'/'.join(parts[1:])}"


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def run() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("ml.main:app", host=host, port=port)


if __name__ == "__main__":
    run()
