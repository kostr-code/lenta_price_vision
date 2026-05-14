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
    enable_ocr: bool | None = None
    enable_qr: bool | None = None
    save_crops: bool = False


class EvaluationRequest(BaseModel):
    data_dir: str = str(DEFAULT_DATA_DIR)
    output_dir: str | None = None
    mode: str = "cpu_safe"
    sample_fps: float | None = Field(default=1.0, gt=0)
    max_frames: int = Field(default=0, ge=0)
    yolo_weights: str | None = None


class YoloDatasetRequest(BaseModel):
    data_dir: str = str(DEFAULT_DATA_DIR)
    output_dir: str
    val_fraction: float = Field(default=0.2, ge=0.0, le=0.8)
    seed: int = 42


class YoloTrainRequest(BaseModel):
    data_yaml: str
    model: str = "yolo11n.pt"
    epochs: int = Field(default=150, ge=1)
    imgsz: int = Field(default=1280, ge=320)
    batch: int = Field(default=4, ge=1)
    device: str = "cpu"
    project: str = "runs/lenta"
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
        enable_ocr=request.enable_ocr,
        enable_qr=request.enable_qr,
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
    enable_ocr: bool | None = ENABLE_OCR_FORM_PARAM,
    enable_qr: bool | None = ENABLE_QR_FORM_PARAM,
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
        enable_ocr=enable_ocr,
        enable_qr=enable_qr,
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
    )
    return run_public_evaluation(Path(request.data_dir), output_dir, config)


@app.post("/dataset/yolo")
def create_yolo_dataset(request: YoloDatasetRequest) -> dict[str, object]:
    result = build_yolo_dataset(
        data_dir=Path(request.data_dir),
        output_dir=Path(request.output_dir),
        val_fraction=request.val_fraction,
        seed=request.seed,
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
    enable_ocr: bool | None,
    enable_qr: bool | None,
    save_crops: bool,
) -> dict[str, object]:
    config = PipelineConfig.from_mode(
        mode,
        sample_fps=sample_fps,
        max_frames=max_frames,
        yolo_weights=yolo_weights,
        enable_ocr=enable_ocr,
        enable_qr=enable_qr,
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
