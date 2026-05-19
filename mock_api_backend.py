from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendSettings(BaseSettings):
    """Runtime configuration for the backend API gateway."""

    ml_url: str = Field(default_factory=lambda: os.getenv("ML_URL", "http://ml:8000"))
    request_timeout_sec: float = 600.0
    health_timeout_sec: float = 5.0
    max_upload_mb: int = 256
    cors_origins: str = "*"

    model_config = SettingsConfigDict(
        env_prefix="BACKEND_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = BackendSettings()
UPLOAD_FILE = File(...)
FORM_MODE = Form("cpu_safe")
FORM_SAMPLE_FPS = Form(None)
FORM_MAX_FRAMES = Form(0)
FORM_YOLO_WEIGHTS = Form(None)
FORM_ENABLE_OCR = Form(None)
FORM_ENABLE_QR = Form(None)
FORM_SAVE_CROPS = Form(False)
DOWNLOAD_PATH_RE = re.compile(r"^/download/(?P<run_id>[^/]+)/(?P<filename>.+)$")


def parse_origins(value: str) -> list[str]:
    raw = value.strip()
    if not raw:
        return ["*"]
    if raw == "*":
        return ["*"]
    parts = [item.strip() for item in raw.split(",")]
    return [item for item in parts if item]


app = FastAPI(title="Price Tag Audit Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_origins(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@dataclass(frozen=True)
class DownloadParts:
    run_id: str
    filename: str


@app.get("/health")
async def health() -> dict[str, Any]:
    upstream = await check_ml_health()
    return {
        "status": "ok",
        "service": "backend",
        "ml": upstream,
    }


@app.get("/api/v1/schema")
async def schema() -> dict[str, Any]:
    response = await request_ml("GET", "/schema")
    return response.json()


@app.get("/api/v1/datasets")
async def datasets(data_dir: str | None = None) -> dict[str, Any]:
    params = {"data_dir": data_dir} if data_dir else None
    response = await request_ml("GET", "/datasets", params=params)
    return response.json()


@app.post("/api/v1/predict/video")
async def predict_video(
    file: UploadFile = UPLOAD_FILE,
    mode: str = FORM_MODE,
    sample_fps: float | None = FORM_SAMPLE_FPS,
    max_frames: int = FORM_MAX_FRAMES,
    yolo_weights: str | None = FORM_YOLO_WEIGHTS,
    enable_ocr: bool | None = FORM_ENABLE_OCR,
    enable_qr: bool | None = FORM_ENABLE_QR,
    save_crops: bool = FORM_SAVE_CROPS,
) -> dict[str, Any]:
    payload = await file.read()
    guard_upload_size(payload)

    files = {
        "file": (
            file.filename or "input.mp4",
            payload,
            file.content_type or "video/mp4",
        )
    }
    form_data: dict[str, str] = {
        "mode": mode,
        "max_frames": str(max_frames),
        "save_crops": bool_to_form(save_crops),
    }
    if sample_fps is not None:
        form_data["sample_fps"] = str(sample_fps)
    if yolo_weights:
        form_data["yolo_weights"] = yolo_weights
    if enable_ocr is not None:
        form_data["enable_ocr"] = bool_to_form(enable_ocr)
    if enable_qr is not None:
        form_data["enable_qr"] = bool_to_form(enable_qr)

    response = await request_ml("POST", "/predict/video", data=form_data, files=files)
    body = response.json()
    enrich_with_backend_downloads(body)
    return body


@app.post("/api/v1/predict/image")
async def predict_image(
    file: UploadFile = UPLOAD_FILE,
    mode: str = FORM_MODE,
    yolo_weights: str | None = FORM_YOLO_WEIGHTS,
    enable_ocr: bool | None = FORM_ENABLE_OCR,
    enable_qr: bool | None = FORM_ENABLE_QR,
    save_crops: bool = FORM_SAVE_CROPS,
) -> dict[str, Any]:
    payload = await file.read()
    guard_upload_size(payload)

    files = {
        "file": (
            file.filename or "input.jpg",
            payload,
            file.content_type or "image/jpeg",
        )
    }
    form_data: dict[str, str] = {
        "mode": mode,
        "save_crops": bool_to_form(save_crops),
    }
    if yolo_weights:
        form_data["yolo_weights"] = yolo_weights
    if enable_ocr is not None:
        form_data["enable_ocr"] = bool_to_form(enable_ocr)
    if enable_qr is not None:
        form_data["enable_qr"] = bool_to_form(enable_qr)

    response = await request_ml("POST", "/predict/image", data=form_data, files=files)
    body = response.json()
    enrich_with_backend_downloads(body)
    return body


@app.post("/api/v1/predict/path")
async def predict_path(request: PathPredictionRequest) -> dict[str, Any]:
    response = await request_ml("POST", "/predict/path", json=request.model_dump(exclude_none=True))
    body = response.json()
    enrich_with_backend_downloads(body)
    return body


@app.post("/api/v1/evaluate/public")
async def evaluate_public(request: dict[str, Any]) -> dict[str, Any]:
    response = await request_ml("POST", "/evaluate/public", json=request)
    return response.json()


@app.post("/api/v1/dataset/yolo")
async def create_yolo_dataset(request: dict[str, Any]) -> dict[str, Any]:
    response = await request_ml("POST", "/dataset/yolo", json=request)
    return response.json()


@app.post("/api/v1/train/yolo")
async def train_yolo(request: dict[str, Any]) -> dict[str, Any]:
    response = await request_ml("POST", "/train/yolo", json=request)
    return response.json()


@app.get("/api/v1/download/{run_id}/{filename:path}")
async def download(run_id: str, filename: str) -> Response:
    response = await request_ml(
        "GET",
        f"/download/{run_id}/{filename}",
        expect_json=False,
    )
    allowed_headers = [
        "content-type",
        "content-length",
        "content-disposition",
    ]
    headers = {
        key: value
        for key, value in response.headers.items()
        if key.lower() in allowed_headers
    }
    return Response(content=response.content, status_code=response.status_code, headers=headers)


def guard_upload_size(payload: bytes) -> None:
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(payload) <= max_bytes:
        return
    raise HTTPException(
        status_code=413,
        detail=f"File is too large ({len(payload)} bytes). Limit: {max_bytes} bytes",
    )


def bool_to_form(value: bool) -> str:
    return "true" if value else "false"


def ml_url(path: str) -> str:
    base = settings.ml_url.rstrip("/")
    suffix = path if path.startswith("/") else f"/{path}"
    return f"{base}{suffix}"


def parse_download_parts(value: str | None) -> DownloadParts | None:
    text = (value or "").strip()
    if not text:
        return None
    match = DOWNLOAD_PATH_RE.match(text)
    if not match:
        return None
    return DownloadParts(
        run_id=match.group("run_id"),
        filename=match.group("filename"),
    )


def enrich_with_backend_downloads(body: dict[str, Any]) -> None:
    download = parse_download_parts(str(body.get("download", "")))
    if download is not None:
        body["backend_download"] = f"/api/v1/download/{download.run_id}/{download.filename}"

    debug = parse_download_parts(str(body.get("debug_download", "")))
    if debug is not None:
        body["backend_debug_download"] = f"/api/v1/download/{debug.run_id}/{debug.filename}"


async def check_ml_health() -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=settings.health_timeout_sec) as client:
            response = await client.get(ml_url("/health"))
        if response.is_success:
            return {
                "reachable": True,
                "status_code": response.status_code,
            }
        return {
            "reachable": False,
            "status_code": response.status_code,
        }
    except httpx.RequestError as exc:
        return {
            "reachable": False,
            "error": str(exc),
        }


async def request_ml(
    method: str,
    path: str,
    *,
    expect_json: bool = True,
    **kwargs: Any,
) -> httpx.Response:
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_sec) as client:
            response = await client.request(method, ml_url(path), **kwargs)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"ML service is unavailable: {exc}") from exc

    if response.is_success:
        return response

    detail = extract_error_detail(response, expect_json)
    status_code = response.status_code if response.status_code < 500 else 502
    raise HTTPException(status_code=status_code, detail=detail)


def extract_error_detail(response: httpx.Response, expect_json: bool) -> str:
    if expect_json:
        try:
            body = response.json()
            if isinstance(body, dict) and body.get("detail"):
                return str(body["detail"])
            return str(body)
        except ValueError:
            pass
    text = response.text.strip()
    if text:
        return text[:1000]
    return f"ML service returned HTTP {response.status_code}"


def run() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()