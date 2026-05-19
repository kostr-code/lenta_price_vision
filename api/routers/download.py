"""api/routers/download.py — GET /download/{run_id}/{filename:path}."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

_MEDIA_TYPES = {
    ".csv": "text/csv",
    ".json": "application/json",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".mp4": "video/mp4",
}


@router.get("/download/{run_id}/{filename:path}")
async def download_file(run_id: str, filename: str) -> FileResponse:
    """Serve any file under runs_dir/run_id/ directly from disk."""
    from api.ml_app import settings as app_settings

    file_path = Path(app_settings.runs_dir) / run_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    media_type = _MEDIA_TYPES.get(file_path.suffix.lower(), "application/octet-stream")
    return FileResponse(path=str(file_path), filename=file_path.name, media_type=media_type)
