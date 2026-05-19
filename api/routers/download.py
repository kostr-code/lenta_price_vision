"""api/routers/download.py — GET /download/{run_id}/{filename:path}."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/download/{run_id}/{filename:path}")
async def download_file(run_id: str, filename: str) -> FileResponse:
    """Serve a result file from runs_dir/run_id/filename."""
    from api.pipeline_bridge import save_run_files  # noqa: F401 (unused but ensures import chain)
    from api.run_store import get_run

    result = get_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    file_path = result.files.get(filename)
    if file_path is None or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    media_type = "text/csv" if filename.endswith(".csv") else "application/json"
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type=media_type,
    )
