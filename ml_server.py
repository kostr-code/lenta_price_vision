"""ml_server.py — Entry point for the ML inference service (port 8000).

Run:
    uv run python ml_server.py

Or via justfile:
    just ml

Environment variables:
    ML_VLM_MODEL_ID      — HuggingFace model ID (default: Qwen/Qwen2.5-VL-3B-Instruct)
    ML_VLM_4BIT          — "true" for 4-bit quantization
    ML_VLM_DEVICE_MAP    — "cuda" / "auto" / "cuda:1"
    ML_WEIGHTS_INSIDE    — path to Stage 2 YOLO weights (optional)
    ML_WEIGHTS_STAGE1    — path to Stage 1 YOLO weights (optional)
    ML_DATA_DIR          — path to labeled data (default: data/Данные)
    ML_RUNS_DIR          — where to store run results (default: runs/api)
    ML_LOG_LEVEL         — INFO / DEBUG / WARNING
    HOST                 — bind host (default: 0.0.0.0)
    PORT                 — bind port (default: 8000)
"""

from __future__ import annotations

import os

import uvicorn

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "api.ml_app:app",
        host=host,
        port=port,
        reload=False,
        timeout_keep_alive=600,
        log_level="warning",  # uvicorn access logs — structlog handles the rest
    )
