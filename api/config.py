"""api/config.py — ML service configuration via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class MLSettings:
    """Runtime settings for the ML inference service.

    All fields can be overridden via environment variables with ML_ prefix:
      ML_VLM_MODEL_ID, ML_VLM_4BIT, ML_VLM_DEVICE_MAP
      ML_WEIGHTS_INSIDE, ML_WEIGHTS_STAGE1
      ML_DATA_DIR, ML_RUNS_DIR
      ML_QUALITY_THRESHOLD, ML_LOG_LEVEL
    """

    # VLM
    vlm_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vlm_4bit: bool = False
    vlm_device_map: str = "cuda"

    # YOLO weights (optional)
    weights_inside: str | None = None   # Stage 2 — sub-region detection
    weights_stage1: str | None = None   # Stage 1 — price tag detection (image mode)

    # Directories
    data_dir: str = "data/Данные"
    runs_dir: str = "runs/api"

    # Pipeline tuning
    quality_threshold: float = 0.2
    track_top_k: int = 3

    # Logging
    log_level: str = "INFO"


def load_settings() -> MLSettings:
    """Build MLSettings from environment variables."""

    def _bool(key: str, default: bool) -> bool:
        v = os.environ.get(key, "").strip().lower()
        if not v:
            return default
        return v in ("1", "true", "yes", "on")

    def _float(key: str, default: float) -> float:
        v = os.environ.get(key, "").strip()
        return float(v) if v else default

    def _int(key: str, default: int) -> int:
        v = os.environ.get(key, "").strip()
        return int(v) if v else default

    def _str(key: str, default: str) -> str:
        return os.environ.get(key, default).strip() or default

    def _opt(key: str) -> str | None:
        v = os.environ.get(key, "").strip()
        return v if v else None

    return MLSettings(
        vlm_model_id=_str("ML_VLM_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct"),
        vlm_4bit=_bool("ML_VLM_4BIT", False),
        vlm_device_map=_str("ML_VLM_DEVICE_MAP", "cuda"),
        weights_inside=_opt("ML_WEIGHTS_INSIDE"),
        weights_stage1=_opt("ML_WEIGHTS_STAGE1"),
        data_dir=_str("ML_DATA_DIR", "data/Данные"),
        runs_dir=_str("ML_RUNS_DIR", "runs/api"),
        quality_threshold=_float("ML_QUALITY_THRESHOLD", 0.2),
        track_top_k=_int("ML_TRACK_TOP_K", 3),
        log_level=_str("ML_LOG_LEVEL", "INFO"),
    )
