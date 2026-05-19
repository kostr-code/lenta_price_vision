"""
pipeline/sr.py — Image upscaling and enhancement utilities.

All functions are independent — nothing is called automatically by the pipeline.
Plug them as ``enhance_steps`` into ``decode_qr`` / ``decode_barcode_linear``,
or call ``apply_enhance_steps`` directly before OCR on any crop.

Step functions (pass as list items):
  upscale_lanczos(img, scale=2)   — fast CPU Lanczos resize
  upscale_lanczos_x2              — pre-bound ×2 variant
  upscale_lanczos_x4              — pre-bound ×4 variant
  upscale_esrgan_x2(img)          — RealESRGAN ×2, lazy-loaded
  upscale_esrgan_x4(img)          — RealESRGAN ×4, lazy-loaded
  enhance_clahe(img)              — CLAHE on luminance channel
  enhance_sharpen(img)            — Unsharp mask σ=1.5

Orchestrator:
  apply_enhance_steps(img, steps: list[EnhanceFn]) → np.ndarray
"""

from __future__ import annotations

import pathlib
import urllib.request
from typing import Callable

import cv2
import numpy as np

EnhanceFn = Callable[[np.ndarray], np.ndarray]

# ── Lanczos upscale ──


def upscale_lanczos(img: np.ndarray, scale: int = 2) -> np.ndarray:
    """Resize image by integer scale using Lanczos4 interpolation."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)


upscale_lanczos_x2: EnhanceFn = lambda img: upscale_lanczos(img, 2)
upscale_lanczos_x4: EnhanceFn = lambda img: upscale_lanczos(img, 4)

# ── CV2 enhancements ──


def enhance_clahe(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    CLAHE contrast enhancement on luminance channel (BGR or grayscale input).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    if img.ndim == 2:
        return clahe.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_sharpen(
    img: np.ndarray, sigma: float = 1.5, strength: float = 1.8
) -> np.ndarray:
    """Unsharp mask: sharpened = orig*strength + blur*(1-strength)."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, strength, blur, 1.0 - strength, 0)


# ── ESRGAN ──

_ESRGAN_MODEL_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"
_ESRGAN_PTH = _ESRGAN_MODEL_DIR / "RealESRGAN_x4plus.pth"
_ESRGAN_URL = (
    "https://github.com/xinntao/Real-ESRGAN"
    "/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)

_esrgan_cache: dict[int, object] = {}


def _load_esrgan(gpu_id: int = 0):
    """
    Lazy-load RealESRGAN ×4 upsampler (downloads ~67 MB on first call).
    """
    if gpu_id in _esrgan_cache:
        return _esrgan_cache[gpu_id]
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError:
        raise RuntimeError("realesrgan not installed — uv add realesrgan basicsr")
    _ESRGAN_MODEL_DIR.mkdir(exist_ok=True)
    if not _ESRGAN_PTH.exists():
        print(f"  ↓ Real-ESRGAN weights (~67 MB): {_ESRGAN_PTH}")
        urllib.request.urlretrieve(_ESRGAN_URL, _ESRGAN_PTH)
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )
    upsampler = RealESRGANer(
        scale=4,
        model_path=str(_ESRGAN_PTH),
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=gpu_id,
    )
    _esrgan_cache[gpu_id] = upsampler
    return upsampler


def upscale_esrgan_x4(img: np.ndarray, gpu_id: int = 0) -> np.ndarray:
    """
    RealESRGAN ×4 upscale. Falls back to Lanczos×4 on any error.
    """
    try:
        out, _ = _load_esrgan(gpu_id).enhance(img, outscale=4)
        return out
    except Exception as exc:
        print(f"  [ESRGAN] x4 failed ({exc}), fallback Lanczos")
        return upscale_lanczos(img, 4)


def upscale_esrgan_x2(img: np.ndarray, gpu_id: int = 0) -> np.ndarray:
    """
    RealESRGAN ×2 upscale (×4 model at outscale=2). Falls back to Lanczos×2.
    """
    try:
        out, _ = _load_esrgan(gpu_id).enhance(img, outscale=2)
        return out
    except Exception as exc:
        print(f"  [ESRGAN] x2 failed ({exc}), fallback Lanczos")
        return upscale_lanczos(img, 2)


# ── Orchestrator ──


def apply_enhance_steps(img: np.ndarray, steps: list[EnhanceFn]) -> np.ndarray:
    """
    Apply a list of enhancement functions sequentially.
    """
    for fn in steps:
        img = fn(img)
    return img
