"""
pipeline/sr.py — Real-ESRGAN super-resolution utilities.
"""
from __future__ import annotations

import pathlib
import urllib.request

import numpy as np

ESRGAN_MODEL_DIR = pathlib.Path("models")
ESRGAN_MODEL_PTH = ESRGAN_MODEL_DIR / "RealESRGAN_x4plus.pth"
ESRGAN_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN"
    "/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)


def load_esrgan(gpu_id: int = 0):
    """
    Load Real-ESRGAN ×4 upsampler (downloads weights ~67MB if needed).

    Returns RealESRGANer instance or raises on failure.
    """
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    ESRGAN_MODEL_DIR.mkdir(exist_ok=True)
    if not ESRGAN_MODEL_PTH.exists():
        print(f"  ↓ Real-ESRGAN weights (~67 MB): {ESRGAN_MODEL_PTH}")
        urllib.request.urlretrieve(ESRGAN_MODEL_URL, ESRGAN_MODEL_PTH)

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32, scale=4,
    )
    return RealESRGANer(
        scale=4,
        model_path=str(ESRGAN_MODEL_PTH),
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=gpu_id,
    )


def upscale_safe(img: np.ndarray, upsampler, scale: int = 4) -> np.ndarray:
    """
    Upscale image with ESRGAN. Returns original on any error.
    """
    try:
        out, _ = upsampler.enhance(img, outscale=scale)
        return out
    except Exception as e:
        print(f"  [ESRGAN] upscale failed: {e}, using original")
        return img
