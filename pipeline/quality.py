"""
pipeline/quality.py — Crop quality assessment for H.264-compressed price tag images.

Laplacian variance alone is insufficient: H.264 DCT blocks create sharp edges at
8-pixel boundaries that inflate the score even on degraded crops.

Three complementary metrics:
  - laplacian_var    : classic sharpness (fast, but fooled by compression blocks)
  - h264_artifact_score : boundary/interior gradient ratio at 8px grid
                         >1.5 -> strong block artifacts -> crop likely poor
  - hf_ratio         : FFT high-frequency energy fraction
                         <0.3 -> too blurry for reliable OCR

estimate_crop_quality() combines all three into a single [0, 1] score.
"""
from __future__ import annotations

import cv2
import numpy as np


def laplacian_var(gray: np.ndarray) -> float:
    """Standard Laplacian variance sharpness (fast baseline)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def h264_artifact_score(gray: np.ndarray, block: int = 8) -> float:
    """
    Detect H.264 block artifacts by comparing gradients at DCT block boundaries
    vs interior pixels.

    Returns ratio > 1.0 means boundaries are sharper than interior — artifact signal.
    Rule of thumb: > 1.5 -> significant artifacts -> crop is low quality.
    """
    g = gray.astype(np.float32)
    boundary_v = np.mean(np.abs(np.diff(g[:, block - 1 :: block], axis=1)))
    interior_v = np.mean(np.abs(np.diff(g, axis=1)))
    boundary_h = np.mean(np.abs(np.diff(g[block - 1 :: block, :], axis=0)))
    interior_h = np.mean(np.abs(np.diff(g, axis=0)))
    return float(
        (boundary_v / (interior_v + 1e-6) + boundary_h / (interior_h + 1e-6)) / 2
    )


def hf_ratio(gray: np.ndarray) -> float:
    """
    FFT high-frequency energy fraction.

    Low-frequency energy is concentrated in the central r=min(h,w)/4 circle.
    Ratio = energy outside that circle / total energy.

    Rule of thumb: < 0.3 -> image is too blurry for reliable OCR.
    """
    f = np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float32))))
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    mask = np.zeros((h, w), dtype=bool)
    mask[cy - r : cy + r, cx - r : cx + r] = True
    total = np.sum(f) + 1e-6
    return float(np.sum(f[~mask]) / total)


def estimate_crop_quality(crop_bgr: np.ndarray) -> float:
    """
    Composite crop quality score in [0, 1]. Higher = better.

    Combines:
      - laplacian_var  (sharpness proxy, normalized by a 4K-typical ceiling)
      - h264_artifact_score  (penalizes block artifacts)
      - hf_ratio  (rewards real high-frequency content)

    Returns a float suitable for ranking crops from the same price tag track.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    lap = laplacian_var(gray)
    h264 = h264_artifact_score(gray)
    hf = hf_ratio(gray)

    # Normalize laplacian: typical good crop ~500–3000, bad ~10–100
    lap_norm = min(lap / 2000.0, 1.0)

    # h264 penalty: score > 1.5 is bad; cap contribution at 1.0
    artifact_penalty = min(max(h264 - 1.0, 0.0) / 2.0, 1.0)

    # hf boost: 0.3 is threshold for "blurry"; 0.6+ is sharp
    hf_norm = min(hf / 0.6, 1.0)

    score = 0.4 * lap_norm + 0.4 * hf_norm - 0.2 * artifact_penalty
    return float(max(0.0, min(1.0, score)))
