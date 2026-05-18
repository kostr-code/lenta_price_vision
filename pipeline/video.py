"""
pipeline/video.py — Video I/O and frame extraction utilities.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

W_ORIG = 3840  # original video width before CCW rotation


def load_df(path: str) -> pd.DataFrame:
    """Load labeled CSV, normalizing bbox coordinates (comma → dot decimal)."""
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()
    for col in ["x_min", "y_min", "x_max", "y_max"]:
        if col in df.columns:
            df[col] = df[col].str.replace(",", ".").astype(float)
    if "frame_timestamp" in df.columns:
        df["frame_timestamp"] = pd.to_numeric(df["frame_timestamp"], errors="coerce")
    return df


def rotate_frame(raw: np.ndarray) -> np.ndarray:
    """Apply 90° CCW rotation (standard for Lenta robot camera)."""
    return cv2.rotate(raw, cv2.ROTATE_90_COUNTERCLOCKWISE)


def find_best_frame(
    video_path: str,
    ts_ms: float,
    n: int = 20,
) -> tuple[np.ndarray | None, float]:
    """
    Find the sharpest frame in a ±n window around ts_ms.

    Returns (rotated_frame, laplacian_variance).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = 1000.0 / fps
    best_var, best_frame = -1.0, None
    for i in range(-n, n + 1):
        cap.set(cv2.CAP_PROP_POS_MSEC, ts_ms + i * step)
        ok, raw = cap.read()
        if not ok:
            continue
        rot = rotate_frame(raw)
        v = cv2.Laplacian(cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if v > best_var:
            best_var, best_frame = v, rot.copy()
    cap.release()
    return best_frame, best_var


def cut_crop_from_row(frame: np.ndarray, row: pd.Series | dict) -> np.ndarray | None:
    """
    Cut price-tag crop from a CCW-rotated frame using labeled bbox coordinates.

    The CSV stores bbox in the original (pre-rotation) coordinate space,
    so we apply the CCW transform: new_x = y, new_y = W_ORIG - x.
    """
    bx1 = int(float(row["y_min"]))
    by1 = int(W_ORIG - 1 - float(row["x_max"]))
    bx2 = int(float(row["y_max"]))
    by2 = int(W_ORIG - 1 - float(row["x_min"]))
    fh, fw = frame.shape[:2]
    crop = frame[max(0, by1) : min(fh, by2), max(0, bx1) : min(fw, bx2)]
    return crop if crop.size > 0 else None


def cut_crop_bbox(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray | None:
    """Cut a crop directly by bbox (for YOLO crops already in rotated space)."""
    fh, fw = frame.shape[:2]
    crop = frame[max(0, y1) : min(fh, y2), max(0, x1) : min(fw, x2)]
    return crop if crop.size > 0 else None


def cut_region(
    crop: np.ndarray,
    x1_frac: float,
    y1_frac: float,
    x2_frac: float,
    y2_frac: float,
) -> np.ndarray | None:
    """
    Cut a sub-region from a crop using fractional [0,1] coordinates.

    Usage examples:
        barcode strip  → cut_region(crop, 0.20, 0.72, 0.90, 0.90)
        QR sub-crop    → cut_region(crop, 0.60, 0.00, 1.00, 0.42)
    """
    h, w = crop.shape[:2]
    x1 = max(0, int(x1_frac * w))
    y1 = max(0, int(y1_frac * h))
    x2 = min(w, int(x2_frac * w))
    y2 = min(h, int(y2_frac * h))
    region = crop[y1:y2, x1:x2]
    return region if region.size > 0 else None
