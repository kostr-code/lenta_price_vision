"""
pipeline/qr.py — QR and linear barcode decoding utilities.

Both decode_qr() and decode_barcode_linear() accept an optional
``enhance_steps`` list (functions from pipeline.sr) applied before the
decode attempt. The raw image is always tried first; enhanced version is
tried additionally when steps are provided.

WeChat (with built-in SRQI super-resolution) is supported via the
``wechat`` parameter — pass the result of load_wechat() once at startup.

Usage:
  from pipeline.qr import load_wechat, decode_qr, decode_barcode_linear
  from pipeline.sr import upscale_lanczos_x2, enhance_clahe

  wechat = load_wechat()

  # QR in top-right corner — WeChat handles SRQI internally
  texts = decode_qr(cut_qr_subcrop(crop), wechat=wechat)

  # Barcode strip — try with CLAHE contrast boost
  codes = decode_barcode_linear(barcode_strip, enhance_steps=[enhance_clahe])
"""
from __future__ import annotations

import pathlib
import urllib.request
from typing import TYPE_CHECKING
from urllib.parse import parse_qs

import cv2
import numpy as np

if TYPE_CHECKING:
    from pipeline.sr import EnhanceFn

try:
    import zxingcpp
    _HAS_ZXING = True
except ImportError:
    _HAS_ZXING = False

try:
    from pyzbar.pyzbar import decode as _pyzbar_decode
    _HAS_PYZBAR = True
except ImportError:
    _HAS_PYZBAR = False

# QR payload field mapping (all known aliases -> CSV column name)
QR_TO_CSV: dict[str, str] = {
    # short aliases
    "b":    "qr_code_barcode",
    "p1":   "price1_qr",
    "p2":   "price2_qr",
    "p3":   "price3_qr",
    "p4":   "price4_qr",
    "wL1C": "wholesale_level_1_count",
    "wL1P": "wholesale_level_1_price",
    "wL2C": "wholesale_level_2_count",
    "wL2P": "wholesale_level_2_price",
    "aP":   "action_price_qr",
    "aC":   "action_code_qr",
    # longer variants used by some QR generators
    "barcode":              "qr_code_barcode",
    "price1":               "price1_qr",
    "price2":               "price2_qr",
    "price3":               "price3_qr",
    "price4":               "price4_qr",
    "wholesaleLevel1Count": "wholesale_level_1_count",
    "wholesaleLevel1Price": "wholesale_level_1_price",
    "wholesaleLevel2Count": "wholesale_level_2_count",
    "wholesaleLevel2Price": "wholesale_level_2_price",
    "actionPrice":          "action_price_qr",
    "actionCode":           "action_code_qr",
}

_WECHAT_MODEL_DIR = pathlib.Path("wechat_models")
_WECHAT_FILES = ["detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel"]
_WECHAT_BASE = "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/"


def load_wechat():
    """Download WeChat QR models if needed and return a detector instance."""
    _WECHAT_MODEL_DIR.mkdir(exist_ok=True)
    for fname in _WECHAT_FILES:
        p = _WECHAT_MODEL_DIR / fname
        if not p.exists():
            print(f"  ↓ WeChat model: {fname}")
            urllib.request.urlretrieve(_WECHAT_BASE + fname, p)
    try:
        det = cv2.wechat_qrcode_WeChatQRCode(
            str(_WECHAT_MODEL_DIR / "detect.prototxt"),
            str(_WECHAT_MODEL_DIR / "detect.caffemodel"),
            str(_WECHAT_MODEL_DIR / "sr.prototxt"),
            str(_WECHAT_MODEL_DIR / "sr.caffemodel"),
        )
        print("  WeChat: NN-режим (детекция + SRQI)")
    except Exception as e:
        det = cv2.wechat_qrcode_WeChatQRCode()
        print(f"  WeChat: базовый режим ({e})")
    return det


def _try_decode_one(
    gray: np.ndarray,
    img_bgr: np.ndarray,
    wechat,
    found: set[str],
) -> None:
    """Run all available decoders on a single image version, adding to found."""
    if _HAS_PYZBAR:
        for r in _pyzbar_decode(gray):
            if r.data:
                found.add(r.data.decode())
    if _HAS_ZXING:
        for r in zxingcpp.read_barcodes(gray):
            if r.text:
                found.add(r.text)
    if wechat is not None:
        data, _ = wechat.detectAndDecode(img_bgr)
        for d in data:
            if d:
                found.add(d)


def decode_qr(
    img_bgr: np.ndarray,
    wechat=None,
    enhance_steps: "list[EnhanceFn]" = (),
) -> list[str]:
    """
    Decode QR codes using all available decoders.

    Tries raw image first; if ``enhance_steps`` given, applies them and tries again.
    WeChat (passed as ``wechat``) uses its built-in SRQI — no external upscaling needed.

    Returns unique decoded strings (sorted).
    """
    found: set[str] = set()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _try_decode_one(gray, img_bgr, wechat, found)

    if enhance_steps and not found:
        from pipeline.sr import apply_enhance_steps
        enhanced = apply_enhance_steps(img_bgr, list(enhance_steps))
        gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        _try_decode_one(gray_enh, enhanced, wechat, found)

    return sorted(found)


def decode_barcode_linear(
    img_bgr: np.ndarray,
    enhance_steps: "list[EnhanceFn]" = (),
) -> list[str]:
    """
    Decode only linear barcodes (EAN-13 etc.) — no WeChat, no QR.

    Faster than decode_qr for bottom-strip barcode regions.
    If ``enhance_steps`` given, applies them and retries on failure.
    """
    found: set[str] = set()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if _HAS_PYZBAR:
        for r in _pyzbar_decode(gray):
            if r.data:
                found.add(r.data.decode())
    if _HAS_ZXING:
        for r in zxingcpp.read_barcodes(gray):
            if r.text:
                found.add(r.text)

    if enhance_steps and not found:
        from pipeline.sr import apply_enhance_steps
        enhanced = apply_enhance_steps(img_bgr, list(enhance_steps))
        gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        if _HAS_PYZBAR:
            for r in _pyzbar_decode(gray_enh):
                if r.data:
                    found.add(r.data.decode())
        if _HAS_ZXING:
            for r in zxingcpp.read_barcodes(gray_enh):
                if r.text:
                    found.add(r.text)

    return sorted(found)


def parse_qr_payload(text: str) -> dict[str, str]:
    """
    Parse QR payload string into CSV field dict.

    Example: "b=4670025474665&p1=252.63&p2=239.99" ->
             {"qr_code_barcode": "4670025474665", "price1_qr": "252.63", ...}
    """
    try:
        parsed = parse_qs(text, keep_blank_values=False)
    except Exception:
        return {}
    result: dict[str, str] = {}
    for short, csv_col in QR_TO_CSV.items():
        if short in parsed:
            result[csv_col] = parsed[short][0]
    return result


def cut_qr_subcrop(crop: np.ndarray) -> np.ndarray | None:
    """
    Extract the QR code sub-region (upper-right corner: x>60%, y<42%).

    QR is always in the top-right on Lenta price tags.
    """
    h, w = crop.shape[:2]
    sub = crop[0 : int(h * 0.42), int(w * 0.60) :]
    return sub if sub.size > 0 else None


def cut_barcode_strip(crop: np.ndarray, y_start: float = 0.70) -> np.ndarray | None:
    """
    Extract the linear barcode strip (lower portion of crop).

    y_start: fraction from top where barcode region begins (default 0.70 = bottom 30%).
    """
    h = crop.shape[0]
    sub = crop[int(h * y_start) :, :]
    return sub if sub.size > 0 else None
