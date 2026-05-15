"""
pipeline/ocr.py — PaddleOCR wrapper, zone splitting, and artifact masking.

Key functions:
  load_ocr()              — load PaddleOCR model
  ocr_to_lines()          — run OCR on single image → List[OCRLine]
  ocr_zoned()             — run OCR on 6 zones + deduplicate → List[OCRLine]
  suppress_code_artifacts() — mask QR/barcode regions before OCR
  enhance_crop()          — CLAHE + unsharp mask
  split_price_tag_zones() — split crop into 6 overlapping zones
  annotate_ocr()          — draw boxes on image (for visualization)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class OCRLine:
    text: str
    confidence: float = 0.0
    box: Any = None
    engine: str = "unknown"


# ── Image enhancement ─────────────────────────────────────────────────────────

def suppress_code_artifacts(image_bgr: np.ndarray) -> np.ndarray:
    """
    Mask dense QR/barcode regions with background color before OCR.

    QR/barcode modules create garbage tokens; this masks them conservatively,
    targeting right/lower regions only to leave price and name areas intact.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    if h < 40 or w < 40:
        return image_bgr

    out = image_bgr.copy()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dark = cv2.inRange(gray, 0, 120)
    edges = cv2.Canny(gray, 45, 160)
    texture = cv2.bitwise_or(dark, edges)
    texture = cv2.morphologyEx(
        texture, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )
    contours, _ = cv2.findContours(texture, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_code_boxes: list[tuple[int, int, int, int]] = []

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 6 or bh < 6:
            continue
        area_ratio = (bw * bh) / max(1.0, float(w * h))
        if not 0.002 <= area_ratio <= 0.45:
            continue
        roi_dark = dark[y: y + bh, x: x + bw]
        roi_edges = edges[y: y + bh, x: x + bw]
        dark_density = float((roi_dark > 0).mean())
        edge_density = float((roi_edges > 0).mean())
        aspect = bw / max(1, bh)

        right_or_lower = x > 0.42 * w or y > 0.48 * h
        qr_like = 0.55 <= aspect <= 1.85 and dark_density >= 0.16 and edge_density >= 0.045
        barcode_like = (
            (aspect >= 2.8 or aspect <= 0.36)
            and dark_density >= 0.11
            and edge_density >= 0.040
            and (y > 0.45 * h or x > 0.50 * w)
        )

        if right_or_lower and qr_like and area_ratio >= 0.002:
            small_code_boxes.append((x, y, x + bw, y + bh))

        if not right_or_lower or not (qr_like or barcode_like):
            continue

        pad = max(2, int(0.03 * max(bw, bh)))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
        bg = np.percentile(out.reshape(-1, 3), 88, axis=0).astype(np.uint8).tolist()
        cv2.rectangle(out, (x1, y1), (x2, y2), bg, thickness=-1)

    # If 6+ small code boxes cluster together → mask the whole region
    if len(small_code_boxes) >= 6:
        x1 = min(box[0] for box in small_code_boxes)
        y1 = min(box[1] for box in small_code_boxes)
        x2 = max(box[2] for box in small_code_boxes)
        y2 = max(box[3] for box in small_code_boxes)
        bw, bh = x2 - x1, y2 - y1
        area_ratio = (bw * bh) / max(1.0, float(w * h))
        aspect = bw / max(1, bh)
        if (0.01 <= area_ratio <= 0.45 and 0.45 <= aspect <= 2.2
                and (x1 > 0.42 * w or y1 > 0.45 * h)):
            pad = max(2, int(0.025 * max(bw, bh)))
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
            bg = np.percentile(out.reshape(-1, 3), 88, axis=0).astype(np.uint8).tolist()
            cv2.rectangle(out, (x1, y1), (x2, y2), bg, thickness=-1)

    return out


def enhance_crop(
    image_bgr: np.ndarray,
    max_side: int = 1600,
    suppress_artifacts: bool = True,
) -> np.ndarray:
    """CLAHE + unsharp mask; optionally upscale small crops and mask QR artifacts."""
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    if suppress_artifacts:
        image_bgr = suppress_code_artifacts(image_bgr)
    h, w = image_bgr.shape[:2]
    scale = 1.0
    if max(h, w) < 650:
        scale = min(3.0, 850.0 / max(h, w))
    elif max(h, w) > max_side:
        scale = max_side / float(max(h, w))
    if abs(scale - 1.0) > 1e-3:
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        image_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=interp)
    # CLAHE on L channel
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    # Unsharp mask
    blur = cv2.GaussianBlur(out, (0, 0), 1.0)
    return cv2.addWeighted(out, 1.45, blur, -0.45, 0)


# ── Zone splitting ────────────────────────────────────────────────────────────

def _crop_zone(image_bgr: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = box
    ix1 = max(0, min(w - 1, int(round(x1 * w))))
    iy1 = max(0, min(h - 1, int(round(y1 * h))))
    ix2 = max(0, min(w, int(round(x2 * w))))
    iy2 = max(0, min(h, int(round(y2 * h))))
    if ix2 <= ix1 or iy2 <= iy1:
        return image_bgr[:0, :0].copy()
    return image_bgr[iy1:iy2, ix1:ix2].copy()


def split_price_tag_zones(image_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """
    Split a price tag crop into 6 overlapping zones for zonal OCR.

    Zones:
      full         — entire crop
      product_top  — top 62% (product name + QR area)
      price_left   — left 58%, middle height (prices)
      price_right  — right 58%, middle height (prices)
      lower_codes  — bottom 45% (barcode, date, code, SKU)
      center       — center 76%×76%
    """
    if image_bgr is None or image_bgr.size == 0:
        return []
    h, w = image_bgr.shape[:2]
    if h < 24 or w < 24:
        return [("full", image_bgr)]

    zones = [
        ("full",         image_bgr),
        ("product_top",  _crop_zone(image_bgr, (0.00, 0.00, 1.00, 0.62))),
        ("price_left",   _crop_zone(image_bgr, (0.00, 0.18, 0.58, 0.92))),
        ("price_right",  _crop_zone(image_bgr, (0.42, 0.18, 1.00, 0.92))),
        ("lower_codes",  _crop_zone(image_bgr, (0.00, 0.55, 1.00, 1.00))),
        ("center",       _crop_zone(image_bgr, (0.12, 0.12, 0.88, 0.88))),
    ]

    out: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[int, int]] = set()
    for name, crop in zones:
        if crop is None or crop.size == 0:
            continue
        ch, cw = crop.shape[:2]
        if ch < 18 or cw < 18:
            continue
        key = (ch, cw)
        if name != "full" and key in seen:
            continue
        out.append((name, crop))
        seen.add(key)
    return out


# ── OCR execution ─────────────────────────────────────────────────────────────

def load_ocr(use_gpu: bool = False, lang: str = "ru"):
    """Load PaddleOCR model."""
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False)


def _paddle_result_to_lines(result, engine_label: str = "paddleocr") -> list[OCRLine]:
    """Convert raw PaddleOCR output to List[OCRLine]."""
    lines: list[OCRLine] = []
    if not result:
        return lines
    # PaddleOCR 3.x: list of dicts with rec_texts/rec_scores/rec_polys
    if isinstance(result, list) and result and hasattr(result[0], "get"):
        for page in result:
            texts = list(page.get("rec_texts") or [])
            scores = list(page.get("rec_scores") or [])
            boxes = list(page.get("rec_polys") or page.get("rec_boxes") or [])
            for i, text in enumerate(texts):
                text = re.sub(r"\s+", " ", str(text).strip())
                if not text:
                    continue
                conf = float(scores[i]) if i < len(scores) else 0.0
                box = boxes[i] if i < len(boxes) else None
                lines.append(OCRLine(text=text, confidence=conf, box=box, engine=engine_label))
        return lines
    # PaddleOCR 2.x: [[box, [text, conf]], ...]
    candidates = result
    if len(result) == 1 and isinstance(result[0], list) and result[0] and isinstance(result[0][0], (list, tuple)):
        candidates = result[0]
    for item in candidates:
        try:
            box = item[0]
            text = item[1][0]
            conf = float(item[1][1])
        except Exception:
            continue
        text = re.sub(r"\s+", " ", str(text).strip())
        if text:
            lines.append(OCRLine(text=text, confidence=conf, box=box, engine=engine_label))
    return lines


def ocr_to_lines(ocr_model, image_bgr: np.ndarray) -> list[OCRLine]:
    """Run PaddleOCR on a single image, return List[OCRLine]."""
    enhanced = enhance_crop(image_bgr)
    try:
        result = ocr_model.ocr(enhanced, cls=True)
    except TypeError:
        result = ocr_model.ocr(enhanced)
    return _paddle_result_to_lines(result)


def _dedupe_lines(lines: list[OCRLine]) -> list[OCRLine]:
    """Remove near-duplicate lines, prefer higher confidence."""
    unique: list[OCRLine] = []
    seen: set[str] = set()
    for line in sorted(lines, key=lambda l: l.confidence, reverse=True):
        key = re.sub(r"\W+", "", line.text.lower())
        if key and key not in seen:
            unique.append(line)
            seen.add(key)
    return unique


def ocr_zoned(ocr_model, image_bgr: np.ndarray) -> list[OCRLine]:
    """
    Run OCR on 6 overlapping zones + deduplicate.

    Catches tokens that full-crop OCR misses due to QR/barcode texture domination.
    """
    all_lines: list[OCRLine] = []
    for zone_name, zone_crop in split_price_tag_zones(image_bgr):
        zone_lines = ocr_to_lines(ocr_model, zone_crop)
        weight = 1.0 if zone_name == "full" else 0.97
        for line in zone_lines:
            all_lines.append(OCRLine(
                text=line.text,
                confidence=line.confidence * weight,
                box=line.box,
                engine=f"{line.engine}|zone:{zone_name}",
            ))
    return _dedupe_lines(all_lines)


# ── Visualization helpers (kept for everything.py) ────────────────────────────

def extract_text_zones(
    ocr_result,
    h: int,
    w: int,
    min_conf: float = 0.35,
) -> dict[str, list[tuple[str, float, float]]]:
    """
    Group raw PaddleOCR result into top/mid/bottom zones.

    Returns: {"top": [(text, conf, y_rel), ...], "mid": [...], "bottom": [...]}
    """
    zones: dict[str, list] = {"top": [], "mid": [], "bottom": []}
    if not ocr_result or not ocr_result[0]:
        return zones
    for line in ocr_result[0]:
        if line is None:
            continue
        pts, (text, conf) = line
        if conf < min_conf:
            continue
        y_center = np.mean([p[1] for p in pts]) / h
        zone = "top" if y_center < 0.35 else ("mid" if y_center < 0.65 else "bottom")
        zones[zone].append((text, round(conf, 3), round(y_center, 3)))
    for z in zones:
        zones[z].sort(key=lambda t: t[2])
    return zones


def annotate_ocr(img: np.ndarray, ocr_result) -> np.ndarray:
    """Draw OCR bounding boxes and text labels on a copy of the image."""
    out = img.copy()
    if not ocr_result or not ocr_result[0]:
        return out
    for line in ocr_result[0]:
        if line is None:
            continue
        pts, (text, conf) = line
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 0), 1)
        x0, y0 = pts[0]
        label = f"{text[:18]}({conf:.2f})"
        cv2.putText(out, label, (int(x0), max(int(y0) - 3, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2)
        cv2.putText(out, label, (int(x0), max(int(y0) - 3, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 200), 1)
    return out


def annotate_lines(img: np.ndarray, lines: list[OCRLine]) -> np.ndarray:
    """Draw OCRLine boxes on image (works with ocr_to_lines / ocr_zoned output)."""
    out = img.copy()
    for line in lines:
        if line.box is None:
            continue
        pts = np.array(line.box, dtype=np.int32)
        cv2.polylines(out, [pts], True, (0, 200, 0), 1)
        x0, y0 = pts[0]
        label = f"{line.text[:16]}({line.confidence:.2f})"
        cv2.putText(out, label, (int(x0), max(int(y0) - 3, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
        cv2.putText(out, label, (int(x0), max(int(y0) - 3, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 180), 1)
    return out


def zones_to_text(zones: dict) -> dict[str, str]:
    """Flatten extract_text_zones output to plain strings."""
    return {z: " | ".join(t for t, _, _ in lines) for z, lines in zones.items()}


def sharpen(img: np.ndarray) -> np.ndarray:
    """Light unsharp mask."""
    blur = cv2.GaussianBlur(img, (0, 0), 1.5)
    return cv2.addWeighted(img, 1.8, blur, -0.8, 0)
