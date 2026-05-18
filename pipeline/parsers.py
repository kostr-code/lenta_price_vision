"""
pipeline/parsers.py — Field extraction from OCR text and QR payloads.

Adapted from competitor analysis. Key function:
  parse_fields(lines, qr_payloads, crop_bgr) → Dict[str, str]

Returns all extractable fields from the 29-column CSV schema.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import cv2
import numpy as np

from .ocr import OCRLine
from .qr import parse_qr_payload, QR_TO_CSV

# ── Constants ─────────────────────────────────────────────────────────────────

ABSENT_VALUE = "нет"

OUTPUT_COLUMNS = [
    "filename", "product_name", "price_default", "price_card", "price_discount",
    "barcode", "discount_amount", "id_sku", "print_datetime", "code",
    "additional_info", "color", "special_symbols",
    "frame_timestamp", "x_min", "y_min", "x_max", "y_max",
    "qr_code_barcode", "price1_qr", "price2_qr", "price3_qr", "price4_qr",
    "wholesale_level_1_count", "wholesale_level_1_price",
    "wholesale_level_2_count", "wholesale_level_2_price",
    "action_price_qr", "action_code_qr",
]

KNOWN_INFO_WORDS = [
    "сухое", "полусухое", "полусладкое", "сладкое", "брют", "экстра",
    "удачная упаковка", "номер на весах", "цена за 1 кг", "цена за 100 г",
    "100г", "1кг", "1 кг",
]

# ── Regex patterns ────────────────────────────────────────────────────────────

PRICE_RE         = re.compile(r"(?<![\d./-])(\d{1,5})\s*[.,]\s*(\d{2})(?![\d./-])")
PRICE_SPACED_RE  = re.compile(r"(?<!\d)(\d{1,5})\s+(\d{2})(?!\d)")
PRICE_COMPACT_RE = re.compile(r"(?<!\d)(\d{3,7})(?!\d)")
DISCOUNT_RE      = re.compile(r"[-−–]?\s*(\d{1,3})\s*%")
DATE_RE          = re.compile(r"(\d{2}[./-]\d{2}[./-]\d{4}\s+\d{1,2}:\d{2})")
ZONE_RE          = re.compile(r"(\d{2}_\d{6}\s*-\s*\d{6})")
EAN_RE           = re.compile(r"(?<!\d)(\d{8,14})(?!\d)")
SKU_RE           = re.compile(r"(?<!\d)(\d{9,13})(?!\d)")
SPECIAL_RE       = re.compile(r"(?:^|\s)([ШШшКкЛл])(?:\s|$)")
CYRILLIC_RE      = re.compile(r"[А-Яа-яЁё]")
VOLUME_CTX_RE    = re.compile(r"(?:л|l|литр|мл|ml|кг|kg|гр|г)(?![а-яa-z])", re.I)
CURRENCY_CTX_RE  = re.compile(r"(?:руб|₽|коп)", re.I)


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def price_to_str(value: Any) -> str:
    try:
        return f"{float(str(value).replace(',', '.')):.2f}"
    except Exception:
        return str(value)


def ean13_is_valid(code: str) -> bool:
    code = re.sub(r"\D", "", str(code))
    if len(code) != 13:
        return False
    digits = [int(c) for c in code]
    checksum = (10 - ((sum(digits[:-1:2]) + 3 * sum(digits[1:-1:2])) % 10)) % 10
    return checksum == digits[-1]


def classify_color(crop_bgr: np.ndarray) -> str:
    """HSV-based color classification for Lenta price tags."""
    if crop_bgr is None or crop_bgr.size == 0:
        return ""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (sat > 45) & (val > 60)
    if mask.mean() < 0.01:
        return "white"
    hue = hsv[:, :, 0][mask]
    if len(hue) == 0:
        return "white"
    hist, bins = np.histogram(hue, bins=18, range=(0, 180))
    dominant = (bins[int(np.argmax(hist))] + bins[int(np.argmax(hist)) + 1]) / 2
    if dominant < 15 or dominant >= 165:
        return "red"
    if dominant < 38:
        return "yellow"
    if dominant < 90:
        return "green"
    if dominant < 135:
        return "blue"
    if dominant < 165:
        return "purple"
    return "red"


# ── Price extraction ──────────────────────────────────────────────────────────

def _looks_like_non_price_context(context: str) -> bool:
    return bool(re.search(r"(?:штрих|barcode|баркод|артикул|sku|id[_\s-]*sku|qr|код)", context, re.I))


def _unit_adjacent(text: str, start: int, end: int) -> bool:
    local = text[max(0, start - 4): min(len(text), end + 5)]
    return bool(VOLUME_CTX_RE.search(local))


def _add_price(prices: list[str], integer_part: str, cents: str, context: str) -> None:
    if not integer_part or not cents:
        return
    try:
        value = float(f"{int(integer_part)}.{cents[:2]}")
    except ValueError:
        return
    if value < 2.0 and not CURRENCY_CTX_RE.search(context):
        return
    if value > 999999:
        return
    if _looks_like_non_price_context(context):
        return
    s = f"{value:.2f}"
    if s not in prices:
        prices.append(s)


def _find_prices(text: str) -> list[str]:
    prices: list[str] = []
    text = normalize_text(text.replace(" ", " "))
    occupied: list[tuple[int, int]] = []

    for match in PRICE_RE.finditer(text):
        if _unit_adjacent(text, match.start(), match.end()):
            occupied.append(match.span())
            continue
        context = text[max(0, match.start() - 18): min(len(text), match.end() + 18)]
        _add_price(prices, match.group(1), match.group(2), context)
        occupied.append(match.span())

    # "129 99" or "129\n99" — OCR splits price into two tokens
    for match in PRICE_SPACED_RE.finditer(text):
        if any(not (match.end() <= a or match.start() >= b) for a, b in occupied):
            continue
        integer_part, cents = match.group(1), match.group(2)
        if len(integer_part) == 1 and int(integer_part) < 2:
            continue
        if _unit_adjacent(text, match.start(), match.end()):
            occupied.append(match.span())
            continue
        context = text[max(0, match.start() - 18): min(len(text), match.end() + 18)]
        before_count = len(prices)
        _add_price(prices, integer_part, cents, context)
        if len(prices) > before_count:
            occupied.append(match.span())

    # Compact: "12999" → 129.99, "378949" → 3789.49
    for match in PRICE_COMPACT_RE.finditer(text):
        if any(not (match.end() <= a or match.start() >= b) for a, b in occupied):
            continue
        raw = match.group(1)
        if len(raw) < 4 or len(raw) > 7:
            continue
        if raw.startswith("0"):
            continue
        prev_ch = text[match.start() - 1] if match.start() > 0 else ""
        next_ch = text[match.end()] if match.end() < len(text) else ""
        if prev_ch in ".,/:;-" or next_ch in ".,/:;-":
            continue
        if _unit_adjacent(text, match.start(), match.end()):
            continue
        context = text[max(0, match.start() - 18): min(len(text), match.end() + 18)]
        if re.search(r"\d{1,2}[./-]\d{1,2}|[:]", context):
            continue
        _add_price(prices, raw[:-2], raw[-2:], context)

    return prices


def _find_prices_from_lines(lines: list[OCRLine], full_text: str) -> list[str]:
    prices = _find_prices(full_text)
    # Adjacent line-pair: integer on one line, cents on next
    texts = [normalize_text(line.text) for line in lines if normalize_text(line.text)]
    for left, right in zip(texts, texts[1:]):
        left_c, right_c = normalize_text(left), normalize_text(right)
        if re.fullmatch(r"\d{1,5}", left_c) and re.fullmatch(r"\d{2}", right_c):
            _add_price(prices, left_c, right_c, f"{left_c} {right_c}")
    return prices


def _find_barcodes(text: str) -> list[str]:
    nums = [re.sub(r"\D", "", m.group(1)) for m in EAN_RE.finditer(text)]
    valid = [n for n in nums if ean13_is_valid(n)]
    if valid:
        return valid
    return []


# ── Product name ──────────────────────────────────────────────────────────────

def _candidate_product_lines(lines: list[OCRLine]) -> list[str]:
    bad = re.compile(r"(руб|коп|цена|скид|карте|штрих|артикул|qr|код|дата|печати|итого|%|₽)", re.I)
    candidates: list[str] = []
    for line in lines:
        text = normalize_text(line.text)
        if len(text) < 5:
            continue
        if not CYRILLIC_RE.search(text):
            continue
        if bad.search(text):
            continue
        if _find_prices(text) or DATE_RE.search(text) or ZONE_RE.search(text):
            continue
        digits = sum(ch.isdigit() for ch in text)
        letters = sum(ch.isalpha() for ch in text)
        if letters < 3 or digits > max(5, letters):
            continue
        candidates.append(text)
    return candidates


def _clean_product_name(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"^[^A-Za-zА-Яа-яЁё]+", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" -;,.|_")


# ── QR payload parsing ────────────────────────────────────────────────────────

def parse_qr_payloads(payloads: list[str]) -> dict[str, str]:
    """Merge QR fields from multiple decoded payload strings."""
    result: dict[str, str] = {}
    for text in payloads:
        result.update(parse_qr_payload(text))
    return result


# ── Main parse function ───────────────────────────────────────────────────────

def parse_fields(
    lines: list[OCRLine],
    qr_payloads: list[str],
    crop_bgr: np.ndarray | None = None,
) -> dict[str, str]:
    """
    Extract all 29 CSV fields from OCR lines + QR payloads.

    Args:
        lines:       OCRLine list from ocr_to_lines() or ocr_zoned()
        qr_payloads: raw decoded QR strings from decode_qr()
        crop_bgr:    original crop image (for color classification)

    Returns:
        dict with column names → string values ("нет" = field absent, "" = not recognized)
    """
    full_text = "\n".join(line.text for line in lines)
    full_text_norm = normalize_text(full_text)
    fields: dict[str, str] = {}

    # ── QR fields (most reliable) ──
    qr_fields = parse_qr_payloads(qr_payloads)
    fields.update(qr_fields)

    # Cross-populate: if QR has barcode code, set barcode too
    if fields.get("qr_code_barcode"):
        fields.setdefault("barcode", re.sub(r"\D", "", fields["qr_code_barcode"]))
    # QR prices (p1≈default, p4≈card/action)
    if fields.get("price1_qr"):
        fields.setdefault("price_default", fields["price1_qr"])
    if fields.get("price4_qr"):
        fields.setdefault("price_card", fields["price4_qr"])
    if fields.get("action_price_qr"):
        fields.setdefault("price_discount", fields["action_price_qr"])

    # ── OCR prices ──
    prices = _find_prices_from_lines(lines, full_text_norm)
    if prices:
        try:
            nums = sorted([float(p) for p in prices])
        except Exception:
            nums = []
        if len(nums) >= 2:
            # Larger = without card (default), smaller = with card
            fields.setdefault("price_default", f"{max(nums):.2f}")
            fields.setdefault("price_card", f"{min(nums):.2f}")
        elif nums:
            fields.setdefault("price_default", prices[0])
        if len(nums) >= 3:
            fields.setdefault("price_discount", f"{min(nums):.2f}")

    # ── Discount ──
    discount = DISCOUNT_RE.search(full_text_norm)
    if discount:
        fields["discount_amount"] = f"-{discount.group(1)}%"

    # ── Date ──
    dt = DATE_RE.search(full_text_norm)
    if dt:
        fields["print_datetime"] = dt.group(1).replace("-", ".")

    # ── Zone code ──
    zone = ZONE_RE.search(full_text_norm)
    if zone:
        fields["code"] = zone.group(1).replace(" ", "")

    # ── Barcode (EAN-13) ──
    barcodes = _find_barcodes(full_text_norm)
    if barcodes:
        fields.setdefault("barcode", barcodes[0])

    # ── SKU ──
    sku_candidates = [re.sub(r"\D", "", m.group(1)) for m in SKU_RE.finditer(full_text_norm)]
    sku_candidates = [s for s in sku_candidates if s != fields.get("barcode")]
    if sku_candidates:
        preferred = [s for s in sku_candidates if len(s) >= 10 and s.startswith(("2", "3"))]
        fields.setdefault("id_sku", (preferred or sku_candidates)[0])

    # ── Special symbols (Ш/К/Л) ──
    special = SPECIAL_RE.search(full_text_norm.replace("|", " "))
    if special:
        fields["special_symbols"] = special.group(1).upper()

    # ── Product name ──
    product_lines = _candidate_product_lines(lines)
    if product_lines:
        merged = " ".join(product_lines[:3])
        fields.setdefault("product_name", _clean_product_name(merged))

    # ── Additional info ──
    low = full_text_norm.lower()
    info = [w for w in KNOWN_INFO_WORDS if w in low]
    if info:
        fields["additional_info"] = "; ".join(dict.fromkeys(info))

    # ── Color (from image) ──
    if crop_bgr is not None:
        color = classify_color(crop_bgr)
        if color:
            fields["color"] = color

    # ── Default absent fields ──
    for col in ["price_discount", "discount_amount", "code", "additional_info", "special_symbols"]:
        fields.setdefault(col, ABSENT_VALUE)

    # ── Normalize price strings ──
    price_cols = [
        "price_default", "price_card", "price_discount",
        "price1_qr", "price2_qr", "price3_qr", "price4_qr",
        "action_price_qr", "wholesale_level_1_price", "wholesale_level_2_price",
    ]
    for col in price_cols:
        if col in fields and fields[col] not in ("", ABSENT_VALUE):
            fields[col] = price_to_str(fields[col])

    # ── Canonicalize all values ──
    def _canonical(v: object) -> str:
        text = normalize_text(str(v or ""))
        if text.lower() in {"нет", "νες", "νετ"}:
            return ABSENT_VALUE
        return text

    return {k: _canonical(v) for k, v in fields.items()}


def make_empty_row() -> dict[str, str]:
    """Return a blank output row with all 29 columns."""
    return {col: "" for col in OUTPUT_COLUMNS}


def merge_field_values(values: list[str]) -> str:
    """
    Merge the same field across multiple observations (track temporal fusion).

    Picks most common non-empty non-absent value, breaking ties by length.
    """
    canonical_vals = [
        normalize_text(str(v)) for v in values
        if v is not None and normalize_text(str(v)) != ""
    ]
    if not canonical_vals:
        return ""
    counts = Counter(canonical_vals)
    return sorted(
        canonical_vals,
        key=lambda v: (counts[v], v != ABSENT_VALUE, len(v)),
        reverse=True,
    )[0]
