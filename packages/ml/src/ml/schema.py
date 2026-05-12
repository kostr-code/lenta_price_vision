from __future__ import annotations

import csv
import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ABSENT_VALUE = "\u043d\u0435\u0442"

OUTPUT_COLUMNS: list[str] = [
    "filename",
    "product_name",
    "price_default",
    "price_card",
    "price_discount",
    "barcode",
    "discount_amount",
    "id_sku",
    "print_datetime",
    "code",
    "additional_info",
    "color",
    "special_symbols",
    "frame_timestamp",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
]

LEGACY_COLUMN_ALIASES: dict[str, str] = {
    "wholesale_level_1_coun": "wholesale_level_1_count",
}

OPTIONAL_ABSENT_DEFAULTS = {
    "price_discount",
    "code",
    "additional_info",
    "special_symbols",
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
}

QR_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "qr_code_barcode": ("barcode", "barCode", "b", "ean", "gtin"),
    "price1_qr": ("price1", "p1"),
    "price2_qr": ("price2", "p2"),
    "price3_qr": ("price3", "p3"),
    "price4_qr": ("price4", "p4"),
    "wholesale_level_1_count": ("wholesaleLevel1Count", "wL1C"),
    "wholesale_level_1_price": ("wholesaleLevel1Price", "wL1P"),
    "wholesale_level_2_count": ("wholesaleLevel2Count", "wL2C"),
    "wholesale_level_2_price": ("wholesaleLevel2Price", "wL2P"),
    "action_price_qr": ("actionPrice", "aP"),
    "action_code_qr": ("actionCode", "aC"),
}

KEY_FIELDS_FOR_QUALITY = [
    "product_name",
    "price_default",
    "price_card",
    "price_discount",
    "barcode",
    "discount_amount",
    "id_sku",
    "print_datetime",
    "code",
    "additional_info",
    "color",
    "special_symbols",
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
]


def empty_record(filename: str = "") -> dict[str, str]:
    record = {column: "" for column in OUTPUT_COLUMNS}
    record["filename"] = filename
    for column in OPTIONAL_ABSENT_DEFAULTS:
        record[column] = ABSENT_VALUE
    return record


def normalize_column_name(column: str) -> str:
    return LEGACY_COLUMN_ALIASES.get(column.strip(), column.strip())


def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def normalize_record(row: Mapping[str, Any], filename: str | None = None) -> dict[str, str]:
    normalized = empty_record(filename or normalize_value(row.get("filename", "")))
    for key, value in row.items():
        column = normalize_column_name(key)
        if column in normalized:
            normalized[column] = normalize_value(value)
    if filename is not None:
        normalized["filename"] = filename
    return normalized


def write_records_csv(records: Iterable[Mapping[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in records:
            writer.writerow(normalize_record(row))
    return output_path


def read_records_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return [normalize_record(row) for row in reader]


def merge_record_values(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, str]:
    merged = normalize_record(base)
    incoming = normalize_record(patch)
    for column in OUTPUT_COLUMNS:
        old = normalize_value(merged.get(column))
        new = normalize_value(incoming.get(column))
        if should_replace_value(old, new):
            merged[column] = new
    return merged


def should_replace_value(old: str, new: str) -> bool:
    if not new:
        return False
    if not old:
        return True
    if old == ABSENT_VALUE and new != ABSENT_VALUE:
        return True
    if new == ABSENT_VALUE:
        return False
    return len(new) > len(old) and not looks_like_noise(new)


def record_completeness(record: Mapping[str, Any]) -> int:
    score = 0
    for column in KEY_FIELDS_FOR_QUALITY:
        value = normalize_value(record.get(column))
        if value and value != ABSENT_VALUE and not looks_like_noise(value):
            score += 1
    return score


def looks_like_noise(value: str) -> bool:
    text = value.strip()
    if not text:
        return True
    return bool(len(text) == 1 and not text.isalnum())


def canonical_text(value: Any) -> str:
    text = normalize_value(value).casefold()
    text = text.replace(",", ".")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^0-9a-zа-яё.% -]+", "", text)
    return text.strip()


def comparable_field_match(left: Any, right: Any) -> bool:
    a = canonical_text(left)
    b = canonical_text(right)
    if not a and not b:
        return True
    if not a or not b:
        return False
    return a == b
