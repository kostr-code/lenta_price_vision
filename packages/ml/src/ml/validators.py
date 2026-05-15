from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .schema import ABSENT_VALUE, normalize_value

PRICE_RE = re.compile(r"^\d{1,6}([,.]\d{2})?$")
DISCOUNT_RE = re.compile(r"^-?\d{1,3}%$")
EAN13_RE = re.compile(r"^\d{13}$")
SKU_RE = re.compile(r"^\d{12}$")
DATETIME_FORMATS = ("%d.%m.%Y %H:%M", "%d.%m.%Y %H.%M", "%d.%m.%Y %H:%M:%S")
SPECIAL_SYMBOLS = {
    "ш": "Ш",
    "л": "Л",
    "к": "К",
    "Ш": "Ш",
    "Л": "Л",
    "К": "К",
}
UNIT_ALIASES = {
    "1шт": "piece",
    "1 шт": "piece",
    "шт": "piece",
    "штука": "piece",
    "1кг": "kg",
    "1 кг": "kg",
    "кг": "kg",
    "100г": "100g",
    "100 г": "100g",
    "100 гр": "100g",
    "вес": "weight",
    "весовой": "weight",
}


def normalize_price(value: Any) -> str:
    text = normalize_value(value).replace(" ", "").replace(".", ",")
    if not text or text == ABSENT_VALUE:
        return text
    match = re.search(r"(\d{1,6})(?:,(\d{1,2}))?", text)
    if not match:
        return ""
    rubles = str(int(match.group(1)))
    kopecks = (match.group(2) or "00").ljust(2, "0")[:2]
    return f"{rubles},{kopecks}"


def validate_price(value: Any) -> bool:
    price = normalize_price(value)
    if not price or price == ABSENT_VALUE or not PRICE_RE.fullmatch(price):
        return False
    try:
        return 0.0 < float(price.replace(",", ".")) < 1_000_000.0
    except ValueError:
        return False


def normalize_discount(value: Any) -> str:
    text = normalize_value(value).replace(" ", "").replace("\u2212", "-")
    if not text or text == ABSENT_VALUE:
        return text
    match = re.search(r"-?\d{1,3}%", text)
    if not match:
        return ""
    discount = match.group(0)
    return discount if discount.startswith("-") else f"-{discount}"


def validate_discount(value: Any) -> bool:
    discount = normalize_discount(value)
    if not discount or discount == ABSENT_VALUE or not DISCOUNT_RE.fullmatch(discount):
        return False
    number = abs(int(discount.rstrip("%")))
    return 0 < number <= 100


def normalize_barcode(value: Any) -> str:
    text = re.sub(r"\D+", "", normalize_value(value))
    return text


def validate_barcode(value: Any) -> bool:
    text = normalize_barcode(value)
    return bool(8 <= len(text) <= 14)


def validate_ean13(value: Any) -> bool:
    text = normalize_barcode(value)
    if not EAN13_RE.fullmatch(text):
        return False
    digits = [int(char) for char in text]
    checksum = (10 - ((sum(digits[0:12:2]) + 3 * sum(digits[1:12:2])) % 10)) % 10
    return checksum == digits[-1]


def validate_sku(value: Any) -> bool:
    text = normalize_barcode(value)
    return bool(SKU_RE.fullmatch(text))


def normalize_datetime(value: Any) -> str:
    text = normalize_value(value).strip()
    if not text or text == ABSENT_VALUE:
        return text
    text = re.sub(r"\s+", " ", text.replace(",", "."))
    text = re.sub(r"(\d{2}\.\d{2}\.\d{4})\s+(\d{1,2})\.(\d{2})", r"\1 \2:\3", text)
    for fmt in DATETIME_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
            date_part = f"{parsed.day:02d}.{parsed.month:02d}.{parsed.year}"
            return f"{date_part} {parsed.hour}:{parsed.minute:02d}"
        except ValueError:
            continue
    match = re.search(r"\d{2}\.\d{2}\.\d{4}\s+\d{1,2}:\d{2}", text)
    return match.group(0) if match else ""


def validate_datetime(value: Any) -> bool:
    return bool(normalize_datetime(value))


def normalize_special_symbol(value: Any) -> str:
    text = normalize_value(value)
    if not text or text == ABSENT_VALUE:
        return text
    symbols: list[str] = []
    for char in text:
        normalized = SPECIAL_SYMBOLS.get(char)
        if normalized and normalized not in symbols:
            symbols.append(normalized)
    return "".join(symbols)


def normalize_unit(value: Any) -> str:
    text = normalize_value(value).casefold().replace(".", "").strip()
    text = re.sub(r"\s+", " ", text)
    return UNIT_ALIASES.get(text, "")


def validate_field(field_name: str, value: Any) -> bool:
    text = normalize_value(value)
    if not text or text == ABSENT_VALUE:
        return False
    if field_name in {"price_default", "price_card", "price_discount"}:
        return validate_price(text)
    if field_name in {
        "price1_qr",
        "price2_qr",
        "price3_qr",
        "price4_qr",
        "action_price_qr",
        "wholesale_level_1_price",
        "wholesale_level_2_price",
    }:
        return validate_price(text)
    if field_name in {"barcode", "qr_code_barcode"}:
        return validate_barcode(text)
    if field_name == "id_sku":
        return validate_sku(text)
    if field_name == "print_datetime":
        return validate_datetime(text)
    if field_name == "discount_amount":
        return validate_discount(text)
    if field_name == "special_symbols":
        return bool(normalize_special_symbol(text))
    return True


def normalize_field_value(field_name: str, value: Any) -> str:
    text = normalize_value(value)
    if not text or text == ABSENT_VALUE:
        return text
    if field_name in {
        "price_default",
        "price_card",
        "price_discount",
        "price1_qr",
        "price2_qr",
        "price3_qr",
        "price4_qr",
        "action_price_qr",
        "wholesale_level_1_price",
        "wholesale_level_2_price",
    }:
        return normalize_price(text)
    if field_name in {"barcode", "qr_code_barcode", "id_sku"}:
        return normalize_barcode(text)
    if field_name == "discount_amount":
        return normalize_discount(text)
    if field_name == "print_datetime":
        return normalize_datetime(text)
    if field_name == "special_symbols":
        return normalize_special_symbol(text)
    return text
