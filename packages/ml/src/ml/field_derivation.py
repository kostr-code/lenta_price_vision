from __future__ import annotations

from typing import Any

from .schema import ABSENT_VALUE, normalize_value
from .validators import normalize_discount, normalize_price, validate_price


def derive_fields(
    record: dict[str, Any], derive_qr_fields_when_missing: bool = False
) -> dict[str, str]:
    derived = {key: normalize_value(value) for key, value in record.items()}
    if derive_qr_fields_when_missing:
        copy_if_missing(derived, "qr_code_barcode", "barcode")
        copy_if_missing(derived, "barcode", "qr_code_barcode")
        copy_if_missing(derived, "price1_qr", "price_default")
        copy_if_missing(derived, "price_default", "price1_qr")
        copy_if_missing(derived, "price4_qr", "price_card")
        copy_if_missing(derived, "price_card", "price4_qr")
        copy_if_missing(derived, "action_price_qr", "price_discount")
        copy_if_missing(derived, "price_discount", "action_price_qr")

    if missing(derived.get("discount_amount")):
        discount = derive_discount(derived.get("price_default"), derived.get("price_card"))
        if discount:
            derived["discount_amount"] = discount
    return derived


def copy_if_missing(record: dict[str, str], target: str, source: str) -> None:
    source_value = normalize_value(record.get(source))
    if missing(record.get(target)) and source_value and source_value != ABSENT_VALUE:
        record[target] = source_value


def derive_discount(price_default: Any, price_card: Any) -> str:
    default = price_number(price_default)
    card = price_number(price_card)
    if default <= 0 or card <= 0 or card >= default:
        return ""
    return normalize_discount(f"-{round((default - card) / default * 100)}%")


def price_number(value: Any) -> float:
    price = normalize_price(value)
    if not validate_price(price):
        return 0.0
    try:
        return float(price.replace(",", "."))
    except ValueError:
        return 0.0


def missing(value: Any) -> bool:
    text = normalize_value(value)
    return not text or text == ABSENT_VALUE
