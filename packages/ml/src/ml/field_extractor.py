from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .qr_tools import QRDecode
from .schema import ABSENT_VALUE, empty_record, normalize_value
from .text_reader import TextLine

PRICE_RE = re.compile(r"(?<!\d)(\d{1,5})\s*[,.]\s*(\d{2})(?![\dA-Za-z\u0400-\u04FF])")
SPLIT_PRICE_RE = re.compile(r"(?<!\d)(\d{1,5})\s+(\d{2})(?![\dA-Za-z\u0400-\u04FF])")
EAN_RE = re.compile(r"(?<!\d)(\d{13})(?!\d)")
SKU_RE = re.compile(r"(?<!\d)(\d{12})(?!\d)")
DATE_TIME_RE = re.compile(r"(\d{2}[.]\d{2}[.]\d{4}\s+\d{1,2}[:.]\d{2})")
DISCOUNT_RE = re.compile(r"[-\u2212]?\s*\d{1,3}\s*%")
ZONE_CODE_RE = re.compile(r"\b\d{2}_\d{5,6}\b")
SPECIAL_RE = re.compile(r"(?<![\w])([\u0428\u041b\u041a\u0448\u043b\u043a])(?![\w])")

ADDITIONAL_KEYWORDS = [
    "\u0441\u0443\u0445\u043e\u0435",
    "\u043f\u043e\u043b\u0443\u0441\u0443\u0445\u043e\u0435",
    "\u0441\u043b\u0430\u0434\u043a\u043e\u0435",
    "\u043f\u043e\u043b\u0443\u0441\u043b\u0430\u0434\u043a\u043e\u0435",
    "\u0431\u0435\u0437 \u043a\u043d\u043e\u043f\u043a\u0438",
    "\u0443\u0434\u0430\u0447\u043d\u0430\u044f \u0443\u043f\u0430\u043a\u043e\u0432\u043a\u0430",
]


@dataclass(frozen=True)
class ExtractionInput:
    filename: str
    text_lines: list[TextLine]
    qr_decodes: list[QRDecode]
    color_hint: str = ""
    crop: Any | None = None


class PriceTagFieldExtractor:
    """Converts QR/OCR evidence into the fixed output CSV schema."""

    def extract(self, item: ExtractionInput) -> dict[str, str]:
        record = empty_record(item.filename)
        text = self._joined_text(item.text_lines)
        qr_fields = self._best_qr_fields(item.qr_decodes)

        record.update(qr_fields)
        record["barcode"] = first_value(qr_fields.get("qr_code_barcode"), self._barcode(text))
        record["id_sku"] = self._sku(text, record["barcode"])
        record["print_datetime"] = self._datetime(text)
        record["code"] = first_value(self._zone_code(text), ABSENT_VALUE)
        record["additional_info"] = first_value(self._additional_info(text), ABSENT_VALUE)
        record["special_symbols"] = first_value(self._special_symbols(text), ABSENT_VALUE)
        record["product_name"] = self._product_name(item.text_lines)

        prices = self._prices(text)
        record["price_default"] = first_value(qr_fields.get("price1_qr"), price_at(prices, "max"))
        record["price_card"] = first_value(
            qr_fields.get("price4_qr"),
            qr_fields.get("price2_qr"),
            price_at(prices, "min"),
        )
        record["price_discount"] = first_value(qr_fields.get("action_price_qr"), ABSENT_VALUE)
        record["discount_amount"] = first_value(
            self._discount_amount(text),
            self._computed_discount(record["price_default"], record["price_card"]),
            ABSENT_VALUE,
        )
        record["color"] = first_value(item.color_hint, estimate_tag_color(item.crop), "")
        return record

    def _joined_text(self, lines: list[TextLine]) -> str:
        return " ".join(line.text for line in lines if line.text)

    def _best_qr_fields(self, decodes: list[QRDecode]) -> dict[str, str]:
        fields: dict[str, str] = {}
        for decode in decodes:
            for key, value in decode.fields.items():
                if value and value != ABSENT_VALUE and not fields.get(key):
                    fields[key] = value
        return fields

    def _barcode(self, text: str) -> str:
        match = EAN_RE.search(text)
        return match.group(1) if match else ""

    def _sku(self, text: str, barcode: str) -> str:
        for match in SKU_RE.finditer(text):
            value = match.group(1)
            if value != barcode:
                return value
        return ""

    def _datetime(self, text: str) -> str:
        match = DATE_TIME_RE.search(text)
        if not match:
            return ""
        return match.group(1)

    def _zone_code(self, text: str) -> str:
        match = ZONE_CODE_RE.search(text)
        return match.group(0) if match else ""

    def _additional_info(self, text: str) -> str:
        lowered = text.casefold()
        for keyword in ADDITIONAL_KEYWORDS:
            if keyword in lowered:
                return keyword[:1].upper() + keyword[1:]
        return ""

    def _special_symbols(self, text: str) -> str:
        values: list[str] = []
        for match in SPECIAL_RE.finditer(text):
            value = match.group(1).upper()
            if value not in values:
                values.append(value)
        return "".join(values)

    def _product_name(self, lines: list[TextLine]) -> str:
        ranked = sorted(
            lines,
            key=lambda line: (self._product_line_score(line.text), line.confidence),
        )
        for line in reversed(ranked):
            text = cleanup_product_text(line.text)
            if self._product_line_score(text) >= 2:
                return text
        return ""

    def _product_line_score(self, text: str) -> int:
        value = text.strip()
        if len(value) < 6:
            return -2
        if PRICE_RE.search(value) or EAN_RE.search(value) or SKU_RE.search(value):
            return -3
        if DISCOUNT_RE.search(value) or DATE_TIME_RE.search(value):
            return -2
        letters = sum(1 for char in value if char.isalpha())
        digits = sum(1 for char in value if char.isdigit())
        score = letters - digits
        if any(char.isupper() for char in value):
            score += 1
        return score

    def _prices(self, text: str) -> list[str]:
        prices: list[str] = []
        for regex in (PRICE_RE, SPLIT_PRICE_RE):
            for match in regex.finditer(text):
                if looks_like_date_fragment(text, match.start(), match.end()):
                    continue
                price = f"{int(match.group(1))},{match.group(2)}"
                if price not in prices:
                    prices.append(price)
        return prices

    def _discount_amount(self, text: str) -> str:
        match = DISCOUNT_RE.search(text)
        if not match:
            return ""
        value = re.sub(r"\s+", "", match.group(0)).replace("\u2212", "-")
        return value if value.startswith("-") else f"-{value}"

    def _computed_discount(self, price_default: str, price_card: str) -> str:
        default = parse_price_number(price_default)
        card = parse_price_number(price_card)
        if default <= 0 or card <= 0 or card >= default:
            return ""
        return f"-{round((default - card) / default * 100)}%"


def first_value(*values: Any) -> str:
    absent_fallback = False
    for value in values:
        text = normalize_value(value)
        if text == ABSENT_VALUE:
            absent_fallback = True
            continue
        if text:
            return text
    return ABSENT_VALUE if absent_fallback else ""


def price_at(prices: list[str], mode: str) -> str:
    if not prices:
        return ""
    parsed = [(parse_price_number(price), price) for price in prices]
    parsed = [item for item in parsed if item[0] > 0]
    if not parsed:
        return ""
    selected = max(parsed) if mode == "max" else min(parsed)
    return selected[1]


def parse_price_number(value: Any) -> float:
    text = normalize_value(value).replace(" ", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return 0.0


def looks_like_date_fragment(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return before == "." or after == "."


def cleanup_product_text(text: str) -> str:
    value = re.sub(r"\s+", " ", text).strip(" .,:;|-")
    value = re.sub(r"\bQR\b", "", value, flags=re.IGNORECASE).strip()
    return value


def estimate_tag_color(image: Any | None) -> str:
    if image is None:
        return ""
    try:
        from .media import import_cv2, import_numpy

        cv2 = import_cv2()
        np = import_numpy()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        total = max(1, int(hsv.shape[0] * hsv.shape[1]))
        masks = {
            "red": cv2.bitwise_or(
                cv2.inRange(hsv, (0, 65, 55), (12, 255, 255)),
                cv2.inRange(hsv, (168, 65, 55), (180, 255, 255)),
            ),
            "yellow": cv2.inRange(hsv, (15, 60, 80), (42, 255, 255)),
            "green": cv2.inRange(hsv, (38, 35, 45), (92, 255, 255)),
            "white": cv2.inRange(hsv, (0, 0, 165), (180, 80, 255)),
        }
        scores = {color: int(np.count_nonzero(mask)) / total for color, mask in masks.items()}
    except Exception:
        return ""
    color, score = max(scores.items(), key=lambda item: item[1])
    return color if score >= 0.08 else ""
