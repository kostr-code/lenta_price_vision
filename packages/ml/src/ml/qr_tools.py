from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlparse

from .schema import ABSENT_VALUE, QR_FIELD_ALIASES, normalize_value


@dataclass(frozen=True)
class QRDecode:
    raw: str
    fields: dict[str, str]
    source: str


class QRDecoder:
    """Multi-backend QR/barcode reader with a no-crash optional dependency policy."""

    def __init__(self, scales: tuple[float, ...] = (1.0, 1.5, 2.0)) -> None:
        self.scales = scales

    def decode(self, image: Any) -> list[QRDecode]:
        payloads: list[tuple[str, str]] = []
        for scaled_image, suffix in self._scaled_images(image):
            payloads.extend(
                (raw, f"{source}{suffix}") for raw, source in self._decode_once(scaled_image)
            )
            if payloads:
                break

        seen: set[str] = set()
        decoded: list[QRDecode] = []
        for raw, source in payloads:
            value = normalize_value(raw)
            if not value or value in seen:
                continue
            seen.add(value)
            decoded.append(QRDecode(raw=value, fields=parse_qr_payload(value), source=source))
        return decoded

    def _decode_once(self, image: Any) -> list[tuple[str, str]]:
        payloads: list[tuple[str, str]] = []
        payloads.extend(self._decode_with_zxingcpp(image))
        payloads.extend(self._decode_with_pyzbar(image))
        payloads.extend(self._decode_with_opencv(image))
        return payloads

    def _scaled_images(self, image: Any) -> list[tuple[Any, str]]:
        variants: list[tuple[Any, str]] = [(image, "")]
        try:
            from .media import import_cv2

            cv2 = import_cv2()
            for scale in self.scales:
                if scale == 1.0:
                    continue
                resized = cv2.resize(
                    image,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
                variants.append((resized, f"@{scale:g}x"))
        except Exception:
            return variants
        return variants

    def _decode_with_zxingcpp(self, image: Any) -> list[tuple[str, str]]:
        try:
            import zxingcpp  # type: ignore
        except ImportError:
            return []
        try:
            results = zxingcpp.read_barcodes(image)
        except Exception:
            return []
        return [(str(item.text), "zxingcpp") for item in results if getattr(item, "text", None)]

    def _decode_with_pyzbar(self, image: Any) -> list[tuple[str, str]]:
        try:
            from pyzbar.pyzbar import decode  # type: ignore
        except ImportError:
            return []
        try:
            results = decode(image)
        except Exception:
            return []
        payloads: list[tuple[str, str]] = []
        for result in results:
            raw = getattr(result, "data", b"")
            if isinstance(raw, bytes):
                payloads.append((raw.decode("utf-8", errors="ignore"), "pyzbar"))
            elif raw:
                payloads.append((str(raw), "pyzbar"))
        return payloads

    def _decode_with_opencv(self, image: Any) -> list[tuple[str, str]]:
        try:
            from .media import import_cv2

            cv2 = import_cv2()
            detector = cv2.QRCodeDetector()
            payloads: list[tuple[str, str]] = []
            try:
                ok, decoded_info, _points, _straight = detector.detectAndDecodeMulti(image)
                if ok:
                    payloads.extend((text, "opencv_qr") for text in decoded_info if text)
            except Exception:
                text, _points, _straight = detector.detectAndDecode(image)
                if text:
                    payloads.append((text, "opencv_qr"))
            return payloads
        except Exception:
            return []


def parse_qr_payload(payload: str) -> dict[str, str]:
    raw_fields = _parse_raw_payload(payload)
    normalized = {column: ABSENT_VALUE for column in QR_FIELD_ALIASES}
    for output_column, aliases in QR_FIELD_ALIASES.items():
        for alias in aliases:
            value = raw_fields.get(alias.casefold())
            if value:
                normalized[output_column] = normalize_qr_value(value)
                break
    return normalized


def _parse_raw_payload(payload: str) -> dict[str, str]:
    text = payload.strip()
    parsed: dict[str, str] = {}

    if text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                for key, value in data.items():
                    parsed[str(key).casefold()] = normalize_value(value)
                return parsed
        except json.JSONDecodeError:
            pass

    query = urlparse(text).query
    if query:
        parsed.update({key.casefold(): value for key, value in parse_qsl(query)})

    if "=" in text:
        parsed.update(
            {key.casefold(): value for key, value in parse_qsl(text, keep_blank_values=True)}
        )

    for key, value in re.findall(r"([A-Za-z][A-Za-z0-9_]+)\s*[:=]\s*([^;,&\s]+)", text):
        parsed[key.casefold()] = value

    if not parsed and re.fullmatch(r"\d{8,14}", text):
        parsed["barcode"] = text
    return parsed


def normalize_qr_value(value: Any) -> str:
    text = normalize_value(value)
    if re.fullmatch(r"\d+[.,]\d+", text):
        return text.replace(",", ".")
    return text
