from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextLine:
    text: str
    confidence: float
    source: str


@dataclass(frozen=True)
class TextReaderConfig:
    enabled: bool = True
    prefer_paddle: bool = True
    language: str = "ru"
    use_gpu: bool = False
    max_variants: int = 6
    zoned: bool = True


class TextReader:
    """OCR facade for crop-only reading with optional PaddleOCR/Tesseract ensemble."""

    def __init__(self, config: TextReaderConfig | None = None) -> None:
        self.config = config or TextReaderConfig()
        self._paddle_reader: Any | None = None
        self._paddle_error: str | None = None
        self._tesseract_error: str | None = None

    @property
    def status(self) -> dict[str, str | bool]:
        return {
            "enabled": self.config.enabled,
            "paddle_loaded": self._paddle_reader is not None,
            "paddle_error": self._paddle_error or "",
            "tesseract_error": self._tesseract_error or "",
        }

    def read(self, image: Any) -> list[TextLine]:
        if not self.config.enabled:
            return []

        lines: list[TextLine] = []
        variants = build_ocr_variants(
            image,
            max_variants=self.config.max_variants,
            zoned=self.config.zoned,
        )
        if not variants:
            variants = [("full", image)]
        for variant_name, variant in variants:
            if self.config.prefer_paddle:
                lines.extend(add_source_context(self._read_paddle(variant), variant_name))
            lines.extend(add_source_context(self._read_tesseract(variant), variant_name))
            if not lines and not self.config.prefer_paddle:
                lines.extend(add_source_context(self._read_paddle(variant), variant_name))
        return deduplicate_lines(lines)

    def _read_paddle(self, image: Any) -> list[TextLine]:
        reader = self._load_paddle()
        if reader is None:
            return []
        try:
            try:
                results = reader.ocr(image, cls=True)
            except TypeError:
                results = reader.ocr(image)
        except Exception as exc:  # pragma: no cover - depends on OCR runtime
            self._paddle_error = str(exc)
            return []

        return parse_paddle_results(results, "paddleocr")

    def _load_paddle(self) -> Any | None:
        if self._paddle_reader is not None:
            return self._paddle_reader
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError:
            self._paddle_error = "paddleocr is not installed"
            return None
        lang = "ru" if self.config.language.lower().startswith("ru") else "en"
        try:
            self._paddle_reader = self._create_paddle_reader(PaddleOCR, lang)
            self._paddle_error = None
        except Exception as exc:  # pragma: no cover - depends on OCR runtime
            self._paddle_error = str(exc)
            return None
        return self._paddle_reader

    def _create_paddle_reader(self, paddle_ocr: Any, lang: str) -> Any:
        configs = [
            {
                "use_textline_orientation": True,
                "lang": lang,
            },
            {
                "use_angle_cls": True,
                "lang": lang,
                "use_gpu": self.config.use_gpu,
                "show_log": False,
            },
            {"lang": lang},
        ]
        last_error: Exception | None = None
        for config in configs:
            try:
                return paddle_ocr(**config)
            except TypeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return paddle_ocr(lang=lang)

    def _read_tesseract(self, image: Any) -> list[TextLine]:
        try:
            import pytesseract  # type: ignore
            from PIL import Image

            from .media import import_cv2
        except ImportError as exc:
            self._tesseract_error = str(exc)
            return []
        try:
            cv2 = import_cv2()
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            lang = choose_tesseract_lang(pytesseract, self.config.language)
            data = pytesseract.image_to_data(
                pil_image,
                lang=lang,
                output_type=pytesseract.Output.DICT,
                config="--psm 6",
            )
        except Exception as exc:  # pragma: no cover - depends on tesseract binary
            self._tesseract_error = str(exc)
            return []
        lines: list[TextLine] = []
        for text, confidence in zip(data.get("text", []), data.get("conf", []), strict=False):
            cleaned = str(text).strip()
            if not cleaned:
                continue
            try:
                score = max(0.0, float(confidence) / 100.0)
            except ValueError:
                score = 0.0
            lines.append(TextLine(cleaned, score, "tesseract"))
        self._tesseract_error = None
        return lines


def parse_paddle_item(item: Any) -> tuple[str, float] | None:
    try:
        text, confidence = item[1]
    except (IndexError, TypeError, ValueError):
        return None
    cleaned = str(text).strip()
    if not cleaned:
        return None
    try:
        score = float(confidence)
    except (TypeError, ValueError):
        score = 0.0
    return cleaned, max(0.0, min(1.0, score))


def parse_paddle_results(results: Any, source: str) -> list[TextLine]:
    lines: list[TextLine] = []
    if not results:
        return lines

    if isinstance(results, list) and results and hasattr(results[0], "get"):
        for page in results:
            texts = list(page.get("rec_texts") or [])
            scores = list(page.get("rec_scores") or [])
            for index, text in enumerate(texts):
                cleaned = str(text).strip()
                if not cleaned:
                    continue
                try:
                    confidence = float(scores[index]) if index < len(scores) else 0.0
                except (TypeError, ValueError):
                    confidence = 0.0
                lines.append(TextLine(cleaned, max(0.0, min(1.0, confidence)), source))
        return lines

    pages = results
    if (
        isinstance(results, list)
        and len(results) == 1
        and isinstance(results[0], list)
        and results[0]
    ):
        pages = results[0]

    for item in pages or []:
        parsed = parse_paddle_item(item)
        if parsed is None:
            continue
        text, confidence = parsed
        lines.append(TextLine(text, confidence, source))
    return lines


def deduplicate_lines(lines: list[TextLine]) -> list[TextLine]:
    by_text: dict[str, TextLine] = {}
    for line in lines:
        key = " ".join(line.text.casefold().split())
        current = by_text.get(key)
        if current is None or line.confidence > current.confidence:
            by_text[key] = line
    return sorted(by_text.values(), key=lambda line: line.confidence, reverse=True)


def add_source_context(lines: list[TextLine], context: str) -> list[TextLine]:
    return [
        TextLine(line.text, line.confidence, f"{line.source}|{context}")
        for line in lines
    ]


def build_ocr_variants(
    image: Any,
    max_variants: int = 6,
    zoned: bool = True,
) -> list[tuple[str, Any]]:
    if image is None or not getattr(image, "size", 0):
        return []

    masked = suppress_code_artifacts(image)
    variants: list[tuple[str, Any]] = [
        ("full_masked", enhance_text_image(masked)),
        ("full", enhance_text_image(image)),
    ]
    if zoned:
        for zone_name, zone in split_price_tag_zones(masked):
            variants.append((zone_name, enhance_text_image(zone)))
            if len(variants) >= max_variants:
                break
    return [
        (name, variant)
        for name, variant in variants[:max_variants]
        if getattr(variant, "size", 0)
    ]


def split_price_tag_zones(image: Any) -> list[tuple[str, Any]]:
    if image is None or not getattr(image, "size", 0):
        return []
    height, width = image.shape[:2]
    if height < 24 or width < 24:
        return [("full_small", image)]
    zones = [
        ("product_top", crop_relative(image, (0.00, 0.00, 1.00, 0.62))),
        ("product_left", crop_relative(image, (0.00, 0.00, 0.72, 0.48))),
        ("price_left", crop_relative(image, (0.00, 0.18, 0.62, 0.92))),
        ("price_right", crop_relative(image, (0.38, 0.18, 1.00, 0.92))),
        ("lower_text", crop_relative(image, (0.00, 0.52, 1.00, 1.00))),
    ]
    result: list[tuple[str, Any]] = []
    seen_shapes: set[tuple[int, int]] = set()
    for name, crop in zones:
        if crop is None or not getattr(crop, "size", 0):
            continue
        crop_height, crop_width = crop.shape[:2]
        if crop_height < 18 or crop_width < 18:
            continue
        shape = (crop_height, crop_width)
        if shape in seen_shapes:
            continue
        result.append((name, crop))
        seen_shapes.add(shape)
    return result


def crop_relative(image: Any, box: tuple[float, float, float, float]) -> Any:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    ix1 = max(0, min(width - 1, int(round(x1 * width))))
    iy1 = max(0, min(height - 1, int(round(y1 * height))))
    ix2 = max(0, min(width, int(round(x2 * width))))
    iy2 = max(0, min(height, int(round(y2 * height))))
    if ix2 <= ix1 or iy2 <= iy1:
        return image[:0, :0].copy()
    return image[iy1:iy2, ix1:ix2].copy()


def suppress_code_artifacts(image: Any) -> Any:
    if image is None or not getattr(image, "size", 0):
        return image
    try:
        from .media import import_cv2, import_numpy

        cv2 = import_cv2()
        np = import_numpy()
    except Exception:
        return image

    height, width = image.shape[:2]
    if height < 40 or width < 40:
        return image
    out = image.copy()
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    dark = cv2.inRange(gray, 0, 125)
    edges = cv2.Canny(gray, 45, 160)
    texture = cv2.bitwise_or(dark, edges)
    texture = cv2.morphologyEx(
        texture,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    contours, _ = cv2.findContours(texture, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    background = np.percentile(out.reshape(-1, 3), 88, axis=0).astype("uint8").tolist()

    cluster_boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < 6 or box_height < 6:
            continue
        area_ratio = (box_width * box_height) / max(1.0, float(width * height))
        if not 0.002 <= area_ratio <= 0.45:
            continue
        roi_dark = dark[y : y + box_height, x : x + box_width]
        roi_edges = edges[y : y + box_height, x : x + box_width]
        dark_density = float((roi_dark > 0).mean())
        edge_density = float((roi_edges > 0).mean())
        aspect = box_width / max(1, box_height)
        right_or_lower = x > 0.42 * width or y > 0.48 * height
        qr_like = 0.55 <= aspect <= 1.85 and dark_density >= 0.16 and edge_density >= 0.04
        barcode_like = (
            (aspect >= 2.8 or aspect <= 0.36)
            and dark_density >= 0.11
            and edge_density >= 0.035
            and (y > 0.45 * height or x > 0.50 * width)
        )
        if right_or_lower and qr_like:
            cluster_boxes.append((x, y, x + box_width, y + box_height))
        if not right_or_lower or not (qr_like or barcode_like):
            continue
        pad = max(2, int(0.03 * max(box_width, box_height)))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(width, x + box_width + pad), min(height, y + box_height + pad)
        cv2.rectangle(out, (x1, y1), (x2, y2), background, thickness=-1)

    if len(cluster_boxes) >= 6:
        x1 = min(box[0] for box in cluster_boxes)
        y1 = min(box[1] for box in cluster_boxes)
        x2 = max(box[2] for box in cluster_boxes)
        y2 = max(box[3] for box in cluster_boxes)
        box_width = x2 - x1
        box_height = y2 - y1
        area_ratio = (box_width * box_height) / max(1.0, float(width * height))
        aspect = box_width / max(1, box_height)
        if 0.01 <= area_ratio <= 0.45 and 0.45 <= aspect <= 2.2:
            pad = max(2, int(0.025 * max(box_width, box_height)))
            cv2.rectangle(
                out,
                (max(0, x1 - pad), max(0, y1 - pad)),
                (min(width, x2 + pad), min(height, y2 + pad)),
                background,
                thickness=-1,
            )
    return out


def enhance_text_image(image: Any, max_side: int = 1800) -> Any:
    if image is None or not getattr(image, "size", 0):
        return image
    try:
        from .media import import_cv2

        cv2 = import_cv2()
    except Exception:
        return image
    out = image.copy()
    height, width = out.shape[:2]
    longest = max(height, width)
    if longest < 650:
        scale = min(3.5, 900.0 / max(1, longest))
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif longest > max_side:
        scale = max_side / float(longest)
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    out = cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    out = cv2.bilateralFilter(out, d=5, sigmaColor=28, sigmaSpace=28)
    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(out, 1.55, blur, -0.55, 0)


def choose_tesseract_lang(pytesseract: Any, requested_language: str) -> str:
    try:
        available = set(pytesseract.get_languages(config=""))
    except Exception:
        available = set()
    wants_russian = requested_language.lower().startswith("ru")
    if wants_russian and {"rus", "eng"}.issubset(available):
        return "rus+eng"
    if wants_russian and "rus" in available:
        return "rus"
    if "eng" in available:
        return "eng"
    return "rus+eng" if wants_russian else "eng"
