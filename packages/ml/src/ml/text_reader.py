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
        if self.config.prefer_paddle:
            lines.extend(self._read_paddle(image))
        lines.extend(self._read_tesseract(image))
        if not lines and not self.config.prefer_paddle:
            lines.extend(self._read_paddle(image))
        return deduplicate_lines(lines)

    def _read_paddle(self, image: Any) -> list[TextLine]:
        reader = self._load_paddle()
        if reader is None:
            return []
        try:
            results = reader.ocr(image, cls=True)
        except Exception as exc:  # pragma: no cover - depends on OCR runtime
            self._paddle_error = str(exc)
            return []

        lines: list[TextLine] = []
        for page in results or []:
            for item in page or []:
                parsed = parse_paddle_item(item)
                if parsed is None:
                    continue
                text, confidence = parsed
                lines.append(TextLine(text, confidence, "paddleocr"))
        return lines

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


def deduplicate_lines(lines: list[TextLine]) -> list[TextLine]:
    by_text: dict[str, TextLine] = {}
    for line in lines:
        key = " ".join(line.text.casefold().split())
        current = by_text.get(key)
        if current is None or line.confidence > current.confidence:
            by_text[key] = line
    return sorted(by_text.values(), key=lambda line: line.confidence, reverse=True)


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
