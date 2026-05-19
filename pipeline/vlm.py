"""
pipeline/vlm.py — VLM abstraction for structured price tag field extraction.

Contract (VLMProvider protocol):
    provider.load()                          — pre-warm model (no-op if already loaded)
    provider.extract_fields(crop_bgr)        — run inference, return dict[str, str]

Built-in implementation:
    Qwen25VLProvider(config: VLMConfig)      — Qwen2.5-VL wrapper

To plug in a different model (e.g. Qwen3-VL), implement the protocol:

    class MyProvider:
        def load(self) -> None:
            ...  # load your model here

        def extract_fields(self, crop_bgr: np.ndarray) -> dict[str, str]:
            ...  # run inference, return keys from VLM_FIELDS

Then pass it to the pipeline instead of Qwen25VLProvider — nothing else changes.

Qwen2.5-VL configuration:

    # bfloat16 (default, ~14 GB VRAM for 7B)
    Qwen25VLProvider(VLMConfig())

    # 4-bit quantization (~5 GB for 7B, ~2.5 GB for 3B)
    Qwen25VLProvider(VLMConfig(model_id="Qwen/Qwen2.5-VL-7B-Instruct", load_in_4bit=True))

    # Small model without quantization (fits 4070 8 GB)
    Qwen25VLProvider(VLMConfig(model_id="Qwen/Qwen2.5-VL-3B-Instruct"))
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np
import structlog

log = structlog.get_logger("vlm")


# ── Provider protocol ──────────────────────────────────────────────────────────


@runtime_checkable
class VLMProvider(Protocol):
    """
    Minimal contract for any VLM backend used in the pipeline.

    Implement both methods to create a custom provider.
    load() is called once before the processing loop to pre-warm the model
    and catch missing weights early. extract_fields() is called per crop.
    """

    def load(self) -> None:
        """Load model weights into memory. No-op if already loaded."""
        ...

    def extract_fields(self, crop_bgr: np.ndarray) -> dict[str, str]:
        """
        Extract structured fields from a price tag crop (BGR image).

        Returns a dict whose keys are a subset of VLM_FIELDS.
        Missing or unrecognized fields should map to "" (empty string), not None.
        """
        ...


# ── Fields the VLM prompt asks for ───────────────────────────────────────────
# Keep in sync with prompts/qwen_extract.md

VLM_FIELDS = [
    "product_name",
    "price_card",
    "price_default",
    "discount_percent",   # mapped → discount_amount in CSV
    "additional_info",
    # "qr_code" — presence indicator only, not a CSV column, intentionally omitted
]

# Mapping: VLM prompt field name → OUTPUT_COLUMNS CSV name.
# Update this dict whenever prompts/qwen_extract.md field names change.
VLM_TO_CSV_MAP: dict[str, str] = {
    "product_name":     "product_name",
    "price_card":       "price_card",
    "price_default":    "price_default",
    "discount_percent": "discount_amount",
    "additional_info":  "additional_info",
}


# ── Default extraction prompt for Qwen2.5-VL ─────────────────────────────────

_PROMPT = """\
На изображении ценник из российского магазина Лента.
Извлеки поля и верни ТОЛЬКО валидный JSON без пояснений и без markdown-блоков:
{
  "product_name": "полное название товара (крупный текст вверху)",
  "price_card": "цена по карте Лента — самое крупное число на ценнике, рубли.копейки",
  "price_default": "цена без карты — число рядом с подписью Без карты",
  "price_discount": "акционная цена если есть, иначе null",
  "discount_amount": "скидка например -32%, иначе null",
  "barcode": "13-значный штрихкод снизу — только цифры без пробелов",
  "id_sku": "артикул формата XXXXXX XXXXXX снизу слева",
  "print_datetime": "дата и время печати ДД.ММ.ГГГГ ЧЧ:ММ",
  "code": "код зоны выкладки вида ДД_XXXXXX-XXXXXX",
  "color": "цвет ценника: red / yellow / blue / white",
  "special_symbols": "символ Ш К или Л если виден, иначе null",
  "additional_info": "доп. информация если есть, иначе null"
}
Если поле не видно на ценнике — ставь null. Только JSON, никаких пояснений.\
"""


# ── VLMConfig: parameters for Qwen25VLProvider ───────────────────────────────


@dataclass
class VLMConfig:
    """Configuration for Qwen25VLProvider."""

    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    # bitsandbytes 4-bit: True → ~5 GB VRAM for 7B, False → ~14 GB bfloat16
    load_in_4bit: bool = False
    # device_map: "cuda" (single GPU), "auto" (split across GPUs/CPU), "cuda:1", etc.
    device_map: str = "cuda"
    # Pixel range for the preprocessor (affects image detail resolution)
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1024 * 28 * 28
    # Maximum tokens to generate per inference call
    max_new_tokens: int = 512
    # Path to .md file containing the extraction prompt.
    # If set and file exists, overrides the default _PROMPT at load() time.
    # Edit the file and restart the service to apply changes.
    prompt_file: str = "prompts/qwen_extract.md"


# ── Qwen2.5-VL provider ───────────────────────────────────────────────────────


class Qwen25VLProvider:
    """
    VLMProvider backed by Qwen2.5-VL (transformers).

    Model is lazily loaded on the first extract_fields() call.
    Call .load() explicitly to pre-warm before a processing loop.

    Example::

        provider = Qwen25VLProvider(VLMConfig(model_id="Qwen/Qwen2.5-VL-3B-Instruct"))
        provider.load()
        fields = provider.extract_fields(crop_bgr)
    """

    def __init__(
        self,
        config: VLMConfig | str = VLMConfig(),
        prompt: str = _PROMPT,
    ) -> None:
        if isinstance(config, str):
            config = VLMConfig(model_id=config)
        self._config = config
        self._prompt = prompt
        self._model: Any = None
        self._processor: Any = None

    # ── Public interface (VLMProvider contract) ──

    def load(self) -> None:
        """Load Qwen2.5-VL model + processor. No-op if already loaded."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        cfg = self._config
        if cfg.prompt_file:
            p = Path(cfg.prompt_file)
            if p.exists():
                self._prompt = p.read_text(encoding="utf-8").strip()
                log.info("vlm.prompt.loaded", file=str(p), chars=len(self._prompt))
            else:
                log.warning("vlm.prompt.file_missing", file=str(p), fallback="built-in prompt")

        log.info("vlm.load.start", model=cfg.model_id, four_bit=cfg.load_in_4bit, device=cfg.device_map)
        t0 = time.perf_counter()

        load_kw: dict[str, Any] = {"device_map": cfg.device_map}
        if cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            load_kw["torch_dtype"] = torch.bfloat16

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.model_id, **load_kw
        )
        self._processor = AutoProcessor.from_pretrained(
            cfg.model_id,
            min_pixels=cfg.min_pixels,
            max_pixels=cfg.max_pixels,
        )

        allocated = torch.cuda.memory_allocated() / 1e9
        log.info("vlm.load.done", model=cfg.model_id, vram_gb=round(allocated, 2), elapsed_s=round(time.perf_counter() - t0, 1))

    def extract_fields(self, crop_bgr: np.ndarray) -> dict[str, str]:
        """Run Qwen2.5-VL on a price tag crop and return extracted fields.

        Return keys are always CSV column names (via VLM_TO_CSV_MAP),
        so the result can be merged directly with OCR output.
        """
        self.load()
        h, w = crop_bgr.shape[:2]
        log.info("vlm.infer.start", h=h, w=w)
        tmp_path = _bgr_to_tmp_jpg(crop_bgr)
        try:
            raw = self._run_qwen(tmp_path)
            log.debug("vlm.infer.raw", text=raw[:400])
        except Exception as exc:
            log.error("vlm.infer.failed", error=str(exc))
            return {}
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        parsed = _parse_json_response(raw)

        # Rename VLM field names → CSV column names via VLM_TO_CSV_MAP.
        # Fields not in the map (e.g. qr_code presence indicator) are discarded.
        result: dict[str, str] = {}
        for vlm_key, csv_key in VLM_TO_CSV_MAP.items():
            val = parsed.get(vlm_key, "")
            if val:
                result[csv_key] = val

        filled = {k: v for k, v in result.items() if v}
        log.info("vlm.infer.done", fields_found=len(filled), values=filled)
        return result

    # ── Internal helpers ──

    def _run_qwen(self, image_path: str) -> str:
        """Single-image inference. Returns raw model text."""
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": self._prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        log.info("vlm.generate.start", max_new_tokens=self._config.max_new_tokens)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self._config.max_new_tokens,
                do_sample=False,
            )
        elapsed = time.perf_counter() - t0

        generated = out[:, inputs.input_ids.shape[1] :]
        decoded = self._processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        log.info("vlm.generate.done", elapsed_s=round(elapsed, 1), out_tokens=generated.shape[1])
        return decoded


# ── Shared helpers (model-agnostic) ──────────────────────────────────────────


def _bgr_to_tmp_jpg(crop_bgr: np.ndarray) -> str:
    """Save BGR crop to a temp JPEG and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return tmp.name


def _extract_json_object(text: str) -> str | None:
    """Find the first complete JSON object by tracking brace depth.

    Unlike a regex, this correctly handles nested objects and braces
    inside string values.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _sanitize_json(text: str) -> str:
    """Remove trailing commas before } or ] — common LLM output error."""
    return re.sub(r",\s*([}\]])", r"\1", text)


def _parse_json_response(raw: str) -> dict[str, str]:
    """Extract a JSON object from raw VLM output and normalise values.

    Handles:
    - <think>...</think> preamble (Qwen reasoning mode)
    - Markdown code fences (```json ... ```)
    - Surrounding text before/after the JSON object
    - Nested objects (brace-depth extraction, not regex)
    - Trailing commas
    - Array values → joined string
    - Nested dict values → JSON string (preserves data)
    - Partial responses (fewer fields than expected) — returned as-is
    """
    # 1. strip <think> block
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1]
    raw = raw.strip()

    # 2. strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    # 3. extract first complete JSON object (brace-depth aware)
    json_str = _extract_json_object(raw)
    if not json_str:
        log.warning("vlm.parse.no_json", raw_preview=raw[:300])
        return {}

    # 4. fix trailing commas
    json_str = _sanitize_json(json_str)

    # 5. parse
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as exc:
        log.warning("vlm.parse.json_error", error=str(exc), raw_preview=json_str[:300])
        return {}

    if not isinstance(parsed, dict):
        log.warning("vlm.parse.not_object", got_type=type(parsed).__name__)
        return {}

    # 6. normalise values to str
    result: dict[str, str] = {}
    for k, v in parsed.items():
        if v is None or str(v).lower() in ("null", "none"):
            result[k] = ""
        elif isinstance(v, list):
            # e.g. additional_info: ["Хит продаж", "Новинка"] → "Хит продаж; Новинка"
            result[k] = "; ".join(str(x).strip() for x in v if x)
        elif isinstance(v, dict):
            # nested object — serialise rather than lose the data
            result[k] = json.dumps(v, ensure_ascii=False)
        else:
            s = str(v).strip()
            result[k] = "" if s.lower() in ("null", "none", "") else s

    log.debug("vlm.parse.ok", keys=list(result.keys()))
    return result
