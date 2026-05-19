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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np


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


# ── Fields any VLM implementation should try to return ───────────────────────

VLM_FIELDS = [
    "product_name",
    "price_card",
    "price_default",
    "price_discount",
    "discount_amount",
    "barcode",
    "id_sku",
    "print_datetime",
    "code",
    "color",
    "special_symbols",
    "additional_info",
]


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
        print(
            f"[vlm] Загрузка {cfg.model_id}"
            f"  4bit={cfg.load_in_4bit}  device={cfg.device_map}"
        )

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
        print(f"[vlm] Загружено. VRAM: {allocated:.2f} GB")

    def extract_fields(self, crop_bgr: np.ndarray) -> dict[str, str]:
        """Run Qwen2.5-VL on a price tag crop and return extracted fields."""
        self.load()
        tmp_path = _bgr_to_tmp_jpg(crop_bgr)
        try:
            raw = self._run_qwen(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        return _parse_json_response(raw)

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

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self._config.max_new_tokens,
                do_sample=False,
            )

        generated = out[:, inputs.input_ids.shape[1] :]
        return self._processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


# ── Shared helpers (model-agnostic) ──────────────────────────────────────────


def _bgr_to_tmp_jpg(crop_bgr: np.ndarray) -> str:
    """Save BGR crop to a temp JPEG and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return tmp.name


def _parse_json_response(raw: str) -> dict[str, str]:
    """Strip <think> blocks, extract JSON, map nulls to empty strings."""
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1]
    raw = raw.strip()

    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    match = re.search(r"\{[\s\S]*?\}", raw)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return {}

    result: dict[str, str] = {}
    for k, v in parsed.items():
        if v is None or str(v).lower() in ("null", "none", ""):
            result[k] = ""
        else:
            result[k] = str(v).strip()
    return result
