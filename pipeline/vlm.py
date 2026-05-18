"""
pipeline/vlm.py — Qwen2.5-VL wrapper for structured price tag field extraction.

Extracts the 12 visually readable fields from a crop image.
The remaining fields (QR data, metadata coords) come from other pipeline stages.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Lazy imports — only resolved when load_vlm() is called
_model: Any = None
_processor: Any = None
_model_id: str = ""

# Fields the VLM can reliably read from the image
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


def load_vlm(model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct") -> None:
    """Load Qwen2.5-VL model + processor into module-level cache. Call once."""
    global _model, _processor, _model_id
    if _model is not None and _model_id == model_id:
        return

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"[vlm] Loading {model_id} ...")
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    _processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=1024 * 28 * 28,
    )
    _model_id = model_id

    import torch as _torch
    allocated = _torch.cuda.memory_allocated() / 1e9
    print(f"[vlm] Loaded. VRAM: {allocated:.2f} GB")


def _bgr_to_tmp_jpg(crop_bgr: np.ndarray) -> str:
    """Save BGR crop to a temp JPEG and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return tmp.name


def _run_qwen(image_path: str, prompt: str) -> str:
    """Single-image inference. Returns raw model text."""
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=512, do_sample=False)

    generated = out[:, inputs.input_ids.shape[1] :]
    return _processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def _parse_json_response(raw: str) -> dict[str, str]:
    """Strip <think> blocks, extract JSON, map nulls to empty strings."""
    # Strip Qwen3-style <think>...</think>
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1]
    raw = raw.strip()

    # Strip ```json ... ``` fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    match = re.search(r"\{[\s\S]*?\}", raw)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return {}

    # Convert null / None / "null" → "" (empty = not recognized)
    result: dict[str, str] = {}
    for k, v in parsed.items():
        if v is None or str(v).lower() in ("null", "none", ""):
            result[k] = ""
        else:
            result[k] = str(v).strip()
    return result


def extract_fields_vlm(
    crop_bgr: np.ndarray,
    prompt: str = _PROMPT,
) -> dict[str, str]:
    """
    Run Qwen2.5-VL on a price tag crop and return extracted fields.

    Args:
        crop_bgr: BGR numpy array (OpenCV format)
        prompt:   override the default extraction prompt

    Returns:
        dict with keys from VLM_FIELDS (missing or null fields → "")

    Raises:
        RuntimeError: if load_vlm() was not called first
    """
    if _model is None:
        raise RuntimeError("Call load_vlm() before extract_fields_vlm()")

    tmp_path = _bgr_to_tmp_jpg(crop_bgr)
    try:
        raw = _run_qwen(tmp_path, prompt)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return _parse_json_response(raw)
