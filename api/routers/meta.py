"""api/routers/meta.py — GET /health, /schema, /datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter

router = APIRouter()

# ── Schema ────────────────────────────────────────────────────────────────────

_FIELD_SCHEMA = [
    {"name": "filename",                 "description": "Имя видеофайла",             "source": "meta"},
    {"name": "product_name",             "description": "Полное название товара",      "source": "vlm|ocr"},
    {"name": "price_default",            "description": "Цена без карты, Р",           "source": "vlm|ocr"},
    {"name": "price_card",               "description": "Цена по карте Лента",         "source": "vlm|ocr"},
    {"name": "price_discount",           "description": "Акционная цена",              "source": "vlm|ocr"},
    {"name": "barcode",                  "description": "EAN-13 штрихкод (13 цифр)",   "source": "vlm|ocr|qr"},
    {"name": "discount_amount",          "description": "Размер скидки (-32%)",        "source": "vlm|ocr"},
    {"name": "id_sku",                   "description": "Артикул товара",              "source": "vlm|ocr"},
    {"name": "print_datetime",           "description": "Дата и время печати ценника", "source": "vlm|ocr"},
    {"name": "code",                     "description": "Код зоны выкладки",           "source": "vlm|ocr"},
    {"name": "additional_info",          "description": "Доп. информация",             "source": "vlm|ocr"},
    {"name": "color",                    "description": "Цвет ценника (red/yellow/...)", "source": "ocr|heuristic"},
    {"name": "special_symbols",          "description": "Тип выкладки Ш/К/Л",         "source": "vlm|ocr"},
    {"name": "frame_timestamp",          "description": "Время кадра, мс",             "source": "meta"},
    {"name": "x_min",                    "description": "bbox x_min",                 "source": "meta"},
    {"name": "y_min",                    "description": "bbox y_min",                 "source": "meta"},
    {"name": "x_max",                    "description": "bbox x_max",                 "source": "meta"},
    {"name": "y_max",                    "description": "bbox y_max",                 "source": "meta"},
    {"name": "qr_code_barcode",          "description": "Штрихкод из QR-кода",        "source": "qr"},
    {"name": "price1_qr",                "description": "Цена 1 из QR",               "source": "qr"},
    {"name": "price2_qr",                "description": "Цена 2 из QR",               "source": "qr"},
    {"name": "price3_qr",                "description": "Цена 3 из QR",               "source": "qr"},
    {"name": "price4_qr",                "description": "Цена 4 из QR",               "source": "qr"},
    {"name": "wholesale_level_1_count",  "description": "Опт. порог 1, кол-во",       "source": "qr"},
    {"name": "wholesale_level_1_price",  "description": "Опт. порог 1, цена",         "source": "qr"},
    {"name": "wholesale_level_2_count",  "description": "Опт. порог 2, кол-во",       "source": "qr"},
    {"name": "wholesale_level_2_price",  "description": "Опт. порог 2, цена",         "source": "qr"},
    {"name": "action_price_qr",          "description": "Акционная цена из QR",       "source": "qr"},
    {"name": "action_code_qr",           "description": "Код акции из QR",            "source": "qr"},
]


@router.get("/health")
async def health() -> dict[str, Any]:
    from api.pipeline_bridge import model_status
    status = model_status()
    return {
        "status": "ok",
        "service": "ml",
        "models": status,
    }


@router.get("/schema")
async def schema() -> dict[str, Any]:
    return {
        "version": "1.0",
        "fields": _FIELD_SCHEMA,
        "field_count": len(_FIELD_SCHEMA),
        "absent_value": "нет",
    }


@router.get("/datasets")
async def datasets(data_dir: str | None = None) -> dict[str, Any]:
    from api.ml_app import settings as app_settings
    import pandas as pd

    base = Path(data_dir or app_settings.data_dir)
    log = structlog.get_logger("meta")

    if not base.exists():
        log.warning("datasets.dir_missing", path=str(base))
        return {"datasets": [], "data_dir": str(base)}

    result = []
    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        videos = sorted(subdir.glob("*.mp4")) + sorted(subdir.glob("*.MP4"))
        csvs = sorted(subdir.glob("*.csv"))
        if not videos:
            continue

        n_labels = 0
        if csvs:
            try:
                n_labels = len(pd.read_csv(csvs[0]))
            except Exception:
                pass

        result.append({
            "name": subdir.name,
            "video": str(videos[0]),
            "csv": str(csvs[0]) if csvs else None,
            "has_labels": bool(csvs),
            "n_labels": n_labels,
        })

    return {"datasets": result, "data_dir": str(base)}
