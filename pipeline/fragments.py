"""
pipeline/fragments.py — FragmentProvider: поставщик sub-регионов ценника.

Абстракция позволяет переключать источник sub-регионов (QR-зона, штрихкод,
текстовые зоны и т.д.) без изменения кода декодеров и OCR.

Два поставщика:
  heuristic_provider(crop)          — фиксированные дробные координаты (быстро)
  YOLO2Provider(weights, ...)       — Stage 2 YOLO детекция (точнее, с fallback)

FragmentMap = dict[str, np.ndarray | None]
  ключи совпадают с именами классов YOLO Stage 2:
    "qr_code", "barcode", "product_name", "price_card", "price_default",
    "price_discount", "discount_amount", "id_sku", "print_datetime",
    "code", "additional_info", "special_symbols"

Использование:
  from pipeline.fragments import heuristic_provider, load_yolo2_provider

  provider = load_yolo2_provider("models/inside_price_tag_yolo.pt")
  # или просто: provider = heuristic_provider

  fragments = provider(crop)
  qr_img      = fragments.get("qr_code")
  barcode_img = fragments.get("barcode")
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# FragmentMap: имя класса → кроп или None если не найдено
FragmentMap = dict[str, "np.ndarray | None"]

# FragmentProvider — протокол: callable crop → FragmentMap
FragmentProvider = Callable[[np.ndarray], FragmentMap]

# ── Эвристические зоны (фиксированные дробные координаты) ──
# Основаны на визуальной структуре ценников Ленты:
#   QR всегда в правом верхнем углу  → top 42% × right 40%
#   Штрихкод всегда внизу по ширине  → bottom 30%
_HEURISTIC_ZONES: dict[str, tuple[float, float, float, float]] = {
    # name: (y_start_frac, y_end_frac, x_start_frac, x_end_frac)
    "qr_code": (0.00, 0.42, 0.60, 1.00),
    "barcode": (0.70, 1.00, 0.00, 1.00),
}


def heuristic_provider(crop: np.ndarray) -> FragmentMap:
    """
    Вырезать sub-регионы по фиксированным дробным координатам.

    Быстро (нет inference), но может быть неточно при fisheye/разных размерах.
    Служит как fallback внутри YOLO2Provider.
    """
    h, w = crop.shape[:2]
    fragments: FragmentMap = {}
    for name, (ys, ye, xs, xe) in _HEURISTIC_ZONES.items():
        sub = crop[int(h * ys) : int(h * ye), int(w * xs) : int(w * xe)]
        fragments[name] = sub if sub.size > 0 else None
    return fragments


# ── YOLO Stage 2 Provider ──


class YOLO2Provider:
    """
    Поставщик фрагментов на основе YOLO Stage 2 детекции.

    Детектирует все 12 семантических регионов ценника.
    Если класс не найден и fallback=True — возвращает эвристическую зону.

    Классы YOLO Stage 2:
      0: product_name   1: qr_code       2: price_default   3: price_card
      4: price_discount 5: barcode       6: discount_amount  7: id_sku
      8: print_datetime 9: code          10: additional_info 11: special_symbols
    """

    def __init__(
        self,
        weights: str,
        conf: float = 0.25,
        imgsz: int = 960,
        device: str | None = None,
        fallback: bool = True,
    ) -> None:
        """
        weights: путь к .pt файлу Stage 2
        conf:    порог уверенности детекции
        imgsz:   размер входа (обучался на 960)
        device:  "0" / "cpu" / None (auto)
        fallback: добавить эвристику для классов, которые YOLO не нашёл
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError("ultralytics не установлен: uv add ultralytics")
        self._model = YOLO(weights)
        self._conf = conf
        self._imgsz = imgsz
        self._device = device
        self._fallback = fallback

    def __call__(self, crop: np.ndarray) -> FragmentMap:
        kw: dict = {"conf": self._conf, "imgsz": self._imgsz, "verbose": False}
        if self._device:
            kw["device"] = self._device

        results = self._model.predict(source=crop, **kw)
        fragments: FragmentMap = {}

        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                cls_name: str = self._model.names[int(box.cls.item())]
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                sub = crop[y1:y2, x1:x2]
                # При нескольких bbox одного класса берём первый (YOLO сортирует по conf ↓)
                if cls_name not in fragments and sub.size > 0:
                    fragments[cls_name] = sub

        if self._fallback:
            for name, val in heuristic_provider(crop).items():
                if name not in fragments:
                    fragments[name] = val

        return fragments


def load_yolo2_provider(
    weights: str,
    conf: float = 0.25,
    imgsz: int = 960,
    device: str | None = None,
) -> YOLO2Provider:
    """Загрузить YOLO Stage 2 как FragmentProvider (с fallback на эвристику)."""
    return YOLO2Provider(weights, conf=conf, imgsz=imgsz, device=device, fallback=True)
