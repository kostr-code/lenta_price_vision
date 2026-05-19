"""
pipeline/detector.py — YOLO Stage 1: детектор ценников.

Два режима:
  detect_price_tags() — model.predict(), без трекинга (для статичных кадров)
  track_price_tags()  — model.track() с ByteTrack, для видео (persist=True)

track_price_tags() требует подачи КАЖДОГО кадра последовательно — трекер ведёт
состояние между вызовами через persist=True. Пропускать кадры нельзя.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    class_id: int
    track_id: int | None = None


_model = None


def load_detector(weights: str | Path) -> None:
    """Загрузить YOLO-модель из .pt файла (вызвать один раз перед inference)."""
    global _model
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics не установлен: uv add ultralytics")
    _model = YOLO(str(weights))
    print(f"[detector] загружена модель: {weights}")


def detect_price_tags(
    frame: np.ndarray,
    conf: float = 0.25,
    imgsz: int = 1280,
    device: str | None = None,
) -> list[Detection]:
    """
    Запустить Stage 1 YOLO по повёрнутому кадру (без трекинга).

    Используется для статичных кадров (labeled mode с GT CSV).
    Возвращает Detection без track_id.
    """
    if _model is None:
        raise RuntimeError("Вызови load_detector() перед detect_price_tags()")
    kw: dict = {"conf": conf, "imgsz": imgsz, "verbose": False}
    if device:
        kw["device"] = device
    results = _model.predict(source=frame, **kw)
    boxes = results[0].boxes
    if boxes is None:
        return []
    out: list[Detection] = []
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        out.append(
            Detection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                conf=float(box.conf.item()),
                class_id=int(box.cls.item()),
            )
        )
    return out


def track_price_tags(
    frame: np.ndarray,
    conf: float = 0.25,
    imgsz: int = 1280,
    device: str | None = None,
    tracker: str = "train/bytetrack_price.yaml",
) -> list[Detection]:
    """
    Запустить Stage 1 YOLO с ByteTrack трекингом по повёрнутому кадру.

    Требует вызова на КАЖДОМ кадре последовательно — persist=True ведёт состояние
    трекера между вызовами. Возвращает Detection с заполненным track_id.

    Трекер инициализируется автоматически при первом вызове и сбрасывается
    при создании новой VideoCapture (новое видео). Для явного сброса трекера
    вызови load_detector() повторно.
    """
    if _model is None:
        raise RuntimeError("Вызови load_detector() перед track_price_tags()")
    kw: dict = {"conf": conf, "imgsz": imgsz, "verbose": False, "persist": True}
    if device:
        kw["device"] = device
    results = _model.track(source=frame, tracker=tracker, **kw)
    boxes = results[0].boxes
    if boxes is None:
        return []
    out: list[Detection] = []
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        track_id = int(box.id.item()) if box.id is not None else None
        out.append(
            Detection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                conf=float(box.conf.item()),
                class_id=int(box.cls.item()),
                track_id=track_id,
            )
        )
    return out
