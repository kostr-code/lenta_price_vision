from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .media import bbox_iou, import_cv2
from .pipeline import DEFAULT_DATA_DIR, discover_labeled_sequences, parse_coord
from .schema import read_records_csv


@dataclass(frozen=True)
class YoloDatasetBuildResult:
    output_dir: str
    data_yaml: str
    train_images: int
    val_images: int
    labels: int


def build_yolo_dataset(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = Path("datasets/lenta_price_tags"),
    val_fraction: float = 0.2,
    seed: int = 42,
) -> YoloDatasetBuildResult:
    """Build a YOLO detection dataset from arbitrary public CSV/video names."""

    cv2 = import_cv2()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dirs = {
        "train": output_dir / "images" / "train",
        "val": output_dir / "images" / "val",
    }
    label_dirs = {
        "train": output_dir / "labels" / "train",
        "val": output_dir / "labels" / "val",
    }
    for directory in [*image_dirs.values(), *label_dirs.values()]:
        directory.mkdir(parents=True, exist_ok=True)

    frames: list[tuple[str, Path, int, list[dict[str, str]]]] = []
    for sequence in discover_labeled_sequences(data_dir):
        rows_by_timestamp: dict[int, list[dict[str, str]]] = {}
        for row in read_records_csv(sequence.csv_path):
            timestamp = int(parse_coord(row.get("frame_timestamp")))
            rows_by_timestamp.setdefault(timestamp, []).append(row)
        for timestamp, rows in rows_by_timestamp.items():
            frames.append((sequence.name, sequence.video_path, timestamp, rows))

    random.Random(seed).shuffle(frames)
    val_count = max(1, int(len(frames) * val_fraction)) if len(frames) > 1 else 0
    val_keys = set(range(val_count))

    image_counts = {"train": 0, "val": 0}
    label_count = 0
    for index, (sequence_name, video_path, timestamp_ms, rows) in enumerate(frames):
        split = "val" if index in val_keys else "train"
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            continue
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0, timestamp_ms))
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            continue

        height, width = frame.shape[:2]
        safe_name = sanitize_name(f"{sequence_name}_{timestamp_ms:08d}")
        image_path = image_dirs[split] / f"{safe_name}.jpg"
        label_path = label_dirs[split] / f"{safe_name}.txt"

        labels: list[str] = []
        for row in rows:
            yolo_box = record_to_yolo(row, width, height)
            if yolo_box is None:
                continue
            labels.append("0 " + " ".join(f"{value:.6f}" for value in yolo_box))
        labels = deduplicate_labels(labels)
        if not labels:
            continue
        cv2.imwrite(str(image_path), frame)
        label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
        image_counts[split] += 1
        label_count += len(labels)

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: price_tag",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return YoloDatasetBuildResult(
        output_dir=str(output_dir),
        data_yaml=str(data_yaml),
        train_images=image_counts["train"],
        val_images=image_counts["val"],
        labels=label_count,
    )


def train_yolo_detector(
    data_yaml: Path,
    model: str = "yolo11n.pt",
    epochs: int = 150,
    imgsz: int = 1280,
    batch: int = 4,
    device: str = "cpu",
    project: str = "runs/lenta",
    name: str = "price_tag_yolo",
) -> Any:
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("ultralytics is required to train YOLO") from exc
    detector = YOLO(model)
    return detector.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
    )


def record_to_yolo(
    row: dict[str, str],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    x_min = parse_coord(row.get("x_min"))
    y_min = parse_coord(row.get("y_min"))
    x_max = parse_coord(row.get("x_max"))
    y_max = parse_coord(row.get("y_max"))
    if x_max <= x_min or y_max <= y_min or width <= 0 or height <= 0:
        return None
    x_min = max(0.0, min(float(width), x_min))
    x_max = max(0.0, min(float(width), x_max))
    y_min = max(0.0, min(float(height), y_min))
    y_max = max(0.0, min(float(height), y_max))
    box_w = x_max - x_min
    box_h = y_max - y_min
    if box_w < 3 or box_h < 3:
        return None
    return (
        (x_min + box_w / 2.0) / width,
        (y_min + box_h / 2.0) / height,
        box_w / width,
        box_h / height,
    )


def deduplicate_labels(labels: list[str]) -> list[str]:
    kept: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    for label in labels:
        parts = label.split()
        if len(parts) != 5:
            continue
        x_center, y_center, width, height = map(float, parts[1:])
        box = yolo_to_corners(x_center, y_center, width, height)
        if any(bbox_iou_tuple(box, other) > 0.92 for other in boxes):
            continue
        boxes.append(box)
        kept.append(label)
    return kept


def yolo_to_corners(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    return (
        x_center - width / 2.0,
        y_center - height / 2.0,
        x_center + width / 2.0,
        y_center + height / 2.0,
    )


def bbox_iou_tuple(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    from .media import BBox

    return bbox_iou(BBox(*left), BBox(*right))


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "frame"
