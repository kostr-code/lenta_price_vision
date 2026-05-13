from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .candidates import iter_tiles
from .media import BBox, bbox_iou, import_cv2
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
    tiled: bool = False,
    tile_size: int = 640,
    tile_stride: int = 512,
    min_box_visibility: float = 0.25,
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

        safe_name = sanitize_name(f"{sequence_name}_{timestamp_ms:08d}")
        if tiled:
            written_images, written_labels = write_tiled_training_frame(
                cv2=cv2,
                frame=frame,
                rows=rows,
                image_dir=image_dirs[split],
                label_dir=label_dirs[split],
                safe_name=safe_name,
                tile_size=tile_size,
                tile_stride=tile_stride,
                min_box_visibility=min_box_visibility,
            )
        else:
            written_images, written_labels = write_full_training_frame(
                cv2=cv2,
                frame=frame,
                rows=rows,
                image_path=image_dirs[split] / f"{safe_name}.jpg",
                label_path=label_dirs[split] / f"{safe_name}.txt",
            )
        image_counts[split] += written_images
        label_count += written_labels

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
    project: str | None = None,
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
        project=project or str(DEFAULT_DATA_DIR.parent / "runs" / "detect"),
        name=name,
    )


def record_to_yolo(
    row: dict[str, str],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    bbox = record_to_bbox(row, width, height)
    if bbox is None:
        return None
    return bbox_to_yolo(bbox, width, height)


def record_to_bbox(row: dict[str, str], width: int, height: int) -> BBox | None:
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
    bbox = BBox(x_min, y_min, x_max, y_max)
    if bbox.width < 3 or bbox.height < 3:
        return None
    return bbox


def bbox_to_yolo(
    bbox: BBox,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    if width <= 0 or height <= 0 or bbox.width < 3 or bbox.height < 3:
        return None
    return (
        (bbox.x_min + bbox.width / 2.0) / width,
        (bbox.y_min + bbox.height / 2.0) / height,
        bbox.width / width,
        bbox.height / height,
    )


def write_full_training_frame(
    cv2: Any,
    frame: Any,
    rows: list[dict[str, str]],
    image_path: Path,
    label_path: Path,
) -> tuple[int, int]:
    height, width = frame.shape[:2]
    labels: list[str] = []
    for row in rows:
        yolo_box = record_to_yolo(row, width, height)
        if yolo_box is None:
            continue
        labels.append("0 " + " ".join(f"{value:.6f}" for value in yolo_box))
    labels = deduplicate_labels(labels)
    if not labels:
        return 0, 0
    cv2.imwrite(str(image_path), frame)
    label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    return 1, len(labels)


def write_tiled_training_frame(
    cv2: Any,
    frame: Any,
    rows: list[dict[str, str]],
    image_dir: Path,
    label_dir: Path,
    safe_name: str,
    tile_size: int,
    tile_stride: int,
    min_box_visibility: float,
) -> tuple[int, int]:
    height, width = frame.shape[:2]
    source_boxes = [
        bbox for bbox in (record_to_bbox(row, width, height) for row in rows) if bbox is not None
    ]
    if not source_boxes:
        return 0, 0

    written_images = 0
    written_labels = 0
    for tile_index, (x_min, y_min, x_max, y_max) in enumerate(
        iter_tiles(width, height, tile_size, tile_stride, max_tiles=0)
    ):
        tile_box = BBox(float(x_min), float(y_min), float(x_max), float(y_max))
        tile_width = max(1, x_max - x_min)
        tile_height = max(1, y_max - y_min)
        labels: list[str] = []

        for source_box in source_boxes:
            clipped = clip_bbox(source_box, tile_box)
            if clipped is None:
                continue
            visibility = clipped.area / max(1.0, source_box.area)
            if visibility < min_box_visibility:
                continue
            local_box = BBox(
                clipped.x_min - x_min,
                clipped.y_min - y_min,
                clipped.x_max - x_min,
                clipped.y_max - y_min,
            )
            yolo_box = bbox_to_yolo(local_box, tile_width, tile_height)
            if yolo_box is None:
                continue
            labels.append("0 " + " ".join(f"{value:.6f}" for value in yolo_box))

        labels = deduplicate_labels(labels)
        if not labels:
            continue
        tile_image = frame[y_min:y_max, x_min:x_max]
        image_path = image_dir / f"{safe_name}_tile_{tile_index:03d}.jpg"
        label_path = label_dir / f"{safe_name}_tile_{tile_index:03d}.txt"
        cv2.imwrite(str(image_path), tile_image)
        label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
        written_images += 1
        written_labels += len(labels)

    return written_images, written_labels


def clip_bbox(bbox: BBox, boundary: BBox) -> BBox | None:
    clipped = BBox(
        max(bbox.x_min, boundary.x_min),
        max(bbox.y_min, boundary.y_min),
        min(bbox.x_max, boundary.x_max),
        min(bbox.y_max, boundary.y_max),
    )
    if clipped.width < 3 or clipped.height < 3:
        return None
    return clipped


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
