from __future__ import annotations

import argparse
import csv
import re
import shutil
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Detection:
    split: str
    source_file: str
    rotated_image: Path
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    original_index: int


@dataclass(frozen=True)
class RemovedDetection:
    removed: Detection
    kept: Detection
    iou: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove duplicated inside-tag YOLO predictions with NMS and export images/labels "
            "in Ultralytics and CVAT YOLO-friendly layouts."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            r"F:\lenta_price_vision\packages\ml\src\ml\runs\datasets"
            r"\lenta_yolo_49_43_26_prop8\crops_best_pt\inside_rot90ccw_best"
        ),
        help="Source folder created by analyze_inside_on_crops_rot90ccw.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            r"F:\lenta_price_vision\packages\ml\src\ml\runs\datasets"
            r"\lenta_yolo_49_43_26_prop8\crops_best_pt\inside_rot90ccw_best_dedup_cvat"
        ),
        help="Output folder for deduplicated labels/images.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.75,
        help="Suppress lower-confidence boxes when IoU is above this value.",
    )
    parser.add_argument(
        "--class-aware",
        action="store_true",
        help="Suppress boxes only when they have the same class id.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete output folder before writing new results.",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Do not create a CVAT zip archive.",
    )
    return parser.parse_args()


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_.") or "image"


def parse_class_names(data_yaml: Path) -> dict[int, str]:
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    names: dict[int, str] = {}
    in_names = False
    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line.strip() == "names:":
            in_names = True
            continue
        if not in_names:
            continue
        if line and not line.startswith(" "):
            break
        match = re.match(r"\s*(\d+)\s*:\s*(.+?)\s*$", line)
        if match:
            names[int(match.group(1))] = match.group(2).strip().strip("'\"")

    if not names:
        raise SystemExit(f"Could not parse class names from: {data_yaml}")
    return names


def read_manifest(source: Path) -> dict[Path, list[Detection]]:
    manifest_path = source / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"manifest.csv not found: {manifest_path}")

    grouped: dict[Path, list[Detection]] = defaultdict(list)
    seen_images: set[Path] = set()

    with manifest_path.open("r", newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        per_image_index: dict[Path, int] = defaultdict(int)

        for row in reader:
            rotated_image_text = (row.get("rotated_image") or "").strip()
            if not rotated_image_text:
                continue

            rotated_image = Path(rotated_image_text)
            seen_images.add(rotated_image)

            class_id_text = (row.get("class_id") or "").strip()
            if not class_id_text:
                grouped.setdefault(rotated_image, [])
                continue

            original_index = per_image_index[rotated_image]
            per_image_index[rotated_image] += 1

            grouped[rotated_image].append(
                Detection(
                    split=(row.get("split") or "train").strip() or "train",
                    source_file=(row.get("source_file") or "").strip(),
                    rotated_image=rotated_image,
                    class_id=int(class_id_text),
                    class_name=(row.get("class_name") or class_id_text).strip(),
                    confidence=float((row.get("confidence") or "0").strip() or 0.0),
                    x1=float((row.get("x1") or "0").strip() or 0.0),
                    y1=float((row.get("y1") or "0").strip() or 0.0),
                    x2=float((row.get("x2") or "0").strip() or 0.0),
                    y2=float((row.get("y2") or "0").strip() or 0.0),
                    original_index=original_index,
                )
            )

    for image_path in seen_images:
        grouped.setdefault(image_path, [])
    return dict(grouped)


def box_iou(left: Detection, right: Detection) -> float:
    inter_x1 = max(left.x1, right.x1)
    inter_y1 = max(left.y1, right.y1)
    inter_x2 = min(left.x2, right.x2)
    inter_y2 = min(left.y2, right.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - inter_area
    return inter_area / union if union > 0 else 0.0


def dedupe_detections(
    detections: list[Detection],
    iou_threshold: float,
    class_aware: bool,
) -> tuple[list[Detection], list[RemovedDetection]]:
    kept: list[Detection] = []
    removed: list[RemovedDetection] = []

    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        best_overlap: tuple[Detection, float] | None = None
        for kept_detection in kept:
            if class_aware and candidate.class_id != kept_detection.class_id:
                continue
            iou = box_iou(candidate, kept_detection)
            if iou > iou_threshold and (best_overlap is None or iou > best_overlap[1]):
                best_overlap = (kept_detection, iou)

        if best_overlap is None:
            kept.append(candidate)
        else:
            removed.append(RemovedDetection(removed=candidate, kept=best_overlap[0], iou=best_overlap[1]))

    return sorted(kept, key=lambda item: item.original_index), removed


def yolo_line(detection: Detection, width: int, height: int) -> str:
    x1 = max(0.0, min(float(width), detection.x1))
    y1 = max(0.0, min(float(height), detection.y1))
    x2 = max(0.0, min(float(width), detection.x2))
    y2 = max(0.0, min(float(height), detection.y2))

    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    norm_width = box_width / width
    norm_height = box_height / height
    return (
        f"{detection.class_id} {x_center:.6f} {y_center:.6f} "
        f"{norm_width:.6f} {norm_height:.6f}"
    )


def color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = [
        (230, 57, 70),
        (29, 53, 87),
        (42, 157, 143),
        (244, 162, 97),
        (131, 56, 236),
        (255, 183, 3),
        (33, 158, 188),
        (214, 40, 40),
        (106, 153, 78),
        (251, 86, 7),
        (67, 97, 238),
        (0, 0, 0),
    ]
    return palette[class_id % len(palette)]


def draw_detections(image_path: Path, detections: list[Detection], output_path: Path) -> None:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        line_width = max(2, min(image.size) // 180)

        for detection in detections:
            color = color_for_class(detection.class_id)
            draw.rectangle(
                [detection.x1, detection.y1, detection.x2, detection.y2],
                outline=color,
                width=line_width,
            )

            label = f"{detection.class_name} {detection.confidence:.2f}"
            text_box = draw.textbbox((0, 0), label, font=font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            text_x = max(0, int(detection.x1))
            text_y = max(0, int(detection.y1) - text_height - 4)
            if text_y <= 0:
                text_y = min(image.height - text_height - 2, int(detection.y1) + 4)

            draw.rectangle(
                [text_x, text_y, text_x + text_width + 6, text_y + text_height + 4],
                fill=color,
            )
            draw.text((text_x + 3, text_y + 2), label, fill=(255, 255, 255), font=font)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, quality=95, subsampling=0)


def write_data_yaml(out_dir: Path, class_names: dict[int, str]) -> Path:
    data_yaml = out_dir / "data.yaml"
    lines = [
        f"path: {out_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines.extend(f"  {class_id}: {class_names[class_id]}" for class_id in sorted(class_names))
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data_yaml


def write_cvat_metadata(cvat_dir: Path, class_names: dict[int, str]) -> None:
    obj_names = cvat_dir / "obj.names"
    obj_names.write_text(
        "\n".join(class_names[class_id] for class_id in sorted(class_names)) + "\n",
        encoding="utf-8",
    )
    (cvat_dir / "obj.data").write_text(
        "\n".join(
            [
                f"classes = {len(class_names)}",
                "train = data/train.txt",
                "valid = data/val.txt",
                "names = data/obj.names",
                "backup = backup/",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def zip_directory(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir))


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    out_dir = args.out_dir.resolve()
    cvat_dir = out_dir / "cvat_yolo"

    if not source.exists():
        raise SystemExit(f"Source folder not found: {source}")

    if args.clear_output and out_dir.exists():
        shutil.rmtree(out_dir)

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "annotated_all" / split).mkdir(parents=True, exist_ok=True)
    (cvat_dir / "obj_train_data").mkdir(parents=True, exist_ok=True)
    (cvat_dir / "labels").mkdir(parents=True, exist_ok=True)

    class_names = parse_class_names(source / "data.yaml")
    grouped = read_manifest(source)

    write_data_yaml(out_dir, class_names)
    write_cvat_metadata(cvat_dir, class_names)

    kept_manifest_path = out_dir / "manifest_dedup.csv"
    removed_manifest_path = out_dir / "manifest_removed_overlaps.csv"
    train_txt_path = cvat_dir / "train.txt"
    val_txt_path = cvat_dir / "val.txt"

    total_raw = 0
    total_kept = 0
    total_removed = 0
    images_processed = 0
    images_with_labels = 0
    train_entries: list[str] = []
    val_entries: list[str] = []

    with (
        kept_manifest_path.open("w", newline="", encoding="utf-8") as kept_file,
        removed_manifest_path.open("w", newline="", encoding="utf-8") as removed_file,
    ):
        kept_writer = csv.DictWriter(
            kept_file,
            fieldnames=[
                "split",
                "image_file",
                "label_file",
                "cvat_image_file",
                "cvat_label_file",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
        )
        removed_writer = csv.DictWriter(
            removed_file,
            fieldnames=[
                "split",
                "image_file",
                "removed_class_id",
                "removed_class_name",
                "removed_confidence",
                "kept_class_id",
                "kept_class_name",
                "kept_confidence",
                "iou",
                "removed_x1",
                "removed_y1",
                "removed_x2",
                "removed_y2",
                "kept_x1",
                "kept_y1",
                "kept_x2",
                "kept_y2",
            ],
        )
        kept_writer.writeheader()
        removed_writer.writeheader()

        for image_path, detections in sorted(grouped.items(), key=lambda item: str(item[0])):
            if not image_path.exists():
                raise SystemExit(f"Image from manifest not found: {image_path}")

            split = detections[0].split if detections else image_path.parent.name
            split = split if split in {"train", "val"} else "train"
            images_processed += 1
            total_raw += len(detections)

            kept, removed = dedupe_detections(
                detections,
                iou_threshold=args.iou_threshold,
                class_aware=args.class_aware,
            )
            total_kept += len(kept)
            total_removed += len(removed)
            if kept:
                images_with_labels += 1

            with Image.open(image_path) as image:
                width, height = image.size

            safe_stem = f"{split}_{safe_name(image_path.stem)}"
            target_image = out_dir / "images" / split / f"{safe_stem}.jpg"
            target_label = out_dir / "labels" / split / f"{safe_stem}.txt"
            target_annotated = out_dir / "annotated_all" / split / f"{safe_stem}.jpg"
            cvat_image = cvat_dir / "obj_train_data" / f"{safe_stem}.jpg"
            cvat_label = cvat_dir / "labels" / f"{safe_stem}.txt"

            shutil.copy2(image_path, target_image)
            shutil.copy2(image_path, cvat_image)

            label_text = "\n".join(yolo_line(detection, width, height) for detection in kept)
            if label_text:
                label_text += "\n"
            target_label.write_text(label_text, encoding="utf-8")
            cvat_label.write_text(label_text, encoding="utf-8")
            draw_detections(image_path, kept, target_annotated)

            cvat_line = f"data/obj_train_data/{safe_stem}.jpg"
            if split == "val":
                val_entries.append(cvat_line)
            else:
                train_entries.append(cvat_line)

            for detection in kept:
                kept_writer.writerow(
                    {
                        "split": split,
                        "image_file": str(target_image),
                        "label_file": str(target_label),
                        "cvat_image_file": str(cvat_image),
                        "cvat_label_file": str(cvat_label),
                        "class_id": detection.class_id,
                        "class_name": detection.class_name,
                        "confidence": f"{detection.confidence:.6f}",
                        "x1": f"{detection.x1:.0f}",
                        "y1": f"{detection.y1:.0f}",
                        "x2": f"{detection.x2:.0f}",
                        "y2": f"{detection.y2:.0f}",
                    }
                )

            for removed_item in removed:
                removed_detection = removed_item.removed
                kept_detection = removed_item.kept
                removed_writer.writerow(
                    {
                        "split": split,
                        "image_file": str(target_image),
                        "removed_class_id": removed_detection.class_id,
                        "removed_class_name": removed_detection.class_name,
                        "removed_confidence": f"{removed_detection.confidence:.6f}",
                        "kept_class_id": kept_detection.class_id,
                        "kept_class_name": kept_detection.class_name,
                        "kept_confidence": f"{kept_detection.confidence:.6f}",
                        "iou": f"{removed_item.iou:.6f}",
                        "removed_x1": f"{removed_detection.x1:.0f}",
                        "removed_y1": f"{removed_detection.y1:.0f}",
                        "removed_x2": f"{removed_detection.x2:.0f}",
                        "removed_y2": f"{removed_detection.y2:.0f}",
                        "kept_x1": f"{kept_detection.x1:.0f}",
                        "kept_y1": f"{kept_detection.y1:.0f}",
                        "kept_x2": f"{kept_detection.x2:.0f}",
                        "kept_y2": f"{kept_detection.y2:.0f}",
                    }
                )

    train_txt_path.write_text("\n".join(train_entries) + ("\n" if train_entries else ""), encoding="utf-8")
    val_txt_path.write_text("\n".join(val_entries) + ("\n" if val_entries else ""), encoding="utf-8")

    zip_path = out_dir / "cvat_yolo_dedup.zip"
    if not args.no_zip:
        zip_directory(cvat_dir, zip_path)

    print("[DONE] Deduplicated inside predictions")
    print(f"  source:       {source}")
    print(f"  output_dir:   {out_dir}")
    print(f"  iou_threshold:{args.iou_threshold}")
    print(f"  class_aware:  {args.class_aware}")
    print(f"  images:       {images_processed}")
    print(f"  images_labeled:{images_with_labels}")
    print(f"  raw_labels:   {total_raw}")
    print(f"  kept_labels:  {total_kept}")
    print(f"  removed:      {total_removed}")
    print(f"  kept_manifest:{kept_manifest_path}")
    print(f"  removed_csv:  {removed_manifest_path}")
    print(f"  cvat_dir:     {cvat_dir}")
    if not args.no_zip:
        print(f"  cvat_zip:     {zip_path}")


if __name__ == "__main__":
    main()
