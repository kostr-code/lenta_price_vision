from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rotate price-tag crops 90 degrees counter-clockwise, run an inside-tag YOLO model, "
            "and save labels plus annotated images."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            r"F:\lenta_price_vision\packages\ml\src\ml\runs\datasets"
            r"\lenta_yolo_49_43_26_prop8\crops_best_pt"
        ),
        help="Folder with crop split folders, usually crops_best_pt.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(r"F:\lenta_price_vision\packages\ml\runs\inside\inside_price_tag_yolo_best"),
        help="Inside-tag YOLO run folder or weights .pt path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            r"F:\lenta_price_vision\packages\ml\src\ml\runs\datasets"
            r"\lenta_yolo_49_43_26_prop8\crops_best_pt\inside_rot90ccw_best"
        ),
        help="Output folder.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size.")
    parser.add_argument("--device", default=None, help='Device for YOLO, for example "0" or "cpu".')
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Split folders inside --source to process.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete output folder before writing new results.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit for a quick test run.",
    )
    return parser.parse_args()


def resolve_weights(model_path: Path) -> Path:
    if model_path.is_file():
        return model_path

    candidates = [
        model_path / "weights" / "best.pt",
        model_path / "best.pt",
        model_path / "weights" / "last.pt",
        model_path / "last.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise SystemExit(f"Could not find YOLO weights under: {model_path}")


def normalize_names(names: object) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_.") or "image"


def iter_split_images(source: Path, split: str) -> list[Path]:
    split_dir = source / split
    if not split_dir.exists():
        raise SystemExit(f"Split folder not found: {split_dir}")

    return sorted(
        path
        for path in split_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def open_rotated_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return image.rotate(90, expand=True)


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


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    line_width = max(2, min(output.size) // 180)

    for detection in detections:
        color = color_for_class(detection.class_id)
        box = [detection.x1, detection.y1, detection.x2, detection.y2]
        draw.rectangle(box, outline=color, width=line_width)

        label = f"{detection.class_name} {detection.confidence:.2f}"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_x = max(0, int(detection.x1))
        text_y = max(0, int(detection.y1) - text_height - 4)
        if text_y <= 0:
            text_y = min(output.height - text_height - 2, int(detection.y1) + 4)

        draw.rectangle(
            [text_x, text_y, text_x + text_width + 6, text_y + text_height + 4],
            fill=color,
        )
        draw.text((text_x + 3, text_y + 2), label, fill=(255, 255, 255), font=font)

    return output


def write_data_yaml(out_dir: Path, class_names: dict[int, str]) -> Path:
    data_yaml = out_dir / "data.yaml"
    lines = [
        f"path: {out_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for class_id in sorted(class_names):
        lines.append(f"  {class_id}: {class_names[class_id]}")
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data_yaml


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    weights = resolve_weights(args.model.resolve())
    out_dir = args.out_dir.resolve()

    if not source.exists():
        raise SystemExit(f"Source folder not found: {source}")

    if args.clear_output and out_dir.exists():
        shutil.rmtree(out_dir)

    for folder in ("images", "labels", "annotated_all", "annotated_single"):
        for split in args.splits:
            (out_dir / folder / split).mkdir(parents=True, exist_ok=True)

    split_images: list[tuple[str, Path]] = []
    for split in args.splits:
        split_images.extend((split, path) for path in iter_split_images(source, split))

    if args.max_images is not None:
        split_images = split_images[: args.max_images]

    if not split_images:
        raise SystemExit(f"No images found in: {source}")

    model = YOLO(str(weights))
    class_names = normalize_names(model.names)
    predict_kwargs = {"conf": args.conf, "imgsz": args.imgsz, "verbose": False}
    if args.device:
        predict_kwargs["device"] = args.device

    data_yaml = write_data_yaml(out_dir, class_names)
    manifest_path = out_dir / "manifest.csv"

    processed_count = 0
    detection_count = 0
    images_with_detections = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "split",
                "source_file",
                "rotated_image",
                "label_file",
                "annotated_all_image",
                "annotated_single_image",
                "detection_index",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
        )
        writer.writeheader()

        for split, image_path in split_images:
            processed_count += 1
            image = open_rotated_rgb(image_path)
            width, height = image.size
            stem = safe_name(image_path.stem)

            rotated_image_path = out_dir / "images" / split / f"{stem}.jpg"
            label_path = out_dir / "labels" / split / f"{stem}.txt"
            annotated_all_path = out_dir / "annotated_all" / split / f"{stem}.jpg"

            image.save(rotated_image_path, quality=95, subsampling=0)

            results = model.predict(source=np.asarray(image), **predict_kwargs)
            boxes = results[0].boxes
            detections: list[Detection] = []

            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls.item()) if box.cls is not None else 0
                    confidence = float(box.conf.item()) if box.conf is not None else 0.0
                    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                    detections.append(
                        Detection(
                            class_id=class_id,
                            class_name=class_names.get(class_id, str(class_id)),
                            confidence=confidence,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                        )
                    )

            label_lines = [yolo_line(detection, width, height) for detection in detections]
            label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

            draw_detections(image, detections).save(annotated_all_path, quality=95, subsampling=0)

            if detections:
                images_with_detections += 1

            for detection_index, detection in enumerate(detections):
                detection_count += 1
                single_path = (
                    out_dir
                    / "annotated_single"
                    / split
                    / f"{stem}_det{detection_index:02d}_{safe_name(detection.class_name)}"
                    f"_conf{detection.confidence:.3f}.jpg"
                )
                draw_detections(image, [detection]).save(single_path, quality=95, subsampling=0)

                writer.writerow(
                    {
                        "split": split,
                        "source_file": str(image_path),
                        "rotated_image": str(rotated_image_path),
                        "label_file": str(label_path),
                        "annotated_all_image": str(annotated_all_path),
                        "annotated_single_image": str(single_path),
                        "detection_index": detection_index,
                        "class_id": detection.class_id,
                        "class_name": detection.class_name,
                        "confidence": f"{detection.confidence:.6f}",
                        "x1": f"{detection.x1:.0f}",
                        "y1": f"{detection.y1:.0f}",
                        "x2": f"{detection.x2:.0f}",
                        "y2": f"{detection.y2:.0f}",
                    }
                )

            if not detections:
                writer.writerow(
                    {
                        "split": split,
                        "source_file": str(image_path),
                        "rotated_image": str(rotated_image_path),
                        "label_file": str(label_path),
                        "annotated_all_image": str(annotated_all_path),
                        "annotated_single_image": "",
                        "detection_index": "",
                        "class_id": "",
                        "class_name": "",
                        "confidence": "",
                        "x1": "",
                        "y1": "",
                        "x2": "",
                        "y2": "",
                    }
                )

            print(
                f"[{processed_count}/{len(split_images)}] "
                f"{len(detections)} inside labels: {split}/{image_path.name}"
            )

    print("Done.")
    print(f"Images processed: {processed_count}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Inside labels: {detection_count}")
    print(f"Output folder: {out_dir}")
    print(f"Dataset yaml: {data_yaml}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
