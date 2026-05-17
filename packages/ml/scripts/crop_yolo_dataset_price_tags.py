from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO over an Ultralytics dataset and save detected price-tag crops."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            r"F:\lenta_price_vision\packages\ml\src\ml\runs\datasets\lenta_yolo_49_43_26_prop8"
        ),
        help="Dataset root with images/train and images/val.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(r"F:\lenta_price_vision\models\best.pt"),
        help="YOLO weights path.",
    )
    parser.add_argument(
        "--out-subdir",
        default="crops_best_pt",
        help="Output folder name inside the dataset root.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size.")
    parser.add_argument("--device", default=None, help='Device for YOLO, for example "0" or "cpu".')
    parser.add_argument(
        "--padding",
        type=float,
        default=0.04,
        help="Extra crop padding as a fraction of detected box size.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to process.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete output folder before saving new crops.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit for a quick test run.",
    )
    return parser.parse_args()


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_.") or "image"


def iter_split_images(dataset: Path, split: str) -> list[Path]:
    image_dir = dataset / "images" / split
    if not image_dir.exists():
        raise SystemExit(f"Image split folder not found: {image_dir}")

    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def open_image_as_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def expand_box(
    xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = box_width * padding_ratio
    pad_y = box_height * padding_ratio

    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(width, int(round(x2 + pad_x)))
    bottom = min(height, int(round(y2 + pad_y)))
    return left, top, right, bottom


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    weights = args.weights.resolve()
    out_dir = dataset / args.out_subdir

    if not dataset.exists():
        raise SystemExit(f"Dataset not found: {dataset}")
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")

    if args.clear_output and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_images: list[tuple[str, Path]] = []
    for split in args.splits:
        images = iter_split_images(dataset, split)
        split_images.extend((split, path) for path in images)

    if args.max_images is not None:
        split_images = split_images[: args.max_images]

    if not split_images:
        raise SystemExit(f"No images found in dataset: {dataset}")

    for split in args.splits:
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    predict_kwargs = {
        "conf": args.conf,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    manifest_path = out_dir / "manifest.csv"
    crop_count = 0
    processed_count = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "crop_file",
                "split",
                "source_file",
                "source_width",
                "source_height",
                "class_id",
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
            image = open_image_as_rgb(image_path)
            width, height = image.size

            results = model.predict(source=np.asarray(image), **predict_kwargs)
            result = results[0]
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                print(f"[{processed_count}/{len(split_images)}] no detections: {split}/{image_path.name}")
                continue

            relative_stem = safe_name(str(image_path.relative_to(dataset / "images" / split).with_suffix("")))
            for detection_index, box in enumerate(boxes):
                confidence = float(box.conf.item()) if box.conf is not None else 0.0
                class_id = int(box.cls.item()) if box.cls is not None else 0
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                left, top, right, bottom = expand_box((x1, y1, x2, y2), width, height, args.padding)

                if right <= left or bottom <= top:
                    continue

                crop = image.crop((left, top, right, bottom))
                crop_count += 1
                crop_name = (
                    f"{crop_count:06d}_{relative_stem}_det{detection_index:02d}"
                    f"_conf{confidence:.3f}.jpg"
                )
                crop_path = out_dir / split / crop_name
                crop.save(crop_path, quality=95, subsampling=0)

                writer.writerow(
                    {
                        "crop_file": str(crop_path),
                        "split": split,
                        "source_file": str(image_path),
                        "source_width": width,
                        "source_height": height,
                        "class_id": class_id,
                        "confidence": f"{confidence:.6f}",
                        "x1": f"{left:.0f}",
                        "y1": f"{top:.0f}",
                        "x2": f"{right:.0f}",
                        "y2": f"{bottom:.0f}",
                    }
                )

            print(f"[{processed_count}/{len(split_images)}] {len(boxes)} detections: {split}/{image_path.name}")

    print(f"Done. Images processed: {processed_count}")
    print(f"Crops saved: {crop_count}")
    print(f"Output folder: {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
