from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".heic", ".heif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO over HEIC images and save detected price-tag crops."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(r"F:\lenta_price_vision\packages\ml\датасет лента"),
        help="Folder with HEIC images.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(r"F:\lenta_price_vision\models\best.pt"),
        help="YOLO weights path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"F:\lenta_price_vision\packages\ml\outputs\crops_custom_data"),
        help="Output folder for crops.",
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


def register_heic_reader() -> None:
    try:
        import pillow_heif
    except ImportError as exc:
        raise SystemExit(
            "HEIC support is missing. Run this script with:\n"
            "uv run --with pillow-heif python scripts\\crop_heic_price_tags.py"
        ) from exc

    pillow_heif.register_heif_opener()


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_.") or "image"


def iter_heic_images(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def open_heic_as_rgb(path: Path) -> Image.Image:
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
    register_heic_reader()

    if not args.source.exists():
        raise SystemExit(f"Source folder not found: {args.source}")
    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}")

    if args.clear_output and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_heic_images(args.source)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    if not image_paths:
        raise SystemExit(f"No HEIC images found in: {args.source}")

    model = YOLO(str(args.weights))
    predict_kwargs = {
        "conf": args.conf,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    manifest_path = args.out_dir / "manifest.csv"
    crop_count = 0
    processed_count = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "crop_file",
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

        for image_path in image_paths:
            processed_count += 1
            image = open_heic_as_rgb(image_path)
            width, height = image.size

            # Ultralytics accepts numpy RGB images; crops are taken from the PIL source.
            results = model.predict(source=np.asarray(image), **predict_kwargs)
            result = results[0]
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                print(f"[{processed_count}/{len(image_paths)}] no detections: {image_path.name}")
                continue

            relative_stem = safe_name(str(image_path.relative_to(args.source).with_suffix("")))
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
                crop_path = args.out_dir / crop_name
                crop.save(crop_path, quality=95, subsampling=0)

                writer.writerow(
                    {
                        "crop_file": str(crop_path),
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

            print(
                f"[{processed_count}/{len(image_paths)}] "
                f"{len(boxes)} detections: {image_path.name}"
            )

    print(f"Done. Images processed: {processed_count}")
    print(f"Crops saved: {crop_count}")
    print(f"Output folder: {args.out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
