from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NAMES = ROOT / "packages" / "ml" / "data" / "inside_data" / "obj.names"
DEFAULT_DATASET = ROOT / "packages" / "ml" / "data" / "inside_yolo_dataset"
DEFAULT_PSEUDO_IMAGES = DEFAULT_DATASET / "images" / "unlabeled"
DEFAULT_EXISTING_IMAGES = DEFAULT_DATASET / "images" / "train"
DEFAULT_EXISTING_LABELS = DEFAULT_DATASET / "labels" / "train"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a CVAT YOLO 1.1 import archive from YOLO prediction output."
    )
    parser.add_argument(
        "--pred-dir",
        required=True,
        type=Path,
        help="YOLO predict directory with labels/*.txt.",
    )
    parser.add_argument(
        "--pseudo-image-dir",
        type=Path,
        default=DEFAULT_PSEUDO_IMAGES,
        help=(
            "Clean images that were pseudo-labeled. "
            "Defaults to inside_yolo_dataset/images/unlabeled."
        ),
    )
    parser.add_argument(
        "--existing-image-dir",
        type=Path,
        default=DEFAULT_EXISTING_IMAGES,
        help="Clean images with manual labels. Defaults to inside_yolo_dataset/images/train.",
    )
    parser.add_argument(
        "--existing-label-dir",
        type=Path,
        default=DEFAULT_EXISTING_LABELS,
        help="Manual YOLO labels. Defaults to inside_yolo_dataset/labels/train.",
    )
    parser.add_argument("--names", type=Path, default=DEFAULT_NAMES)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--zip-name", default="inside_combined_clean_cvat_yolo.zip")
    return parser


def read_classes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def format_coord(value: str) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def normalize_label(src: Path | None, dst: Path, class_count: int) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src is None or not src.exists():
        dst.write_text("", encoding="utf-8")
        return 0, 0

    rows: list[str] = []
    dropped_extra_columns = 0
    for line_number, raw_line in enumerate(src.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"{src}:{line_number}: expected at least 5 YOLO columns")
        class_id = int(float(parts[0]))
        if not 0 <= class_id < class_count:
            raise ValueError(f"{src}:{line_number}: class id {class_id} is out of range")
        coords = [format_coord(value) for value in parts[1:5]]
        rows.append(" ".join([str(class_id), *coords]))
        if len(parts) > 5:
            dropped_extra_columns += len(parts) - 5

    dst.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows), dropped_extra_columns


def write_zip(source_dir: Path, zip_path: Path) -> int:
    count = 0
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            archive.write(path, path.relative_to(source_dir).as_posix())
            count += 1
    return count


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def copy_clean_image_with_label(
    image_path: Path,
    label_src: Path | None,
    data_dir: Path,
    class_count: int,
) -> tuple[int, int]:
    image_dst = data_dir / image_path.name
    label_dst = data_dir / f"{image_path.stem}.txt"
    shutil.copy2(image_path, image_dst)
    return normalize_label(label_src, label_dst, class_count)


def run(args: argparse.Namespace) -> dict[str, object]:
    pred_dir = args.pred_dir
    labels_dir = pred_dir / "labels"
    out_dir = args.out_dir or (pred_dir / "cvat_yolo_import_combined_clean")
    data_dir = out_dir / "obj_train_data"
    zip_path = pred_dir / args.zip_name

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Prediction labels directory not found: {labels_dir}")
    if not args.existing_label_dir.exists():
        raise FileNotFoundError(f"Existing labels directory not found: {args.existing_label_dir}")

    classes = read_classes(args.names)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    pseudo_images = list_images(args.pseudo_image_dir)
    existing_images = list_images(args.existing_image_dir)
    seen_names: set[str] = set()
    train_lines: list[str] = []
    label_files = 0
    empty_labels = 0
    boxes = 0
    dropped_extra_columns = 0
    source_counts = {"existing": 0, "pseudo": 0}

    def add_item(image_path: Path, label_src: Path | None, source: str) -> None:
        nonlocal boxes, dropped_extra_columns, empty_labels, label_files
        if image_path.name in seen_names:
            raise ValueError(f"Duplicate image name in combined dataset: {image_path.name}")
        seen_names.add(image_path.name)
        box_count, dropped = copy_clean_image_with_label(
            image_path=image_path,
            label_src=label_src,
            data_dir=data_dir,
            class_count=len(classes),
        )
        label_files += 1
        boxes += box_count
        dropped_extra_columns += dropped
        if box_count == 0:
            empty_labels += 1
        train_lines.append(f"obj_train_data/{image_path.name}")
        source_counts[source] += 1

    for image_path in existing_images:
        add_item(
            image_path=image_path,
            label_src=args.existing_label_dir / f"{image_path.stem}.txt",
            source="existing",
        )

    for image_path in pseudo_images:
        add_item(
            image_path=image_path,
            label_src=labels_dir / f"{image_path.stem}.txt",
            source="pseudo",
        )

    (out_dir / "obj.names").write_text("\n".join(classes) + "\n", encoding="utf-8")
    (out_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (out_dir / "obj.data").write_text(
        "\n".join(
            [
                f"classes = {len(classes)}",
                "train = train.txt",
                "valid = train.txt",
                "names = obj.names",
                "backup = backup/",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary: dict[str, object] = {
        "pred_dir": str(pred_dir),
        "out_dir": str(out_dir),
        "zip_path": str(zip_path),
        "classes": classes,
        "images": len(train_lines),
        "existing_images": source_counts["existing"],
        "pseudo_images": source_counts["pseudo"],
        "pseudo_image_dir": str(args.pseudo_image_dir),
        "existing_image_dir": str(args.existing_image_dir),
        "existing_label_dir": str(args.existing_label_dir),
        "label_files": label_files,
        "empty_label_files": empty_labels,
        "boxes": boxes,
        "dropped_extra_columns": dropped_extra_columns,
        "format": "CVAT YOLO 1.1",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["zip_files"] = write_zip(out_dir, zip_path)
    (out_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
