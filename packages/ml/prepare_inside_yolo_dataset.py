from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "packages" / "ml" / "data" / "inside_data"
DEFAULT_OUT = ROOT / "packages" / "ml" / "data" / "inside_yolo_dataset"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a no-split YOLO dataset for price-tag internals."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--include-empty-labels",
        action="store_true",
        help=(
            "Put empty-label images into train as background images. "
            "Default keeps them as unlabeled."
        ),
    )
    return parser


def posix(path: Path) -> str:
    return path.resolve().as_posix()


def read_classes(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [name for name in names if name]


def find_images(source: Path) -> dict[str, Path]:
    images_root = source / "images"
    images: dict[str, Path] = {}
    for path in sorted(images_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images[path.stem] = path
    return images


def validate_label(label_path: Path, class_count: int) -> tuple[int, dict[int, int]]:
    boxes = 0
    class_hist: dict[int, int] = {}
    lines = label_path.read_text(encoding="utf-8").splitlines()
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{label_path}:{line_number}: expected 5 YOLO columns, got {len(parts)}"
            )
        class_id = int(float(parts[0]))
        if not 0 <= class_id < class_count:
            raise ValueError(f"{label_path}:{line_number}: class id {class_id} is out of range")
        coords = [float(value) for value in parts[1:]]
        if any(value < 0 or value > 1 for value in coords):
            raise ValueError(f"{label_path}:{line_number}: bbox values must be normalized to 0..1")
        boxes += 1
        class_hist[class_id] = class_hist.get(class_id, 0) + 1
    return boxes, class_hist


def copy_pair(image_path: Path, label_path: Path, image_out: Path, label_out: Path) -> None:
    image_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_out)
    text = label_path.read_text(encoding="utf-8").strip()
    label_out.write_text(text + ("\n" if text else ""), encoding="utf-8")


def write_data_yaml(out: Path, classes: list[str]) -> None:
    lines = [
        f"path: {posix(out)}",
        "train: images/train",
        "val: images/train",
        "",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(classes))
    (out / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_list(path: Path, paths: list[Path], base: Path) -> None:
    lines = [item.relative_to(base).as_posix() for item in paths]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    source = args.source
    out = args.out
    label_dir = source / "obj_train_data"
    classes = read_classes(source / "obj.names")
    images = find_images(source)

    train_images: list[Path] = []
    unlabeled_images: list[Path] = []
    class_hist = {index: 0 for index in range(len(classes))}
    total_boxes = 0
    empty_labels = 0
    missing_images: list[str] = []

    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = images.get(label_path.stem)
        if image_path is None:
            missing_images.append(label_path.name)
            continue

        box_count, label_hist = validate_label(label_path, len(classes))
        if box_count > 0 or args.include_empty_labels:
            image_out = out / "images" / "train" / image_path.name
            label_out = out / "labels" / "train" / f"{image_path.stem}.txt"
            copy_pair(image_path, label_path, image_out, label_out)
            train_images.append(image_out)
            total_boxes += box_count
            for class_id, count in label_hist.items():
                class_hist[class_id] += count
        else:
            empty_labels += 1
            unlabeled_out = out / "images" / "unlabeled" / image_path.name
            unlabeled_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, unlabeled_out)
            unlabeled_images.append(unlabeled_out)

    labeled_stems = {path.stem for path in label_dir.glob("*.txt")}
    for stem, image_path in sorted(images.items()):
        if stem in labeled_stems:
            continue
        unlabeled_out = out / "images" / "unlabeled" / image_path.name
        unlabeled_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, unlabeled_out)
        unlabeled_images.append(unlabeled_out)

    write_data_yaml(out, classes)
    write_list(out / "train.txt", train_images, out)
    write_list(out / "unlabeled.txt", unlabeled_images, out)

    summary: dict[str, object] = {
        "source": str(source),
        "out": str(out),
        "classes": classes,
        "class_counts": {classes[index]: class_hist[index] for index in range(len(classes))},
        "train_images": len(train_images),
        "unlabeled_images": len(unlabeled_images),
        "empty_label_files": empty_labels,
        "total_boxes": total_boxes,
        "missing_images_for_labels": missing_images,
        "split": "none; val points to train in data.yaml",
    }
    (out / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
