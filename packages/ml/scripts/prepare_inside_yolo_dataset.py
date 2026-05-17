from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a CVAT YOLO export into an Ultralytics train/val dataset."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(r"F:\lenta_price_vision\packages\ml\src\ml\data\inside_combined_clean_cvat_yolo"),
        help="CVAT YOLO export folder with obj_train_data, labels, obj.names.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            r"F:\lenta_price_vision\packages\ml\src\ml\runs\datasets\inside_price_tag_yolo"
        ),
        help="Output Ultralytics dataset folder.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete output folder before writing the prepared dataset.",
    )
    return parser.parse_args()


def read_class_names(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [name for name in names if name]


def read_label_classes(label_path: Path) -> list[int]:
    if not label_path.exists():
        return []

    classes: list[int] = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Bad YOLO label line in {label_path}:{line_no}: {line}")

        try:
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"Bad YOLO label line in {label_path}:{line_no}: {line}") from exc

        if any(value < 0.0 or value > 1.0 for value in coords):
            raise ValueError(f"YOLO coords outside 0..1 in {label_path}:{line_no}: {line}")

        classes.append(class_id)
    return classes


def find_images(image_dir: Path) -> list[Path]:
    return sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def choose_split(
    image_paths: list[Path],
    image_classes: dict[str, list[int]],
    val_ratio: float,
    seed: int,
) -> tuple[set[str], int]:
    if not 0 < val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1.")

    target_val_count = max(1, round(len(image_paths) * val_ratio))
    all_class_counts = Counter(
        class_id for classes in image_classes.values() for class_id in set(classes)
    )
    classes_to_cover = {class_id for class_id, count in all_class_counts.items() if count >= 2}
    stems = [path.stem for path in image_paths]

    # Retry deterministic random splits until every present class appears in train and val.
    # This matters for the rare inner-tag fields.
    for offset in range(1000):
        rng = random.Random(seed + offset)
        shuffled = stems[:]
        rng.shuffle(shuffled)
        val_stems = set(shuffled[:target_val_count])
        train_stems = set(shuffled[target_val_count:])

        val_classes = {
            class_id
            for stem in val_stems
            for class_id in image_classes.get(stem, [])
        }
        train_classes = {
            class_id
            for stem in train_stems
            for class_id in image_classes.get(stem, [])
        }

        if classes_to_cover.issubset(val_classes) and classes_to_cover.issubset(train_classes):
            return val_stems, seed + offset

    # Fallback: still return a stable split rather than failing the whole preparation step.
    rng = random.Random(seed)
    shuffled = stems[:]
    rng.shuffle(shuffled)
    return set(shuffled[:target_val_count]), seed


def write_data_yaml(out_dir: Path, class_names: list[str]) -> Path:
    data_yaml = out_dir / "data.yaml"
    lines = [
        f"path: {out_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(class_names))
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data_yaml


def main() -> None:
    args = parse_args()
    source = args.source
    image_dir = source / "obj_train_data"
    label_dir = source / "labels"
    names_path = source / "obj.names"

    if not image_dir.exists():
        raise SystemExit(f"Image folder not found: {image_dir}")
    if not label_dir.exists():
        raise SystemExit(f"Label folder not found: {label_dir}")
    if not names_path.exists():
        raise SystemExit(f"Class names file not found: {names_path}")

    if args.clear_output and args.out_dir.exists():
        shutil.rmtree(args.out_dir)

    class_names = read_class_names(names_path)
    image_paths = find_images(image_dir)
    if not image_paths:
        raise SystemExit(f"No images found in: {image_dir}")

    image_classes: dict[str, list[int]] = {}
    missing_labels: list[str] = []
    object_counts = Counter()

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels.append(image_path.name)
        classes = read_label_classes(label_path)
        image_classes[image_path.stem] = classes
        object_counts.update(classes)

    bad_classes = sorted(class_id for class_id in object_counts if class_id < 0 or class_id >= len(class_names))
    if bad_classes:
        raise SystemExit(f"Label files contain class ids outside obj.names: {bad_classes}")

    val_stems, split_seed = choose_split(image_paths, image_classes, args.val_ratio, args.seed)

    split_counts = {
        "train": {"images": 0, "objects": 0, "empty_labels": 0, "class_counts": Counter()},
        "val": {"images": 0, "objects": 0, "empty_labels": 0, "class_counts": Counter()},
    }

    for split in ("train", "val"):
        (args.out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        split = "val" if image_path.stem in val_stems else "train"
        target_image = args.out_dir / "images" / split / image_path.name
        target_label = args.out_dir / "labels" / split / f"{image_path.stem}.txt"
        source_label = label_dir / f"{image_path.stem}.txt"

        shutil.copy2(image_path, target_image)
        if source_label.exists():
            shutil.copy2(source_label, target_label)
        else:
            target_label.write_text("", encoding="utf-8")

        classes = image_classes[image_path.stem]
        split_counts[split]["images"] += 1
        split_counts[split]["objects"] += len(classes)
        split_counts[split]["class_counts"].update(classes)
        if not classes:
            split_counts[split]["empty_labels"] += 1

    data_yaml = write_data_yaml(args.out_dir, class_names)
    report = {
        "source": str(source),
        "out_dir": str(args.out_dir),
        "data_yaml": str(data_yaml),
        "classes": class_names,
        "split_seed": split_seed,
        "val_ratio": args.val_ratio,
        "total_images": len(image_paths),
        "total_objects": sum(object_counts.values()),
        "missing_labels": missing_labels,
        "class_counts": {class_names[index]: object_counts.get(index, 0) for index in range(len(class_names))},
        "splits": {
            split: {
                "images": values["images"],
                "objects": values["objects"],
                "empty_labels": values["empty_labels"],
                "class_counts": {
                    class_names[index]: values["class_counts"].get(index, 0)
                    for index in range(len(class_names))
                },
            }
            for split, values in split_counts.items()
        },
    }
    report_path = args.out_dir / "prepare_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE] Inside-price-tag YOLO dataset prepared")
    print(f"  output_dir: {args.out_dir}")
    print(f"  data_yaml:  {data_yaml}")
    print(f"  images:     {len(image_paths)}")
    print(f"  objects:    {sum(object_counts.values())}")
    print(f"  train:      {split_counts['train']['images']} images")
    print(f"  val:        {split_counts['val']['images']} images")
    print(f"  empty labels created: {len(missing_labels)}")
    print(f"  report:     {report_path}")


if __name__ == "__main__":
    main()
