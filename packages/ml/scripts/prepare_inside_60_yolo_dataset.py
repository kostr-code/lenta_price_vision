from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an Ultralytics YOLO dataset from the 60_inside_data CVAT export. "
            "The export contains labels only, so matching images are copied from an external image root."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            r"F:\lenta_price_vision\packages\ml\src\ml\runs\datasets"
            r"\lenta_yolo_49_43_26_prop8\crops_best_pt\60_inside_data"
        ),
        help="CVAT YOLO export folder with obj.names and label txt files.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path(
            r"F:\lenta_price_vision\packages\ml\src\ml\runs\datasets"
            r"\lenta_yolo_49_43_26_prop8\crops_best_pt"
            r"\inside_rot90ccw_best_dedup_cvat\images"
        ),
        help="Root with matching images in train/ and val/ folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"F:\lenta_price_vision\packages\ml\src\ml\runs\datasets\inside_60_yolo"),
        help="Output Ultralytics dataset folder.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete output folder before writing the prepared dataset.",
    )
    return parser.parse_args()


def find_label_dir(source: Path) -> Path:
    candidates = [
        source / "labels",
        source / "obj_train_data" / "obj_train_data",
        source / "obj_train_data",
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*.txt")):
            return candidate
    raise SystemExit(f"Could not find YOLO label txt files under: {source}")


def read_class_names(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Class names file not found: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [name for name in names if name]


def infer_split(stem: str) -> str:
    if stem.startswith("val_"):
        return "val"
    return "train"


def find_image(image_root: Path, split: str, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = image_root / split / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    for extension in IMAGE_EXTENSIONS:
        candidate = image_root / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def validate_label_file(label_path: Path, class_count: int) -> tuple[int, Counter[int]]:
    line_count = 0
    class_counts: Counter[int] = Counter()

    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Bad YOLO line in {label_path}:{line_no}: {line}")
        try:
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"Bad YOLO line in {label_path}:{line_no}: {line}") from exc
        if class_id < 0 or class_id >= class_count:
            raise ValueError(f"Class id outside obj.names in {label_path}:{line_no}: {line}")
        if any(value < 0.0 or value > 1.0 for value in coords):
            raise ValueError(f"YOLO coord outside 0..1 in {label_path}:{line_no}: {line}")
        line_count += 1
        class_counts[class_id] += 1

    return line_count, class_counts


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
    source = args.source.resolve()
    image_root = args.image_root.resolve()
    out_dir = args.out_dir.resolve()

    if not source.exists():
        raise SystemExit(f"Source folder not found: {source}")
    if not image_root.exists():
        raise SystemExit(f"Image root not found: {image_root}")

    label_dir = find_label_dir(source)
    class_names = read_class_names(source / "obj.names")
    label_paths = sorted(label_dir.glob("*.txt"))
    if not label_paths:
        raise SystemExit(f"No label files found in: {label_dir}")

    if args.clear_output and out_dir.exists():
        shutil.rmtree(out_dir)

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    report = {
        "source": str(source),
        "label_dir": str(label_dir),
        "image_root": str(image_root),
        "out_dir": str(out_dir),
        "classes": class_names,
        "splits": {
            "train": {"images": 0, "labels": 0, "objects": 0, "class_counts": Counter()},
            "val": {"images": 0, "labels": 0, "objects": 0, "class_counts": Counter()},
        },
        "missing_images": [],
    }

    for label_path in label_paths:
        split = infer_split(label_path.stem)
        image_path = find_image(image_root, split, label_path.stem)
        if image_path is None:
            report["missing_images"].append(str(label_path))
            continue

        object_count, class_counts = validate_label_file(label_path, len(class_names))
        target_image = out_dir / "images" / split / image_path.name
        target_label = out_dir / "labels" / split / label_path.name

        shutil.copy2(image_path, target_image)
        shutil.copy2(label_path, target_label)

        split_report = report["splits"][split]
        split_report["images"] += 1
        split_report["labels"] += 1
        split_report["objects"] += object_count
        split_report["class_counts"].update(class_counts)

    if report["missing_images"]:
        missing_preview = "\n".join(report["missing_images"][:20])
        raise SystemExit(
            f"Missing images for {len(report['missing_images'])} label files. First missing:\n"
            f"{missing_preview}"
        )

    data_yaml = write_data_yaml(out_dir, class_names)
    report["data_yaml"] = str(data_yaml)
    report["total_images"] = sum(values["images"] for values in report["splits"].values())
    report["total_objects"] = sum(values["objects"] for values in report["splits"].values())
    report["class_counts"] = {
        class_name: sum(values["class_counts"].get(index, 0) for values in report["splits"].values())
        for index, class_name in enumerate(class_names)
    }
    report["splits"] = {
        split: {
            "images": values["images"],
            "labels": values["labels"],
            "objects": values["objects"],
            "class_counts": {
                class_names[index]: values["class_counts"].get(index, 0)
                for index in range(len(class_names))
            },
        }
        for split, values in report["splits"].items()
    }

    report_path = out_dir / "prepare_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE] 60_inside_data prepared for Ultralytics YOLO")
    print(f"  output_dir: {out_dir}")
    print(f"  data_yaml:  {data_yaml}")
    print(f"  train:      {report['splits']['train']['images']} images")
    print(f"  val:        {report['splits']['val']['images']} images")
    print(f"  objects:    {report['total_objects']}")
    print(f"  report:     {report_path}")


if __name__ == "__main__":
    main()
