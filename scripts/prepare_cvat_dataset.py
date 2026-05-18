"""
scripts/prepare_cvat_dataset.py — CVAT YOLO export → clean YOLO dataset.

Handles the standard CVAT YOLO export layout:
    obj_train_data/   (or obj.data, obj.names)
    labels/
    train.txt

Usage:
    uv run python scripts/prepare_cvat_dataset.py \\
        --source path/to/cvat_export \\
        --out-dir runs/datasets/lenta_inside_yolo \\
        --val-ratio 0.2 --seed 42

For the special "60_inside_data" layout where images live elsewhere:
    uv run python scripts/prepare_cvat_dataset.py \\
        --source path/to/cvat_export \\
        --image-root path/to/images \\
        --out-dir runs/datasets/lenta_inside_yolo

Output:
    data.yaml with names read from obj.names (or classes.txt)
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path


def read_class_names(source: Path) -> list[str]:
    for candidate in ["obj.names", "classes.txt", "labels.txt"]:
        f = source / candidate
        if f.exists():
            return [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    return ["price_tag_element"]  # fallback


def collect_image_label_pairs(source: Path, image_root: Path | None) -> list[tuple[Path, Path | None]]:
    """Find all (image, label) pairs from CVAT export layout."""
    pairs = []
    img_extensions = {".jpg", ".jpeg", ".png"}

    # Standard CVAT: images under obj_train_data/ or obj.data/
    for img_dir_name in ["obj_train_data", "obj.data", "images"]:
        img_dir = source / img_dir_name
        if img_dir.exists():
            for img in sorted(img_dir.rglob("*")):
                if img.suffix.lower() in img_extensions:
                    lbl = img.with_suffix(".txt")
                    pairs.append((img, lbl if lbl.exists() else None))
            break

    # Special case: images from external image_root, labels inside source
    if not pairs and image_root:
        lbl_dir = source / "labels"
        if not lbl_dir.exists():
            lbl_dir = source
        for lbl in sorted(lbl_dir.rglob("*.txt")):
            stem = lbl.stem
            for ext in img_extensions:
                img = image_root / (stem + ext)
                if img.exists():
                    pairs.append((img, lbl))
                    break

    return pairs


def write_dataset(
    pairs: list[tuple[Path, Path | None]],
    names: list[str],
    out_dir: Path,
    val_fraction: float,
    seed: int,
) -> None:
    random.Random(seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_fraction)) if len(pairs) > 1 else 0
    val_set = set(range(n_val))

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for idx, (img_path, lbl_path) in enumerate(pairs):
        split = "val" if idx in val_set else "train"
        shutil.copy2(img_path, out_dir / "images" / split / img_path.name)
        if lbl_path and lbl_path.exists():
            shutil.copy2(lbl_path, out_dir / "labels" / split / img_path.with_suffix(".txt").name)
        else:
            # Empty label file (background sample)
            (out_dir / "labels" / split / img_path.with_suffix(".txt").name).write_text("")

    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n{names_yaml}\n",
        encoding="utf-8",
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="CVAT YOLO export directory")
    p.add_argument("--image-root", help="External image root (for 60_inside_data layout)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clear-output", action="store_true")
    args = p.parse_args()

    source = Path(args.source)
    out_dir = Path(args.out_dir)
    if not source.exists():
        print(f"Error: --source not found: {source}")
        return 1
    if args.clear_output and out_dir.exists():
        shutil.rmtree(out_dir)

    image_root = Path(args.image_root) if args.image_root else None
    names = read_class_names(source)
    pairs = collect_image_label_pairs(source, image_root)

    if not pairs:
        print("Error: no image/label pairs found in source")
        return 1

    print(f"[cvat] {len(pairs)} pairs  classes: {names}")
    write_dataset(pairs, names, out_dir, args.val_ratio, args.seed)
    print(f"[cvat] dataset → {out_dir / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
