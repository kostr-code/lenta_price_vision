"""
scripts/crop_detections.py — run YOLO detector over a dataset and save detected crops.

Used after Stage 1 training to extract price-tag crops for Stage 2 pseudo-labeling.

Usage:
    uv run python scripts/crop_detections.py \\
        --dataset runs/datasets/lenta_yolo_tiled \\
        --weights models/price_tag_yolo.pt \\
        --out-subdir crops_for_stage2 \\
        --conf 0.25 --imgsz 1280 --device 0

Output:
    <dataset>/crops_for_stage2/train/<stem>_crop<N>.jpg
    <dataset>/crops_for_stage2/val/<stem>_crop<N>.jpg
    <dataset>/crops_for_stage2/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


def run_detector(weights: Path, img_dir: Path, out_dir: Path, conf: float, imgsz: int, device: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics required: uv add ultralytics")
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python required: uv add opencv-python")

    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    records = []

    for img_path in sorted(img_dir.glob("*.jpg")):
        results = model.predict(str(img_path), conf=conf, imgsz=imgsz, device=device, verbose=False)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        for res in results:
            for crop_idx, box in enumerate(res.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                stem = img_path.stem
                out_name = f"{stem}_crop{crop_idx:02d}.jpg"
                out_path = out_dir / out_name
                cv2.imwrite(str(out_path), crop)
                records.append({
                    "filename": out_name,
                    "source": img_path.name,
                    "conf": f"{float(box.conf[0]):.4f}",
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })

    return records


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Dataset root (contains images/train, images/val)")
    p.add_argument("--weights", required=True, help="Path to trained YOLO .pt weights")
    p.add_argument("--out-subdir", default="crops_for_stage2")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="0")
    p.add_argument("--clear-output", action="store_true", help="Delete out-subdir before writing")
    args = p.parse_args()

    dataset = Path(args.dataset)
    weights = Path(args.weights)
    if not weights.exists():
        print(f"Error: weights not found: {weights}")
        return 1

    out_root = dataset / args.out_subdir
    if args.clear_output and out_root.exists():
        shutil.rmtree(out_root)

    all_records = []
    for split in args.splits:
        img_dir = dataset / "images" / split
        if not img_dir.exists():
            print(f"  [skip] {img_dir} not found")
            continue
        out_dir = out_root / split
        print(f"[crops] Processing {img_dir} → {out_dir}")
        records = run_detector(weights, img_dir, out_dir, args.conf, args.imgsz, args.device)
        for r in records:
            r["split"] = split
        all_records.extend(records)
        print(f"  saved {len(records)} crops")

    # Write manifest
    manifest = out_root / "manifest.csv"
    if all_records:
        with manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "split", "source", "conf", "x1", "y1", "x2", "y2"])
            writer.writeheader()
            writer.writerows(all_records)
        print(f"[crops] manifest → {manifest}  ({len(all_records)} total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
