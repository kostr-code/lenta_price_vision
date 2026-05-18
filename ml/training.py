"""
ml/training.py — YOLO dataset builder for Lenta price-tag detection.

Builds a YOLO-format dataset (images/ + labels/ + data.yaml) from
labeled video+CSV pairs.  Supports full-frame and tiled (640px) modes.

Usage:
    uv run python -m ml.training \\
        --data-dir data/Данные \\
        --out-dir  runs/datasets/lenta_yolo_tiled \\
        --tiled \\
        --val-ratio 0.2
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path


# ── Geometry ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        return self.width * self.height


def bbox_iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a.x_min, b.x_min), max(a.y_min, b.y_min)
    ix2, iy2 = min(a.x_max, b.x_max), min(a.y_max, b.y_max)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


# ── Tiling ────────────────────────────────────────────────────────────────────

def iter_tiles(
    width: int,
    height: int,
    tile_size: int = 640,
    stride: int = 512,
) -> list[tuple[int, int, int, int]]:
    """Yield (x1, y1, x2, y2) tiles covering width×height."""
    def starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        pts = list(range(0, length - tile_size + 1, stride))
        if pts[-1] != length - tile_size:
            pts.append(length - tile_size)
        return pts

    return [
        (x, y, min(width, x + tile_size), min(height, y + tile_size))
        for y in starts(height)
        for x in starts(width)
    ]


# ── Data discovery ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LabeledSequence:
    name: str
    video_path: Path
    csv_path: Path


def discover_sequences(data_dir: Path) -> list[LabeledSequence]:
    """Find all (video, csv) pairs inside data_dir subdirectories."""
    seqs = []
    for d in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        csvs = sorted(d.glob("*.csv"))
        vids = sorted(d.glob("*.mp4"))
        if not csvs or not vids:
            continue
        csv_path = csvs[0]
        same = [v for v in vids if v.stem == csv_path.stem]
        video_path = same[0] if same else max(vids, key=lambda p: p.stat().st_size)
        seqs.append(LabeledSequence(name=d.name, video_path=video_path, csv_path=csv_path))
    return seqs


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return [{k.strip(): str(v).strip() for k, v in row.items()} for row in csv.DictReader(f)]


def parse_float(value: object) -> float:
    try:
        return float(str(value or "").strip().replace(",", "."))
    except ValueError:
        return 0.0


# ── Coordinate helpers ────────────────────────────────────────────────────────

def row_to_bbox(row: dict[str, str], frame_w: int, frame_h: int) -> BBox | None:
    x1, y1 = parse_float(row.get("x_min")), parse_float(row.get("y_min"))
    x2, y2 = parse_float(row.get("x_max")), parse_float(row.get("y_max"))
    x1, x2 = sorted([max(0.0, min(float(frame_w), x1)), max(0.0, min(float(frame_w), x2))])
    y1, y2 = sorted([max(0.0, min(float(frame_h), y1)), max(0.0, min(float(frame_h), y2))])
    bbox = BBox(x1, y1, x2, y2)
    return bbox if bbox.width >= 3 and bbox.height >= 3 else None


def bbox_to_yolo(bbox: BBox, w: int, h: int) -> tuple[float, float, float, float] | None:
    if bbox.width < 3 or bbox.height < 3:
        return None
    return (
        (bbox.x_min + bbox.width / 2) / w,
        (bbox.y_min + bbox.height / 2) / h,
        bbox.width / w,
        bbox.height / h,
    )


def clip_to_tile(bbox: BBox, tile: tuple[int, int, int, int]) -> BBox | None:
    tx1, ty1, tx2, ty2 = tile
    cx1, cy1 = max(bbox.x_min, tx1), max(bbox.y_min, ty1)
    cx2, cy2 = min(bbox.x_max, tx2), min(bbox.y_max, ty2)
    clipped = BBox(cx1, cy1, cx2, cy2)
    return clipped if clipped.width >= 3 and clipped.height >= 3 else None


def dedup_labels(labels: list[str], iou_thr: float = 0.92) -> list[str]:
    kept, boxes = [], []
    for line in labels:
        parts = line.split()
        if len(parts) != 5:
            continue
        cx, cy, bw, bh = map(float, parts[1:])
        b = BBox(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
        if any(bbox_iou(b, prev) > iou_thr for prev in boxes):
            continue
        boxes.append(b)
        kept.append(line)
    return kept


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "frame"


# ── Frame writing ─────────────────────────────────────────────────────────────

def write_full_frame(cv2, frame, rows, img_path: Path, lbl_path: Path) -> tuple[int, int]:
    h, w = frame.shape[:2]
    labels = [
        "0 " + " ".join(f"{v:.6f}" for v in yolo)
        for row in rows
        if (bbox := row_to_bbox(row, w, h)) and (yolo := bbox_to_yolo(bbox, w, h))
    ]
    labels = dedup_labels(labels)
    if not labels:
        return 0, 0
    cv2.imwrite(str(img_path), frame)
    lbl_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    return 1, len(labels)


def write_tiled_frame(
    cv2, frame, rows,
    img_dir: Path, lbl_dir: Path, stem: str,
    tile_size: int, tile_stride: int, min_visibility: float,
) -> tuple[int, int]:
    h, w = frame.shape[:2]
    bboxes = [b for row in rows if (b := row_to_bbox(row, w, h))]
    if not bboxes:
        return 0, 0

    written_imgs, written_lbls = 0, 0
    for i, tile in enumerate(iter_tiles(w, h, tile_size, tile_stride)):
        tx1, ty1, tx2, ty2 = tile
        tw, th = tx2 - tx1, ty2 - ty1
        labels = []
        for src in bboxes:
            clipped = clip_to_tile(src, tile)
            if clipped is None:
                continue
            if clipped.area / max(src.area, 1.0) < min_visibility:
                continue
            local = BBox(
                clipped.x_min - tx1, clipped.y_min - ty1,
                clipped.x_max - tx1, clipped.y_max - ty1,
            )
            yolo = bbox_to_yolo(local, tw, th)
            if yolo:
                labels.append("0 " + " ".join(f"{v:.6f}" for v in yolo))
        labels = dedup_labels(labels)
        if not labels:
            continue
        tile_img = frame[ty1:ty2, tx1:tx2]
        cv2.imwrite(str(img_dir / f"{stem}_t{i:03d}.jpg"), tile_img)
        (lbl_dir / f"{stem}_t{i:03d}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        written_imgs += 1
        written_lbls += len(labels)
    return written_imgs, written_lbls


# ── Main builder ──────────────────────────────────────────────────────────────

def build_yolo_dataset(
    data_dir: Path,
    output_dir: Path,
    val_fraction: float = 0.2,
    seed: int = 42,
    tiled: bool = False,
    tile_size: int = 640,
    tile_stride: int = 512,
    min_box_visibility: float = 0.25,
) -> dict[str, object]:
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python required: uv add opencv-python")

    output_dir.mkdir(parents=True, exist_ok=True)
    img_dirs = {s: output_dir / "images" / s for s in ("train", "val")}
    lbl_dirs = {s: output_dir / "labels" / s for s in ("train", "val")}
    for d in [*img_dirs.values(), *lbl_dirs.values()]:
        d.mkdir(parents=True, exist_ok=True)

    # Collect all (sequence, timestamp, rows) triples
    frames = []
    for seq in discover_sequences(data_dir):
        by_ts: dict[int, list[dict]] = {}
        for row in read_csv(seq.csv_path):
            ts = int(parse_float(row.get("frame_timestamp", "0")))
            by_ts.setdefault(ts, []).append(row)
        for ts, rows in by_ts.items():
            frames.append((seq, ts, rows))

    random.Random(seed).shuffle(frames)
    n_val = max(1, int(len(frames) * val_fraction)) if len(frames) > 1 else 0
    val_set = set(range(n_val))

    counts = {"train": 0, "val": 0}
    total_labels = 0

    for idx, (seq, ts_ms, rows) in enumerate(frames):
        split = "val" if idx in val_set else "train"
        cap = cv2.VideoCapture(str(seq.video_path))
        if not cap.isOpened():
            print(f"  [warn] Cannot open {seq.video_path}")
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0, ts_ms))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            continue

        stem = safe_name(f"{seq.name}_{ts_ms:08d}")
        if tiled:
            ni, nl = write_tiled_frame(
                cv2, frame, rows,
                img_dirs[split], lbl_dirs[split], stem,
                tile_size, tile_stride, min_box_visibility,
            )
        else:
            ni, nl = write_full_frame(
                cv2, frame, rows,
                img_dirs[split] / f"{stem}.jpg",
                lbl_dirs[split] / f"{stem}.txt",
            )
        counts[split] += ni
        total_labels += nl
        print(f"  [{split}] {stem}: {ni} img, {nl} labels")

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {output_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: price_tag\n",
        encoding="utf-8",
    )

    return {
        "data_yaml": str(data_yaml),
        "train_images": counts["train"],
        "val_images": counts["val"],
        "total_labels": total_labels,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Build YOLO dataset from labeled video/CSV pairs")
    p.add_argument("--data-dir", required=True, help="Directory with subdirs containing .mp4 + .csv")
    p.add_argument("--out-dir", required=True, help="Output dataset directory")
    p.add_argument("--tiled", action="store_true", help="Split frames into 640px tiles")
    p.add_argument("--tile-size", type=int, default=640)
    p.add_argument("--tile-stride", type=int, default=512)
    p.add_argument("--min-visibility", type=float, default=0.25, help="Min bbox area fraction in tile")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: --data-dir not found: {data_dir}")
        return 1

    print(f"[dataset] data_dir={data_dir}  out={args.out_dir}  tiled={args.tiled}")
    result = build_yolo_dataset(
        data_dir=data_dir,
        output_dir=Path(args.out_dir),
        val_fraction=args.val_ratio,
        seed=args.seed,
        tiled=args.tiled,
        tile_size=args.tile_size,
        tile_stride=args.tile_stride,
        min_box_visibility=args.min_visibility,
    )
    print(f"\n[done] train={result['train_images']}  val={result['val_images']}"
          f"  labels={result['total_labels']}")
    print(f"[done] data.yaml → {result['data_yaml']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
