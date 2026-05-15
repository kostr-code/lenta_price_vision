"""
track_and_crop.py — YOLO tracking → лучший кроп на трек (memory-efficient).

Оптимизации:
  - Кропы НЕ хранятся в RAM во время трекинга — только score + bbox + frame_idx
  - После трекинга делаем второй проход по видео и вырезаем только победителей
  - Каждый нужный кадр читается ровно один раз

Использование:
    python track_and_crop.py \\
        --model   models/best.pt \\
        --video   данные/Unlabeled/26_2-10.mp4 \\
        --tracker configs/bytetrack_price.yaml \\
        --out     crops/26_2-10
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ── scoring без создания отдельного кропа ────────────────────────────────────


def score_frame_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    """Считает score прямо из региона кадра — crop не создаётся в памяти."""
    if x2 <= x1 or y2 <= y1:
        return 0.0
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    overexposed = np.mean(gray > 240)
    exposure_ok = max(0.0, 1.0 - overexposed * 5)
    brightness_ok = 1.0 if 40 < gray.mean() < 220 else 0.5
    size_bonus = min(gray.size / (150 * 100), 3.0)
    return sharpness * exposure_ok * brightness_ok * size_bonus


# ── лёгкий TrackBuffer — только числа, никаких кропов ───────────────────────


@dataclass
class Candidate:
    score: float
    frame_idx: int
    timestamp_ms: float
    conf: float
    bbox: tuple  # (x1, y1, x2, y2) после поворота


class TrackBuffer:
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.best: Candidate | None = None
        self.length = 0

    def update(self, c: Candidate):
        self.length += 1
        if self.best is None or c.score > self.best.score:
            self.best = c


# ── пересчёт bbox после CCW поворота ────────────────────────────────────────


def rotate_bbox_ccw(
    x1: int, y1: int, x2: int, y2: int, orig_w: int, orig_h: int
) -> tuple[int, int, int, int]:
    """
    При ROTATE_90_COUNTERCLOCKWISE:
      new_x = y,  new_y = (orig_w - x)
    Новый размер кадра: ширина = orig_h, высота = orig_w
    """
    return y1, orig_w - x2, y2, orig_w - x1


# ── проход 1: трекинг — только метаданные в памяти ──────────────────────────


def pass1_track(
    model: YOLO,
    video_path: str,
    tracker_config: str,
    imgsz: int,
    conf_thr: float,
    iou: float,
    device: str,
    fps: float,
) -> dict[int, TrackBuffer]:

    buffers: dict[int, TrackBuffer] = {}
    frame_idx = 0

    for result in model.track(
        source=video_path,
        imgsz=imgsz,
        conf=conf_thr,
        iou=iou,
        device=device,
        tracker=tracker_config,
        stream=True,
        persist=True,
        verbose=False,
    ):
        frame = cv2.rotate(result.orig_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        orig_h, orig_w = result.orig_img.shape[:2]
        pad = 6
        fh, fw = frame.shape[:2]

        if result.boxes is not None and result.boxes.id is not None:
            for box, tid, conf_val in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.id.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
            ):
                x1, y1, x2, y2 = map(int, box)
                rx1, ry1, rx2, ry2 = rotate_bbox_ccw(x1, y1, x2, y2, orig_w, orig_h)
                rx1 = max(0, rx1 - pad)
                ry1 = max(0, ry1 - pad)
                rx2 = min(fw, rx2 + pad)
                ry2 = min(fh, ry2 + pad)

                tid = int(tid)
                score = score_frame_region(frame, rx1, ry1, rx2, ry2)

                if tid not in buffers:
                    buffers[tid] = TrackBuffer(tid)
                buffers[tid].update(
                    Candidate(
                        score=score,
                        frame_idx=frame_idx,
                        timestamp_ms=frame_idx / fps * 1000,
                        conf=float(conf_val),
                        bbox=(rx1, ry1, rx2, ry2),
                    )
                )

        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"  [pass1] кадр {frame_idx} | треков: {len(buffers)}")

    return buffers


# ── проход 2: читаем только нужные кадры, вырезаем победителей ──────────────


def pass2_extract(
    video_path: str,
    buffers: dict[int, TrackBuffer],
    out_path: Path,
    min_track_len: int,
    min_score: float,
) -> list[dict]:

    # отбираем валидные треки
    valid = [
        (tid, buf.best)
        for tid, buf in buffers.items()
        if buf.length >= min_track_len and buf.best is not None and buf.best.score >= min_score
    ]

    if not valid:
        print("Нет валидных треков.")
        return []

    # группируем по frame_idx — один кадр читаем один раз
    from collections import defaultdict

    frame_to_tracks: dict[int, list] = defaultdict(list)
    for tid, cand in valid:
        frame_to_tracks[cand.frame_idx].append((tid, cand))

    print(f"[pass2] {len(valid)} треков, {len(frame_to_tracks)} уникальных кадров")

    cap = cv2.VideoCapture(video_path)
    rows = []

    for frame_idx in sorted(frame_to_tracks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, raw = cap.read()
        if not ok:
            continue
        frame = cv2.rotate(raw, cv2.ROTATE_90_COUNTERCLOCKWISE)

        for tid, cand in frame_to_tracks[frame_idx]:
            x1, y1, x2, y2 = cand.bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            stem = f"track_{tid:04d}_frame_{frame_idx:05d}_score_{int(cand.score):06d}"
            img_path = out_path / f"{stem}.jpg"
            cv2.imwrite(str(img_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            meta = {
                "track_id": tid,
                "frame_idx": frame_idx,
                "timestamp_ms": round(cand.timestamp_ms),
                "score": round(cand.score, 2),
                "conf": round(cand.conf, 3),
                "bbox_pixels": list(cand.bbox),
                "crop_size": [crop.shape[1], crop.shape[0]],
                "track_length": buffers[tid].length,
                "video": str(video_path),
                "image": str(img_path),
            }
            (out_path / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
            rows.append(meta)

            print(
                f"  [ok] track={tid:04d}  frame={frame_idx:05d}  "
                f"ts={int(cand.timestamp_ms):6d}ms  "
                f"score={cand.score:8.0f}  "
                f"size={crop.shape[1]}×{crop.shape[0]}  "
                f"len={buffers[tid].length}"
            )

    cap.release()
    return rows


# ── точка входа ──────────────────────────────────────────────────────────────


def run(
    model_path: str,
    video_path: str,
    out_dir: str,
    tracker_config: str,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    min_track_len: int,
    min_score: float,
):

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Видео: {total_frames} кадров @ {fps:.2f} fps\n")

    model = YOLO(model_path)

    print("=== Проход 1: трекинг ===")
    buffers = pass1_track(model, video_path, tracker_config, imgsz, conf, iou, device, fps)
    valid_count = sum(
        1
        for b in buffers.values()
        if b.length >= min_track_len and b.best is not None and b.best.score >= min_score
    )
    print(f"Треков всего: {len(buffers)}, пройдут фильтр: {valid_count}\n")

    print("=== Проход 2: извлечение кропов ===")
    rows = pass2_extract(video_path, buffers, out_path, min_track_len, min_score)

    print(f"\n{'=' * 55}")
    print(f"Сохранено : {len(rows)} | Пропущено : {len(buffers) - len(rows)}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_path / "summary.csv", index=False, encoding="utf-8-sig")

        print(f"\nРаспределение score:")
        bins = [0, 100, 500, 1000, 5000, 10000, float("inf")]
        labels = ["<100", "100-500", "500-1k", "1k-5k", "5k-10k", "10k+"]
        for i, label in enumerate(labels):
            n = int(((df.score >= bins[i]) & (df.score < bins[i + 1])).sum())
            print(f"  {label:>10}: {'█' * n} ({n})")

    print(f"\nГотово → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--min-track", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=50.0)
    args = parser.parse_args()

    run(
        args.model,
        args.video,
        args.out,
        args.tracker,
        args.imgsz,
        args.conf,
        args.iou,
        args.device,
        args.min_track,
        args.min_score,
    )
