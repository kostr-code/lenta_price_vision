"""
track_and_crop.py — YOLO tracking → TrackBuffer → лучшие кропы.
Оптимизирован по памяти: кропы не хранятся в RAM во время трекинга,
только score + frame_idx + bbox. Финальные кропы извлекаются за один проход.

Использование:
    python track_and_crop.py \\
        --model   models/best.pt \\
        --video   данные/Unlabeled/26_2-10.mp4 \\
        --tracker configs/bytetrack_price.yaml \\
        --out     crops/26_2-10

Зависимости:
    pip install ultralytics opencv-python numpy pandas
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# ── оценка резкости (grayscale, без хранения цветного кропа) ────────────────

def score_gray(gray: np.ndarray) -> float:
    """Laplacian variance + штрафы за засветку и темноту."""
    if gray is None or gray.size == 0:
        return 0.0
    sharpness     = cv2.Laplacian(gray, cv2.CV_64F).var()
    overexposed   = np.mean(gray > 240)
    exposure_ok   = max(0.0, 1.0 - overexposed * 5)
    brightness_ok = 1.0 if 40 < gray.mean() < 220 else 0.5
    size_bonus    = min(gray.size / (150 * 100), 3.0)
    return sharpness * exposure_ok * brightness_ok * size_bonus


# ── лёгкая запись о кандидате (без пикселей) ────────────────────────────────

@dataclass
class Candidate:
    score:        float
    frame_idx:    int
    timestamp_ms: float
    conf:         float
    bbox:         tuple   # (x1, y1, x2, y2) в повёрнутых координатах


# ── TrackBuffer — только метаданные, никаких пикселей ───────────────────────

class TrackBuffer:
    def __init__(self, track_id: int, max_candidates: int = 20):
        self.track_id       = track_id
        self.max_candidates = max_candidates
        self.candidates: list[Candidate] = []
        self.total_seen     = 0  # сколько кадров видел этот трек

    def add(self, frame_idx: int, gray_crop: np.ndarray,
            bbox: tuple, conf: float, timestamp_ms: float):
        self.total_seen += 1
        score = score_gray(gray_crop)
        if score <= 0:
            return
        self.candidates.append(
            Candidate(score, frame_idx, timestamp_ms, conf, bbox)
        )
        # держим только топ N
        if len(self.candidates) > self.max_candidates * 2:
            self.candidates.sort(key=lambda c: c.score, reverse=True)
            self.candidates = self.candidates[:self.max_candidates]

    def best(self) -> Candidate | None:
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda c: c.score)

    def __len__(self):
        return self.total_seen


# ── пересчёт bbox после CCW поворота ────────────────────────────────────────

def rotate_bbox_ccw(x1, y1, x2, y2, orig_w, orig_h):
    """
    После ROTATE_90_COUNTERCLOCKWISE:
      new_x = y_old
      new_y = orig_w - x_old
    Новый размер: ширина=orig_h, высота=orig_w
    """
    return y1, orig_w - x2, y2, orig_w - x1


# ── проход 1: трекинг — только score + bbox в памяти ────────────────────────

def tracking_pass(
    model: YOLO,
    video_path: str,
    tracker_config: str,
    imgsz: int,
    conf_thresh: float,
    iou: float,
    device: str,
    pad: int = 6,
) -> tuple[dict[int, TrackBuffer], float, int]:

    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    rot_w, rot_h = raw_h, raw_w  # после CCW поворота
    print(f"Видео: {n_frames} кадров @ {fps:.2f} fps  "
          f"({raw_w}×{raw_h} → {rot_w}×{rot_h} после CCW поворота)")

    buffers: dict[int, TrackBuffer] = {}
    frame_idx = 0

    for result in model.track(
        source=video_path,
        imgsz=imgsz,
        conf=conf_thresh,
        iou=iou,
        device=device,
        tracker=tracker_config,
        stream=True,    # генератор — не грузит всё видео в RAM
        persist=True,   # треки живут между кадрами
        verbose=False,
    ):
        if result.boxes is not None and result.boxes.id is not None:
            # grayscale сразу — в 3x меньше памяти чем BGR
            frame_gray = cv2.cvtColor(
                cv2.rotate(result.orig_img, cv2.ROTATE_90_COUNTERCLOCKWISE),
                cv2.COLOR_BGR2GRAY,
            )

            for box, tid, conf_val in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.id.cpu().numpy().astype(int),
                result.boxes.conf.cpu().numpy(),
            ):
                x1, y1, x2, y2 = map(int, box)
                rx1, ry1, rx2, ry2 = rotate_bbox_ccw(x1, y1, x2, y2, raw_w, raw_h)

                # padding + clamp
                rx1 = max(0,     rx1 - pad)
                ry1 = max(0,     ry1 - pad)
                rx2 = min(rot_w, rx2 + pad)
                ry2 = min(rot_h, ry2 + pad)

                gray_crop = frame_gray[ry1:ry2, rx1:rx2]

                if tid not in buffers:
                    buffers[tid] = TrackBuffer(tid)
                buffers[tid].add(
                    frame_idx, gray_crop,
                    (rx1, ry1, rx2, ry2),
                    float(conf_val),
                    frame_idx / fps * 1000,
                )

            del frame_gray

        del result  # освобождаем тензоры YOLO

        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"  [{frame_idx:>5}/{n_frames}]  треков: {len(buffers)}")

    return buffers, fps, n_frames


# ── проход 2: читаем только нужные кадры, вырезаем цветные кропы ────────────

def extract_best_crops(
    buffers: dict[int, TrackBuffer],
    video_path: str,
    out_path: Path,
    min_track_len: int,
    min_score: float,
) -> list[dict]:

    # отбираем треки-победители
    winners: list[tuple[int, Candidate]] = [
        (tid, buf.best())
        for tid, buf in sorted(buffers.items())
        if len(buf) >= min_track_len
        and buf.best() is not None
        and buf.best().score >= min_score
    ]

    skipped = len(buffers) - len(winners)
    print(f"Треков отобрано: {len(winners)}  пропущено: {skipped}")

    if not winners:
        return []

    # один seek на кадр даже если несколько треков попали в один кадр
    frame_to_tracks: dict[int, list] = defaultdict(list)
    for tid, cand in winners:
        frame_to_tracks[cand.frame_idx].append((tid, cand))

    cap   = cv2.VideoCapture(video_path)
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rows  = []

    for frame_idx in sorted(frame_to_tracks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, raw = cap.read()
        if not ok:
            print(f"  [warn] кадр {frame_idx} недоступен")
            continue

        frame = cv2.rotate(raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
        del raw

        for tid, cand in frame_to_tracks[frame_idx]:
            x1, y1, x2, y2 = cand.bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            stem = (f"track_{tid:04d}"
                    f"_frame_{frame_idx:05d}"
                    f"_score_{int(cand.score):06d}")

            cv2.imwrite(
                str(out_path / f"{stem}.jpg"), crop,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

            meta = {
                "track_id":     tid,
                "frame_idx":    frame_idx,
                "timestamp_ms": round(cand.timestamp_ms),
                "score":        round(cand.score, 2),
                "conf":         round(cand.conf, 3),
                "bbox_pixels":  list(cand.bbox),
                "crop_size":    [crop.shape[1], crop.shape[0]],
                "track_length": len(buffers[tid]),
                "video":        str(video_path),
            }
            (out_path / f"{stem}.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2)
            )
            rows.append(meta)

            print(f"  [ok] track={tid:04d}  frame={frame_idx:05d}  "
                  f"ts={int(cand.timestamp_ms):6d}ms  "
                  f"score={cand.score:8.0f}  "
                  f"size={crop.shape[1]}×{crop.shape[0]}  "
                  f"len={len(buffers[tid])}")
        del frame

    cap.release()
    return rows


# ── main ─────────────────────────────────────────────────────────────────────

def run(
    model_path: str,
    video_path: str,
    out_dir: str,
    tracker_config: str = "bytetrack.yaml",
    imgsz: int = 1280,
    conf: float = 0.15,
    iou: float = 0.5,
    device: str = "0",
    min_track_len: int = 3,
    min_score: float = 50.0,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    print("=== Проход 1: трекинг ===")
    buffers, fps, n_frames = tracking_pass(
        model, video_path, tracker_config, imgsz, conf, iou, device
    )
    print(f"Всего треков обнаружено: {len(buffers)}")

    print("\n=== Проход 2: извлечение кропов ===")
    rows = extract_best_crops(buffers, video_path, out_path, min_track_len, min_score)

    print(f"\n{'='*55}")
    print(f"Сохранено кропов: {len(rows)}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_path / "summary.csv", index=False, encoding="utf-8-sig")
        print(f"Сводка → {out_path / 'summary.csv'}")

        print("\nРаспределение score:")
        bins   = [0, 100, 500, 1000, 5000, 10000, float("inf")]
        labels = ["<100", "100-500", "500-1k", "1k-5k", "5k-10k", "10k+"]
        for i, lbl in enumerate(labels):
            n = int(((df.score >= bins[i]) & (df.score < bins[i+1])).sum())
            print(f"  {lbl:>10}: {'█' * n} ({n})")

    print(f"\nГотово → {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="YOLO tracking → лучший кроп на трек (memory-efficient, 2 прохода)"
    )
    p.add_argument("--model",      required=True)
    p.add_argument("--video",      required=True)
    p.add_argument("--out",        required=True)
    p.add_argument("--tracker",    default="bytetrack.yaml")
    p.add_argument("--imgsz",      type=int,   default=1280)
    p.add_argument("--conf",       type=float, default=0.15)
    p.add_argument("--iou",        type=float, default=0.5)
    p.add_argument("--device",     default="0")
    p.add_argument("--min-track",  type=int,   default=3,    dest="min_track")
    p.add_argument("--min-score",  type=float, default=50.0, dest="min_score")
    args = p.parse_args()

    run(
        model_path    = args.model,
        video_path    = args.video,
        out_dir       = args.out,
        tracker_config= args.tracker,
        imgsz         = args.imgsz,
        conf          = args.conf,
        iou           = args.iou,
        device        = args.device,
        min_track_len = args.min_track,
        min_score     = args.min_score,
    )
