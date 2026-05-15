"""
track_and_crop.py — YOLO tracking → кэш всех кандидатов + лучший кроп на трек.

Pass 1: YOLO+ByteTrack по всему видео.
        Для каждого детекта записываем метрики (laplacian, tenengrad, flow, conf, track_age).
        После pass1 сохраняем кэш — ALL кропы всех кандидатов + tracks.json.

Pass 2: по кэшу выбираем победителя на трек (можно менять формулу скоринга)
        и вырезаем финальные кропы в --out директорию.

Для визуального тестирования scoring — запускай crop_explorer.py --cache {out}/cache

Использование:
    uv run python packages/ml/track_and_crop.py \\
        --model models/best.pt \\
        --video "packages/ml/src/ml/data/Unlabeled/26_2-10.mp4" \\
        --tracker packages/ml/src/ml/configs/bytetrack_price.yaml \\
        --out runs/crops/26_2-10
"""

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ── метрики резкости ─────────────────────────────────────────────────────────


def score_metrics(
    frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
) -> tuple[float, float]:
    """Возвращает (laplacian_variance, tenengrad) для региона кадра."""
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    ten = float(np.mean(gx**2 + gy**2))
    return lap, ten


DEFAULT_WEIGHTS = {"sharp": 0.3, "conf": 0.3, "flow": 0.3, "age": 0.1}


def composite_score(c: "Candidate", weights: dict = DEFAULT_WEIGHTS) -> float:
    """Composite scoring: чем больше — тем лучше.
    flow_mag=0 (робот стоит) даёт максимальный бонус."""
    motion_ok = 1.0 / (1.0 + c.flow_mag * 0.5)
    sharpness = c.tenengrad / 1000
    age_bonus = min(c.track_age / 5, 2.0)
    return (
        weights["sharp"] * sharpness
        + weights["conf"] * c.conf**2 * 100
        + weights["flow"] * motion_ok * 100
        + weights["age"] * age_bonus * 100
    )


# ── структуры данных ──────────────────────────────────────────────────────────


@dataclass
class Candidate:
    score: float
    frame_idx: int
    timestamp_ms: float
    conf: float
    bbox: tuple  # (x1, y1, x2, y2) после CCW поворота
    laplacian: float = 0.0
    tenengrad: float = 0.0
    flow_mag: float = 0.0   # 0 = робот стоит
    track_age: int = 0      # сколько кадров трек уже живёт


class TrackBuffer:
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.all_candidates: list[Candidate] = []
        self.length = 0

    def update(self, c: Candidate) -> None:
        self.length += 1
        self.all_candidates.append(c)

    @property
    def best(self) -> Candidate | None:
        if not self.all_candidates:
            return None
        return max(self.all_candidates, key=lambda c: c.score)


# ── пересчёт bbox после CCW поворота ─────────────────────────────────────────


def rotate_bbox_ccw(
    x1: int, y1: int, x2: int, y2: int, orig_w: int
) -> tuple[int, int, int, int]:
    """При ROTATE_90_COUNTERCLOCKWISE: new_x=y, new_y=(orig_w-x)."""
    return y1, orig_w - x2, y2, orig_w - x1


# ── проход 1: трекинг — метрики + все кандидаты ──────────────────────────────


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
    track_ages: dict[int, int] = {}
    prev_gray: np.ndarray | None = None

    for frame_idx, result in enumerate(model.track(
        source=video_path,
        imgsz=imgsz,
        conf=conf_thr,
        iou=iou,
        device=device,
        tracker=tracker_config,
        stream=True,
        persist=True,
        verbose=False,
    )):
        curr_frame = cv2.rotate(result.orig_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # optical flow по даунсемплированному кадру (320px по короткой стороне)
        # Farneback на 4K = минуты; на 320p = миллисекунды, magnitude та же
        flow_mag = 0.0
        small = cv2.resize(curr_gray, (320, 180))
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, small, None,
                pyr_scale=0.5, levels=2, winsize=15,
                iterations=2, poly_n=5, poly_sigma=1.2, flags=0,
            )
            flow_mag = float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())
        prev_gray = small

        _, orig_w = result.orig_img.shape[:2]
        pad = 6
        fh, fw = curr_frame.shape[:2]

        if result.boxes is not None and result.boxes.id is not None:
            for box, tid, conf_val in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.id.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                strict=False,
            ):
                tid = int(tid)
                track_ages[tid] = track_ages.get(tid, 0) + 1

                x1, y1, x2, y2 = map(int, box)
                rx1, ry1, rx2, ry2 = rotate_bbox_ccw(x1, y1, x2, y2, orig_w)
                rx1 = max(0, rx1 - pad)
                ry1 = max(0, ry1 - pad)
                rx2 = min(fw, rx2 + pad)
                ry2 = min(fh, ry2 + pad)

                lap, ten = score_metrics(curr_frame, rx1, ry1, rx2, ry2)

                if tid not in buffers:
                    buffers[tid] = TrackBuffer(tid)
                buffers[tid].update(
                    Candidate(
                        score=ten,  # для фильтрации в pass2; explorer пересчитывает
                        frame_idx=frame_idx,
                        timestamp_ms=frame_idx / fps * 1000,
                        conf=float(conf_val),
                        bbox=(rx1, ry1, rx2, ry2),
                        laplacian=lap,
                        tenengrad=ten,
                        flow_mag=flow_mag,
                        track_age=track_ages[tid],
                    )
                )

        if frame_idx % 200 == 0:
            print(f"  [pass1] кадр {frame_idx} | треков: {len(buffers)}")

    return buffers


# ── сохранение кэша — все кропы + tracks.json ────────────────────────────────


def save_pass1_cache(
    buffers: dict[int, TrackBuffer],
    video_path: str,
    out_dir: str,
    fps: float,
) -> Path:
    """Сохраняет все кропы всех кандидатов и tracks.json в {out_dir}/cache/."""
    cache_dir = Path(out_dir) / "cache"
    crops_dir = cache_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # группируем по frame_idx — один проход по видео
    frame_to_cands: dict[int, list[tuple[int, Candidate]]] = defaultdict(list)
    for tid, buf in buffers.items():
        for c in buf.all_candidates:
            frame_to_cands[c.frame_idx].append((tid, c))

    total_cands = sum(len(v) for v in frame_to_cands.values())
    print(f"  [cache] {len(buffers)} треков, {total_cands} кандидатов, "
          f"{len(frame_to_cands)} уникальных кадров")

    cap = cv2.VideoCapture(video_path)
    tracks_data: dict[str, dict] = {}

    for frame_idx in sorted(frame_to_cands):  # frame_idx is dict key, not loop counter
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, raw = cap.read()
        if not ok:
            continue
        frame = cv2.rotate(raw, cv2.ROTATE_90_COUNTERCLOCKWISE)

        for tid, c in frame_to_cands[frame_idx]:
            x1, y1, x2, y2 = c.bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            fname = f"track_{tid:04d}_frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(crops_dir / fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

            entry = {**asdict(c), "crop_path": str(crops_dir / fname)}
            entry["bbox"] = list(entry["bbox"])  # tuple → list for JSON

            tid_key = str(tid)
            if tid_key not in tracks_data:
                tracks_data[tid_key] = {
                    "track_id": tid,
                    "total_frames": buffers[tid].length,
                    "candidates": [],
                }
            tracks_data[tid_key]["candidates"].append(entry)

    cap.release()

    cache_json = {
        "video": str(video_path),
        "fps": fps,
        "tracks": tracks_data,
    }
    (cache_dir / "tracks.json").write_text(
        json.dumps(cache_json, ensure_ascii=False, indent=2)
    )
    print(f"  [cache] → {cache_dir}")
    return cache_dir


# ── проход 2: вырезаем победителей ───────────────────────────────────────────


def pass2_extract(
    video_path: str,
    buffers: dict[int, TrackBuffer],
    out_path: Path,
    min_track_len: int,
    min_score: float,
) -> list[dict]:

    valid = [
        (tid, buf.best)
        for tid, buf in buffers.items()
        if buf.length >= min_track_len and buf.best is not None and buf.best.score >= min_score
    ]

    if not valid:
        print("Нет валидных треков.")
        return []

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
                "laplacian": round(cand.laplacian, 1),
                "tenengrad": round(cand.tenengrad, 1),
                "flow_mag": round(cand.flow_mag, 3),
                "track_age": cand.track_age,
                "bbox_pixels": list(cand.bbox),
                "crop_size": [crop.shape[1], crop.shape[0]],
                "track_length": buffers[tid].length,
                "video": str(video_path),
                "image": str(img_path),
            }
            (out_path / f"{stem}.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2)
            )
            rows.append(meta)

            print(
                f"  [ok] track={tid:04d}  frame={frame_idx:05d}  "
                f"ts={int(cand.timestamp_ms):6d}ms  "
                f"score={cand.score:8.0f}  flow={cand.flow_mag:.2f}  "
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
) -> None:

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Видео: {total_frames} кадров @ {fps:.2f} fps\n")

    model = YOLO(model_path)

    print("=== Проход 1: трекинг + метрики ===")
    buffers = pass1_track(model, video_path, tracker_config, imgsz, conf, iou, device, fps)
    valid_count = sum(
        1
        for b in buffers.values()
        if b.length >= min_track_len and b.best is not None and b.best.score >= min_score
    )
    print(f"Треков всего: {len(buffers)}, пройдут фильтр: {valid_count}\n")

    print("=== Сохранение кэша кандидатов ===")
    cache_dir = save_pass1_cache(buffers, video_path, out_dir, fps)
    print(f"Кэш для crop_explorer.py: {cache_dir}\n")

    print("=== Проход 2: извлечение победителей ===")
    rows = pass2_extract(video_path, buffers, out_path, min_track_len, min_score)

    print("\n" + "=" * 55)
    print(f"Сохранено : {len(rows)} | Пропущено : {len(buffers) - len(rows)}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_path / "summary.csv", index=False, encoding="utf-8-sig")

        print("\nРаспределение score:")
        bins = [0, 100, 500, 1000, 5000, 10000, float("inf")]
        labels = ["<100", "100-500", "500-1k", "1k-5k", "5k-10k", "10k+"]
        for i, label in enumerate(labels):
            n = int(((df.score >= bins[i]) & (df.score < bins[i + 1])).sum())
            print(f"  {label:>10}: {'█' * n} ({n})")

    print(f"\nГотово → {out_path}")
    print(f"Визуальный explorer: uv run python packages/ml/crop_explorer.py --cache {cache_dir}")


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
