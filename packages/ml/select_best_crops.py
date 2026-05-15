"""
select_best_crops.py — выбирает лучший кроп на каждый трек из YOLO tracking output.

Использование:
    python select_best_crops.py \\
        --labels  runs/track/video_26_2_10_bytetrack_custom/labels \\
        --video   данные/Unlabeled/26_2-10.mp4 \\
        --out     crops/26_2-10

Выход:
    crops/26_2-10/
        track_0042_frame_0318_score_1847.jpg   ← имя содержит всё нужное
        track_0042_frame_0318_score_1847.json  ← bbox + meta для следующего шага
        ...
    crops/26_2-10/summary.csv                  ← сводка по всем трекам

Зависимости:
    pip install opencv-python numpy pandas
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ── качество кропа ───────────────────────────────────────────────────────────

def score_crop(crop_bgr: np.ndarray) -> float:
    """
    Чем выше — тем лучше кроп для OCR/QR.
    Три компонента: резкость, экспозиция, размер.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Резкость — Laplacian variance
    #    Главная метрика: смазанный кадр → низкая дисперсия градиентов
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. Засветка — штрафуем если >15% пикселей overexposed
    overexposed_ratio = np.mean(gray > 240)
    exposure_ok = max(0.0, 1.0 - overexposed_ratio * 5)

    # 3. Недосветка — штрафуем если слишком тёмный кроп
    mean_brightness = gray.mean()
    brightness_ok = 1.0 if 40 < mean_brightness < 220 else 0.5

    # 4. Размер — больше пикселей = больше деталей (до разумного предела)
    pixel_count = h * w
    size_bonus = min(pixel_count / (150 * 100), 3.0)

    return sharpness * exposure_ok * brightness_ok * size_bonus


# ── парсинг YOLO label txt ───────────────────────────────────────────────────

def parse_label_file(txt_path: Path):
    """
    Парсит один YOLO tracking label файл.
    Формат строки: class_id cx cy w h conf track_id
    Возвращает список dict.
    """
    detections = []
    try:
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                det = {
                    "class_id": int(parts[0]),
                    "cx":       float(parts[1]),
                    "cy":       float(parts[2]),
                    "w":        float(parts[3]),
                    "h":        float(parts[4]),
                    "conf":     float(parts[5]),
                    "track_id": int(parts[6]) if len(parts) > 6 else -1,
                }
                detections.append(det)
    except Exception as e:
        print(f"[warn] Не могу прочитать {txt_path}: {e}")
    return detections


def frame_idx_from_name(txt_path: Path) -> int:
    """
    Извлекает номер кадра из имени файла.
    "26_2-10_318.txt" → 318
    """
    m = re.search(r"_(\d+)\.txt$", txt_path.name)
    return int(m.group(1)) if m else -1


# ── извлечение кропа из видео ────────────────────────────────────────────────

def get_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    """Возвращает кадр по индексу (с поворотом 90°)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    # видео снято боком
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)


def yolo_to_pixel_bbox(cx, cy, w, h, frame_w, frame_h, padding=6):
    """
    YOLO нормализованные координаты → пиксельный bbox с отступом.
    Возвращает (x1, y1, x2, y2).
    """
    x1 = int((cx - w / 2) * frame_w) - padding
    y1 = int((cy - h / 2) * frame_h) - padding
    x2 = int((cx + w / 2) * frame_w) + padding
    y2 = int((cy + h / 2) * frame_h) + padding
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w, x2), min(frame_h, y2)
    return x1, y1, x2, y2


# ── TrackBuffer ───────────────────────────────────────────────────────────────

class TrackBuffer:
    """Накапливает кандидатов для одного track_id."""

    def __init__(self, track_id: int, max_candidates: int = 15):
        self.track_id = track_id
        self.max_candidates = max_candidates
        # (score, frame_idx, crop, bbox_pixels, det_meta)
        self.candidates: list[tuple] = []

    def add(self, frame_idx: int, crop: np.ndarray,
            bbox_pixels: tuple, det: dict):
        score = score_crop(crop)
        if score <= 0:
            return
        self.candidates.append((score, frame_idx, crop, bbox_pixels, det))
        # держим только топ N чтобы не жрать память
        if len(self.candidates) > self.max_candidates * 2:
            self.candidates.sort(key=lambda x: x[0], reverse=True)
            self.candidates = self.candidates[:self.max_candidates]

    def best(self) -> tuple | None:
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda x: x[0])

    def __len__(self):
        return len(self.candidates)


# ── основной pipeline ────────────────────────────────────────────────────────

def run(labels_dir: str, video_path: str, out_dir: str,
        min_track_len: int = 3, min_score: float = 50.0,
        save_all_candidates: bool = False):

    labels_path = Path(labels_dir)
    out_path    = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── собираем все label файлы ──────────────────────────────
    txt_files = sorted(labels_path.glob("*.txt"))
    if not txt_files:
        print(f"[error] Нет .txt файлов в {labels_path}")
        return

    print(f"Label файлов: {len(txt_files)}")

    # ── группируем детекты по track_id ───────────────────────
    # track_id → список (frame_idx, det)
    track_frames: dict[int, list] = defaultdict(list)

    for txt_path in txt_files:
        frame_idx = frame_idx_from_name(txt_path)
        if frame_idx < 0:
            continue
        for det in parse_label_file(txt_path):
            tid = det["track_id"]
            track_frames[tid].append((frame_idx, det))

    print(f"Уникальных треков: {len(track_frames)}")

    # фильтруем короткие треки (скорее всего шум)
    long_tracks = {
        tid: frames
        for tid, frames in track_frames.items()
        if len(frames) >= min_track_len
    }
    print(f"Треков ≥ {min_track_len} кадров: {len(long_tracks)}")

    # ── открываем видео ───────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[error] Не могу открыть видео: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)

    # размер кадра после поворота (90°) — ширина и высота меняются местами
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w, frame_h = raw_h, raw_w  # после rotate 90°

    print(f"Видео: {total_frames} кадров, {fps:.1f} fps, "
          f"размер после поворота: {frame_w}×{frame_h}")

    # ── кэш кадров — не перечитываем одно и то же ────────────
    # Собираем все уникальные frame_idx которые нам нужны
    needed_frames: set[int] = set()
    for frames in long_tracks.values():
        for frame_idx, _ in frames:
            needed_frames.add(frame_idx)

    print(f"Уникальных кадров для загрузки: {len(needed_frames)}")

    # Загружаем нужные кадры (по порядку — быстрее чем вразброс)
    frame_cache: dict[int, np.ndarray] = {}
    for frame_idx in sorted(needed_frames):
        frame = get_frame(cap, frame_idx)
        if frame is not None:
            frame_cache[frame_idx] = frame

    cap.release()
    print(f"Кадров загружено: {len(frame_cache)}")

    # ── строим TrackBuffer для каждого трека ─────────────────
    buffers: dict[int, TrackBuffer] = {}

    for tid, frames in long_tracks.items():
        buf = TrackBuffer(tid)
        for frame_idx, det in frames:
            frame = frame_cache.get(frame_idx)
            if frame is None:
                continue
            x1, y1, x2, y2 = yolo_to_pixel_bbox(
                det["cx"], det["cy"], det["w"], det["h"],
                frame_w, frame_h
            )
            crop = frame[y1:y2, x1:x2]
            buf.add(frame_idx, crop, (x1, y1, x2, y2), det)
        buffers[tid] = buf

    # ── сохраняем лучший кроп на трек ────────────────────────
    summary_rows = []
    saved = 0
    skipped_score = 0

    for tid, buf in sorted(buffers.items()):
        best = buf.best()
        if best is None:
            continue

        score, frame_idx, crop, (x1, y1, x2, y2), det = best
        timestamp_ms = int(frame_idx / fps * 1000)

        if score < min_score:
            skipped_score += 1
            print(f"  [skip] track={tid:04d} score={score:.0f} < {min_score} "
                  f"(кадров в треке: {len(buf)})")
            continue

        # имя файла кодирует все нужные данные
        stem = (f"track_{tid:04d}"
                f"_frame_{frame_idx:05d}"
                f"_score_{int(score):06d}")

        # сохраняем кроп
        img_path = out_path / f"{stem}.jpg"
        cv2.imwrite(str(img_path), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

        # сохраняем мета-данные для следующего шага pipeline
        meta = {
            "track_id":     tid,
            "frame_idx":    frame_idx,
            "timestamp_ms": timestamp_ms,
            "score":        round(score, 2),
            "conf":         round(det["conf"], 3),
            "bbox_pixels":  [x1, y1, x2, y2],
            "bbox_norm":    [det["cx"], det["cy"], det["w"], det["h"]],
            "crop_size":    [crop.shape[1], crop.shape[0]],  # w, h
            "track_length": len(buf),
            "video":        str(video_path),
            "image":        str(img_path),
        }
        json_path = out_path / f"{stem}.json"
        json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

        summary_rows.append({
            "track_id":     tid,
            "frame_idx":    frame_idx,
            "timestamp_ms": timestamp_ms,
            "score":        round(score, 2),
            "conf":         round(det["conf"], 3),
            "track_length": len(buf),
            "crop_w":       crop.shape[1],
            "crop_h":       crop.shape[0],
            "img_path":     str(img_path),
        })
        saved += 1

        print(f"  [ok]   track={tid:04d}  frame={frame_idx:05d}  "
              f"ts={timestamp_ms:6d}ms  score={score:8.0f}  "
              f"size={crop.shape[1]}×{crop.shape[0]}  "
              f"len={len(buf)}")

    # ── сводка ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Сохранено кропов : {saved}")
    print(f"Пропущено (score) : {skipped_score}")
    print(f"Папка            : {out_path}")

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_path = out_path / "summary.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"Сводка           : {csv_path}")

        print(f"\nСтатистика score:")
        print(f"  min  = {df.score.min():.0f}")
        print(f"  median = {df.score.median():.0f}")
        print(f"  max  = {df.score.max():.0f}")

        print(f"\nСтатистика размера кропа:")
        print(f"  мин  = {df.crop_w.min()}×{df.crop_h.min()}")
        print(f"  медиана = {df.crop_w.median():.0f}×{df.crop_h.median():.0f}")

        # гистограмма score в консоли
        print(f"\nРаспределение score (каждый █ = 1 трек):")
        bins = [0, 100, 500, 1000, 5000, 10000, float("inf")]
        labels = ["<100", "100-500", "500-1k", "1k-5k", "5k-10k", "10k+"]
        for i, label in enumerate(labels):
            count = ((df.score >= bins[i]) & (df.score < bins[i+1])).sum()
            bar = "█" * count
            print(f"  {label:>10}: {bar} ({count})")

    print(f"\nГотово. Следующий шаг:")
    print(f"  python ocr_pipeline.py --crops {out_path}")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Выбирает лучший кроп на трек из YOLO tracking output"
    )
    parser.add_argument("--labels",    required=True,
                        help="папка с .txt label файлами от YOLO")
    parser.add_argument("--video",     required=True,
                        help="путь к MP4 видео")
    parser.add_argument("--out",       required=True,
                        help="куда сохранять кропы")
    parser.add_argument("--min-track", type=int, default=3,
                        help="минимальная длина трека (кадров), дефолт=3")
    parser.add_argument("--min-score", type=float, default=50.0,
                        help="минимальный score кропа, дефолт=50")
    args = parser.parse_args()

    run(
        labels_dir=args.labels,
        video_path=args.video,
        out_dir=args.out,
        min_track_len=args.min_track,
        min_score=args.min_score,
    )
