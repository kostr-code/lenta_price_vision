"""
main.py — Распознавание ценников Ленты.

Два режима:

  Labeled (с GT CSV):
    uv run python main.py \\
        --video data/Данные/43_15/43_15.mp4 \\
        --csv   data/Данные/43_15/43_15.csv \\
        --out   results/43_15_out.csv

  Unlabeled (YOLO + ByteTrack):
    uv run python main.py \\
        --video data/Данные/Unlabeled/video.mp4 \\
        --detect \\
        --weights models/price_tag_yolo.pt \\
        --out results/unlabeled_out.csv

Пайплайн labeled:
  load_df -> find_best_frame per timestamp -> cut_crop_from_row
  -> quality_score -> VLM -> OCR fallback -> parse_fields -> CSV

Пайплайн unlabeled (ByteTrack):
  VideoCapture -> КАЖДЫЙ кадр -> rotate_frame -> track_price_tags() (ByteTrack)
  -> для каждого track_id: хранить топ-3 кропа по _rank_crop()
  -> после видео: VLM + OCR на топ-1 кропе каждого трека -> одна строка на ценник
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pipeline.parsers import (
    OUTPUT_COLUMNS,
    ABSENT_VALUE,
    make_empty_row,
    merge_field_values,
    parse_fields,
    normalize_text,
)
from pipeline.quality import estimate_crop_quality
from pipeline.video import (
    cut_crop_bbox,
    cut_crop_from_row,
    find_best_frame,
    load_df,
    rotate_frame,
)
from pipeline.vlm import VLM_FIELDS, extract_fields_vlm, load_vlm

# ── OCR is optional — import only if paddleocr is installed ──────────────────
try:
    from pipeline.ocr import enhance_crop, load_ocr, ocr_zoned

    _HAS_OCR = True
except ImportError:
    _HAS_OCR = False

# ── YOLO detector is optional — import only if ultralytics is installed ───────
try:
    from pipeline.detector import (
        Detection,
        load_detector,
        detect_price_tags,
        track_price_tags,
    )

    _HAS_DETECTOR = True
except ImportError:
    _HAS_DETECTOR = False

_TRACK_TOP_K = 3  # кол-во лучших кропов на трек для ранжирования


@dataclass
class _CropCandidate:
    """Один кроп-кандидат от трека — для выбора лучшего перед VLM."""

    crop: np.ndarray
    score: float
    ts_ms: float
    det: object  # Detection


def _rank_crop(crop: np.ndarray, det: object) -> float:
    """
    Скоринг кропа для ранжирования топ-K внутри трека.

    Сейчас: только composite quality score (h264 + FFT + Laplacian).
    Можно расширить: умножить на det.conf, добавить бонус за размер bbox и т.д.
    """
    return estimate_crop_quality(crop)


def _merge_vlm_and_ocr(
    vlm_fields: dict[str, str],
    ocr_fields: dict[str, str],
) -> dict[str, str]:
    """
    Merge VLM and OCR results per-field.

    VLM is primary; OCR supplements empty VLM fields.
    """
    merged = dict(vlm_fields)
    for k, v in ocr_fields.items():
        if k not in merged or not merged[k]:
            merged[k] = v
    return merged


def process_video(
    video_path: str,
    csv_path: str,
    out_path: str,
    model_id: str,
    use_ocr: bool,
    quality_threshold: float,
    scan_frames: int,
) -> None:
    print(f"[main] Loading {csv_path}")
    df = load_df(csv_path)
    video_name = Path(video_path).name

    print(f"[main] Loading VLM: {model_id}")
    load_vlm(model_id)

    ocr_model = None
    if use_ocr and _HAS_OCR:
        print("[main] Loading PaddleOCR (fallback) ...")
        ocr_model = load_ocr(use_gpu=True)

    # Cache frames by timestamp — avoid re-seeking the same position
    frame_cache: dict[float, tuple] = {}

    def _get_frame(ts_ms: float):
        if ts_ms not in frame_cache:
            frame_cache[ts_ms] = find_best_frame(video_path, ts_ms, n=scan_frames)
        return frame_cache[ts_ms]

    output_rows: list[dict[str, str]] = []
    total = len(df)

    for idx, row in df.iterrows():
        ts_ms = float(row.get("frame_timestamp", 0))
        print(
            f"[{idx + 1}/{total}] ts={ts_ms:.0f}ms  {row.get('product_name', '')[:40]}"
        )

        frame, lap_var = _get_frame(ts_ms)
        if frame is None:
            print(f"  [warn] Could not read frame at {ts_ms}ms")
            output_rows.append(make_empty_row())
            continue

        crop = cut_crop_from_row(frame, row)
        if crop is None or crop.size == 0:
            print("  [warn] Empty crop, skipping")
            output_rows.append(make_empty_row())
            continue

        quality = estimate_crop_quality(crop)
        print(f"  quality={quality:.3f}  lap={lap_var:.0f}")

        if quality < quality_threshold:
            print(
                f"  [warn] Quality {quality:.3f} < {quality_threshold} — crop likely degraded"
            )

        # ── Primary: VLM ──────────────────────────────────────────────────────
        vlm_result = extract_fields_vlm(crop)
        print(f"  VLM -> {len([v for v in vlm_result.values() if v])} fields extracted")

        # ── Fallback: PaddleOCR ───────────────────────────────────────────────
        ocr_fields: dict[str, str] = {}
        if ocr_model is not None:
            try:
                enhanced = enhance_crop(crop)
                lines = ocr_zoned(ocr_model, enhanced)
                ocr_fields = parse_fields(lines, [], crop)
            except Exception as exc:
                print(f"  [warn] OCR failed: {exc}")

        # ── Merge ─────────────────────────────────────────────────────────────
        combined = _merge_vlm_and_ocr(vlm_result, ocr_fields)

        # ── Build output row ──────────────────────────────────────────────────
        out_row = make_empty_row()

        # Metadata from source CSV (passthrough)
        for meta_col in (
            "filename",
            "frame_timestamp",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
        ):
            if meta_col in row:
                out_row[meta_col] = (
                    str(row[meta_col]) if pd.notna(row[meta_col]) else ""
                )
        if not out_row["filename"]:
            out_row["filename"] = video_name

        # Fill recognized fields
        for col in OUTPUT_COLUMNS:
            if col in combined and combined[col]:
                out_row[col] = normalize_text(combined[col])

        # Apply absent-value defaults for optional fields
        for col in (
            "price_discount",
            "discount_amount",
            "code",
            "additional_info",
            "special_symbols",
        ):
            if not out_row[col]:
                out_row[col] = ABSENT_VALUE

        out_row["_quality"] = f"{quality:.3f}"
        output_rows.append(out_row)

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_cols = OUTPUT_COLUMNS + ["_quality"]
    out_df = pd.DataFrame(output_rows, columns=out_cols)
    out_df.to_csv(out_path, index=False)
    print(f"\n[main] Saved {len(out_df)} rows -> {out_path}")

    # Quick stats
    filled = {
        col: out_df[col].apply(lambda v: bool(v) and v != ABSENT_VALUE).sum()
        for col in (
            "product_name",
            "price_card",
            "price_default",
            "barcode",
            "print_datetime",
        )
    }
    print("\n[stats] Field fill rate:")
    for col, n in filled.items():
        print(f"  {col:20s}: {n}/{total} ({100 * n // max(total, 1)}%)")


def process_video_unlabeled(
    video_path: str,
    out_path: str,
    weights: str,
    model_id: str,
    use_ocr: bool,
    quality_threshold: float,
    conf_det: float,
    device: str | None,
    tracker: str,
) -> None:
    if not _HAS_DETECTOR:
        print("Error: ultralytics не установлен (uv add ultralytics)")
        return

    print(f"[main] Загрузка детектора: {weights}")
    load_detector(weights)

    print(f"[main] Загрузка VLM: {model_id}")
    load_vlm(model_id)

    ocr_model = None
    if use_ocr and _HAS_OCR:
        print("[main] Загрузка PaddleOCR (fallback) ...")
        ocr_model = load_ocr(use_gpu=True)

    video_name = Path(video_path).name
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: не удалось открыть видео: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"[main] Видео: {video_name}  ({total_frames} кадров)")
    print(f"       трекер: {tracker}  conf={conf_det}")

    # track_id -> список топ-K кандидатов
    track_candidates: dict[int, list[_CropCandidate]] = defaultdict(list)
    frame_idx = 0

    while True:
        ok, raw = cap.read()
        if not ok:
            break
        frame = rotate_frame(raw)
        ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        detections = track_price_tags(
            frame, conf=conf_det, device=device, tracker=tracker
        )

        for det in detections:
            if det.track_id is None:
                continue
            crop = cut_crop_bbox(frame, det.x1, det.y1, det.x2, det.y2)
            if crop is None or crop.size == 0:
                continue
            score = _rank_crop(crop, det)
            bucket = track_candidates[det.track_id]
            bucket.append(_CropCandidate(crop=crop, score=score, ts_ms=ts_ms, det=det))
            if len(bucket) > _TRACK_TOP_K:
                bucket.sort(key=lambda c: c.score, reverse=True)
                del bucket[_TRACK_TOP_K:]

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  кадр {frame_idx}/{total_frames}  треков: {len(track_candidates)}")

    cap.release()
    print(f"\n[track] {len(track_candidates)} уникальных ценников найдено")

    output_rows: list[dict[str, str]] = []

    for track_id, candidates in sorted(track_candidates.items()):
        best = max(candidates, key=lambda c: c.score)
        if best.score < quality_threshold:
            print(
                f"  трек {track_id}: score={best.score:.3f} < {quality_threshold}, пропуск"
            )
            continue

        print(
            f"  трек {track_id}: score={best.score:.3f}  ts={best.ts_ms:.0f}ms", end=""
        )

        vlm_result = extract_fields_vlm(best.crop)
        print(f"  VLM->{len([v for v in vlm_result.values() if v])} полей")

        ocr_fields: dict[str, str] = {}
        if ocr_model is not None:
            try:
                enhanced = enhance_crop(best.crop)
                lines = ocr_zoned(ocr_model, enhanced)
                ocr_fields = parse_fields(lines, [], best.crop)
            except Exception as exc:
                print(f"    [warn] OCR failed: {exc}")

        combined = _merge_vlm_and_ocr(vlm_result, ocr_fields)

        out_row = make_empty_row()
        out_row["filename"] = video_name
        out_row["frame_timestamp"] = str(int(best.ts_ms))
        out_row["x_min"] = str(best.det.x1)
        out_row["y_min"] = str(best.det.y1)
        out_row["x_max"] = str(best.det.x2)
        out_row["y_max"] = str(best.det.y2)

        for col in OUTPUT_COLUMNS:
            if col in combined and combined[col]:
                out_row[col] = normalize_text(combined[col])

        for col in (
            "price_discount",
            "discount_amount",
            "code",
            "additional_info",
            "special_symbols",
        ):
            if not out_row[col]:
                out_row[col] = ABSENT_VALUE

        out_row["_quality"] = f"{best.score:.3f}"
        out_row["_track_id"] = str(track_id)
        output_rows.append(out_row)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_cols = OUTPUT_COLUMNS + ["_quality", "_track_id"]
    out_df = pd.DataFrame(output_rows, columns=out_cols)
    out_df.to_csv(out_path, index=False)
    print(f"\n[main] Сохранено {len(out_df)} строк -> {out_path}")

    if output_rows:
        filled = {
            col: out_df[col].apply(lambda v: bool(v) and v != ABSENT_VALUE).sum()
            for col in (
                "product_name",
                "price_card",
                "price_default",
                "barcode",
                "print_datetime",
            )
        }
        n = len(output_rows)
        print("\n[stats] Заполненность полей:")
        for col, cnt in filled.items():
            print(f"  {col:20s}: {cnt}/{n} ({100 * cnt // max(n, 1)}%)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Lenta price tag recognition")
    parser.add_argument("--video", required=True, help="Путь к .mp4 видеофайлу")
    parser.add_argument("--csv", default=None, help="Путь к GT CSV (labeled mode)")
    parser.add_argument("--out", default="results/output.csv", help="Выходной CSV")
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="HuggingFace model ID"
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Включить PaddleOCR fallback (требует paddleocr)",
    )
    parser.add_argument(
        "--quality-thr", type=float, default=0.2, help="Порог качества кропа"
    )
    parser.add_argument(
        "--scan",
        type=int,
        default=20,
        help="+-N кадров для поиска резкого кадра (labeled mode)",
    )
    # Unlabeled (YOLO + ByteTrack) mode
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Unlabeled mode: YOLO+ByteTrack вместо GT CSV",
    )
    parser.add_argument(
        "--weights",
        default="models/price_tag_yolo.pt",
        help="Веса Stage 1 детектора (--detect mode)",
    )
    parser.add_argument(
        "--tracker",
        default="train/bytetrack_price.yaml",
        help="Конфиг ByteTrack трекера (--detect mode)",
    )
    parser.add_argument(
        "--conf-det",
        type=float,
        default=0.25,
        help="Порог уверенности YOLO (--detect mode)",
    )
    parser.add_argument(
        "--device", default=None, help="YOLO device: '0', 'cpu', etc. (--detect mode)"
    )
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: видео не найдено: {args.video}")
        return 1

    if args.detect:
        if not Path(args.weights).exists():
            print(f"Error: веса детектора не найдены: {args.weights}")
            return 1
        process_video_unlabeled(
            video_path=args.video,
            out_path=args.out,
            weights=args.weights,
            model_id=args.model,
            use_ocr=args.ocr,
            quality_threshold=args.quality_thr,
            conf_det=args.conf_det,
            device=args.device,
            tracker=args.tracker,
        )
    else:
        if not args.csv:
            print(
                "Error: укажи --csv для labeled mode, или используй --detect для unlabeled"
            )
            return 1
        if not Path(args.csv).exists():
            print(f"Error: CSV не найден: {args.csv}")
            return 1
        process_video(
            video_path=args.video,
            csv_path=args.csv,
            out_path=args.out,
            model_id=args.model,
            use_ocr=args.ocr,
            quality_threshold=args.quality_thr,
            scan_frames=args.scan,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
