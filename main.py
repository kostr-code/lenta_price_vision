"""
main.py — Price tag recognition pipeline (labeled-data mode).

Usage:
  uv run python main.py \\
      --video data/Данные/43_15/43_15.mp4 \\
      --csv   data/Данные/43_15/43_15.csv \\
      --out   results/43_15_out.csv

The pipeline:
  load_df → per unique timestamp: find_best_frame
  → per bbox row: cut_crop → quality_score
  → VLM (Qwen2.5-VL): extract visual fields
  → parse_fields: regex-fill remaining fields
  → merge → save CSV

For unlabeled video (no GT CSV), YOLO detection is needed first.
That path will be added once the YOLO weights are wired in.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
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
from pipeline.video import cut_crop_from_row, find_best_frame, load_df
from pipeline.vlm import VLM_FIELDS, extract_fields_vlm, load_vlm

# ── OCR is optional — import only if paddleocr is installed ──────────────────
try:
    from pipeline.ocr import enhance_crop, load_ocr, ocr_zoned
    _HAS_OCR = True
except ImportError:
    _HAS_OCR = False


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
        print(f"[{idx + 1}/{total}] ts={ts_ms:.0f}ms  {row.get('product_name', '')[:40]}")

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
            print(f"  [warn] Quality {quality:.3f} < {quality_threshold} — crop likely degraded")

        # ── Primary: VLM ──────────────────────────────────────────────────────
        vlm_result = extract_fields_vlm(crop)
        print(f"  VLM → {len([v for v in vlm_result.values() if v])} fields extracted")

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
        for meta_col in ("filename", "frame_timestamp", "x_min", "y_min", "x_max", "y_max"):
            if meta_col in row:
                out_row[meta_col] = str(row[meta_col]) if pd.notna(row[meta_col]) else ""
        if not out_row["filename"]:
            out_row["filename"] = video_name

        # Fill recognized fields
        for col in OUTPUT_COLUMNS:
            if col in combined and combined[col]:
                out_row[col] = normalize_text(combined[col])

        # Apply absent-value defaults for optional fields
        for col in ("price_discount", "discount_amount", "code", "additional_info", "special_symbols"):
            if not out_row[col]:
                out_row[col] = ABSENT_VALUE

        out_row["_quality"] = f"{quality:.3f}"
        output_rows.append(out_row)

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_cols = OUTPUT_COLUMNS + ["_quality"]
    out_df = pd.DataFrame(output_rows, columns=out_cols)
    out_df.to_csv(out_path, index=False)
    print(f"\n[main] Saved {len(out_df)} rows → {out_path}")

    # Quick stats
    filled = {
        col: out_df[col].apply(lambda v: bool(v) and v != ABSENT_VALUE).sum()
        for col in ("product_name", "price_card", "price_default", "barcode", "print_datetime")
    }
    print("\n[stats] Field fill rate:")
    for col, n in filled.items():
        print(f"  {col:20s}: {n}/{total} ({100*n//max(total,1)}%)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lenta price tag recognition — labeled-data mode"
    )
    parser.add_argument("--video", required=True, help="Path to .mp4 video file")
    parser.add_argument("--csv", required=True, help="Path to labeled GT CSV")
    parser.add_argument("--out", default="results/output.csv", help="Output CSV path")
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="HuggingFace model ID"
    )
    parser.add_argument(
        "--ocr", action="store_true", help="Enable PaddleOCR fallback (requires paddleocr)"
    )
    parser.add_argument(
        "--quality-thr", type=float, default=0.2, help="Quality score threshold (warn below)"
    )
    parser.add_argument(
        "--scan", type=int, default=20, help="+-N frames to scan for sharpest frame"
    )
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: video not found: {args.video}")
        return 1
    if not Path(args.csv).exists():
        print(f"Error: CSV not found: {args.csv}")
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
