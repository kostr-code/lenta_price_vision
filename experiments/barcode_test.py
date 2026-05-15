#!/usr/bin/env python3
"""
Linear barcode detection from the bottom region of labeled price tag crops.

Unlike QR (upper-right corner, tiny modules), EAN-13 linear barcodes at the
bottom of price tags have much wider bars → more pixels per element → higher
decode probability even with video compression.

Pipeline per CSV row:
  1. Find sharpest frame (Laplacian scan ±N)
  2. Cut price-tag crop (CCW bbox transform)
  3. Cut barcode region (bottom ~20% of crop)
  4. Try pyzbar + zxing on raw and 2×/4× upscaled region
  5. Compare against GT qr_code_barcode (same EAN number as the linear barcode)

Run: uv run python barcode_test.py [--limit N] [--scan N]
"""
import argparse
import pathlib

import cv2
import numpy as np
import pandas as pd
from pyzbar.pyzbar import decode as pyzbar_decode
import zxingcpp

W_ORIG = 3840

DATASETS = {
    "43_15": {
        "csv":   "Данные/43_15/43_15.csv",
        "video": "Данные/43_15/43_15.mp4",
    },
    "25_12-20": {
        "csv":   "Данные/25_12-20/25_12-20.csv",
        "video": "Данные/25_12-20/25_12-20.mp4",
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_df(path):
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()
    for col in ["x_min", "y_min", "x_max", "y_max"]:
        df[col] = df[col].str.replace(",", ".").astype(float)
    df["frame_timestamp"] = pd.to_numeric(df["frame_timestamp"], errors="coerce")
    return df


def find_best_frame(video_path, ts_ms, n=20):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = 1000.0 / fps
    best_var, best_frame = -1.0, None
    for i in range(-n, n + 1):
        cap.set(cv2.CAP_PROP_POS_MSEC, ts_ms + i * step)
        ok, raw = cap.read()
        if not ok:
            continue
        rot = cv2.rotate(raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
        v = cv2.Laplacian(cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if v > best_var:
            best_var, best_frame = v, rot.copy()
    cap.release()
    return best_frame, best_var


def cut_crop(frame, row):
    bx1 = int(row["y_min"])
    by1 = int(W_ORIG - 1 - row["x_max"])
    bx2 = int(row["y_max"])
    by2 = int(W_ORIG - 1 - row["x_min"])
    fh, fw = frame.shape[:2]
    c = frame[max(0, by1):min(fh, by2), max(0, bx1):min(fw, bx2)]
    return c if c.size > 0 else None


def cut_barcode_region(crop):
    """Bottom band of the price tag where EAN-13 linear barcode lives."""
    h, w = crop.shape[:2]
    # Barcode is roughly in y=[0.72..0.90], x=[0.20..0.90]
    region = crop[int(h * 0.72):int(h * 0.90), int(w * 0.20):int(w * 0.90)]
    return region if region.size > 0 else None


def cut_qr_region(crop):
    """Upper-right corner where QR lives — for comparison."""
    h, w = crop.shape[:2]
    region = crop[0:int(h * 0.42), int(w * 0.60):]
    return region if region.size > 0 else None


def decode_region(img_bgr):
    """Try pyzbar + zxing on image at multiple scales. Returns list of decoded strings."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    found = set()
    for scale in [1, 2, 4]:
        if scale > 1:
            g = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        else:
            g = gray
        for r in pyzbar_decode(g):
            if r.data:
                found.add(r.data.decode())
        for r in zxingcpp.read_barcodes(g):
            if r.text:
                found.add(r.text)
    return sorted(found)


def save_debug_strip(crop, barcode_region, qr_region, save_path):
    """Save full crop with highlighted regions for visual inspection."""
    h, w = crop.shape[:2]
    annotated = crop.copy()
    # Draw barcode region
    by1, by2 = int(h * 0.72), int(h * 0.90)
    bx1, bx2 = int(w * 0.20), int(w * 0.90)
    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
    cv2.putText(annotated, "barcode", (bx1+3, by1+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # Draw QR region
    qy2, qx1 = int(h * 0.42), int(w * 0.60)
    cv2.rectangle(annotated, (qx1, 0), (w, qy2), (0, 120, 255), 2)
    cv2.putText(annotated, "QR", (qx1+3, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2)
    cv2.imwrite(str(save_path), annotated)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Rows per dataset (0=all)")
    ap.add_argument("--scan",  type=int, default=20, help="±N frame sharpness scan")
    ap.add_argument("--save-crops", action="store_true", help="Save annotated crop images")
    args = ap.parse_args()

    out_dir   = pathlib.Path("results/barcode")
    crops_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_crops:
        crops_dir.mkdir(exist_ok=True)

    rows = []

    for ds_name, cfg in DATASETS.items():
        csv_path   = pathlib.Path(cfg["csv"])
        video_path = pathlib.Path(cfg["video"])
        if not csv_path.exists() or not video_path.exists():
            print(f"SKIP {ds_name}")
            continue

        df = load_df(str(csv_path))
        if args.limit:
            df = df.head(args.limit)
        print(f"\n=== {ds_name}: {len(df)} строк ===")

        for pos, (_, row) in enumerate(df.iterrows()):
            ts    = float(row["frame_timestamp"])
            gt_qr = str(row.get("qr_code_barcode", "")).strip()

            frame, _ = find_best_frame(str(video_path), ts, args.scan)
            if frame is None:
                print(f"  [{pos:3d}] кадр не найден")
                continue

            crop = cut_crop(frame, row)
            if crop is None:
                print(f"  [{pos:3d}] кроп пустой")
                continue

            barcode_region = cut_barcode_region(crop)
            qr_region      = cut_qr_region(crop)

            barcode_decoded = decode_region(barcode_region) if barcode_region is not None else []
            qr_decoded      = decode_region(qr_region)      if qr_region      is not None else []

            # Barcode match: the EAN-13 code should be same as qr_code_barcode GT
            bc_match = bool(gt_qr and any(gt_qr in d for d in barcode_decoded))
            qr_match = bool(gt_qr and any(gt_qr in d for d in qr_decoded))

            status_bc = "✅" if barcode_decoded else "❌"
            status_qr = "✅" if qr_decoded else "❌"
            print(
                f"  [{pos:3d}] barcode:{status_bc}{barcode_decoded}  "
                f"QR:{status_qr}{qr_decoded}  gt={gt_qr[:16] or '—'}"
            )

            if args.save_crops:
                save_debug_strip(
                    crop, barcode_region, qr_region,
                    crops_dir / f"{ds_name}_{pos:03d}.jpg",
                )

            h_crop, w_crop = crop.shape[:2]
            rows.append({
                "dataset":        ds_name,
                "row_idx":        pos,
                "gt_qr":          gt_qr,
                "crop_h":         h_crop,
                "crop_w":         w_crop,
                "barcode_decoded":"|".join(barcode_decoded),
                "qr_decoded":     "|".join(qr_decoded),
                "barcode_ok":     bool(barcode_decoded),
                "qr_ok":          bool(qr_decoded),
                "barcode_match":  bc_match,
                "qr_match":       qr_match,
            })

    df_res = pd.DataFrame(rows)
    df_res.to_csv(out_dir / "barcode_results.csv", index=False)

    total = len(df_res)
    def pct(col): return f"{int(df_res[col].sum())}/{total} ({100*df_res[col].mean():.1f}%)"

    print(f"\n{'='*55}")
    print(f"Итого: {total} ценников")
    print(f"  Linear barcode decoded:   {pct('barcode_ok')}")
    print(f"  Linear barcode GT match:  {pct('barcode_match')}")
    print(f"  QR decoded:               {pct('qr_ok')}")
    print(f"  QR GT match:              {pct('qr_match')}")
    if total > 0:
        sizes = df_res[["crop_h", "crop_w"]].describe()
        print(f"\nРазмер кропов (h×w):")
        print(f"  mean: {sizes['crop_h']['mean']:.0f}×{sizes['crop_w']['mean']:.0f} px")
        print(f"  min:  {sizes['crop_h']['min']:.0f}×{sizes['crop_w']['min']:.0f} px")
        print(f"  max:  {sizes['crop_h']['max']:.0f}×{sizes['crop_w']['max']:.0f} px")
    print(f"\nРезультаты: {out_dir}/barcode_results.csv")


if __name__ == "__main__":
    main()
