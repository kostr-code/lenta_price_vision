#!/usr/bin/env python3
"""
ESRGAN ×4 upscale of the full price-tag crop → PaddleOCR.

The price-tag crops are only ~200×100 px at video resolution, which is too small
for reliable barcode / OCR reading. Upscaling ×4 gives ~800×400 px which is
in PaddleOCR's sweet spot.

Pipeline per CSV row:
  1. Find sharpest frame (Laplacian ±N)
  2. Cut full price-tag crop (CCW bbox transform)
  3. Upscale ×4 with Real-ESRGAN (GPU)
  4. PaddleOCR on upscaled crop
  5. Try linear barcode on upscaled bottom region
  6. Save annotated crops + results CSV

Run: uv run python -u ocr_esrgan.py [--limit N] [--scan N] [--no-esrgan] [--gpu-ocr]
"""
import argparse
import pathlib
import urllib.request

import cv2
import numpy as np
import pandas as pd
import zxingcpp
from pyzbar.pyzbar import decode as pyzbar_decode

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

ESRGAN_MODEL_DIR = pathlib.Path("models")
ESRGAN_MODEL_PTH = ESRGAN_MODEL_DIR / "RealESRGAN_x4plus.pth"
ESRGAN_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN"
    "/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)

# ── Loaders ────────────────────────────────────────────────────────────────────

def load_esrgan():
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    ESRGAN_MODEL_DIR.mkdir(exist_ok=True)
    if not ESRGAN_MODEL_PTH.exists():
        print(f"  ↓ Real-ESRGAN weights (~67 MB)...")
        urllib.request.urlretrieve(ESRGAN_MODEL_URL, ESRGAN_MODEL_PTH)
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32, scale=4,
    )
    return RealESRGANer(
        scale=4, model_path=str(ESRGAN_MODEL_PTH),
        model=model, tile=512, tile_pad=10, pre_pad=0,
        half=True, gpu_id=0,
    )


def load_ocr(use_gpu=False):
    from paddleocr import PaddleOCR
    return PaddleOCR(
        use_angle_cls=True,
        lang="ru",
        use_gpu=use_gpu,
        show_log=False,
    )

# ── Video helpers ──────────────────────────────────────────────────────────────

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

# ── Decoding ───────────────────────────────────────────────────────────────────

def decode_barcode(img_bgr):
    """Try pyzbar + zxing on image (no WeChat — too slow for batch)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    found = set()
    for r in pyzbar_decode(gray):
        if r.data: found.add(r.data.decode())
    for r in zxingcpp.read_barcodes(gray):
        if r.text: found.add(r.text)
    return sorted(found)


def extract_ocr_fields(ocr_result, h, w):
    """
    Map OCR lines to approximate price-tag zones.
    Returns dict: zone → list of (text, conf, y_rel).
    """
    zones = {"top": [], "mid": [], "bottom": []}
    if not ocr_result or not ocr_result[0]:
        return zones
    for line in ocr_result[0]:
        if line is None:
            continue
        pts, (text, conf) = line
        if conf < 0.35:
            continue
        y_center = np.mean([p[1] for p in pts]) / h
        zone = "top" if y_center < 0.35 else ("mid" if y_center < 0.65 else "bottom")
        zones[zone].append((text, round(conf, 3), round(y_center, 3)))
    for z in zones:
        zones[z].sort(key=lambda t: t[2])  # sort by y
    return zones


def annotate(img, ocr_result):
    out = img.copy()
    if not ocr_result or not ocr_result[0]:
        return out
    for line in ocr_result[0]:
        if line is None: continue
        pts, (text, conf) = line
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 0), 1)
        x0, y0 = pts[0]
        cv2.putText(out, f"{text[:18]}({conf:.2f})",
                    (int(x0), max(int(y0)-3, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 2)
        cv2.putText(out, f"{text[:18]}({conf:.2f})",
                    (int(x0), max(int(y0)-3, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,255,200), 1)
    return out

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",     type=int, default=0,   help="Rows per dataset (0=all)")
    ap.add_argument("--scan",      type=int, default=20,  help="±N frame scan")
    ap.add_argument("--no-esrgan", action="store_true",   help="Skip ESRGAN (OCR on raw crop)")
    ap.add_argument("--gpu-ocr",   action="store_true",   help="Use GPU for PaddleOCR")
    args = ap.parse_args()

    out_dir      = pathlib.Path("results/ocr_esrgan")
    annotated_dir = out_dir / "annotated"
    raw_dir      = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)

    print("=== ESRGAN ×4 + PaddleOCR pipeline ===\n")

    upsampler = None
    if not args.no_esrgan:
        print("Загружаем Real-ESRGAN...")
        try:
            upsampler = load_esrgan()
            print("  Real-ESRGAN: OK (GPU FP16)")
        except Exception as e:
            print(f"  Real-ESRGAN: ОШИБКА — {e}. Продолжаем без SR.")

    print("Загружаем PaddleOCR (lang=ru)...")
    ocr = load_ocr(use_gpu=args.gpu_ocr)
    print(f"  PaddleOCR: OK (gpu={args.gpu_ocr})\n")

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
        print(f"=== {ds_name}: {len(df)} строк ===")

        for pos, (_, row) in enumerate(df.iterrows()):
            ts       = float(row["frame_timestamp"])
            gt_name  = str(row.get("product_name", "")).strip()
            gt_price = str(row.get("price_card", "")).strip()
            gt_qr    = str(row.get("qr_code_barcode", "")).strip()

            frame, lap = find_best_frame(str(video_path), ts, args.scan)
            if frame is None:
                print(f"  [{pos:3d}] кадр не найден")
                continue

            crop = cut_crop(frame, row)
            if crop is None:
                print(f"  [{pos:3d}] кроп пустой")
                continue

            raw_h, raw_w = crop.shape[:2]

            # Upscale with ESRGAN
            if upsampler is not None:
                try:
                    img_for_ocr, _ = upsampler.enhance(crop, outscale=4)
                    sr_label = f"ESRGAN×4 {img_for_ocr.shape[1]}x{img_for_ocr.shape[0]}"
                except Exception as e:
                    print(f"  [{pos:3d}] ESRGAN err: {e}, using raw crop")
                    img_for_ocr = crop
                    sr_label = f"raw {raw_w}x{raw_h}"
            else:
                img_for_ocr = cv2.resize(crop, None, fx=4, fy=4,
                                         interpolation=cv2.INTER_LANCZOS4)
                sr_label = f"lanczos×4 {img_for_ocr.shape[1]}x{img_for_ocr.shape[0]}"

            # OCR
            h_up, w_up = img_for_ocr.shape[:2]
            ocr_result = ocr.ocr(img_for_ocr, cls=True)
            zones = extract_ocr_fields(ocr_result, h_up, w_up)

            # Linear barcode on upscaled bottom region
            bc_y1 = int(h_up * 0.72)
            bc_y2 = int(h_up * 0.90)
            bc_x1 = int(w_up * 0.20)
            bc_x2 = int(w_up * 0.90)
            barcode_region = img_for_ocr[bc_y1:bc_y2, bc_x1:bc_x2]
            barcode_decoded = decode_barcode(barcode_region) if barcode_region.size > 0 else []

            all_text = [t for t, _, _ in (zones["top"] + zones["mid"] + zones["bottom"])]
            n_lines  = len(all_text)

            name_words  = set(gt_name.lower().split()) if gt_name else set()
            top_text_lw = " ".join(t for t, _, _ in zones["top"]).lower()
            name_match  = (
                sum(1 for w_ in name_words if w_ in top_text_lw) / max(len(name_words), 1)
                if name_words else 0.0
            )

            bc_status = "✅" if barcode_decoded else "❌"
            print(
                f"  [{pos:3d}] raw={raw_w}×{raw_h}→{sr_label}  "
                f"OCR={n_lines}lines  barcode:{bc_status}  name={name_match:.0%}"
            )
            if zones["top"]:
                print(f"    TOP:    {' | '.join(t for t,_,_ in zones['top'])[:80]}")
            if zones["mid"]:
                print(f"    MID:    {' | '.join(t for t,_,_ in zones['mid'])[:80]}")
            if zones["bottom"]:
                print(f"    BOT:    {' | '.join(t for t,_,_ in zones['bottom'])[:80]}")
            if gt_name:
                print(f"    GT:     name={gt_name[:50]}  price={gt_price}  qr={gt_qr}")

            # Save images
            fname = f"{ds_name}_{pos:03d}.jpg"
            cv2.imwrite(str(raw_dir      / fname), crop)
            cv2.imwrite(str(annotated_dir / fname), annotate(img_for_ocr, ocr_result))

            rows.append({
                "dataset":        ds_name,
                "row_idx":        pos,
                "frame_ts_ms":    ts,
                "lap_var":        round(lap, 1),
                "raw_w":          raw_w,
                "raw_h":          raw_h,
                "up_w":           img_for_ocr.shape[1],
                "up_h":           img_for_ocr.shape[0],
                "gt_product_name":gt_name,
                "gt_price_card":  gt_price,
                "gt_qr":          gt_qr,
                "ocr_lines":      n_lines,
                "ocr_top":        " | ".join(t for t,_,_ in zones["top"]),
                "ocr_mid":        " | ".join(t for t,_,_ in zones["mid"]),
                "ocr_bottom":     " | ".join(t for t,_,_ in zones["bottom"]),
                "barcode_decoded":"|".join(barcode_decoded),
                "name_match_frac":round(name_match, 3),
                "has_ocr":        n_lines > 0,
                "barcode_ok":     bool(barcode_decoded),
            })

    df_res = pd.DataFrame(rows)
    df_res.to_csv(out_dir / "ocr_esrgan_results.csv", index=False)

    total     = len(df_res)
    has_ocr   = int(df_res["has_ocr"].sum())    if total else 0
    bc_ok     = int(df_res["barcode_ok"].sum()) if total else 0
    avg_lines = df_res["ocr_lines"].mean()       if total else 0
    avg_match = df_res["name_match_frac"].mean() if total else 0

    print(f"\n{'='*60}")
    print(f"Итого: {total} ценников")
    print(f"  OCR что-то нашёл:         {has_ocr}/{total} ({100*has_ocr/max(total,1):.1f}%)")
    print(f"  Среднее строк OCR:        {avg_lines:.1f}")
    print(f"  Совпадение product_name:  {avg_match:.0%}")
    print(f"  Линейный штрихкод:        {bc_ok}/{total} ({100*bc_ok/max(total,1):.1f}%)")
    print(f"\nРезультаты:  {out_dir}/ocr_esrgan_results.csv")
    print(f"Аннотации:   {annotated_dir}/")


if __name__ == "__main__":
    main()
