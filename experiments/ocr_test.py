#!/usr/bin/env python3
"""
PaddleOCR test on labeled price tag crops.

Reads each labeled bbox from CSV, finds the sharpest frame, cuts the price-tag
crop, runs PaddleOCR (Russian model), and prints detected text regions with
confidence scores. Saves annotated crop images for visual review.

Run: uv run python -u ocr_test.py [--limit N] [--scan N] [--gpu]
"""

import argparse
import pathlib

import cv2
import numpy as np
import pandas as pd

W_ORIG = 3840

DATASETS = {
    "43_15": {
        "csv": "Данные/43_15/43_15.csv",
        "video": "Данные/43_15/43_15.mp4",
    },
    "25_12-20": {
        "csv": "Данные/25_12-20/25_12-20.csv",
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
    c = frame[max(0, by1) : min(fh, by2), max(0, bx1) : min(fw, bx2)]
    return c if c.size > 0 else None


def sharpen_crop(img):
    """Light sharpening to help OCR on slightly blurry video frames."""
    blur = cv2.GaussianBlur(img, (0, 0), 1.5)
    return cv2.addWeighted(img, 1.8, blur, -0.8, 0)


def annotate_ocr(img, ocr_results):
    """Draw OCR bboxes and text on a copy of the image."""
    out = img.copy()
    if not ocr_results or not ocr_results[0]:
        return out
    for line in ocr_results[0]:
        if line is None:
            continue
        pts, (text, conf) = line
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 0), 1)
        x, y = pts[0]
        cv2.putText(
            out,
            f"{text[:20]} {conf:.2f}",
            (int(x), max(int(y) - 3, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            out,
            f"{text[:20]} {conf:.2f}",
            (int(x), max(int(y) - 3, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
        )
    return out


def extract_text_blocks(ocr_results, crop_h, crop_w):
    """
    Group OCR results by vertical position (relative to crop height).
    Returns dict with approximate field positions:
      'top'    → y < 35%  (product name + QR zone)
      'mid'    → 35-65%   (prices area)
      'bottom' → 65-100%  (barcode, date, code, footer)
    """
    blocks = {"top": [], "mid": [], "bottom": []}
    if not ocr_results or not ocr_results[0]:
        return blocks
    for line in ocr_results[0]:
        if line is None:
            continue
        pts, (text, conf) = line
        if conf < 0.4:
            continue
        y_center = np.mean([p[1] for p in pts]) / crop_h
        region = "top" if y_center < 0.35 else ("mid" if y_center < 0.65 else "bottom")
        blocks[region].append({"text": text, "conf": round(conf, 3), "y_rel": round(y_center, 3)})
    return blocks


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Rows per dataset (0=all)")
    ap.add_argument("--scan", type=int, default=20, help="±N frame sharpness scan")
    ap.add_argument("--gpu", action="store_true", help="Use GPU for PaddleOCR")
    ap.add_argument("--lang", default="ru", help="PaddleOCR lang (default: ru)")
    args = ap.parse_args()

    print("=== PaddleOCR test pipeline ===\n")
    print(f"Загружаем PaddleOCR (lang={args.lang}, gpu={args.gpu})...")
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        use_angle_cls=True,
        lang=args.lang,
        use_gpu=args.gpu,
        show_log=False,
    )
    print("  PaddleOCR: OK\n")

    out_dir = pathlib.Path("results/ocr")
    crops_dir = out_dir / "annotated"
    raw_dir = out_dir / "raw_crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)

    all_rows = []

    for ds_name, cfg in DATASETS.items():
        csv_path = pathlib.Path(cfg["csv"])
        video_path = pathlib.Path(cfg["video"])
        if not csv_path.exists() or not video_path.exists():
            print(f"SKIP {ds_name}")
            continue

        df = load_df(str(csv_path))
        if args.limit:
            df = df.head(args.limit)
        print(f"=== {ds_name}: {len(df)} строк ===")

        for pos, (_, row) in enumerate(df.iterrows()):
            ts = float(row["frame_timestamp"])
            gt_name = str(row.get("product_name", "")).strip()
            gt_qr = str(row.get("qr_code_barcode", "")).strip()
            gt_price = str(row.get("price_card", "")).strip()

            frame, lap = find_best_frame(str(video_path), ts, args.scan)
            if frame is None:
                print(f"  [{pos:3d}] кадр не найден")
                continue

            crop = cut_crop(frame, row)
            if crop is None:
                print(f"  [{pos:3d}] кроп пустой")
                continue

            # Slight sharpen before OCR
            crop_sharp = sharpen_crop(crop)
            h, w = crop.shape[:2]

            # Run OCR
            result = ocr.ocr(crop_sharp, cls=True)
            blocks = extract_text_blocks(result, h, w)

            all_text = []
            if result and result[0]:
                all_text = [line[1][0] for line in result[0] if line and line[1][1] > 0.4]

            # Simple GT match: check if product_name words appear in top-block text
            name_words = set(gt_name.lower().split()) if gt_name else set()
            top_text = " ".join(b["text"] for b in blocks["top"]).lower()
            name_match_frac = (
                sum(1 for w_ in name_words if w_ in top_text) / max(len(name_words), 1)
                if name_words
                else 0.0
            )

            print(
                f"  [{pos:3d}] lap={lap:.0f}  {w}×{h}px  "
                f"OCR lines={len(all_text)}  "
                f"name_match={name_match_frac:.0%}"
            )
            # Print top zone text (product name area)
            if blocks["top"]:
                top_str = " | ".join(b["text"] for b in blocks["top"])
                print(f"    TOP:    {top_str[:80]}")
            if blocks["mid"]:
                mid_str = " | ".join(b["text"] for b in blocks["mid"])
                print(f"    MID:    {mid_str[:80]}")
            if blocks["bottom"]:
                bot_str = " | ".join(b["text"] for b in blocks["bottom"])
                print(f"    BOTTOM: {bot_str[:80]}")
            print(f"    GT name:  {gt_name[:60]}")
            print(f"    GT price: {gt_price}  GT barcode: {gt_qr}")

            # Save annotated + raw crops
            fname = f"{ds_name}_{pos:03d}.jpg"
            cv2.imwrite(str(raw_dir / fname), crop)
            cv2.imwrite(str(crops_dir / fname), annotate_ocr(crop, result))

            all_rows.append(
                {
                    "dataset": ds_name,
                    "row_idx": pos,
                    "frame_ts_ms": ts,
                    "lap_var": round(lap, 1),
                    "crop_h": h,
                    "crop_w": w,
                    "gt_product_name": gt_name,
                    "gt_price_card": gt_price,
                    "gt_qr": gt_qr,
                    "ocr_lines": len(all_text),
                    "ocr_top": " | ".join(b["text"] for b in blocks["top"]),
                    "ocr_mid": " | ".join(b["text"] for b in blocks["mid"]),
                    "ocr_bottom": " | ".join(b["text"] for b in blocks["bottom"]),
                    "name_match_frac": round(name_match_frac, 3),
                    "has_ocr": len(all_text) > 0,
                }
            )

    df_res = pd.DataFrame(all_rows)
    df_res.to_csv(out_dir / "ocr_results.csv", index=False)

    total = len(df_res)
    has_ocr = int(df_res["has_ocr"].sum()) if total else 0
    avg_lines = df_res["ocr_lines"].mean() if total else 0
    avg_match = df_res["name_match_frac"].mean() if total else 0

    print(f"\n{'=' * 55}")
    print(f"Итого: {total} ценников")
    print(f"  Получили хоть что-то:     {has_ocr}/{total} ({100 * has_ocr / max(total, 1):.1f}%)")
    print(f"  Среднее строк OCR:        {avg_lines:.1f}")
    print(f"  Совпадение product_name:  {avg_match:.0%} (средн. по словам)")
    print(f"\nРезультаты:    {out_dir}/ocr_results.csv")
    print(f"С разметкой:   {crops_dir}/")
    print(f"Сырые кропы:   {raw_dir}/")


if __name__ == "__main__":
    main()
