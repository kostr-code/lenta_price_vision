#!/usr/bin/env python3
"""
Real-ESRGAN ×4 super-resolution on QR sub-crops from labeled price tag video.

Pipeline per CSV row:
  1. Find sharpest frame in ±N window (Laplacian variance)
  2. Cut price-tag crop (CCW bbox transform)
  3. Cut QR sub-crop (upper-right corner: x>60%, y<40%)
  4. Decode original QR sub-crop with all decoders
  5. Upscale QR sub-crop ×4 with Real-ESRGAN
  6. Decode upscaled image
  7. Save side-by-side comparison + results CSV

Run: uv run python qr_esrgan.py [--limit N] [--scan N] [--no-esrgan]
"""
import argparse
import pathlib
import urllib.request
from typing import Any

import cv2
import numpy as np
import pandas as pd
from pyzbar.pyzbar import decode as pyzbar_decode
import zxingcpp

# ── Config ─────────────────────────────────────────────────────────────────────

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
    "26_12-20": {
        "csv":   "Данные/26_12-20/26_12-20.csv",
        "video": "Данные/26_12-20/26_12-20.mp4",
    },
}

WECHAT_MODEL_DIR = pathlib.Path("wechat_models")
WECHAT_FILES = ["detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel"]
WECHAT_BASE  = "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/"

ESRGAN_MODEL_DIR = pathlib.Path("models")
ESRGAN_MODEL_PTH = ESRGAN_MODEL_DIR / "RealESRGAN_x4plus.pth"
# Official Real-ESRGAN x4+ model (trained on real-world degradations incl. video compression)
ESRGAN_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN"
    "/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)

# ── Loaders ────────────────────────────────────────────────────────────────────

def load_wechat() -> cv2.wechat_qrcode_WeChatQRCode:
    WECHAT_MODEL_DIR.mkdir(exist_ok=True)
    for fname in WECHAT_FILES:
        p = WECHAT_MODEL_DIR / fname
        if not p.exists():
            print(f"  ↓ WeChat model: {fname}")
            urllib.request.urlretrieve(WECHAT_BASE + fname, p)
    try:
        det = cv2.wechat_qrcode_WeChatQRCode(
            str(WECHAT_MODEL_DIR / "detect.prototxt"),
            str(WECHAT_MODEL_DIR / "detect.caffemodel"),
            str(WECHAT_MODEL_DIR / "sr.prototxt"),
            str(WECHAT_MODEL_DIR / "sr.caffemodel"),
        )
        print("  WeChat: NN-режим (детекция + SRQI)")
    except Exception as e:
        det = cv2.wechat_qrcode_WeChatQRCode()
        print(f"  WeChat: базовый режим ({e})")
    return det


def load_esrgan() -> Any:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    ESRGAN_MODEL_DIR.mkdir(exist_ok=True)
    if not ESRGAN_MODEL_PTH.exists():
        print(f"  ↓ Real-ESRGAN weights (~67 MB): {ESRGAN_MODEL_PTH}")
        urllib.request.urlretrieve(ESRGAN_MODEL_URL, ESRGAN_MODEL_PTH)
        print("  ✓ Скачан")

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32, scale=4,
    )
    upsampler = RealESRGANer(
        scale=4,
        model_path=str(ESRGAN_MODEL_PTH),
        model=model,
        tile=0,        # без тайлинга (QR-кропы маленькие)
        tile_pad=10,
        pre_pad=0,
        half=True,     # FP16 → быстрее на RTX
        gpu_id=0,
    )
    return upsampler

# ── CSV loading ────────────────────────────────────────────────────────────────

def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = df.columns.str.strip()
    for col in ["x_min", "y_min", "x_max", "y_max"]:
        df[col] = df[col].str.replace(",", ".").astype(float)
    df["frame_timestamp"] = pd.to_numeric(df["frame_timestamp"], errors="coerce")
    return df

# ── Frame / crop helpers ───────────────────────────────────────────────────────

def find_best_frame(video_path: str, ts_ms: float, n: int = 20) -> tuple[np.ndarray | None, float]:
    """Return (rotated_frame, laplacian_variance) for the sharpest frame in ±n window."""
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


def cut_crop(frame, row) -> np.ndarray | None:
    """Cut price-tag bbox from CCW-rotated frame."""
    bx1 = int(row["y_min"])
    by1 = int(W_ORIG - 1 - row["x_max"])
    bx2 = int(row["y_max"])
    by2 = int(W_ORIG - 1 - row["x_min"])
    fh, fw = frame.shape[:2]
    c = frame[max(0, by1):min(fh, by2), max(0, bx1):min(fw, bx2)]
    return c if c.size > 0 else None


def cut_qr_subcrop(crop: np.ndarray) -> np.ndarray | None:
    """Upper-right corner where QR lives (x>60%, y<42%)."""
    h, w = crop.shape[:2]
    sub = crop[0:int(h * 0.42), int(w * 0.60):]
    return sub if sub.size > 0 else None

# ── Decoding ───────────────────────────────────────────────────────────────────

def decode_all(img_bgr: np.ndarray, wechat) -> list[str]:
    """Decode QR/barcodes with pyzbar, zxing, WeChat. Returns list of unique values."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    found: set[str] = set()

    for r in pyzbar_decode(gray):
        if r.data:
            found.add(r.data.decode())
    for r in zxingcpp.read_barcodes(gray):
        if r.text:
            found.add(r.text)

    data, _ = wechat.detectAndDecode(img_bgr)
    for d in data:
        if d:
            found.add(d)

    # Also try 2× resize (cheap, sometimes helps barcode detectors)
    gray2 = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    for r in pyzbar_decode(gray2):
        if r.data:
            found.add(r.data.decode())
    for r in zxingcpp.read_barcodes(gray2):
        if r.text:
            found.add(r.text)

    return sorted(found)

# ── Image utils ────────────────────────────────────────────────────────────────

def _add_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(out, text, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 80), 2)
    return out


def save_comparison(panels_labels: list[tuple[np.ndarray, str]], path: pathlib.Path) -> None:
    """Save horizontally stacked panels with labels."""
    target_h = max(p.shape[0] for p, _ in panels_labels)
    imgs = []
    for p, label in panels_labels:
        scale = target_h / p.shape[0]
        resized = cv2.resize(p, (int(p.shape[1] * scale), target_h))
        imgs.append(_add_label(resized, label))
    cv2.imwrite(str(path), np.hstack(imgs))

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",     type=int, default=0,   help="Rows per dataset (0=all)")
    ap.add_argument("--scan",      type=int, default=20,  help="±N frames sharpness scan")
    ap.add_argument("--no-esrgan", action="store_true",   help="Skip ESRGAN (baseline only)")
    args = ap.parse_args()

    out_dir   = pathlib.Path("results/esrgan")
    crops_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(exist_ok=True)

    print("=== QR-ESRGAN pipeline ===\n")
    print("Загружаем WeChat...")
    wechat = load_wechat()

    upsampler = None
    if not args.no_esrgan:
        print("Загружаем Real-ESRGAN...")
        try:
            upsampler = load_esrgan()
            print("  Real-ESRGAN: OK")
        except Exception as e:
            print(f"  Real-ESRGAN: ОШИБКА — {e}\n  Продолжаем без ESRGAN.")

    rows = []

    for ds_name, cfg in DATASETS.items():
        csv_path   = pathlib.Path(cfg["csv"])
        video_path = pathlib.Path(cfg["video"])
        if not csv_path.exists() or not video_path.exists():
            print(f"\nSKIP {ds_name}: файлы не найдены")
            continue

        df = load_df(str(csv_path))
        if args.limit:
            df = df.head(args.limit)
        print(f"\n=== {ds_name}: {len(df)} строк ===")

        for pos, (_, row) in enumerate(df.iterrows()):
            ts    = float(row["frame_timestamp"])
            gt_qr = str(row.get("qr_code_barcode", "")).strip()
            print(f"  [{pos:3d}] ts={ts:7.0f}ms  gt_qr={gt_qr[:24] or '—'}", end="  ")

            frame, lap = find_best_frame(str(video_path), ts, args.scan)
            if frame is None:
                print("→ кадр не найден")
                continue

            crop = cut_crop(frame, row)
            if crop is None:
                print("→ кроп пустой")
                continue

            qr_sub = cut_qr_subcrop(crop)
            if qr_sub is None:
                print("→ QR sub-crop пустой")
                continue

            orig_decoded   = decode_all(qr_sub, wechat)
            esrgan_decoded = []
            qr_up          = None

            if upsampler is not None:
                try:
                    qr_up, _ = upsampler.enhance(qr_sub, outscale=4)
                    esrgan_decoded = decode_all(qr_up, wechat)
                except Exception as e:
                    print(f"[ESRGAN err: {e}] ", end="")

            status = (
                "✅ ESRGAN" if esrgan_decoded else
                "✅ ORIG"   if orig_decoded   else
                "❌"
            )
            print(f"→ {status}  orig={orig_decoded}  esrgan={esrgan_decoded}")

            # Save comparison image
            panels = [
                (qr_sub, f"Orig {qr_sub.shape[1]}x{qr_sub.shape[0]}"),
            ]
            if qr_up is not None:
                panels.append((
                    cv2.resize(qr_up, (qr_sub.shape[1] * 2, qr_sub.shape[0] * 2)),
                    f"ESRGAN×4 decoded={bool(esrgan_decoded)}",
                ))
            save_comparison(panels, crops_dir / f"{ds_name}_{pos:03d}.jpg")

            rows.append({
                "dataset":       ds_name,
                "row_idx":       pos,
                "frame_ts_ms":   ts,
                "lap_var":       round(lap, 1),
                "gt_qr":         gt_qr,
                "orig_decoded":  "|".join(orig_decoded),
                "esrgan_decoded":"|".join(esrgan_decoded),
                "orig_ok":       bool(orig_decoded),
                "esrgan_ok":     bool(esrgan_decoded),
                "gt_match":      bool(orig_decoded and gt_qr and gt_qr in orig_decoded)
                                 or bool(esrgan_decoded and gt_qr and gt_qr in esrgan_decoded),
            })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_dir / "esrgan_results.csv", index=False)

    total      = len(result_df)
    orig_ok    = int(result_df["orig_ok"].sum())    if total else 0
    esrgan_ok  = int(result_df["esrgan_ok"].sum())  if total else 0
    gt_match   = int(result_df["gt_match"].sum())   if total else 0

    print(f"\n{'='*55}")
    print(f"Итого: {total} ценников")
    print(f"  Baseline (orig+2×):  {orig_ok}/{total}  ({100*orig_ok/max(total,1):.1f}%)")
    print(f"  Real-ESRGAN ×4:      {esrgan_ok}/{total}  ({100*esrgan_ok/max(total,1):.1f}%)")
    print(f"  Совпало с GT QR:     {gt_match}/{total}  ({100*gt_match/max(total,1):.1f}%)")
    print(f"\nРезультаты:  {out_dir}/esrgan_results.csv")
    print(f"Кропы:       {crops_dir}/")


if __name__ == "__main__":
    main()
