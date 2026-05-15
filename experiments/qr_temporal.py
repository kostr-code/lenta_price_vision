#!/usr/bin/env python3
"""
Temporal multi-frame fusion for QR decoding.

For each labeled price-tag bbox:
  1. Collect frames in ±N window around CSV timestamp
  2. Cut the same bbox crop from each frame (CCW transform)
  3. Cut QR sub-crop (upper-right) and score by Laplacian sharpness
  4. Take top-K sharpest crops
  5. Align them to the reference (sharpest) using ECC (translation mode)
  6. Average aligned crops → fused image
  7. Optionally apply Real-ESRGAN ×4 on fused result
  8. Try all decoders on fused (and optionally ESRGAN-fused) crop
  9. Save results + comparison images

Run: uv run python qr_temporal.py [--limit N] [--win N] [--topk N] [--esrgan]
"""
import argparse
import pathlib
import urllib.request

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
}

WECHAT_MODEL_DIR = pathlib.Path("wechat_models")
WECHAT_FILES     = ["detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel"]
WECHAT_BASE      = "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/"

ESRGAN_MODEL_DIR = pathlib.Path("models")
ESRGAN_MODEL_PTH = ESRGAN_MODEL_DIR / "RealESRGAN_x4plus.pth"
ESRGAN_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN"
    "/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)

ECC_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-5)

# ── Loaders ────────────────────────────────────────────────────────────────────

def load_wechat():
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
        print("  WeChat: NN-режим")
    except Exception as e:
        det = cv2.wechat_qrcode_WeChatQRCode()
        print(f"  WeChat: базовый ({e})")
    return det


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
        model=model, tile=0, tile_pad=10, pre_pad=0,
        half=True, gpu_id=0,
    )

# ── CSV loading ────────────────────────────────────────────────────────────────

def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = df.columns.str.strip()
    for col in ["x_min", "y_min", "x_max", "y_max"]:
        df[col] = df[col].str.replace(",", ".").astype(float)
    df["frame_timestamp"] = pd.to_numeric(df["frame_timestamp"], errors="coerce")
    return df

# ── Video / crop helpers ───────────────────────────────────────────────────────

def collect_frames(video_path: str, ts_ms: float, n: int) -> list[np.ndarray]:
    """Return all rotated frames in [ts_ms - n*step, ts_ms + n*step]."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = 1000.0 / fps
    frames = []
    for i in range(-n, n + 1):
        cap.set(cv2.CAP_PROP_POS_MSEC, ts_ms + i * step)
        ok, raw = cap.read()
        if ok:
            frames.append(cv2.rotate(raw, cv2.ROTATE_90_COUNTERCLOCKWISE))
    cap.release()
    return frames


def cut_crop(frame: np.ndarray, row) -> np.ndarray | None:
    bx1 = int(row["y_min"])
    by1 = int(W_ORIG - 1 - row["x_max"])
    bx2 = int(row["y_max"])
    by2 = int(W_ORIG - 1 - row["x_min"])
    fh, fw = frame.shape[:2]
    c = frame[max(0, by1):min(fh, by2), max(0, bx1):min(fw, bx2)]
    return c if c.size > 0 else None


def cut_qr_subcrop(crop: np.ndarray) -> np.ndarray | None:
    h, w = crop.shape[:2]
    sub = crop[0:int(h * 0.42), int(w * 0.60):]
    return sub if sub.size > 0 else None


def lap_var(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

# ── Temporal fusion ────────────────────────────────────────────────────────────

def ecc_align(reference: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    """Align target to reference using ECC (translation only). Returns warped target or None."""
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tgt_gray = cv2.cvtColor(target,    cv2.COLOR_BGR2GRAY).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(
            ref_gray, tgt_gray, warp,
            cv2.MOTION_TRANSLATION, ECC_CRITERIA,
        )
        h, w = reference.shape[:2]
        aligned = cv2.warpAffine(
            target, warp, (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
        return aligned
    except cv2.error:
        return None


def temporal_fuse(crops: list[np.ndarray], top_k: int) -> np.ndarray | None:
    """
    Select top-K sharpest crops, ECC-align to the sharpest, average.
    Returns fused BGR image or None.
    """
    if not crops:
        return None

    # Score and sort by QR-region sharpness
    scored = sorted(
        [(lap_var(c), c) for c in crops],
        key=lambda x: x[0], reverse=True,
    )
    best_crops = [c for _, c in scored[:top_k]]
    reference  = best_crops[0]

    # Resize all to reference size (bbox size may vary a pixel across frames)
    h, w = reference.shape[:2]
    aligned = [reference]
    for crop in best_crops[1:]:
        resized = cv2.resize(crop, (w, h))
        result  = ecc_align(reference, resized)
        aligned.append(result if result is not None else resized)

    # Weighted average (weight = Laplacian variance)
    weights = np.array([lap_var(a) for a in aligned], dtype=np.float64)
    weights /= weights.sum()

    fused = np.zeros_like(reference, dtype=np.float64)
    for weight, img in zip(weights, aligned):
        fused += weight * img.astype(np.float64)

    return np.clip(fused, 0, 255).astype(np.uint8)

# ── Decoding ───────────────────────────────────────────────────────────────────

def decode_all(img_bgr: np.ndarray, wechat) -> list[str]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    found: set[str] = set()

    for r in pyzbar_decode(gray):
        if r.data: found.add(r.data.decode())
    for r in zxingcpp.read_barcodes(gray):
        if r.text: found.add(r.text)
    data, _ = wechat.detectAndDecode(img_bgr)
    for d in data:
        if d: found.add(d)

    # 2× upscale for barcode detectors
    gray2 = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    for r in pyzbar_decode(gray2):
        if r.data: found.add(r.data.decode())
    for r in zxingcpp.read_barcodes(gray2):
        if r.text: found.add(r.text)

    return sorted(found)

# ── Visualisation ──────────────────────────────────────────────────────────────

def _labeled(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (5, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0),   4)
    cv2.putText(out, text, (5, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,80), 2)
    return out


def save_comparison(items: list[tuple[np.ndarray, str]], path: pathlib.Path):
    target_h = max(p.shape[0] for p, _ in items)
    panels = []
    for p, label in items:
        scale   = target_h / p.shape[0]
        resized = cv2.resize(p, (int(p.shape[1] * scale), target_h))
        panels.append(_labeled(resized, label))
    cv2.imwrite(str(path), np.hstack(panels))

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",  type=int,  default=0,     help="Rows per dataset (0=all)")
    ap.add_argument("--win",    type=int,  default=25,    help="±N frames to collect")
    ap.add_argument("--topk",   type=int,  default=7,     help="Top-K sharpest for fusion")
    ap.add_argument("--esrgan", action="store_true",      help="Apply ESRGAN after fusion")
    args = ap.parse_args()

    out_dir   = pathlib.Path("results/temporal")
    crops_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(exist_ok=True)

    print("=== QR Temporal Fusion pipeline ===\n")
    print(f"Окно: ±{args.win} кадров  |  топ-{args.topk} для fusion")

    print("Загружаем WeChat...")
    wechat = load_wechat()

    upsampler = None
    if args.esrgan:
        print("Загружаем Real-ESRGAN...")
        try:
            upsampler = load_esrgan()
            print("  Real-ESRGAN: OK")
        except Exception as e:
            print(f"  Real-ESRGAN: ОШИБКА — {e}")

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
            print(f"  [{pos:3d}] ts={ts:7.0f}ms  gt={gt_qr[:24] or '—'}", end="  ")

            # Collect frames
            all_frames = collect_frames(str(video_path), ts, args.win)
            if not all_frames:
                print("→ нет кадров")
                continue

            # Cut bbox crops from each frame
            all_crops = []
            for f in all_frames:
                c = cut_crop(f, row)
                if c is not None:
                    all_crops.append(c)
            if not all_crops:
                print("→ нет кропов")
                continue

            # Cut QR sub-crops for selection/scoring
            qr_subs = [cut_qr_subcrop(c) for c in all_crops]
            valid   = [(sub, crop) for sub, crop in zip(qr_subs, all_crops)
                       if sub is not None]
            if not valid:
                print("→ QR sub-crop пустой")
                continue

            # Single-frame baseline: sharpest crop
            best_sub = sorted(valid, key=lambda x: lap_var(x[0]), reverse=True)[0][0]
            single_decoded = decode_all(best_sub, wechat)

            # Temporal fusion on QR sub-crops
            fused_qr = temporal_fuse([s for s, _ in valid], args.topk)
            fused_decoded = decode_all(fused_qr, wechat) if fused_qr is not None else []

            # Optional: ESRGAN on fused result
            esrgan_decoded = []
            fused_up       = None
            if upsampler is not None and fused_qr is not None:
                try:
                    fused_up, _ = upsampler.enhance(fused_qr, outscale=4)
                    esrgan_decoded = decode_all(fused_up, wechat)
                except Exception as e:
                    print(f"[ESRGAN: {e}] ", end="")

            any_decoded = single_decoded or fused_decoded or esrgan_decoded
            status = (
                "✅ ESRGAN+fused" if esrgan_decoded else
                "✅ Fused"        if fused_decoded  else
                "✅ Single"       if single_decoded  else
                "❌"
            )
            print(
                f"→ {status}  "
                f"frames={len(all_crops)}  "
                f"single={single_decoded}  "
                f"fused={fused_decoded}  "
                f"esrgan={esrgan_decoded}"
            )

            # Save comparison image
            items = [
                (best_sub, f"Single best ({best_sub.shape[1]}x{best_sub.shape[0]})"),
            ]
            if fused_qr is not None:
                items.append((fused_qr, f"Fused top-{args.topk} decoded={bool(fused_decoded)}"))
            if fused_up is not None:
                h0, w0 = (best_sub.shape[0], best_sub.shape[1])
                items.append((
                    cv2.resize(fused_up, (w0 * 2, h0 * 2)),
                    f"ESRGAN×4 decoded={bool(esrgan_decoded)}",
                ))
            save_comparison(items, crops_dir / f"{ds_name}_{pos:03d}.jpg")

            rows.append({
                "dataset":        ds_name,
                "row_idx":        pos,
                "frame_ts_ms":    ts,
                "n_frames_used":  len(valid),
                "gt_qr":          gt_qr,
                "single_decoded": "|".join(single_decoded),
                "fused_decoded":  "|".join(fused_decoded),
                "esrgan_decoded": "|".join(esrgan_decoded),
                "single_ok":      bool(single_decoded),
                "fused_ok":       bool(fused_decoded),
                "esrgan_ok":      bool(esrgan_decoded),
                "gt_match":       bool(
                    any_decoded and gt_qr and any(gt_qr in d for d in any_decoded)
                ),
            })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_dir / "temporal_results.csv", index=False)

    total     = len(result_df)
    def pct(col): return f"{int(result_df[col].sum())}/{total} ({100*result_df[col].mean():.1f}%)"

    print(f"\n{'='*60}")
    print(f"Итого: {total} ценников")
    print(f"  Single-frame best:  {pct('single_ok')}")
    print(f"  Temporal fusion:    {pct('fused_ok')}")
    if args.esrgan:
        print(f"  Fusion + ESRGAN:    {pct('esrgan_ok')}")
    print(f"  Совпало с GT:       {pct('gt_match')}")
    print(f"\nРезультаты:  {out_dir}/temporal_results.csv")
    print(f"Кропы:       {crops_dir}/")


if __name__ == "__main__":
    main()
