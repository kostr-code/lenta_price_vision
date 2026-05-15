#!/usr/bin/env python3
"""
ADNet (EG-Restormer) QR-code deblurring on labeled price tag crops.

Downloads of pretrained weights (Google Drive, see adnet_repo/README.md):
  EG-Restormer (GoPro+QRData): https://drive.google.com/file/d/12NCPyL2lvga3z85WRjc-LzZb18qcUbgY/view
  Place at: adnet_repo/experiment/egrestormer/models/net_g_qrdata.pth

Run:
  uv run python adnet_test.py --weights adnet_repo/experiment/egrestormer/models/net_g_qrdata.pth
  uv run python adnet_test.py --weights <path> --limit 10
"""
import argparse
import pathlib
import sys
import warnings

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent / "adnet_repo"))

from qr_esrgan import (
    DATASETS,
    cut_crop,
    cut_qr_subcrop,
    decode_all,
    find_best_frame,
    load_df,
    load_wechat,
    save_comparison,
)

YAML_CFG = pathlib.Path(__file__).parent / "adnet_repo/options/train/train_egrestormer_qrdataset.yml"
FACTOR = 8  # EG-Restormer requires input dims divisible by 8


def load_egrestormer(weights_path: str):
    import torch
    import torch.nn as nn
    import yaml

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    if not YAML_CFG.exists():
        raise FileNotFoundError(f"ADNet yaml config not found: {YAML_CFG}")

    cfg = yaml.load(YAML_CFG.read_text(), Loader=Loader)
    cfg["network_g"].pop("type")

    from basicsr.archs.eg_restormer_arch import EGRestormer

    model = EGRestormer(**cfg["network_g"])
    ckpt = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(ckpt["params"])
    model.cuda()
    model = nn.DataParallel(model)
    model.eval()
    print(f"  EG-Restormer: loaded from {weights_path}")
    return model


def restore(model, img_bgr: np.ndarray) -> np.ndarray:
    """Run EG-Restormer on a BGR image, return BGR result."""
    import torch
    import torch.nn.functional as F

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(np.float32(img_rgb) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()

    h, w = t.shape[2], t.shape[3]
    H = ((h + FACTOR - 1) // FACTOR) * FACTOR
    W = ((w + FACTOR - 1) // FACTOR) * FACTOR
    padh, padw = H - h, W - w
    t = F.pad(t, (0, padw, 0, padh), "reflect")

    with torch.no_grad():
        out = model(t)

    out = out[:, :, :h, :w]
    out_np = torch.clamp(out, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
    return cv2.cvtColor((out_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to EG-Restormer .pth weights")
    ap.add_argument("--limit", type=int, default=0, help="Rows per dataset (0=all)")
    ap.add_argument("--scan", type=int, default=20, help="±N frames sharpness scan")
    args = ap.parse_args()

    if not pathlib.Path(args.weights).exists():
        print(f"ERROR: weights not found: {args.weights}")
        print("Download from Google Drive (see adnet_repo/README.md section 3.2)")
        sys.exit(1)

    out_dir = pathlib.Path("results/adnet")
    crops_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(exist_ok=True)

    print("=== ADNet (EG-Restormer) QR pipeline ===\n")
    print("Загружаем WeChat...")
    wechat = load_wechat()

    print("Загружаем EG-Restormer...")
    model = load_egrestormer(args.weights)

    rows = []

    for ds_name, cfg in DATASETS.items():
        csv_path = pathlib.Path(cfg["csv"])
        video_path = pathlib.Path(cfg["video"])
        if not csv_path.exists() or not video_path.exists():
            print(f"\nSKIP {ds_name}: файлы не найдены")
            continue

        df = load_df(str(csv_path))
        if args.limit:
            df = df.head(args.limit)
        print(f"\n=== {ds_name}: {len(df)} строк ===")

        for pos, (_, row) in enumerate(df.iterrows()):
            ts = float(row["frame_timestamp"])
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

            orig_decoded = decode_all(qr_sub, wechat)

            try:
                sharp = restore(model, qr_sub)
                adnet_decoded = decode_all(sharp, wechat)
            except Exception as e:
                print(f"[ADNet err: {e}] ", end="")
                sharp = None
                adnet_decoded = []

            status = (
                "✅ ADNet" if adnet_decoded else
                "✅ ORIG"  if orig_decoded  else
                "❌"
            )
            print(f"→ {status}  orig={orig_decoded}  adnet={adnet_decoded}")

            panels = [(qr_sub, f"Orig {qr_sub.shape[1]}x{qr_sub.shape[0]}")]
            if sharp is not None:
                panels.append((sharp, f"ADNet dec={bool(adnet_decoded)}"))
            save_comparison(panels, crops_dir / f"{ds_name}_{pos:03d}.jpg")

            rows.append({
                "dataset":        ds_name,
                "row_idx":        pos,
                "frame_ts_ms":    ts,
                "lap_var":        round(lap, 1),
                "gt_qr":          gt_qr,
                "orig_decoded":   "|".join(orig_decoded),
                "adnet_decoded":  "|".join(adnet_decoded),
                "orig_ok":        bool(orig_decoded),
                "adnet_ok":       bool(adnet_decoded),
                "gt_match":       bool(orig_decoded and gt_qr and gt_qr in orig_decoded)
                                  or bool(adnet_decoded and gt_qr and gt_qr in adnet_decoded),
            })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_dir / "adnet_results.csv", index=False)

    total = len(result_df)
    orig_ok = int(result_df["orig_ok"].sum()) if total else 0
    adnet_ok = int(result_df["adnet_ok"].sum()) if total else 0
    gt_match = int(result_df["gt_match"].sum()) if total else 0

    print(f"\n{'='*55}")
    print(f"Итого: {total} ценников")
    print(f"  Baseline:    {orig_ok}/{total}  ({100*orig_ok/max(total,1):.1f}%)")
    print(f"  ADNet:       {adnet_ok}/{total}  ({100*adnet_ok/max(total,1):.1f}%)")
    print(f"  Совпало GT:  {gt_match}/{total}  ({100*gt_match/max(total,1):.1f}%)")
    print(f"\nРезультаты: {out_dir}/adnet_results.csv")
    print(f"Кропы:      {crops_dir}/")


if __name__ == "__main__":
    main()
