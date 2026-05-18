#!/usr/bin/env bash
# train/train_yolo2.sh — Stage 2: inside price-tag elements detector
#
# Usage:
#   bash train/train_yolo2.sh [path/to/data.yaml]
#
# May require Stage 1 weights at models/price_tag_yolo.pt (or set YOLO2_BASE_MODEL).
#
# TODO(коллега):
#   1. Подтвердить датасет: runs/datasets/lenta_inside_yolo/data.yaml
#      (собирается через: just crop-for-stage2 → разметка в CVAT → just prepare-cvat)
#   2. Стартовать с Stage 1 весов (трансфер) или с pretrained yolo26n.pt?
#      Сейчас default = Stage 1 веса (трансфер обычно лучше для похожего домена)
#   3. Классы внутри ценника — что именно детектируем? (barcode, qr, price_zone, ...)

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA_YAML="${1:-$REPO/runs/datasets/lenta_inside_yolo/data.yaml}"
RUNS_DIR="$REPO/runs/detect"

# Default: start from Stage 1 weights (transfer learning)
# Override: YOLO2_BASE_MODEL=yolo11n.pt bash train/train_yolo2.sh
MODEL="${YOLO2_BASE_MODEL:-$REPO/models/price_tag_yolo.pt}"

if [[ ! -f "$DATA_YAML" ]]; then
    echo "Error: data.yaml not found: $DATA_YAML"
    echo "Prepare inside dataset first:"
    echo "  just crop-for-stage2"
    echo "  # annotate crops in CVAT"
    echo "  just prepare-cvat --source <cvat_export> --out-dir runs/datasets/lenta_inside_yolo"
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Warning: base model not found: $MODEL"
    echo "Falling back to yolo26n.pt (pretrained)"
    MODEL="yolo26n.pt"
fi

echo "[train] Stage 2 inside elements detector"
echo "  data: $DATA_YAML"
echo "  model: $MODEL"
echo "  runs: $RUNS_DIR"

cd "$REPO"
uv run yolo detect train \
    model="$MODEL" \
    data="$DATA_YAML" \
    epochs=200 \
    imgsz=960 \
    batch=4 \
    device=0 \
    project="$RUNS_DIR" \
    name=inside_price_tag_yolo \
    patience=40 \
    cos_lr=True \
    close_mosaic=20 \
    hsv_h=0.010 \
    hsv_s=0.35 \
    hsv_v=0.25 \
    degrees=3.0 \
    translate=0.05 \
    scale=0.25 \
    shear=1.0 \
    perspective=0.0002 \
    fliplr=0.0 \
    mosaic=0.25 \
    mixup=0.0 \
    copy_paste=0.0 \
    workers=2 \
    seed=42 \
    cache=True \
    plots=True

echo ""
echo "[done] Best weights → $RUNS_DIR/inside_price_tag_yolo/weights/best.pt"
echo "       Run: just save-yolo2   to copy to models/inside_price_tag_yolo.pt"
