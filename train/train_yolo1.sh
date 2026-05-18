#!/usr/bin/env bash
# train/train_yolo1.sh — Stage 1: price tag detector training
#
# Usage:
#   bash train/train_yolo1.sh [path/to/data.yaml]
#
# Defaults to runs/datasets/lenta_yolo/data.yaml if not specified.
# Override base model via: YOLO1_BASE_MODEL=yolo11n.pt bash train/train_yolo1.sh
#

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA_YAML="${1:-$REPO/runs/datasets/lenta_yolo/data.yaml}"
RUNS_DIR="$REPO/runs/detect"

MODEL="${YOLO1_BASE_MODEL:-yolo26n.pt}"

if [[ ! -f "$DATA_YAML" ]]; then
    echo "Error: data.yaml not found: $DATA_YAML"
    echo "Build dataset first: just dataset-build"
    exit 1
fi

echo "[train] Stage 1 price-tag detector"
echo "  data: $DATA_YAML"
echo "  model: $MODEL"
echo "  runs: $RUNS_DIR"

cd "$REPO"
uv run yolo detect train \
    model="$MODEL" \
    data="$DATA_YAML" \
    epochs=150 \
    imgsz=1280 \
    batch=4 \
    device=0 \
    cache=disk \
    project="$RUNS_DIR" \
    name=price_tag_yolo \
    patience=30 \
    cos_lr=True \
    close_mosaic=15 \
    hsv_h=0.015 \
    hsv_s=0.50 \
    hsv_v=0.30 \
    degrees=8.0 \
    translate=0.08 \
    scale=0.45 \
    shear=2.0 \
    perspective=0.0008 \
    fliplr=0.0 \
    mosaic=0.55 \
    mixup=0.05 \
    copy_paste=0.0 \
    workers=2 \
    seed=42

echo ""
echo "[done] Best weights → $RUNS_DIR/price_tag_yolo/weights/best.pt"
echo "       Run: just save-yolo1   to copy to models/price_tag_yolo.pt"
