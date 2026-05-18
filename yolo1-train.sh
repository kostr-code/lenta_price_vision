uv run yolo detect train \
    model=yolo26n.pt \
    data="<repo>/packages/ml/src/ml/runs/datasets/lenta_yolo_full_prop10/data.yaml" \
    epochs=150 \
    imgsz=1280 \
    batch=4 \
    device=0 \
    project="<repo>/packages/ml/runs/lenta" \
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
    seed=4 \

# Доп команды для разметки данных

# для сборки датасета:


cd <repo>/packages/ml
uv run python -m ml.training \
  --data-dir <repo>/packages/ml/src/ml/data \
  --out-dir <repo>/packages/ml/src/ml/runs/datasets/lenta_yolo_full_prop10 \
  --propagate 10 `
  --val-ratio 0.2 `
  --match-threshold 0.42 `
  --search-pad 80 `
  --split-mode frame_id_mod

# для просмотра датасета

uv run ml-view-annotations \
  --dataset <repo>/packages/ml/src/ml/runs/datasets/lenta_yolo_full_prop10/data.yaml \
  --split train \
  --limit 300 \
  --out-dir <repo>/packages/ml/src/ml/runs/annotation_preview_full_prop10_train