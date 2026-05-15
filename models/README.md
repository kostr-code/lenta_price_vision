# Модели

## training

```shell
yolo detect train \
    model=yolo26n.pt \
    data=$LENTA_REPO/packages/ml/src/ml/runs/datasets/lenta_yolo_full_prop10/data.yaml \
    epochs=150 \
    imgsz=1280 \
    batch=4 \
    device=0 \
    project=$LENTA_REPO/packages/ml/runs/lenta \
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
```

## inference

требование: `ultralytics>=8.3.0`

```shell
yolo detect track \
    model=$LENTA_REPO/models/best.pt \
    source=$LENTA_REPO/packages/ml/src/ml/data/Unlabeled/26_2-10.mp4 \
    imgsz=1280 \
    conf=0.15 \
    iou=0.5 \
    device=0 \
    tracker=$LENTA_REPO/packages/ml/src/ml/configs/bytetrack_price.yaml \
    save=True \
    save_txt=True \
    save_conf=True \
    project=$LENTA_REPO/packages/ml/runs/track \ 
    name=video_26_2_10_bytetrack_custom
```

```shell
yolo detect track model=$LENTA_REPO/models/best.pt source=$LENTA_REPO/packages/ml/src/ml/data/Unlabeled/26_2-10.mp4 imgsz=1280 conf=0.15 iou=0.5 device=0 tracker=$LENTA_REPO/packages/ml/src/ml/configs/bytetrack_price.yaml save=True save_txt=True save_conf=True project=$LENTA_REPO/packages/ml/runs/track name=video_26_2_10_bytetrack_custom
```