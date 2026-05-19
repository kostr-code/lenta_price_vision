# How to YOLO

Короткая bash-шпаргалка по циклу подготовки YOLO-модели из ветки коллеги.
Команды ниже рассчитаны на запуск из ML-пакета проекта:

```bash
cd <repo>/packages/ml
uv sync --extra quality
```

Плейсхолдеры вида `<path_...>` нужно заменить на реальные локальные пути.

## 1. Подготовить YOLO-датасет из public CSV/video

Вход: папка с подпапками public-разметки, где рядом лежат `.mp4` и `.csv`.

```bash
uv run python -m ml.training \
  --data-dir <path_public_video_csv_data> \
  --out-dir <path_1_результат_разметки_датасета> \
  --propagate 8 \
  --val-ratio 0.2 \
  --match-threshold 0.42 \
  --search-pad 80 \
  --split-mode frame_id_mod
```

Результат:

```text
<path_1_результат_разметки_датасета>/data.yaml
<path_1_результат_разметки_датасета>/images/train
<path_1_результат_разметки_датасета>/images/val
<path_1_результат_разметки_датасета>/labels/train
<path_1_результат_разметки_датасета>/labels/val
```

Если bbox после просмотра плывут, пробовать строже:

```bash
uv run python -m ml.training \
  --data-dir <path_public_video_csv_data> \
  --out-dir <path_1_результат_разметки_датасета> \
  --propagate 0 \
  --val-ratio 0.2 \
  --match-threshold 0.72 \
  --search-pad 80 \
  --split-mode frame_id_mod
```

## 2. Проверить разметку глазами

```bash
uv run ml-view-annotations \
  --dataset <path_1_результат_разметки_датасета>/data.yaml \
  --split train \
  --limit 300 \
  --out-dir <path_preview_train>
```

```bash
uv run ml-view-annotations \
  --dataset <path_1_результат_разметки_датасета>/data.yaml \
  --split val \
  --limit 300 \
  --out-dir <path_preview_val>
```

Открывать:

```text
<path_preview_train>/index.html
<path_preview_val>/index.html
```

## 3. Обучить первую YOLO-модель

Вариант из скрипта коллеги для детектора ценников:

```bash
uv run yolo detect train \
  model=<path_base_weights_or_yolo_model> \
  data=<path_1_результат_разметки_датасета>/data.yaml \
  epochs=150 \
  imgsz=1280 \
  batch=4 \
  device=0 \
  cache=disk \
  project=<path_yolo_runs> \
  name=price_tag_yolo_1 \
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

Результат:

```text
<path_yolo_runs>/price_tag_yolo_1/weights/best.pt
```

## 4. Сделать crops для псевдолейблинга

Прогоняем обученный детектор по датасету и сохраняем найденные ценники как отдельные картинки.

```bash
uv run python scripts/crop_yolo_dataset_price_tags.py \
  --dataset <path_1_результат_разметки_датасета> \
  --weights <path_yolo_runs>/price_tag_yolo_1/weights/best.pt \
  --out-subdir crops_best_pt \
  --conf 0.25 \
  --imgsz 1280 \
  --device 0 \
  --splits train val \
  --clear-output
```

Результат:

```text
<path_1_результат_разметки_датасета>/crops_best_pt/train
<path_1_результат_разметки_датасета>/crops_best_pt/val
<path_1_результат_разметки_датасета>/crops_best_pt/manifest.csv
```

Дальше эти crops можно разметить вручную или использовать как вход для следующего этапа псевдолейблинга.

## 5. Псевдолейблинг внутренностей ценника

Если уже есть модель для внутренних полей ценника, можно прогнать ее по crops и получить YOLO-labels.

```bash
uv run python scripts/analyze_inside_on_crops_rot90ccw.py \
  --source <path_1_результат_разметки_датасета>/crops_best_pt \
  --model <path_inside_yolo_weights_or_run> \
  --out-dir <path_2_псевдоразметка_inside_raw> \
  --conf 0.25 \
  --imgsz 960 \
  --device 0 \
  --splits train val \
  --clear-output
```

Убрать дублирующиеся предсказания и подготовить layout, удобный для CVAT/YOLO:

```bash
uv run python scripts/dedupe_inside_predictions_for_cvat.py \
  --source <path_2_псевдоразметка_inside_raw> \
  --out-dir <path_3_псевдоразметка_inside_dedup> \
  --iou-threshold 0.75 \
  --class-aware \
  --clear-output
```

Результат можно проверять/исправлять в CVAT, затем экспортировать как YOLO.

## 6. Подготовить YOLO-датасет из CVAT-экспорта

Для обычного CVAT YOLO export, где есть `obj_train_data`, `labels`, `obj.names`:

```bash
uv run python scripts/prepare_inside_yolo_dataset.py \
  --source <path_cvat_yolo_export> \
  --out-dir <path_4_inside_yolo_dataset> \
  --val-ratio 0.2 \
  --seed 42 \
  --clear-output
```

Для кейса коллеги с `60_inside_data`, где labels лежат отдельно, а картинки берутся из другого root:

```bash
uv run python scripts/prepare_inside_60_yolo_dataset.py \
  --source <path_60_inside_cvat_export> \
  --image-root <path_images_for_60_inside> \
  --out-dir <path_4_inside_yolo_dataset> \
  --clear-output
```

Результат:

```text
<path_4_inside_yolo_dataset>/data.yaml
```

## 7. Дообучить на новом датасете

Первое обучение inside-модели:

```bash
uv run yolo detect train \
  model=<path_yolo_runs>/price_tag_yolo_1/weights/best.pt \
  data=<path_4_inside_yolo_dataset>/data.yaml \
  epochs=200 \
  imgsz=960 \
  batch=4 \
  device=0 \
  project=<path_inside_yolo_runs> \
  name=inside_price_tag_yolo_1 \
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
```

Дообучение inside-модели на дополнительной разметке:

```bash
uv run yolo detect train \
  model=<path_inside_yolo_runs>/inside_price_tag_yolo_1/weights/best.pt \
  data=<path_4_inside_yolo_dataset>/data.yaml \
  epochs=250 \
  imgsz=960 \
  batch=4 \
  device=0 \
  project=<path_inside_yolo_runs> \
  name=inside_price_tag_yolo_2_finetune \
  patience=60 \
  cos_lr=True \
  close_mosaic=25 \
  lr0=0.003 \
  lrf=0.01 \
  hsv_h=0.010 \
  hsv_s=0.35 \
  hsv_v=0.25 \
  degrees=3.0 \
  translate=0.05 \
  scale=0.25 \
  shear=1.0 \
  perspective=0.0002 \
  fliplr=0.0 \
  mosaic=0.20 \
  mixup=0.0 \
  copy_paste=0.0 \
  workers=2 \
  seed=42 \
  cache=True \
  plots=True
```

## 8. Сохранить итоговые веса

Для пайплайна детекции ценников:

```bash
mkdir -p <repo>/models
cp <path_yolo_runs>/price_tag_yolo_1/weights/best.pt <repo>/models/price_tag_yolo.pt
```

Для inside-модели:

```bash
mkdir -p <repo>/models
cp <path_inside_yolo_runs>/inside_price_tag_yolo_2_finetune/weights/best.pt <repo>/models/inside_price_tag_yolo.pt
```
