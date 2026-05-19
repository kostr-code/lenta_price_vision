# sol_main — Production Pipeline

Путь: `video + labeled CSV -> VLM/OCR -> 29-field CSV`.

Решение работает локально, без cloud API. Две независимые части:
- **Inference pipeline** (`main.py` + `pipeline/`) — распознавание ценников через Qwen2.5-VL + PaddleOCR
- **YOLO training pipeline** (`ml/` + `scripts/` + `train/`) — подготовка данных и обучение детекторов

---

## Структура

```text
sol_main/
  main.py              # CLI: video + labeled CSV -> recognized CSV
  pipeline/
    video.py           # VideoCapture, find_best_frame, cut_crop, rotate CCW
    parsers.py         # parse_fields, merge_field_values, OUTPUT_COLUMNS (29 полей)
    ocr.py             # PaddleOCR zoned, enhance_crop, OCRLine
    qr.py              # pyzbar / zxingcpp QR + barcode decode
    quality.py         # estimate_crop_quality: h264_artifact_score + FFT hf_ratio
    vlm.py             # Qwen2.5-VL-7B: load_vlm, extract_fields_vlm
  ml/
    training.py        # build_yolo_dataset: BBox, iter_tiles, full/tiled frames
  scripts/
    build_dataset.py   # CLI: labeled CSV/video -> YOLO dataset + template propagation
    preview_dataset.py # HTML-галерея с bbox поверх тайлов (проверка разметки)
    crop_detections.py # Прогнать YOLO -> сохранить кропы ценников (для Stage 2)
    dedupe_predictions.py     # IoU NMS деdup predictions перед импортом в CVAT
    prepare_cvat_dataset.py   # CVAT YOLO export -> финальный YOLO датасет
  train/
    train_yolo1.sh     # Stage 1: детектор ценников (price_tag)
    train_yolo2.sh     # Stage 2: внутренние элементы (barcode, qr, price zones...)
  justfile             # рецепты для всех этапов
  data/
    Данные/            # labeled видео + CSV (не в git)
    testdata/          # небольшие тестовые кропы
  runs/                # датасеты, checkpoints, результаты (не в git)
  models/              # веса YOLO (не в git)
```

---

## Установка

```bash
cd sol_main
uv sync
```

Для OCR и QR (опционально):

```bash
uv add paddleocr paddlepaddle pyzbar zxing-cpp
```

Для YOLO training:

```bash
uv add ultralytics
```

---

## Inference: распознавание ценников

```bash
uv run python main.py \
    --video data/Данные/43_15/43_15.mp4 \
    --csv   data/Данные/43_15/43_15.csv \
    --out   runs/results/43_15_recognized.csv
```

Опции:

```text
--model      Qwen/Qwen2.5-VL-7B-Instruct   HuggingFace model ID
--ocr                                        включить PaddleOCR fallback
--quality-thr  0.2                           порог качества кропа (warn below)
--scan         20                            +-N кадров для поиска резкого
```

Pipeline внутри:

```text
load_df (CSV) -> find_best_frame (+-scan) -> cut_crop_from_row
-> estimate_crop_quality (h264 + FFT)
-> extract_fields_vlm (Qwen2.5-VL -> JSON -> 12 полей)
-> [опционально] ocr_zoned (PaddleOCR -> parse_fields -> остаток полей)
-> merge + normalize -> save CSV (29 полей + _quality)
```

Метрики качества кропа (`pipeline/quality.py`):

```text
h264_artifact_score   ratio DCT-boundary / interior gradients; >1.5 -> блочные артефакты
hf_ratio              FFT HF energy fraction; <0.3 -> слишком размыто для OCR
estimate_crop_quality  composite [0,1] = 0.4*lap_norm + 0.4*hf_norm - 0.2*artifact_penalty
```

---

## YOLO Stage 1: детектор ценников

### Шаг 1. Собрать датасет

```bash
# Тайловый датасет (рекомендуется для 4K видео)
just dataset-build

# Или вручную:
uv run python scripts/build_dataset.py \
    --data-dir data/Данные \
    --out-dir  runs/datasets/lenta_yolo_tiled \
    --tiled \
    --propagate 8 \
    --val-ratio 0.2
```

Параметры `build_dataset.py`:

```text
--data-dir          Папка с подпапками, где лежат .mp4 + .csv
--out-dir           Куда писать датасет
--tiled             Режим 640px тайлов (иначе полные кадры)
--tile-size    640  Размер тайла
--tile-stride  512  Шаг между тайлами (перекрытие = tile_size - stride)
--min-visibility 0.25  Мин. доля bbox в тайле чтобы взять label
--propagate    8    Кол-во соседних кадров для template propagation (0=выкл)
--match-threshold 0.42  cv2.matchTemplate score threshold
                        0.42 = лояльный; 0.72 = строгий (если bbox плывут)
--search-pad   80   Поиск в bbox +- N пикселей
--val-ratio    0.2  Доля val
```

Template propagation: для каждого кадра с bbox ищет те же ценники в
соседних кадрах через `cv2.matchTemplate`. Умножает тренировочные данные
без ручной разметки. Используется только для train split.

Результат:

```text
runs/datasets/lenta_yolo_tiled/
  data.yaml
  images/train/   *.jpg
  images/val/     *.jpg
  labels/train/   *.txt  (YOLO format: class cx cy w h, normalized)
  labels/val/     *.txt
```

### Шаг 2. Проверить разметку глазами

```bash
just dataset-preview  # генерирует runs/datasets/lenta_yolo_tiled/preview_train/index.html

# Или вручную:
uv run python scripts/preview_dataset.py \
    --dataset runs/datasets/lenta_yolo_tiled/data.yaml \
    --split train --limit 300 \
    --out-dir runs/preview_train
```

Открыть `runs/preview_train/index.html`. Если bbox плывут -> повысить
`--match-threshold` до `0.72` или убрать `--propagate 0`.

### Шаг 3. Обучить

```bash
just train-yolo1

# Или вручную:
bash train/train_yolo1.sh runs/datasets/lenta_yolo_tiled/data.yaml
```

Параметры тренировки Stage 1 (`train/train_yolo1.sh`):

```text
model     yolo11n.pt     базовая модель (см. TODO ниже)
epochs    150
imgsz     1280
batch     4
device    0              GPU
patience  30
```

Аугментации (для ценников с горизонтальным текстом): `fliplr=0.0`, умеренный `hsv`, слабые геометрические (`degrees=8`, `perspective=0.0008`).

Результат:

```text
runs/detect/price_tag_yolo/weights/best.pt
```

### Шаг 4. Сохранить веса

```bash
just save-yolo1
# -> models/price_tag_yolo.pt
```

---

## YOLO Stage 2: внутренние элементы ценника

Stage 2 детектирует sub-regions внутри кропа ценника (barcode strip, QR zone,
price zones и т.д.). Требует Stage 1 весов и разметки внутренностей.

### Шаг 1. Извлечь кропы ценников из датасета

```bash
just crop-for-stage2

# Или вручную:
uv run python scripts/crop_detections.py \
    --dataset  runs/datasets/lenta_yolo_tiled \
    --weights  models/price_tag_yolo.pt \
    --out-subdir crops_for_stage2 \
    --conf 0.25 --imgsz 1280 --device 0
```

Результат:

```text
runs/datasets/lenta_yolo_tiled/crops_for_stage2/
  train/   *.jpg   (кропы ценников с train split)
  val/     *.jpg
  manifest.csv     (filename, split, conf, x1, y1, x2, y2)
```

### Шаг 2. Разметить в CVAT

Импортировать `crops_for_stage2/train/` в CVAT, разметить классы внутренних
элементов, экспортировать в формате YOLO.

### Шаг 3. Деdup предсказаний (если использовался inside-псевдолейблер)

```bash
uv run python scripts/dedupe_predictions.py \
    --source  runs/inside_pseudo_raw \
    --out-dir runs/inside_pseudo_dedup \
    --iou-threshold 0.75 --class-aware
```

### Шаг 4. Подготовить датасет из CVAT export

```bash
just prepare-cvat \
    source=runs/cvat_inside_export \
    out=runs/datasets/lenta_inside_yolo

# Или вручную (стандартный CVAT YOLO layout):
uv run python scripts/prepare_cvat_dataset.py \
    --source path/to/cvat_export \
    --out-dir runs/datasets/lenta_inside_yolo \
    --val-ratio 0.2
```

Для нестандартного layout где картинки лежат отдельно:

```bash
uv run python scripts/prepare_cvat_dataset.py \
    --source   path/to/cvat_export \
    --image-root path/to/images \
    --out-dir  runs/datasets/lenta_inside_yolo
```

### Шаг 5. Обучить Stage 2

```bash
just train-yolo2

# Или вручную:
bash train/train_yolo2.sh runs/datasets/lenta_inside_yolo/data.yaml
```

Параметры (`train/train_yolo2.sh`):

```text
model     models/price_tag_yolo.pt   стартуем с Stage 1 весов (transfer)
epochs    200
imgsz     960
batch     4
```

### Шаг 6. Сохранить веса

```bash
just save-yolo2
# -> models/inside_price_tag_yolo.pt
```

---

## Justfile — все рецепты

```text
just dataset-build        Собрать tiled датасет (prop=8)
just dataset-build-full   То же, но полные кадры без тайлов
just dataset-preview      HTML-галерея разметки train split
just train-yolo1          Обучить Stage 1
just save-yolo1           Скопировать best.pt -> models/price_tag_yolo.pt
just crop-for-stage2      Кропы ценников для разметки Stage 2
just prepare-cvat         CVAT export -> YOLO датасет
just train-yolo2          Обучить Stage 2
just save-yolo2           Скопировать best.pt -> models/inside_price_tag_yolo.pt
just yolo1-full           Весь Stage 1 одной командой (build->preview->train)
just qwen-debug [path]    Прогнать Qwen-VL на картинках для отладки
```

Параметры рецептов переопределяются на месте:

```bash
just dataset-build data=data/custom_data out=runs/my_dataset prop=4
just train-yolo1 dataset=runs/my_dataset
```

---

## Где лежат данные

```text
data/Данные/
  <sequence_name>/
    *.mp4   видео (90° CCW rotated, 4K)
    *.csv   labeled bbox + timestamps
  Unlabeled/
    *.mp4   видео без разметки
```

Видео не кладём в git (`.gitignore`). Пайплайн ищет в каждой подпапке
первый CSV и подходящее MP4 (по stem или по размеру файла).

---

## Где лежат модели

```text
models/
  price_tag_yolo.pt        Stage 1: детектор ценников
  inside_price_tag_yolo.pt Stage 2: внутренние элементы (после обучения)
```

Веса не кладём в git (`.gitignore`). Путь к модели можно задать через
env или явно в скриптах.

---

## Формат YOLO label файлов

```text
class_id cx cy w h       (все значения normalized 0–1 относительно тайла/кадра)
0 0.5234 0.4500 0.1456 0.3400
```

Stage 1: один класс `0: price_tag`.
Stage 2: несколько классов — зависит от разметки коллеги.

---

## CSV-схема выхода (29 полей)

```text
filename, product_name, price_default, price_card, price_discount,
barcode, discount_amount, id_sku, print_datetime, code, additional_info,
color, special_symbols, frame_timestamp, x_min, y_min, x_max, y_max,
qr_code_barcode, price1_qr, price2_qr, price3_qr, price4_qr,
wholesale_level_1_count, wholesale_level_1_price,
wholesale_level_2_count, wholesale_level_2_price,
action_price_qr, action_code_qr
```

Соглашение: `нет` = поле точно отсутствует, пусто = не распознано.

---

## TODO для коллеги

Следующие вещи не реализованы или требуют уточнения:

```text

[ ] Stage 2 классы внутренних элементов
    train/train_yolo2.sh готов, но нужен датасет.
    Какие классы детектируем внутри ценника?
    (barcode_strip, qr_zone, price_card_zone, price_default_zone, ...)

[ ] ByteTrack трекинг для непрерывного видео
    43_15 и другие датасеты с остановками робота работают без трекинга.
    Для видео с движением нужен ByteTrack трек + temporal field voting.
    Есть в sol1/packages/ml/evidence_fusion.py — решение коллеги.

[ ] pHash dedup кропов
    В ml_readme (старом) упоминался crop_phash_dedup чтобы не гонять OCR
    по одинаковым кропам. Не перенесено.

[ ] Rail ROI
    Поиск полочной рейки для ограничения зоны детекции.
    Есть в sol1/packages/ml/rail_roi.py — решение коллеги.
```
