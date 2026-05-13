# ML service

ML-часть отвечает за путь `video -> detections -> QR/OCR -> field voting -> CSV`.
Решение работает локально, без cloud OCR/CV API. Ручная разметка не используется:
YOLO-датасет собирается автоматически из public CSV/video, которые лежат в `data`.

## Структура

```text
ml/
  main.py              # FastAPI: загрузка/путь к видео, eval, сборка датасета, train
  pipeline.py          # основной пайплайн: видео -> CSV
  candidates.py        # YOLO + tiled YOLO + QR-seed + color/geometry fallback
  rail_roi.py          # поиск полосы полочной рейки и ограничение зоны детекции
  qr_tools.py          # zxing-cpp / pyzbar / OpenCV QR decoding, multi-scale early exit
  text_reader.py       # PaddleOCR + Tesseract ensemble
  field_extractor.py   # парсинг полей ценника из OCR/QR
  field_voting.py      # голосование по полям между кадрами и источниками
  validators.py        # нормализация/валидация цен, barcode, SKU, дат, скидок, символов
  crop_bank.py         # оценка crop: sharpness/area/conf/QR + pHash dedup
  evidence_fusion.py   # объединение одного ценника между кадрами
  field_derivation.py  # опциональный вывод QR-полей из OCR-полей
  schema.py            # фиксированная CSV-схема задания
  training.py          # сборка YOLO/tiled YOLO датасета и запуск обучения
  media.py             # видео, кадры, bbox, crop enhancement
  data/                # public-разметка и локальные видео для инференса
  runs/                # результаты запусков, CSV, debug JSON, датасеты
```

Аналоги отдельных `scripts/*.py` сделаны как API endpoints:

```text
POST /predict/path       # inference по локальному пути к видео
POST /predict/video      # upload mp4 -> CSV
POST /evaluate/public    # локальная proxy-оценка на public CSV/video
POST /dataset/yolo       # сборка YOLO или tiled YOLO датасета
POST /train/yolo         # обучение YOLO через ultralytics
```

## Установка

Команды удобнее запускать из `packages/ml`:

```powershell
cd packages\ml
uv sync --extra quality
```

`quality` extra ставит Python-зависимости для OCR/QR:

```text
PaddleOCR + paddlepaddle
pytesseract
zxing-cpp
pyzbar
```

Важно: `pytesseract` это Python-обертка. Для Tesseract нужен системный бинарник
и языки `rus`/`eng`. Проверка:

```powershell
tesseract --list-langs
```

Проверка импортов:

```powershell
uv run python -c "import paddleocr, paddle, pytesseract, zxingcpp; import pyzbar.pyzbar; print('quality deps ok')"
```

## Запуск сервиса

```powershell
uv run ml-service
```

После запуска:

```text
http://localhost:8000/docs
```

Быстрая проверка без сервера:

```powershell
uv run python -c "from ml.main import app; print(app.title)"
```

## Где должны лежать данные

Public-разметка ожидается здесь:

```text
packages/ml/src/ml/data/
  25_12-20/
    *.mp4
    *.csv
  26_12-20/
    *.mp4
    *.csv
  43_15/
    *.mp4
    *.csv
```

Названия файлов условные: пайплайн ищет в каждой папке первый CSV и подходящее MP4.
Если stem не совпадает, берется самое крупное видео в папке.

Видео без разметки для обычного инференса удобно класть сюда:

```text
packages/ml/src/ml/data/Unlabeled/
```

Для `POST /predict/path` видео может лежать где угодно, если передать полный путь.

## Где должны лежать модели

Рекомендуемое место для обученного YOLO-детектора:

```text
models/price_tag_yolo.pt
```

Пайплайн ищет веса автоматически в:

```text
<текущая рабочая папка>/models/price_tag_yolo.pt
<корень репозитория>/models/price_tag_yolo.pt
```

Надежнее всего явно передавать путь:

```json
{
  "yolo_weights": "models/price_tag_yolo.pt"
}
```

Или через env:

```powershell
$env:ML_YOLO_WEIGHTS="models/price_tag_yolo.pt"
uv run ml-service
```

Веса и экспортированные модели не кладем в git: `.pt`, `.onnx`, `.rknn`, `.engine`
игнорируются.

## Inference

Пример `POST /predict/path`:

```json
{
  "video_path": "packages/ml/src/ml/data/Unlabeled/demo.mp4",
  "mode": "accurate",
  "sample_fps": 2.0,
  "max_frames": 0,
  "yolo_weights": "models/price_tag_yolo.pt",
  "enable_ocr": true,
  "enable_qr": true,
  "tiled_yolo": true,
  "rail_roi_enabled": true,
  "crop_phash_dedup": true,
  "derive_qr_fields_when_missing": false,
  "save_crops": false
}
```

Результаты сохраняются в:

```text
packages/ml/src/ml/runs/<run_id>/
  *_recognized.csv
  *_debug.json
```

`debug.json` содержит настройки, rail ROI, bbox, качество crop, QR/OCR-статус и
сводку по трекам.

## Режимы

```text
fast      быстрый smoke/inference: меньше FPS, OCR по умолчанию отключен
cpu_safe  безопасный локальный режим без обязательных YOLO-весов
accurate  целевой режим: YOLO + tiled YOLO + rail ROI + QR + OCR + fusion
quality   alias accurate
```

Если весов нет, будут работать QR-seed и color/geometry fallback. Это полезно для
smoke-теста, но нормальное качество ожидается после обучения YOLO.

## Что уже добавлено из сильных идей

```text
tiled YOLO inference      ценники крупнее для модели на 4K/широких кадрах
tiled YOLO dataset        train labels пересчитываются из full-frame bbox в tile bbox
rail ROI                  поиск горизонтальной полочной рейки и детекция внутри ROI
multi-scale QR            zxing-cpp / pyzbar / OpenCV, остановка после успешного decode
PaddleOCR + Tesseract     ensemble OCR с graceful fallback
crop pHash dedup          не гоняем OCR по одинаковым crop
field voting              выбираем лучшее значение поля между кадрами/источниками
validators                отбрасываем шумные цены, barcode, SKU, даты, скидки
debug tracks              видно, почему строка CSV получилась именно такой
```

## Сборка YOLO-датасета

Обычный full-frame датасет:

```json
{
  "data_dir": "packages/ml/src/ml/data",
  "output_dir": "packages/ml/src/ml/runs/lenta_yolo_dataset",
  "val_fraction": 0.2,
  "seed": 42,
  "tiled": false
}
```

Рекомендуемый tiled dataset для 4K/мелких ценников:

```json
{
  "data_dir": "packages/ml/src/ml/data",
  "output_dir": "packages/ml/src/ml/runs/lenta_yolo_tiled_dataset",
  "val_fraction": 0.2,
  "seed": 42,
  "tiled": true,
  "tile_size": 640,
  "tile_stride": 512,
  "min_box_visibility": 0.25
}
```

На выходе:

```text
images/train
images/val
labels/train
labels/val
data.yaml
```

Логика tiled dataset: кадр берется по `frame_timestamp` из CSV, режется на тайлы,
bbox клипается границами тайла и переводится в YOLO-формат. Тайлы без ценников не
сохраняются.

## Обучение YOLO

Через API `POST /train/yolo`:

```json
{
  "data_yaml": "packages/ml/src/ml/runs/lenta_yolo_tiled_dataset/data.yaml",
  "model": "yolo11n.pt",
  "epochs": 150,
  "imgsz": 640,
  "batch": 16,
  "device": "0",
  "project": "F:/lenta_price_vision/packages/ml/src/ml/runs/detect",
  "name": "price_tag_yolo"
}
```

Для CPU вместо GPU:

```json
{
  "device": "cpu",
  "batch": 4
}
```

Напрямую через ultralytics CLI:

```powershell
uv run yolo detect train model=yolo11n.pt data=F:/lenta_price_vision/packages/ml/src/ml/runs/lenta_yolo_tiled_dataset/data.yaml imgsz=640 epochs=120 batch=16 device=0 cache=disk project=F:/lenta_price_vision/packages/ml/src/ml/runs/detect name=price_tag_yolo11n_tiled_640
```

После обучения лучший вес обычно здесь:

```text
F:/lenta_price_vision/packages/ml/src/ml/runs/detect/price_tag_yolo11n_tiled_640/weights/best.pt
```

Скопировать в стандартное место:

```powershell
New-Item -ItemType Directory -Force F:\lenta_price_vision\models
Copy-Item F:\lenta_price_vision\packages\ml\src\ml\runs\detect\price_tag_yolo11n_tiled_640\weights\best.pt F:\lenta_price_vision\models\price_tag_yolo.pt
```

## Оценка на public-разметке

`POST /evaluate/public`:

```json
{
  "data_dir": "packages/ml/src/ml/data",
  "mode": "accurate",
  "sample_fps": 1.0,
  "max_frames": 0,
  "yolo_weights": "models/price_tag_yolo.pt",
  "tiled_yolo": true,
  "rail_roi_enabled": true
}
```

Отчет появится в:

```text
runs/<run_id>/evaluation_report.json
```

## CSV-схема

Итоговый CSV всегда пишется в фиксированном порядке:

```text
filename, product_name, price_default, price_card, price_discount,
barcode, discount_amount, id_sku, print_datetime, code, additional_info,
color, special_symbols, frame_timestamp, x_min, y_min, x_max, y_max,
qr_code_barcode, price1_qr, price2_qr, price3_qr, price4_qr,
wholesale_level_1_count, wholesale_level_1_price,
wholesale_level_2_count, wholesale_level_2_price,
action_price_qr, action_code_qr
```

Legacy-колонка public-разметки:

```text
wholesale_level_1_coun -> wholesale_level_1_count
```

Правила заполнения:

```text
нет      параметр точно отсутствует
пусто    параметр есть, но не распознан
```

## Практичный порядок работы

```text
1. Положить public mp4/csv в packages/ml/src/ml/data/<sequence>/
2. Собрать tiled YOLO dataset через POST /dataset/yolo
3. Обучить YOLO через POST /train/yolo или uv run yolo ...
4. Положить best.pt в models/price_tag_yolo.pt
5. Прогнать POST /evaluate/public
6. Прогнать POST /predict/path на Unlabeled видео
7. Смотреть *_recognized.csv и *_debug.json
```
