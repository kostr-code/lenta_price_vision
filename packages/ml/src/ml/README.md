# ML service

ML-часть отвечает за обработку видео с полок: находит ценники, пытается извлечь QR/OCR-поля, объединяет повторы по кадрам и сохраняет итоговый CSV в фиксированной схеме хакатона.

## Структура

```text
ml/
  main.py              # FastAPI API: загрузка/путь к видео -> скачать CSV
  pipeline.py          # основной пайплайн: видео -> детекция -> QR/OCR -> фьюжн -> CSV
  candidates.py        # YOLO + QR-seed + color/geometry fallback для поиска ценников
  qr_tools.py          # zxing-cpp / pyzbar / OpenCV QR decoding + парсинг QR-полей
  text_reader.py       # OCR: PaddleOCR + Tesseract ensemble
  field_extractor.py   # поля ценника: цены, скидка, barcode, SKU, дата, зона, цвет
  evidence_fusion.py   # объединение одного ценника между кадрами
  media.py             # чтение видео, sampling кадров, bbox, crop enhancement
  schema.py            # фиксированная CSV-схема и legacy-алиасы колонок
  training.py          # сборка YOLO-датасета и запуск обучения detector-а
  README.md            # краткая инструкция по ML-части

  data/
    25_12-20/          # public-разметка: mp4 + csv
    26_12-20/          # public-разметка: mp4 + csv
    43_15/             # public-разметка: mp4 + csv
    Unlabeled/         # видео без разметки для inference/self-training

  runs/                # создается при запуске: CSV, debug JSON, датасеты, результаты eval
```

Аналоги классических скриптов сделаны как API endpoints:

```text
POST /dataset/yolo     # build_yolo_dataset.py
POST /train/yolo       # train_detector.py
POST /predict/video    # infer_video.py для загруженного mp4
POST /predict/path     # infer_video.py для локального пути
POST /evaluate/public  # evaluate_on_public.py
```

## Быстрый старт

Запускать команды удобнее из папки `packages/ml`:

```powershell
cd packages\ml
uv run ml-service
```

Для полного OCR/QR-стека установи quality-extra:

```powershell
uv sync --extra quality
```

Он подключает Python-пакеты:

```text
PaddleOCR + paddlepaddle
pytesseract
zxing-cpp
pyzbar
```

Важно: `pytesseract` — это Python-обертка. Для реального Tesseract OCR на машине ещё должен быть установлен системный бинарник Tesseract и языковые данные `rus`/`eng`.

Для `pyzbar` на Windows/сервере может понадобиться системная библиотека ZBar. Если её нет, QR всё равно будет пробоваться через `zxing-cpp` и OpenCV QR.

Проверка Python-зависимостей:

```powershell
uv run python -c "import paddleocr, paddle, pytesseract, zxingcpp; import pyzbar.pyzbar; print('quality deps ok')"
```

Проверка языков Tesseract:

```powershell
tesseract --list-langs
```

Для русского fallback-OCR в списке должен быть `rus`. Если есть только `eng`, PaddleOCR всё равно может читать русский, но Tesseract fallback для русского текста будет ограничен.

По умолчанию сервис поднимается на:

```text
http://localhost:8000
```

Проверка:

```powershell
uv run python -c "from ml.main import app; print(app.title)"
```

## Основные endpoints

```text
GET  /health             статус сервиса и CSV-схема
GET  /schema             список колонок итогового CSV
GET  /datasets           найденные размеченные видео/CSV в src/ml/data
POST /predict/video      загрузить mp4 и получить CSV
POST /predict/path       обработать mp4 по локальному пути
POST /evaluate/public    прогнать proxy-оценку на размеченных public-данных
POST /dataset/yolo       собрать YOLO-датасет из public CSV/video
POST /train/yolo         запустить обучение YOLO через ultralytics
```

Документация FastAPI после запуска:

```text
http://localhost:8000/docs
```

## Инференс по локальному видео

Пример через Python-клиент или Swagger `POST /predict/path`:

```json
{
  "video_path": "F:/lenta_price_vision/packages/ml/src/ml/data/Unlabeled/26_2-10.mp4",
  "mode": "fast",
  "sample_fps": 1.0,
  "max_frames": 30,
  "enable_ocr": false,
  "enable_qr": true
}
```

Результат сохраняется в `src/ml/runs/<run_id>/`:

```text
*_recognized.csv
*_debug.json
```

## Режимы

```text
fast      быстро: детекция + QR, OCR можно отключить
cpu_safe  локальный режим без обязательных YOLO-весов
accurate  целевой режим: YOLO-веса + QR + OCR + temporal fusion
```

Если есть обученные веса, передайте путь:

```json
{
  "yolo_weights": "F:/lenta_price_vision/models/price_tag_yolo.pt"
}
```

Без весов сервис использует fallback-детекцию по QR/color/geometry. Это полезно для smoke-теста, но качество ниже.

## Public proxy-eval

Через Swagger `POST /evaluate/public`:

```json
{
  "data_dir": "F:/lenta_price_vision/packages/ml/src/ml/data",
  "mode": "fast",
  "sample_fps": 1.0,
  "max_frames": 0
}
```

Отчет появится в `src/ml/runs/<run_id>/evaluation_report.json`.

## Сборка YOLO-датасета

Через `POST /dataset/yolo`:

```json
{
  "data_dir": "F:/lenta_price_vision/packages/ml/src/ml/data",
  "output_dir": "F:/lenta_price_vision/packages/ml/src/ml/runs/lenta_yolo_dataset",
  "val_fraction": 0.2,
  "seed": 42
}
```

На выходе будет структура:

```text
images/train
images/val
labels/train
labels/val
data.yaml
```

## Обучение YOLO

Через `POST /train/yolo`:

```json
{
  "data_yaml": "F:/lenta_price_vision/packages/ml/src/ml/runs/lenta_yolo_dataset/data.yaml",
  "model": "yolo11n.pt",
  "epochs": 150,
  "imgsz": 1280,
  "batch": 4,
  "device": "cpu",
  "project": "runs/lenta",
  "name": "price_tag_yolo"
}
```

Для GPU поменять:

```json
{
  "device": "0"
}
```

После обучения лучший вес обычно лежит в:

```text
runs/lenta/price_tag_yolo/weights/best.pt
```

Его нужно использовать как `yolo_weights` для accurate-инференса.

## CSV-схема

Итоговый CSV всегда пишется в фиксированном порядке колонок. Legacy-колонка `wholesale_level_1_coun` из public-разметки автоматически нормализуется в:

```text
wholesale_level_1_count
```

Правила заполнения:

```text
нет      параметр точно отсутствует
пусто    параметр есть, но не распознан
```
