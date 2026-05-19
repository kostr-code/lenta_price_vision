# Lenta Price Vision

Сервис автоматической оцифровки ценников из видеопотока робота.
Принимает MP4 или изображение, возвращает структурированный CSV (29 полей).

## Требования

| | |
|---|---|
| GPU | ≥ 6 GB VRAM (Qwen2.5-VL-3B без квантизации; 4-bit режим — ≈ 3 GB) |
| CUDA | ≥ 12.x |
| Python | 3.12+ |
| Менеджер пакетов | [uv](https://docs.astral.sh/uv/) |
| Node.js | 18+ (только для фронтенда) |
| ffmpeg | для генерации аннотированного видео |

## Установка

```bash
# Python-зависимости (uv создаёт venv автоматически)
uv sync

# Фронтенд
cd frontend && npm install && cd ..
```

## Данные

Ожидаемая структура:

```
data/Данные/
├── 25_12-20/
│   ├── 25_12-20.csv      # разметка: bbox + 29 полей ценника
│   └── 2.mp4
├── 43_15/
│   ├── 43_15.csv
│   └── 43_15.mp4
└── Unlabeled/
    ├── 25_12-20.mp4
    ├── 26_12-20.mp4
    └── ...
```

## Веса YOLO

Скачать с Яндекс.Диска: **https://disk.yandex.ru/d/u9GjAQhiGbhoDg**

Положить в `models/`:

```
models/
├── price_tag_yolo.pt           # Stage 1 — детектор ценников
└── inside_price_tag_yolo.pt    # Stage 2 — QR/barcode субрегионы (опционально)
```

VLM (`Qwen/Qwen2.5-VL-3B-Instruct`) скачивается автоматически с HuggingFace при первом запуске.

## Запуск

В трёх отдельных терминалах:

```bash
just ml      # ML-сервис  :8000  (загружает модели при старте, ~1–2 мин)
just gw      # Gateway    :8001
just front   # React UI   :5173
```

```bash
just health  # проверить что ML-сервис отвечает
```

Открыть **http://localhost:5173** → загрузить видео или фото → нажать **Run Recognition**.

## Конфигурация

### ML-сервис (`ML_*`)

| Переменная | Дефолт | Описание |
|---|---|---|
| `ML_VLM_MODEL_ID` | `Qwen/Qwen2.5-VL-3B-Instruct` | HuggingFace model id |
| `ML_VLM_4BIT` | `false` | 4-bit квантизация (≈ 3 GB VRAM) |
| `ML_VLM_DEVICE_MAP` | `cuda` | device_map для transformers |
| `ML_WEIGHTS_STAGE1` | `models/price_tag_yolo.pt` | YOLO Stage 1 веса |
| `ML_WEIGHTS_INSIDE` | — | YOLO Stage 2 веса (опционально) |
| `ML_RUNS_DIR` | `runs/api` | папка для CSV / JSON / видео результатов |
| `ML_QUALITY_THRESHOLD` | `0.2` | минимальный quality score кропа |
| `ML_LOG_LEVEL` | `INFO` | уровень логирования |

### Gateway

| Переменная | Дефолт | Описание |
|---|---|---|
| `ML_URL` | `http://ml:8000` | адрес ML-сервиса; для локальной разработки `just gw` передаёт `http://localhost:8000` автоматически |

## Архитектура

```
[React :5173]
      │  POST /api/v1/predict/{video,image}
      ▼
[Gateway :8001]  api_gateway.py
      │  проксирует на ML; переписывает /download/ → /api/v1/download/
      ▼
[ML Service :8000]  ml_server.py + api/
      │  держит модели в памяти (load_models при старте)
      ▼
pipeline/  — переиспользуемые модули
```

## Pipeline

### Видео (unlabeled)

```
MP4
  └─ rotate 90° → YOLO Stage 1 + ByteTrack
        └─ top-K кропов на трек → выбрать best по quality score
              ├─ YOLO Stage 2 → QR/barcode субрегионы → decode
              ├─ Qwen2.5-VL  → JSON (6 текстовых полей)
              └─ PaddleOCR   → parse_fields → 29 полей
        └─ merge (VLM primary, OCR fillback)

Результат: output.csv  debug.json  crops/*.jpg  annotated.mp4
```

### Изображение

```
JPG/PNG → rotate 90° → [YOLO Stage 1 detect, если есть веса] → кропы
        └─ тот же процессинг (QR + VLM + OCR + merge)
```

## VLM-промпт

Редактировать в `prompts/qwen_extract.md` — перезагружается при рестарте ML-сервиса без изменения кода.

## Выходной CSV

29 полей на строку (один уникальный ценник):

| Группа | Поля |
|---|---|
| Координаты | `filename`, `frame_timestamp`, `x_min`, `y_min`, `x_max`, `y_max` |
| Текст (VLM/OCR) | `product_name`, `price_card`, `price_default`, `price_discount`, `discount_amount`, `barcode`, `id_sku`, `print_datetime`, `code`, `additional_info`, `color`, `special_symbols` |
| Из QR-кода | `qr_code_barcode`, `price1_qr`, `price2_qr`, `price3_qr`, `price4_qr`, `wholesale_level_1_count`, `wholesale_level_1_price`, `wholesale_level_2_count`, `wholesale_level_2_price`, `action_price_qr`, `action_code_qr` |

## Структура кода

```
sol_main/
  ml_server.py              — uvicorn entry point (ML :8000)
  api_gateway.py            — gateway proxy (:8001)
  prompts/
    qwen_extract.md         — VLM extraction prompt
  api/
    ml_app.py               — FastAPI app + lifespan (load_models)
    config.py               — MLSettings из env
    pipeline_bridge.py      — singletons + process_video_file / process_single_crop
    routers/
      predict.py            — POST /predict/{video,image,path}
      download.py           — GET /download/{run_id}/{path:path}
      meta.py               — GET /health, /schema
  pipeline/
    detector.py             — YOLO Stage 1: detect_price_tags, track_price_tags
    vlm.py                  — VLMProvider protocol + Qwen25VLProvider
    ocr.py                  — PaddleOCR: enhance_crop, ocr_zoned
    parsers.py              — parse_fields, OUTPUT_COLUMNS
    qr.py                   — decode_qr, decode_barcode_linear
    fragments.py            — YOLO Stage 2: FragmentProvider
    quality.py              — estimate_crop_quality
    video.py                — rotate_frame, cut_crop_bbox
  models/
    price_tag_yolo.pt
    inside_price_tag_yolo.pt
  frontend/
    src/App.jsx             — React UI
    src/styles.css
```
