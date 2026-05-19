# Lenta Price Vision — ML Service

Сервис автоматической оцифровки ценников из видеопотока.

## Архитектура

```
[React Frontend :5173]
        │  POST /api/v1/predict/video
        │  POST /api/v1/predict/image
        ▼
[Gateway :8001]  api_gateway.py
        │  принимает файлы, проксирует на ML,
        │  обогащает ответ: download → backend_download
        ▼
[ML Service :8000]  ml_server.py + api/
        │  FastAPI, грузит модели при старте (lifespan)
        ▼
[pipeline/]  — переиспользуемые модули
```

**Gateway** (`api_gateway.py`) — stateless прокси. Принимает загрузки от фронта,
пересылает на ML, добавляет `backend_download` / `backend_debug_download` URL.

**ML Service** (`ml_server.py` + `api/`) — держит модели в памяти, запускает inference.

## Pipeline — видео (unlabeled mode)

```
MP4-файл
  │
  ├─ rotate_frame() — поворот 90° CCW (ориентация камеры робота)
  │
  ├─ track_price_tags() — YOLO Stage 1 + ByteTrack
  │     └─ накапливает top-K кропов на трек (по quality score)
  │
  └─ per track: best crop
        ├─ QR decode — WeChat / pyzbar / zxing-cpp → qr_payloads
        ├─ VLM inference — Qwen2.5-VL → 12 текстовых полей (JSON)
        ├─ OCR — PaddleOCR зональный → parse_fields() → 29 полей
        └─ merge — VLM primary, OCR fillback → output row
```

## Pipeline — изображение

```
JPG/PNG
  │
  ├─ rotate_frame()
  ├─ [опц.] detect_price_tags() — YOLO Stage 1, если weights_stage1 задан
  │     иначе: весь кадр = один кроп
  └─ process_single_crop()
        ├─ QR decode
        ├─ VLM inference
        └─ OCR + merge
```

## Модели (load_models при старте)

| Модель | Файл / ID | Назначение |
|--------|-----------|------------|
| YOLO Stage 1 | `models/price_tag_yolo.pt` | детектор ценников |
| YOLO Stage 2 | `models/inside_price_tag_yolo.pt` | QR/barcode регионы |
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-3B-Instruct` | первичное распознавание, 12 полей |
| PaddleOCR | — | вторичный OCR, fillback |
| WeChat QRCode | opencv-contrib | QR декодер (опционально) |

## API эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/predict/video` | видео-файл → CSV строки по трекам |
| `POST` | `/predict/image` | изображение → строки по детекциям |
| `POST` | `/predict/path` | путь на сервере (для тестов) |
| `GET`  | `/health` | статус сервиса + какие модели загружены |
| `GET`  | `/schema` | список 29 полей с описаниями |
| `GET`  | `/datasets` | датасеты из data_dir |
| `GET`  | `/download/{run_id}/{filename}` | скачать output.csv / debug.json |

## Запуск

```bash
just ml          # ML сервис :8000
just gw          # Gateway :8001
just health      # curl /health
```

Фронт:
```bash
cd frontend && npm install && npm run dev   # :5173
```

## Конфигурация (env vars)

| Переменная | Дефолт | Описание |
|------------|--------|----------|
| `ML_VLM_MODEL_ID` | `Qwen/Qwen2.5-VL-3B-Instruct` | HF model id |
| `ML_VLM_4BIT` | `false` | bitsandbytes 4-bit квантизация |
| `ML_VLM_DEVICE_MAP` | `cuda` | device_map для transformers |
| `ML_WEIGHTS_STAGE1` | `models/price_tag_yolo.pt` | YOLO Stage 1 веса |
| `ML_WEIGHTS_INSIDE` | — | YOLO Stage 2 веса |
| `ML_RUNS_DIR` | `runs/api` | директория для output.csv / debug.json |
| `ML_QUALITY_THRESHOLD` | `0.2` | минимальный quality score кропа |
| `ML_LOG_LEVEL` | `INFO` | уровень логирования |
| `ML_URL` | `http://ml:8000` | URL ML сервиса (для gateway) |

## Структура кода

```
sol_main/
  ml_server.py              — uvicorn entry point
  api_gateway.py       — gateway proxy
  api/
    ml_app.py               — FastAPI app + lifespan
    config.py               — MLSettings из env
    logging_setup.py        — structlog + rich
    pipeline_bridge.py      — module singletons + process_*
    run_store.py            — UUID → RunResult
    routers/
      predict.py            — /predict/*
      meta.py               — /health, /schema, /datasets
      download.py           — /download/{run_id}/{file}
  pipeline/
    video.py                — rotate_frame, cut_crop_bbox, find_best_frame
    detector.py             — YOLO Stage 1: detect/track_price_tags
    vlm.py                  — VLMProvider protocol + Qwen25VLProvider
    ocr.py                  — PaddleOCR: enhance_crop, ocr_zoned
    parsers.py              — parse_fields, OUTPUT_COLUMNS (29 полей)
    qr.py                   — decode_qr, decode_barcode_linear
    quality.py              — estimate_crop_quality
    fragments.py            — YOLO Stage 2: FragmentProvider
    sr.py                   — CLAHE, sharpen (опционально)
  models/
    price_tag_yolo.pt       — Stage 1 веса
    inside_price_tag_yolo.pt — Stage 2 веса
```
