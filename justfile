default:
    just --list

# ── Inference ────────────────────────────────────────────────────────────────

# Qwen-VL для одной картинки или папки с картинками.
qwen-debug path="data/testdata/crops_mid_01":
    uv run python test_qwen3vl.py "{{path}}"

# то же самое, но искать картинки еще и во вложенных папках.
qwen-debug-recursive path="data/testdata/crops_mid_01":
    uv run python test_qwen3vl.py --recursive "{{path}}"

# ── Dataset ───────────────────────────────────────────────────────────────────

# Собрать тайловый YOLO-датасет из labeled CSV+MP4.
# Параметры: data (источник), out (выход), prop (template propagation, 0=выкл).
dataset-build data="data/Данные" out="runs/datasets/lenta_yolo_tiled" prop="8":
    uv run python scripts/build_dataset.py \
      --data-dir {{data}} \
      --out-dir {{out}} \
      --tiled \
      --propagate {{prop}} \
      --val-ratio 0.2

# То же, но без тайлов (полные кадры).
dataset-build-full data="data/Данные" out="runs/datasets/lenta_yolo_full" prop="8":
    uv run python scripts/build_dataset.py \
      --data-dir {{data}} \
      --out-dir {{out}} \
      --propagate {{prop}} \
      --val-ratio 0.2

# Проверить разметку глазами — генерирует index.html с bbox поверх тайлов.
dataset-preview dataset="runs/datasets/lenta_yolo_tiled" split="train":
    uv run python scripts/preview_dataset.py \
      --dataset {{dataset}}/data.yaml \
      --split {{split}} \
      --limit 300 \
      --out-dir {{dataset}}/preview_{{split}}
    @echo "Открыть: {{dataset}}/preview_{{split}}/index.html"

# ── Stage 1: детектор ценников ────────────────────────────────────────────────

# Обучить Stage 1 (price-tag detector).
train-yolo1 dataset="runs/datasets/lenta_yolo_tiled":
    bash train/train_yolo1.sh {{dataset}}/data.yaml

# Скопировать лучшие веса Stage 1 в models/.
save-yolo1 run="price_tag_yolo":
    mkdir -p models
    cp runs/detect/{{run}}/weights/best.pt models/price_tag_yolo.pt
    @echo "Сохранено → models/price_tag_yolo.pt"

# ── Stage 2: внутренние элементы ценника ─────────────────────────────────────

# Прогнать Stage 1 по датасету и сохранить кропы ценников для Stage 2.
crop-for-stage2 dataset="runs/datasets/lenta_yolo_tiled":
    uv run python scripts/crop_detections.py \
      --dataset {{dataset}} \
      --weights models/price_tag_yolo.pt \
      --out-subdir crops_for_stage2 \
      --conf 0.25 --imgsz 1280 --device 0

# Подготовить датасет из CVAT export (standard layout).
# Использовать после ручной разметки crops_for_stage2 в CVAT.
prepare-cvat source="runs/cvat_inside_export" out="runs/datasets/lenta_inside_yolo":
    uv run python scripts/prepare_cvat_dataset.py \
      --source {{source}} \
      --out-dir {{out}} \
      --val-ratio 0.2

# Обучить Stage 2 (inside elements detector).
train-yolo2 dataset="runs/datasets/lenta_inside_yolo":
    bash train/train_yolo2.sh {{dataset}}/data.yaml

# Скопировать лучшие веса Stage 2 в models/.
save-yolo2 run="inside_price_tag_yolo":
    mkdir -p models
    cp runs/detect/{{run}}/weights/best.pt models/inside_price_tag_yolo.pt
    @echo "Сохранено → models/inside_price_tag_yolo.pt"

# ── Полный цикл Stage 1 (одной командой) ─────────────────────────────────────

# Весь Stage 1 pipeline: датасет → проверка → тренировка.
# Веса НЕ копируются автоматически — сделать just save-yolo1 вручную после проверки.
yolo1-full: dataset-build dataset-preview train-yolo1
