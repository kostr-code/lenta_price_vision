"""
scripts/preview_dataset.py — нарисовать bbox поверх изображений датасета и сгенерировать index.html.

Читает data.yaml, находит изображения нужного split-а, рисует зелёные прямоугольники
по YOLO-меткам и сохраняет HTML-галерею с inline base64 картинками.

Запуск:
    uv run python scripts/preview_dataset.py \\
        --dataset runs/datasets/lenta_yolo_tiled/data.yaml \\
        --split train --limit 200 --out-dir runs/preview_train

Затем открыть runs/preview_train/index.html в браузере.
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import yaml  # pip/uv: pyyaml


def load_dataset(data_yaml: Path) -> tuple[Path, dict]:
    """Прочитать data.yaml и вернуть (корневая папка датасета, конфиг)."""
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", data_yaml.parent)).resolve()
    return root, cfg


def draw_boxes(cv2, img_path: Path, lbl_path: Path | None):
    """Нарисовать YOLO-bbox поверх изображения; вернуть base64-строку JPEG или None.

    YOLO-координаты (cx, cy, bw, bh) нормированы [0,1] — переводим в пиксели
    перед вызовом cv2.rectangle.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    if lbl_path and lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf.tobytes()).decode()


def build_html(items: list[tuple[str, str]]) -> str:
    """Сгенерировать HTML-галерею: сетка карточек с inline base64-изображениями."""
    cards = "".join(
        f'<div class="card"><img src="data:image/jpeg;base64,{b64}" />'
        f"<p>{name}</p></div>"
        for name, b64 in items
    )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>body{font-family:sans-serif;background:#111;color:#eee}"
        ".grid{display:flex;flex-wrap:wrap;gap:6px;padding:8px}"
        ".card{background:#222;padding:4px;border-radius:4px;max-width:320px}"
        ".card img{max-width:100%;display:block}"
        ".card p{margin:2px 0;font-size:11px;word-break:break-all}</style></head>"
        f"<body><div class='grid'>{cards}</div></body></html>"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Путь к data.yaml датасета")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument(
        "--limit", type=int, default=200, help="Максимальное кол-во картинок в галерее"
    )
    p.add_argument("--out-dir", required=True, help="Папка для index.html")
    args = p.parse_args()

    try:
        import cv2
    except ImportError:
        print("Error: opencv-python required")
        return 1

    data_yaml = Path(args.dataset)
    if not data_yaml.exists():
        print(f"Error: not found: {data_yaml}")
        return 1

    root, cfg = load_dataset(data_yaml)
    img_dir = root / cfg.get(
        "train" if args.split == "train" else "val", f"images/{args.split}"
    )
    lbl_dir = Path(str(img_dir).replace("images", "labels"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_dir.glob("*.jpg"))[: args.limit]
    print(f"[preview] {len(img_files)} images from {img_dir}")

    items = []
    for img_path in img_files:
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        b64 = draw_boxes(cv2, img_path, lbl_path)
        if b64:
            items.append((img_path.name, b64))

    html_path = out_dir / "index.html"
    html_path.write_text(build_html(items), encoding="utf-8")
    print(f"[preview] → {html_path}  ({len(items)} cards)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
