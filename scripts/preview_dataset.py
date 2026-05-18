"""
scripts/preview_dataset.py — HTML-галерея с отрисованной YOLO-разметкой.

Сохраняет изображения с bbox на диск и генерирует index.html со ссылками на них.
В отличие от inline base64, такой подход быстрее открывается в браузере для больших
датасетов.

Запуск:
    uv run python scripts/preview_dataset.py \\
        --dataset runs/datasets/lenta_yolo/data.yaml \\
        --split train --limit 200 --out-dir runs/preview_train

Затем открыть runs/preview_train/index.html в браузере.
"""

from __future__ import annotations

import argparse
import html
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.training import BBox

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_COLOR = (40, 220, 40)
WARNING_COLOR = (0, 180, 255)


# ── Структуры данных ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DatasetLayout:
    root: Path
    images_dir: Path
    labels_dir: Path
    split: str
    data_yaml: Path


@dataclass(frozen=True)
class AnnotationPreview:
    image_path: Path
    label_path: Path
    output_path: Path
    boxes: int
    warnings: list[str]


# ── Разрешение путей датасета ─────────────────────────────────────────────────


def resolve_dataset_layout(dataset: Path, split: str) -> DatasetLayout:
    """Найти реальные папки images/<split> и labels/<split>.

    Пробует несколько корней: path: из data.yaml, папку рядом с data.yaml,
    переданный dataset и cwd. Выбирает первый вариант где обе папки существуют.
    """
    dataset = dataset.expanduser()
    data_yaml = dataset if dataset.is_file() else dataset / "data.yaml"
    data_yaml = data_yaml.resolve()

    config = _read_data_yaml(data_yaml)
    split_value = str(config.get(split, f"images/{split}")).strip() or f"images/{split}"

    root_candidates = _build_root_candidates(
        raw_root=config.get("path"),
        data_yaml=data_yaml,
        dataset=dataset,
    )

    checked: list[str] = []
    for root in root_candidates:
        images_dir = _resolve_split_path(root, split_value)
        labels_dir = _infer_labels_dir(images_dir, root, split_value)
        checked.append(f"images={images_dir} | labels={labels_dir}")

        if images_dir.exists() and labels_dir.exists():
            return DatasetLayout(
                root=root,
                images_dir=images_dir,
                labels_dir=labels_dir,
                split=split,
                data_yaml=data_yaml,
            )

    lines = [
        f"Cannot resolve YOLO dataset layout for split={split!r}.",
        f"data_yaml: {data_yaml}",
        f"data_yaml path field: {config.get('path')!r}",
        f"{split} field: {split_value!r}",
        "",
        "Checked candidates:",
        *[f"  - {item}" for item in checked],
        "",
        "Expected structure: <root>/images/train  <root>/labels/train",
    ]
    raise FileNotFoundError("\n".join(lines))


def _build_root_candidates(
    raw_root: object, data_yaml: Path, dataset: Path
) -> list[Path]:
    candidates: list[Path] = []

    def add(path: Path) -> None:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            resolved = path.expanduser()
        if resolved not in candidates:
            candidates.append(resolved)

    yaml_dir = data_yaml.parent
    if raw_root is not None and str(raw_root).strip():
        root = Path(str(raw_root).strip())
        if root.is_absolute():
            add(root)
        else:
            add(yaml_dir / root)
            add(Path.cwd() / root)
    if dataset.is_dir():
        add(dataset)
    add(yaml_dir)
    return candidates


def _resolve_split_path(root: Path, split_value: str) -> Path:
    p = Path(split_value)
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def _infer_labels_dir(images_dir: Path, root: Path, split_value: str) -> Path:
    normalized = split_value.replace("\\", "/")
    if normalized.startswith("images/"):
        return (root / normalized.replace("images/", "labels/", 1)).resolve()
    parts = list(Path(split_value).parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return (root / Path(*parts)).resolve()
    if images_dir.name in {"train", "val"} and images_dir.parent.name == "images":
        return (images_dir.parent.parent / "labels" / images_dir.name).resolve()
    return (root / "labels" / images_dir.name).resolve()


def _read_data_yaml(path: Path) -> dict[str, Any]:
    """Прочитать data.yaml; fallback-парсер если pyyaml не установлен."""
    if not path.exists():
        raise FileNotFoundError(f"data.yaml not found: {path}")

    text = path.read_text(encoding="utf-8-sig")
    try:
        import yaml

        data = yaml.safe_load(text) or {}
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    data: dict[str, Any] = {}
    current_key: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" in line and not line.startswith((" ", "\t")):
            key, value = line.split(":", 1)
            current_key = key.strip()
            data[current_key] = value.strip()
        elif current_key == "names" and ":" in stripped:
            names = data.setdefault("names", {})
            if isinstance(names, dict):
                key, value = stripped.split(":", 1)
                names[key.strip()] = value.strip()
    return data


# ── Чтение и отрисовка ────────────────────────────────────────────────────────


def read_yolo_boxes(label_path: Path) -> tuple[list[tuple[int, BBox]], list[str]]:
    """Прочитать YOLO label-файл; вернуть (боксы, предупреждения о битых строках)."""
    if not label_path.exists():
        return [], [f"missing label: {label_path.name}"]

    boxes: list[tuple[int, BBox]] = []
    warnings: list[str] = []

    for line_no, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        text = line.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) < 5:
            warnings.append(f"line {line_no}: expected 5 values, got {len(parts)}")
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
        except ValueError:
            warnings.append(f"line {line_no}: cannot parse numbers")
            continue
        if bw <= 0 or bh <= 0:
            warnings.append(f"line {line_no}: non-positive box size")
            continue
        if not all(0 <= v <= 1 for v in (cx, cy, bw, bh)):
            warnings.append(f"line {line_no}: values outside [0, 1]")
        boxes.append((cls, BBox(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)))

    return boxes, warnings


def _normalized_to_pixels(box: BBox, width: int, height: int) -> BBox:
    return BBox(
        max(0.0, min(float(width), box.x_min * width)),
        max(0.0, min(float(height), box.y_min * height)),
        max(0.0, min(float(width), box.x_max * width)),
        max(0.0, min(float(height), box.y_max * height)),
    )


def draw_preview(
    image_path: Path,
    output_path: Path,
    boxes: list[tuple[int, BBox]],
    warnings: list[str],
) -> None:
    """Нарисовать bbox поверх изображения и сохранить в output_path."""
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    h, w = image.shape[:2]
    thickness = max(2, round(min(w, h) / 300))
    font_scale = max(0.45, min(w, h) / 1100)

    for cls, normalized_box in boxes:
        box = _normalized_to_pixels(normalized_box, w, h)
        x1, y1, x2, y2 = box.as_int_tuple()
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(image, (x1, y1), (x2, y2), DEFAULT_COLOR, thickness)
        cv2.putText(
            image,
            f"cls:{cls}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            DEFAULT_COLOR,
            thickness,
            cv2.LINE_AA,
        )

    if warnings:
        cv2.putText(
            image,
            f"warnings: {len(warnings)}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            WARNING_COLOR,
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


# ── Генерация HTML ────────────────────────────────────────────────────────────


def build_annotation_gallery(
    dataset: Path,
    out_dir: Path,
    split: str = "train",
    limit: int = 200,
    seed: int = 42,
    shuffle: bool = True,
    include_empty: bool = True,
) -> Path:
    """Собрать HTML-галерею: нарисовать bbox на каждом изображении, сохранить index.html."""
    layout = resolve_dataset_layout(dataset, split)
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p
        for p in layout.images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if shuffle:
        random.Random(seed).shuffle(images)
    if limit > 0:
        images = images[:limit]

    previews: list[AnnotationPreview] = []
    for image_path in images:
        label_path = layout.labels_dir / f"{image_path.stem}.txt"
        boxes, warnings = read_yolo_boxes(label_path)
        if not boxes and not include_empty:
            continue
        output_path = out_images / image_path.name
        draw_preview(image_path, output_path, boxes, warnings)
        previews.append(
            AnnotationPreview(
                image_path=image_path,
                label_path=label_path,
                output_path=output_path,
                boxes=len(boxes),
                warnings=warnings,
            )
        )

    index_path = out_dir / "index.html"
    index_path.write_text(_render_html(layout, previews, out_dir), encoding="utf-8")
    return index_path


def _render_html(
    layout: DatasetLayout, previews: list[AnnotationPreview], out_dir: Path
) -> str:
    cards = "\n".join(_render_card(item, out_dir) for item in previews)
    total_boxes = sum(item.boxes for item in previews)
    warning_count = sum(1 for item in previews if item.warnings)

    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YOLO annotation viewer</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #111; color: #eee; }}
    header {{ position: sticky; top: 0; z-index: 1; background: #191919; padding: 16px 20px; border-bottom: 1px solid #333; }}
    h1 {{ margin: 0 0 8px; font-size: 20px; }}
    .meta {{ color: #bbb; font-size: 13px; line-height: 1.5; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 14px; padding: 14px; }}
    .card {{ background: #1d1d1d; border: 1px solid #333; border-radius: 8px; overflow: hidden; }}
    .card img {{ width: 100%; display: block; background: #000; }}
    .caption {{ padding: 10px 12px; font-size: 13px; color: #ddd; }}
    .path {{ color: #9bd; word-break: break-all; }}
    .warn {{ color: #ffc266; margin-top: 6px; }}
  </style>
</head>
<body>
  <header>
    <h1>YOLO annotation viewer</h1>
    <div class="meta">
      split: {html.escape(layout.split)}<br>
      images: {len(previews)} | boxes: {total_boxes} | images with warnings: {warning_count}<br>
      dataset root: {html.escape(str(layout.root))}<br>
      data.yaml: {html.escape(str(layout.data_yaml))}
    </div>
  </header>
  <main class="grid">
    {cards}
  </main>
</body>
</html>
"""


def _render_card(item: AnnotationPreview, out_dir: Path) -> str:
    image_src = item.output_path.relative_to(out_dir).as_posix()
    warnings_html = ""
    if item.warnings:
        warnings_html = (
            '<div class="warn">'
            + "<br>".join(map(html.escape, item.warnings))
            + "</div>"
        )
    return f"""<section class="card">
  <a href="{html.escape(image_src)}"><img src="{html.escape(image_src)}" loading="lazy"></a>
  <div class="caption">
    boxes: {item.boxes}<br>
    <span class="path">{html.escape(item.image_path.name)}</span>
    {warnings_html}
  </div>
</section>"""


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description="Галерея YOLO-разметки: bbox поверх изображений → index.html"
    )
    p.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Путь к data.yaml или папке датасета",
    )
    p.add_argument(
        "--out-dir", type=Path, required=True, help="Куда писать index.html и images/"
    )
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument(
        "--limit", type=int, default=200, help="Максимум изображений; 0 = все"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-shuffle", action="store_true", help="Не перемешивать (взять первые N)"
    )
    p.add_argument(
        "--hide-empty", action="store_true", help="Не показывать изображения без bbox"
    )
    args = p.parse_args()

    try:
        import cv2  # noqa: F401
    except ImportError:
        print("Error: opencv-python required (uv add opencv-python)")
        return 1

    index_path = build_annotation_gallery(
        dataset=args.dataset,
        out_dir=args.out_dir,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        shuffle=not args.no_shuffle,
        include_empty=not args.hide_empty,
    )
    print(f"[preview] → {index_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
