"""
scripts/review_dataset.py — интерактивная проверка YOLO-разметки через cv2.

Открывает изображения по одному с нарисованными bbox. Клавиши:
    Space / Enter / K  — оставить
    D / Backspace      — удалить (переместить в trash-dir)
    Q / Esc            — выйти

Результат записывается в JSONL-лог; можно продолжить с --start-at.

Запуск:
    uv run python scripts/review_dataset.py \\
        --dataset runs/datasets/lenta_yolo \\
        --split train

    # продолжить с конкретного файла:
    uv run python scripts/review_dataset.py \\
        --dataset runs/datasets/lenta_yolo \\
        --split train --start-at 43_15_00006822.jpg

Удалённые файлы по умолчанию перемещаются в <dataset_root>/.review_removed/<split>/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.preview_dataset import (
    DEFAULT_COLOR,
    IMAGE_EXTENSIONS,
    DatasetLayout,
    _normalized_to_pixels,
    read_yolo_boxes,
    resolve_dataset_layout,
)
from ml.training import BBox

Decision = Literal["keep", "remove", "quit"]
RemoveMode = Literal["move", "delete"]

WINDOW_NAME = "YOLO annotation review"
HELP_LINES = (
    "Space/Enter/K: keep",
    "D/Backspace: remove",
    "Q/Esc: quit",
)
WARNING_COLOR = (0, 180, 255)
TEXT_COLOR = (255, 255, 255)
TEXT_SHADOW = (0, 0, 0)


# ── Структуры данных ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReviewItem:
    image_path: Path
    label_path: Path
    boxes: list[tuple[int, BBox]]
    warnings: list[str]


@dataclass(frozen=True)
class ReviewEvent:
    timestamp: str
    decision: str
    image_path: str
    label_path: str
    moved_image_path: str | None = None
    moved_label_path: str | None = None
    boxes: int = 0
    warnings: list[str] | None = None


@dataclass(frozen=True)
class ReviewSummary:
    dataset: str
    split: str
    reviewed: int
    kept: int
    removed: int
    skipped: int
    trash_dir: str | None
    log_path: str


# ── Сбор элементов для проверки ───────────────────────────────────────────────


def collect_review_items(
    layout: DatasetLayout,
    include_empty: bool,
    start_at: str | None = None,
) -> list[ReviewItem]:
    images = [
        p
        for p in sorted(layout.images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if start_at:
        start_name = Path(start_at).name
        images = [p for p in images if p.name >= start_name]

    items: list[ReviewItem] = []
    for image_path in images:
        label_path = layout.labels_dir / f"{image_path.stem}.txt"
        boxes, warnings = read_yolo_boxes(label_path)
        if not boxes and not include_empty:
            continue
        items.append(
            ReviewItem(
                image_path=image_path,
                label_path=label_path,
                boxes=boxes,
                warnings=warnings,
            )
        )
    return items


# ── Основная функция проверки ─────────────────────────────────────────────────


def review_dataset(
    dataset: Path,
    split: str,
    trash_dir: Path | None = None,
    remove_mode: RemoveMode = "move",
    include_empty: bool = True,
    max_width: int = 1600,
    max_height: int = 950,
    start_at: str | None = None,
    log_path: Path | None = None,
) -> ReviewSummary:
    import cv2

    layout = resolve_dataset_layout(dataset, split)
    items = collect_review_items(layout, include_empty=include_empty, start_at=start_at)

    if trash_dir is None and remove_mode == "move":
        trash_dir = layout.root / ".review_removed" / split
    if log_path is None:
        log_path = layout.root / f"annotation_review_{split}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    kept = removed = skipped = reviewed = 0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        total = len(items)
        for index, item in enumerate(items, start=1):
            if not item.image_path.exists():
                skipped += 1
                continue

            image = cv2.imread(str(item.image_path))
            if image is None:
                skipped += 1
                _write_event(
                    log_path,
                    ReviewEvent(
                        timestamp=_utc_now(),
                        decision="skip_unreadable",
                        image_path=str(item.image_path),
                        label_path=str(item.label_path),
                        boxes=len(item.boxes),
                        warnings=["Cannot read image"],
                    ),
                )
                continue

            rendered = _draw_review_image(
                image, item, index, total, max_width, max_height
            )
            cv2.imshow(WINDOW_NAME, rendered)
            decision = _read_decision(cv2)

            if decision == "quit":
                break

            reviewed += 1

            if decision == "keep":
                kept += 1
                _write_event(
                    log_path,
                    ReviewEvent(
                        timestamp=_utc_now(),
                        decision="keep",
                        image_path=str(item.image_path),
                        label_path=str(item.label_path),
                        boxes=len(item.boxes),
                        warnings=item.warnings,
                    ),
                )
                continue

            moved_img, moved_lbl = _remove_item(item, trash_dir, remove_mode)
            removed += 1
            _write_event(
                log_path,
                ReviewEvent(
                    timestamp=_utc_now(),
                    decision=f"remove_{remove_mode}",
                    image_path=str(item.image_path),
                    label_path=str(item.label_path),
                    moved_image_path=str(moved_img) if moved_img else None,
                    moved_label_path=str(moved_lbl) if moved_lbl else None,
                    boxes=len(item.boxes),
                    warnings=item.warnings,
                ),
            )
    finally:
        cv2.destroyWindow(WINDOW_NAME)

    return ReviewSummary(
        dataset=str(layout.data_yaml),
        split=split,
        reviewed=reviewed,
        kept=kept,
        removed=removed,
        skipped=skipped,
        trash_dir=str(trash_dir) if trash_dir else None,
        log_path=str(log_path),
    )


# ── Отрисовка ─────────────────────────────────────────────────────────────────


def _draw_review_image(
    image: Any,
    item: ReviewItem,
    index: int,
    total: int,
    max_width: int,
    max_height: int,
) -> Any:
    import cv2

    rendered = image.copy()
    h, w = rendered.shape[:2]
    thickness = max(2, round(min(w, h) / 300))
    font_scale = max(0.55, min(w, h) / 1000)

    for cls, normalized_box in item.boxes:
        box = _normalized_to_pixels(normalized_box, w, h)
        x1, y1, x2, y2 = box.as_int_tuple()
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(rendered, (x1, y1), (x2, y2), DEFAULT_COLOR, thickness)
        _draw_text(
            rendered,
            f"cls:{cls}",
            (x1, max(24, y1 - 8)),
            scale=font_scale,
            color=DEFAULT_COLOR,
            thickness=thickness,
        )

    status = f"{index}/{total} | boxes: {len(item.boxes)} | {item.image_path.name}"
    _draw_text(rendered, status, (14, 30), scale=0.8, color=TEXT_COLOR, thickness=2)

    y = 62
    for line in HELP_LINES:
        _draw_text(rendered, line, (14, y), scale=0.65, color=TEXT_COLOR, thickness=2)
        y += 28
    for warning in item.warnings[:3]:
        _draw_text(
            rendered, warning, (14, y), scale=0.6, color=WARNING_COLOR, thickness=2
        )
        y += 26

    return _fit_to_screen(rendered, max_width, max_height)


def _draw_text(
    image: Any,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    import cv2

    x, y = origin
    # тень для читаемости на любом фоне
    cv2.putText(
        image,
        text,
        (x + 2, y + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        TEXT_SHADOW,
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _fit_to_screen(image: Any, max_width: int, max_height: int) -> Any:
    import cv2

    h, w = image.shape[:2]
    if w <= max_width and h <= max_height:
        return image
    scale = min(max_width / w, max_height / h)
    target = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(image, target, interpolation=cv2.INTER_AREA)


def _read_decision(cv2: Any) -> Decision:
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            return "quit"
        if key in (ord("k"), ord(" "), 13):
            return "keep"
        if key in (ord("d"), 8, 127):
            return "remove"


# ── Удаление файлов ───────────────────────────────────────────────────────────


def _remove_item(
    item: ReviewItem,
    trash_dir: Path | None,
    mode: RemoveMode,
) -> tuple[Path | None, Path | None]:
    if mode == "delete":
        _unlink_if_exists(item.image_path)
        _unlink_if_exists(item.label_path)
        return None, None
    if trash_dir is None:
        raise ValueError("trash_dir required when remove_mode='move'")
    return (
        _move_to_trash(item.image_path, trash_dir / "images"),
        _move_to_trash(item.label_path, trash_dir / "labels"),
    )


def _move_to_trash(path: Path, target_dir: Path) -> Path | None:
    if not path.exists():
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    target = _unique_path(target_dir / path.name)
    shutil.move(str(path), str(target))
    return target


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot build unique path for {path}")


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


# ── Лог ──────────────────────────────────────────────────────────────────────


def _write_event(log_path: Path, event: ReviewEvent) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description="Интерактивная проверка YOLO-разметки: keep / remove по одному изображению"
    )
    p.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Путь к data.yaml или папке датасета",
    )
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument(
        "--trash-dir",
        type=Path,
        default=None,
        help="Куда перемещать отклонённые файлы (по умолч. .review_removed/<split>)",
    )
    p.add_argument(
        "--remove-mode",
        choices=["move", "delete"],
        default="move",
        help="move = в trash-dir; delete = удалить насовсем",
    )
    p.add_argument(
        "--hide-empty", action="store_true", help="Пропустить изображения без bbox"
    )
    p.add_argument("--max-width", type=int, default=1600)
    p.add_argument("--max-height", type=int, default=950)
    p.add_argument("--start-at", default=None, help="Начать с конкретного имени файла")
    p.add_argument(
        "--log-path", type=Path, default=None, help="Путь к JSONL-логу решений"
    )
    args = p.parse_args()

    try:
        import cv2  # noqa: F401
    except ImportError:
        print("Error: opencv-python required (uv add opencv-python)")
        return 1

    summary = review_dataset(
        dataset=args.dataset,
        split=args.split,
        trash_dir=args.trash_dir,
        remove_mode=args.remove_mode,
        include_empty=not args.hide_empty,
        max_width=args.max_width,
        max_height=args.max_height,
        start_at=args.start_at,
        log_path=args.log_path,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
