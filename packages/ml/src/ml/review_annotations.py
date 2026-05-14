from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .annotation_viewer import (
    DEFAULT_COLOR,
    IMAGE_EXTENSIONS,
    DatasetLayout,
    normalized_to_pixels,
    read_yolo_boxes,
    resolve_dataset_layout,
)
from .media import BBox, import_cv2

Decision = Literal["keep", "remove", "quit"]
RemoveMode = Literal["move", "delete"]

WINDOW_NAME = "YOLO annotation review"
HELP_LINES = (
    "Space/Enter/K: keep",
    "D/Backspace: remove from dataset",
    "Q/Esc: quit",
)
WARNING_COLOR = (0, 180, 255)
TEXT_COLOR = (255, 255, 255)
TEXT_SHADOW = (0, 0, 0)


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


def collect_review_items(
    layout: DatasetLayout,
    include_empty: bool,
    start_at: str | None = None,
) -> list[ReviewItem]:
    images = [
        path
        for path in sorted(layout.images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if start_at:
        start_name = Path(start_at).name
        images = [path for path in images if path.name >= start_name]

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
    layout = resolve_dataset_layout(dataset, split)
    items = collect_review_items(layout, include_empty=include_empty, start_at=start_at)

    if trash_dir is None and remove_mode == "move":
        trash_dir = layout.root / ".review_removed" / split

    if log_path is None:
        log_path = layout.root / f"annotation_review_{split}.jsonl"

    log_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    removed = 0
    skipped = 0
    reviewed = 0

    cv2 = import_cv2()
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
                write_event(
                    log_path,
                    ReviewEvent(
                        timestamp=utc_now(),
                        decision="skip_unreadable",
                        image_path=str(item.image_path),
                        label_path=str(item.label_path),
                        boxes=len(item.boxes),
                        warnings=["Cannot read image"],
                    ),
                )
                continue

            rendered = draw_review_image(
                image=image,
                item=item,
                index=index,
                total=total,
                max_width=max_width,
                max_height=max_height,
            )

            cv2.imshow(WINDOW_NAME, rendered)
            decision = read_decision(cv2)

            if decision == "quit":
                break

            reviewed += 1

            if decision == "keep":
                kept += 1
                write_event(
                    log_path,
                    ReviewEvent(
                        timestamp=utc_now(),
                        decision="keep",
                        image_path=str(item.image_path),
                        label_path=str(item.label_path),
                        boxes=len(item.boxes),
                        warnings=item.warnings,
                    ),
                )
                continue

            moved_image_path, moved_label_path = remove_item(
                item=item,
                trash_dir=trash_dir,
                mode=remove_mode,
            )
            removed += 1
            write_event(
                log_path,
                ReviewEvent(
                    timestamp=utc_now(),
                    decision=f"remove_{remove_mode}",
                    image_path=str(item.image_path),
                    label_path=str(item.label_path),
                    moved_image_path=str(moved_image_path) if moved_image_path else None,
                    moved_label_path=str(moved_label_path) if moved_label_path else None,
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


def draw_review_image(
    image: Any,
    item: ReviewItem,
    index: int,
    total: int,
    max_width: int,
    max_height: int,
) -> Any:
    cv2 = import_cv2()
    rendered = image.copy()
    height, width = rendered.shape[:2]
    thickness = max(2, round(min(width, height) / 300))
    font_scale = max(0.55, min(width, height) / 1000)

    for cls, normalized_box in item.boxes:
        box = normalized_to_pixels(normalized_box, width, height)
        x_min, y_min, x_max, y_max = box.as_int_tuple()

        if x_max <= x_min or y_max <= y_min:
            continue

        cv2.rectangle(rendered, (x_min, y_min), (x_max, y_max), DEFAULT_COLOR, thickness)
        draw_text(
            rendered,
            f"class:{cls}",
            (x_min, max(24, y_min - 8)),
            scale=font_scale,
            color=DEFAULT_COLOR,
            thickness=thickness,
        )

    status = f"{index}/{total} | boxes: {len(item.boxes)} | {item.image_path.name}"
    draw_text(rendered, status, (14, 30), scale=0.8, color=TEXT_COLOR, thickness=2)

    y = 62
    for line in HELP_LINES:
        draw_text(rendered, line, (14, y), scale=0.65, color=TEXT_COLOR, thickness=2)
        y += 28

    for warning in item.warnings[:3]:
        draw_text(rendered, warning, (14, y), scale=0.6, color=WARNING_COLOR, thickness=2)
        y += 26

    return fit_to_screen(rendered, max_width=max_width, max_height=max_height)


def draw_text(
    image: Any,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2 = import_cv2()
    x, y = origin
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


def fit_to_screen(image: Any, max_width: int, max_height: int) -> Any:
    cv2 = import_cv2()
    height, width = image.shape[:2]

    if width <= max_width and height <= max_height:
        return image

    scale = min(max_width / width, max_height / height)
    target_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def read_decision(cv2: Any) -> Decision:
    while True:
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            return "quit"

        if key in (ord("k"), ord(" "), 13):
            return "keep"

        if key in (ord("d"), 8, 127):
            return "remove"


def remove_item(
    item: ReviewItem,
    trash_dir: Path | None,
    mode: RemoveMode,
) -> tuple[Path | None, Path | None]:
    if mode == "delete":
        unlink_if_exists(item.image_path)
        unlink_if_exists(item.label_path)
        return None, None

    if trash_dir is None:
        raise ValueError("trash_dir is required when remove_mode='move'")

    image_target = move_to_trash(item.image_path, trash_dir / "images")
    label_target = move_to_trash(item.label_path, trash_dir / "labels")
    return image_target, label_target


def move_to_trash(path: Path, target_dir: Path) -> Path | None:
    if not path.exists():
        return None

    target_dir.mkdir(parents=True, exist_ok=True)
    target = unique_path(target_dir / path.name)
    shutil.move(str(path), str(target))
    return target


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Cannot build unique path for {path}")


def unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def write_event(log_path: Path, event: ReviewEvent) -> None:
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively review YOLO annotations and keep or remove image/label pairs."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset directory or data.yaml",
    )
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Dataset split")
    parser.add_argument(
        "--trash-dir",
        type=Path,
        default=None,
        help="Where rejected files are moved",
    )
    parser.add_argument(
        "--remove-mode",
        choices=["move", "delete"],
        default="move",
        help="Move rejected files to trash-dir or delete them permanently",
    )
    parser.add_argument("--hide-empty", action="store_true", help="Skip images without boxes")
    parser.add_argument("--max-width", type=int, default=1600)
    parser.add_argument("--max-height", type=int, default=950)
    parser.add_argument("--start-at", default=None, help="Start from image filename")
    parser.add_argument("--log-path", type=Path, default=None, help="JSONL review log path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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


if __name__ == "__main__":
    main()
