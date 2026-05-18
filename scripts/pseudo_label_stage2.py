"""
scripts/pseudo_label_stage2.py — псевдолейблинг Stage 2 на кропах ценников.

Берёт кропы из crops_for_stage2/ (выход crop_detections.py), прогоняет
обученный детектор внутренних элементов (Stage 2) и сохраняет YOLO-метки
плюс preview-изображения с bbox для визуальной проверки.

Рабочий процесс:
    1. just crop-for-stage2      ← нарезать кропы ценников Stage 1-детектором
    2. Разметить часть кропов в CVAT вручную
    3. just train-yolo2          ← первое обучение Stage 2
    4. just save-yolo2           ← models/inside_price_tag_yolo.pt
    5. just pseudo-label-stage2  ← NEW: прогнать Stage 2 по оставшимся кропам
    6. Проверить аннотации: just dataset-review --dataset <out-dir>
    7. Исправить плохие детекции в CVAT → переобучить Stage 2

Запуск:
    uv run python scripts/pseudo_label_stage2.py \\
        --source  runs/datasets/lenta_yolo/crops_for_stage2 \\
        --weights models/inside_price_tag_yolo.pt \\
        --out-dir runs/pseudo/stage2_inside

Примечание о повороте:
    В оригинальном пайплайне коллеги кропы снимались с неповёрнутого видео
    и требовали поворота 90° CCW перед инференсом. В нашем пайплайне кропы
    уже правильно ориентированы — флаг --rotate-ccw по умолчанию выключен.

Выход:
    <out-dir>/
      images/train/*.jpg     кропы (повёрнутые если --rotate-ccw)
      labels/train/*.txt     YOLO-метки внутренних элементов
      annotated/train/*.jpg  preview с нарисованными bbox
      data.yaml
      manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# цветовая палитра для классов (BGR для cv2)
_CLASS_PALETTE = [
    (70, 57, 230),
    (87, 53, 29),
    (143, 157, 42),
    (97, 162, 244),
    (236, 56, 131),
    (3, 183, 255),
    (188, 158, 33),
    (40, 40, 214),
    (78, 153, 106),
    (7, 86, 251),
    (238, 97, 67),
    (0, 0, 0),
]


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


def _resolve_weights(model_path: Path) -> Path:
    """
    Принять .pt файл или папку runs/detect/<name>/; вернуть weights/best.pt.
    """
    if model_path.is_file():
        return model_path
    for candidate in [
        model_path / "weights" / "best.pt",
        model_path / "best.pt",
        model_path / "weights" / "last.pt",
    ]:
        if candidate.exists():
            return candidate
    raise SystemExit(f"Не найдены веса YOLO: {model_path}")


def _normalize_names(names: object) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def _safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_.") or "image"


def _color(class_id: int) -> tuple[int, int, int]:
    return _CLASS_PALETTE[class_id % len(_CLASS_PALETTE)]


def _draw_detections(cv2, image, detections: list[Detection]):
    """Нарисовать bbox + подпись класса поверх изображения."""
    h, w = image.shape[:2]
    thickness = max(2, min(w, h) // 180)
    font_scale = max(0.4, min(w, h) / 900)
    out = image.copy()
    for det in detections:
        color = _color(det.class_id)
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{det.class_name} {det.confidence:.2f}"
        ty = max(y1 - 4, 14)
        cv2.putText(
            out,
            label,
            (x1, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return out


def _yolo_line(det: Detection, width: int, height: int) -> str:
    x1 = max(0.0, min(float(width), det.x1))
    y1 = max(0.0, min(float(height), det.y1))
    x2 = max(0.0, min(float(width), det.x2))
    y2 = max(0.0, min(float(height), det.y2))
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{det.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def pseudo_label_crops(
    source: Path,
    weights: Path,
    out_dir: Path,
    splits: list[str],
    conf: float,
    imgsz: int,
    device: str | None,
    rotate_ccw: bool,
    max_images: int,
) -> None:
    """Прогнать Stage 2 YOLO по кропам и сохранить YOLO-метки + аннотированные preview."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python required: uv add opencv-python")
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics required: uv add ultralytics")

    model = YOLO(str(weights))
    class_names = _normalize_names(model.names)
    predict_kw: dict = {"conf": conf, "imgsz": imgsz, "verbose": False}
    if device:
        predict_kw["device"] = device

    # создаём папки
    for folder in ("images", "labels", "annotated"):
        for split in splits:
            (out_dir / folder / split).mkdir(parents=True, exist_ok=True)

    # data.yaml
    names_yaml = "\n".join(f"  {i}: {n}" for i, n in sorted(class_names.items()))
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\n"
        f"names:\n{names_yaml}\n",
        encoding="utf-8",
    )

    manifest_path = out_dir / "manifest.csv"
    manifest_fields = [
        "split",
        "source_file",
        "image",
        "label_file",
        "annotated_image",
        "detections",
        "class_ids",
    ]

    processed = 0
    total_detections = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=manifest_fields)
        writer.writeheader()

        for split in splits:
            split_dir = source / split
            if not split_dir.exists():
                print(f"  [skip] не найдена папка: {split_dir}")
                continue

            images = sorted(
                p
                for p in split_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if max_images:
                images = images[:max_images]

            print(f"[stage2-pseudo] {split}: {len(images)} изображений")

            for image_path in images:
                # читаем и при необходимости поворачиваем
                frame = cv2.imread(str(image_path))
                if frame is None:
                    continue
                if rotate_ccw:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                h, w = frame.shape[:2]
                stem = _safe_name(image_path.stem)

                img_out = out_dir / "images" / split / f"{stem}.jpg"
                lbl_out = out_dir / "labels" / split / f"{stem}.txt"
                ann_out = out_dir / "annotated" / split / f"{stem}.jpg"

                cv2.imwrite(str(img_out), frame)

                results = model.predict(source=frame, **predict_kw)
                boxes = results[0].boxes
                detections: list[Detection] = []

                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls.item()) if box.cls is not None else 0
                        confidence = (
                            float(box.conf.item()) if box.conf is not None else 0.0
                        )
                        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                        detections.append(
                            Detection(
                                class_id=cls_id,
                                class_name=class_names.get(cls_id, str(cls_id)),
                                confidence=confidence,
                                x1=x1,
                                y1=y1,
                                x2=x2,
                                y2=y2,
                            )
                        )

                label_lines = [_yolo_line(d, w, h) for d in detections]
                lbl_out.write_text(
                    "\n".join(label_lines) + ("\n" if label_lines else ""),
                    encoding="utf-8",
                )

                cv2.imwrite(str(ann_out), _draw_detections(cv2, frame, detections))

                processed += 1
                total_detections += len(detections)

                writer.writerow(
                    {
                        "split": split,
                        "source_file": str(image_path),
                        "image": str(img_out),
                        "label_file": str(lbl_out),
                        "annotated_image": str(ann_out),
                        "detections": len(detections),
                        "class_ids": " ".join(str(d.class_id) for d in detections),
                    }
                )

                print(
                    f"  {processed}: {len(detections)} детекций | {split}/{image_path.name}"
                )

    print(f"\n[done]  обработано: {processed} изображений")
    print(f"        всего детекций: {total_detections}")
    print(f"        выход: {out_dir}")
    print(f"        manifest: {manifest_path}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Псевдолейблинг Stage 2: прогнать inside-детектор по кропам ценников"
    )
    p.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Папка с кропами (выход crop_detections.py: содержит train/, val/)",
    )
    p.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Веса Stage 2: models/inside_price_tag_yolo.pt или папка runs/detect/<name>",
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Куда писать результат")
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Сплиты внутри --source для обработки",
    )
    p.add_argument(
        "--conf", type=float, default=0.25, help="Порог уверенности детектора"
    )
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--rotate-ccw",
        action="store_true",
        help="Повернуть кропы 90° CCW перед инференсом (если кропы с неповёрнутого видео)",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Лимит изображений на сплит (0 = без лимита)",
    )
    p.add_argument(
        "--clear-output", action="store_true", help="Очистить out-dir перед записью"
    )
    args = p.parse_args()

    if not args.source.exists():
        print(f"Error: --source не найден: {args.source}")
        return 1

    weights = _resolve_weights(args.weights)
    print(f"[stage2-pseudo] кропы:  {args.source}")
    print(f"                веса:   {weights}")
    print(f"                выход:  {args.out_dir}")
    print(
        f"                conf={args.conf}  imgsz={args.imgsz}  rotate_ccw={args.rotate_ccw}"
    )

    if args.clear_output and args.out_dir.exists():
        shutil.rmtree(args.out_dir)

    pseudo_label_crops(
        source=args.source,
        weights=weights,
        out_dir=args.out_dir,
        splits=args.splits,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        rotate_ccw=args.rotate_ccw,
        max_images=args.max_images,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
