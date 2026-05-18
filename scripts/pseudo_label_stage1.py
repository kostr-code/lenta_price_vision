"""
scripts/pseudo_label_stage1.py — псевдолейблинг Stage 1 на unlabeled видео.

Прогоняет обученный детектор ценников (Stage 1) по кадрам unlabeled видео,
сохраняет кадры с детекциями как YOLO датасет для последующей проверки в CVAT.

Рабочий процесс:
    1. just dataset-build        ← базовый датасет из labeled CSV
    2. just train-yolo1          ← обучить Stage 1
    3. just pseudo-label-stage1  ← NEW: прогнать по unlabeled видео
    4. Проверить just dataset-review --dataset <out-dir>
    5. Добавить хорошие кадры в основной датасет и переобучить

Запуск:
    uv run python scripts/pseudo_label_stage1.py \\
        --video-dir data/Данные/Unlabeled \\
        --weights   models/price_tag_yolo.pt \\
        --out-dir   runs/pseudo/stage1_unlabeled \\
        --sample-every 25 --conf 0.35

Выход:
    runs/pseudo/stage1_unlabeled/
      images/train/*.jpg    кадры с детекциями
      labels/train/*.txt    YOLO-метки (class cx cy w h)
      data.yaml
      manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _resolve_weights(model_path: Path) -> Path:
    """Принять .pt файл или папку runs/detect/<name>/; вернуть weights/best.pt."""
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


def _collect_videos(video_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in video_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def pseudo_label_videos(
    video_dir: Path,
    weights: Path,
    out_dir: Path,
    sample_every: int,
    conf: float,
    imgsz: int,
    device: str,
    split: str,
    min_detections: int,
    max_frames: int,
) -> None:
    """Прогнать Stage 1 YOLO по кадрам видео и сохранить результат как YOLO датасет."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python required: uv add opencv-python")
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics required: uv add ultralytics")

    videos = _collect_videos(video_dir)
    if not videos:
        raise SystemExit(f"Видео не найдены в: {video_dir}")

    img_dir = out_dir / "images" / split
    lbl_dir = out_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    predict_kw: dict = {"conf": conf, "imgsz": imgsz, "verbose": False}
    if device:
        predict_kw["device"] = device

    records: list[dict] = []
    saved = 0

    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  [skip] не открывается: {video_path.name}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        video_stem = video_path.stem
        frame_idx = 0
        saved_this_video = 0

        print(f"[video] {video_path.name}  ({total_frames} кадров)")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % sample_every != 0:
                frame_idx += 1
                continue

            results = model.predict(source=frame, **predict_kw)
            boxes = results[0].boxes
            detections = boxes.xyxy.cpu().numpy() if boxes is not None else []
            classes = boxes.cls.cpu().numpy().astype(int) if boxes is not None else []

            if len(detections) < min_detections:
                frame_idx += 1
                continue

            # имя файла: <video_stem>_f<frame_idx:07d>
            stem = f"{video_stem}_f{frame_idx:07d}"
            img_path = img_dir / f"{stem}.jpg"
            lbl_path = lbl_dir / f"{stem}.txt"

            cv2.imwrite(str(img_path), frame)

            h, w = frame.shape[:2]
            label_lines: list[str] = []
            for cls, (x1, y1, x2, y2) in zip(classes, detections):
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            lbl_path.write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8"
            )

            records.append(
                {
                    "split": split,
                    "video": video_path.name,
                    "frame": frame_idx,
                    "image": img_path.name,
                    "detections": len(detections),
                }
            )
            saved += 1
            saved_this_video += 1

            if max_frames and saved >= max_frames:
                break
            frame_idx += 1

        cap.release()
        print(f"  сохранено кадров: {saved_this_video}")
        if max_frames and saved >= max_frames:
            print(f"[info] достигнут лимит --max-frames {max_frames}, остановка")
            break

    # data.yaml
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: price_tag\n",
        encoding="utf-8",
    )

    # manifest
    manifest = out_dir / "manifest.csv"
    if records:
        with manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["split", "video", "frame", "image", "detections"]
            )
            writer.writeheader()
            writer.writerows(records)

    print(f"\n[done] сохранено {saved} кадров → {out_dir}")
    print(f"       manifest: {manifest}")
    print(f"       data.yaml: {out_dir / 'data.yaml'}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Псевдолейблинг Stage 1: прогнать детектор ценников по unlabeled видео"
    )
    p.add_argument(
        "--video-dir",
        type=Path,
        required=True,
        help="Папка с unlabeled .mp4 (Данные/Unlabeled или другая)",
    )
    p.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Веса Stage 1: models/price_tag_yolo.pt или папка runs/detect/<name>",
    )
    p.add_argument(
        "--out-dir", type=Path, required=True, help="Куда писать псевдолейблы"
    )
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument(
        "--sample-every",
        type=int,
        default=25,
        help="Брать каждый N-й кадр (25 ≈ 1fps при 25fps видео)",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Порог уверенности детектора (выше → меньше шума)",
    )
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="0")
    p.add_argument(
        "--min-detections",
        type=int,
        default=1,
        help="Пропускать кадры с меньшим числом детекций",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Лимит сохранённых кадров (0 = без лимита)",
    )
    p.add_argument(
        "--clear-output", action="store_true", help="Очистить out-dir перед записью"
    )
    args = p.parse_args()

    if not args.video_dir.exists():
        print(f"Error: --video-dir не найдена: {args.video_dir}")
        return 1

    weights = _resolve_weights(args.weights)
    print(f"[stage1-pseudo] видео: {args.video_dir}")
    print(f"                веса:  {weights}")
    print(f"                выход: {args.out_dir}")
    print(f"                sample_every={args.sample_every}  conf={args.conf}")

    if args.clear_output and args.out_dir.exists():
        shutil.rmtree(args.out_dir)

    pseudo_label_videos(
        video_dir=args.video_dir,
        weights=weights,
        out_dir=args.out_dir,
        sample_every=args.sample_every,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        split=args.split,
        min_detections=args.min_detections,
        max_frames=args.max_frames,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
