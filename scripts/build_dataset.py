"""
scripts/build_dataset.py — CLI сборки датасета с template propagation.

Обёртка над ml.training.build_yolo_dataset. Дополнительно расширяет датасет:
для каждого размеченного кадра ищет те же bbox в +-N соседних кадрах через
cv2.matchTemplate и добавляет найденные кадры в train-split.

Запуск:
    uv run python scripts/build_dataset.py \\
        --data-dir data/Данные \\
        --out-dir  runs/datasets/lenta_yolo_tiled \\
        --tiled --propagate 8 --val-ratio 0.2

Template propagation:
    Позволяет умножить тренировочные данные без ручной разметки.
    Если bbox после проверки "плывут" — повысить --match-threshold (0.42 → 0.72).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# добавляем корень sol_main/ в sys.path чтобы пакет ml был виден при запуске из scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.training import (
    BBox,
    LabeledSequence,
    bbox_to_yolo,
    build_yolo_dataset,
    dedup_labels,
    discover_sequences,
    iter_tiles,
    parse_float,
    read_csv,
    row_to_bbox,
    safe_name,
    write_full_frame,
    write_tiled_frame,
)


def propagate_frame(
    cv2,
    base_frame,
    base_bboxes: list[BBox],
    video_path: Path,
    base_ts_ms: int,
    fps: float,
    n_frames: int,
    threshold: float,
    search_pad: int,
) -> list[tuple[object, list[BBox]]]:
    """
    Найти каждый bbox из base_frame в +-n_frames соседних кадрах через template matching.

    Для каждого кадра-соседа:
      1. Вырезаем шаблон (grayscale) из base_frame по bbox
      2. В соседнем кадре ищем шаблон в области bbox +- search_pad пикселей
      3. Если cv2.TM_CCOEFF_NORMED score >= threshold — сохраняем найденный bbox

    Возвращает список (кадр, список найденных bbox) для кадров где нашли хотя бы 1 bbox.

    Оптимизация seek: вместо 2*n_frames отдельных cap.set() делаем один seek к началу
    окна и читаем кадры последовательно. Для H.264 это на порядок быстрее, т.к. каждый
    random seek требует декодирования с ближайшего I-frame.
    """
    import numpy as np  # noqa: F401 — нужен для cv2.matchTemplate внутри

    results = []
    step_ms = 1000.0 / max(fps, 1.0)
    end_ms = base_ts_ms + n_frames * step_ms

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return results

    # один seek к началу окна; дальше читаем последовательно
    start_ms = max(0.0, base_ts_ms - n_frames * step_ms)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)

    gray_base = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)

    while True:
        frame_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_ms > end_ms + step_ms:
            break

        # пропускаем базовый кадр (совпадает по времени с base_ts_ms)
        if abs(frame_ms - base_ts_ms) < step_ms * 0.5:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = frame.shape[:2]
        matched = []

        for bbox in base_bboxes:
            x1, y1, x2, y2 = (
                int(bbox.x_min),
                int(bbox.y_min),
                int(bbox.x_max),
                int(bbox.y_max),
            )
            tmpl = gray_base[y1:y2, x1:x2]
            if tmpl.size == 0:
                continue

            # область поиска = bbox +- search_pad, ограниченная размерами кадра
            sx1 = max(0, x1 - search_pad)
            sy1 = max(0, y1 - search_pad)
            sx2 = min(fw, x2 + search_pad)
            sy2 = min(fh, y2 + search_pad)
            region = gray_frame[sy1:sy2, sx1:sx2]
            if region.shape[0] < tmpl.shape[0] or region.shape[1] < tmpl.shape[1]:
                continue

            res = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            if score < threshold:
                continue

            # loc — координаты левого верхнего угла совпадения внутри region;
            # переводим обратно в координаты полного кадра
            mx, my = loc[0] + sx1, loc[1] + sy1
            matched.append(
                BBox(float(mx), float(my), float(mx + (x2 - x1)), float(my + (y2 - y1)))
            )

        if matched:
            results.append((frame.copy(), matched))

    cap.release()
    return results


def build_with_propagation(
    data_dir: Path,
    output_dir: Path,
    val_fraction: float,
    seed: int,
    tiled: bool,
    tile_size: int,
    tile_stride: int,
    min_visibility: float,
    n_propagate: int,
    match_threshold: float,
    search_pad: int,
) -> None:
    """
    Собрать датасет и опционально расширить его через template propagation.

    Фаза 1: базовый датасет из размеченных кадров (только GT timestamp'ы из CSV).
    Фаза 2: для каждого размеченного кадра ищем те же bbox в +-n_propagate соседних
        кадрах; найденные кадры добавляются только в train (val не трогаем).
    """
    # фаза 1: базовый датасет — только размеченные кадры
    result = build_yolo_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        val_fraction=val_fraction,
        seed=seed,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
        min_box_visibility=min_visibility,
    )
    print(
        f"\n[base] train={result['train_images']}  val={result['val_images']}"
        f"  labels={result['total_labels']}"
    )

    if n_propagate == 0:
        return

    # фаза 2: template propagation в соседние кадры (только train split)
    try:
        import cv2
    except ImportError:
        print("[warn] opencv не найден, propagation пропущен")
        return

    img_dir = output_dir / "images" / "train"
    lbl_dir = output_dir / "labels" / "train"
    prop_imgs, prop_lbls = 0, 0

    for seq in discover_sequences(data_dir):
        cap = cv2.VideoCapture(str(seq.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        by_ts: dict[int, list] = {}
        for row in read_csv(seq.csv_path):
            ts = int(parse_float(row.get("frame_timestamp", "0")))
            by_ts.setdefault(ts, []).append(row)

        print(f"[prop] {seq.name}: {len(by_ts)} timestamp(s), fps={fps:.1f}")

        for ts_ms, rows in by_ts.items():
            print(f"  ts={ts_ms}ms ...", end=" ", flush=True)
            # загружаем базовый (размеченный) кадр
            cap2 = cv2.VideoCapture(str(seq.video_path))
            cap2.set(cv2.CAP_PROP_POS_MSEC, ts_ms)
            ok, base_frame = cap2.read()
            cap2.release()
            if not ok or base_frame is None:
                print("пропущен (кадр не прочитан)")
                continue

            fh, fw = base_frame.shape[:2]
            bboxes = [b for row in rows if (b := row_to_bbox(row, fw, fh))]
            if not bboxes:
                print("пропущен (нет bbox)")
                continue

            prop_frames = propagate_frame(
                cv2,
                base_frame,
                bboxes,
                seq.video_path,
                ts_ms,
                fps,
                n_propagate,
                match_threshold,
                search_pad,
            )

            print(f"{len(prop_frames)} кадров")

            for offset_idx, (frame, matched_bboxes) in enumerate(prop_frames):
                stem = safe_name(f"{seq.name}_{ts_ms:08d}_prop{offset_idx:02d}")

                # конвертируем BBox обратно в dict-строки чтобы переиспользовать write_*
                fake_rows = [
                    {
                        "x_min": str(b.x_min),
                        "y_min": str(b.y_min),
                        "x_max": str(b.x_max),
                        "y_max": str(b.y_max),
                    }
                    for b in matched_bboxes
                ]

                if tiled:
                    ni, nl = write_tiled_frame(
                        cv2,
                        frame,
                        fake_rows,
                        img_dir,
                        lbl_dir,
                        stem,
                        tile_size,
                        tile_stride,
                        min_visibility,
                    )
                else:
                    ni, nl = write_full_frame(
                        cv2,
                        frame,
                        fake_rows,
                        img_dir / f"{stem}.jpg",
                        lbl_dir / f"{stem}.txt",
                    )
                prop_imgs += ni
                prop_lbls += nl

    print(
        f"[propagation] добавлено в train: {prop_imgs} изображений, {prop_lbls} меток"
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Собрать YOLO-датасет с опциональным template propagation"
    )
    p.add_argument("--data-dir", required=True, help="Папка с подпапками .mp4 + .csv")
    p.add_argument("--out-dir", required=True, help="Куда писать датасет")
    p.add_argument("--tiled", action="store_true", help="Тайловый режим 640px")
    p.add_argument("--tile-size", type=int, default=640)
    p.add_argument("--tile-stride", type=int, default=512)
    p.add_argument("--min-visibility", type=float, default=0.25)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--propagate",
        type=int,
        default=0,
        help="Расширить разметку на +-N соседних кадров через template matching (0=выкл)",
    )
    p.add_argument(
        "--match-threshold",
        type=float,
        default=0.42,
        help="Порог cv2.matchTemplate: 0.42 лояльный, 0.72 строгий (если bbox плывут)",
    )
    p.add_argument(
        "--search-pad",
        type=int,
        default=80,
        help="Радиус поиска в пикселях вокруг bbox при template matching",
    )
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: --data-dir not found: {data_dir}")
        return 1

    print(
        f"[build] data={data_dir}  out={args.out_dir}"
        f"  tiled={args.tiled}  propagate={args.propagate}"
    )

    build_with_propagation(
        data_dir=data_dir,
        output_dir=Path(args.out_dir),
        val_fraction=args.val_ratio,
        seed=args.seed,
        tiled=args.tiled,
        tile_size=args.tile_size,
        tile_stride=args.tile_stride,
        min_visibility=args.min_visibility,
        n_propagate=args.propagate,
        match_threshold=args.match_threshold,
        search_pad=args.search_pad,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
