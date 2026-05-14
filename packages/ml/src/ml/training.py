from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd


VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

DEFAULT_DATA_DIR_CANDIDATES = (
    Path('data/Данные'),
    Path('data'),
    Path('src/ml/data/Данные'),
    Path('src/ml/data'),
)

COLUMN_ALIASES = {
    'frame_timestamp_ms': 'frame_timestamp',
    'timestamp': 'frame_timestamp',
    'time_ms': 'frame_timestamp',
    'xmin': 'x_min',
    'x1': 'x_min',
    'left': 'x_min',
    'ymin': 'y_min',
    'y1': 'y_min',
    'top': 'y_min',
    'xmax': 'x_max',
    'x2': 'x_max',
    'right': 'x_max',
    'ymax': 'y_max',
    'y2': 'y_max',
    'bottom': 'y_max',
}


@dataclass(frozen=True)
class YoloDatasetBuildResult:
    '''Результат сборки YOLO-датасета.'''

    output_dir: str
    data_yaml: str
    train_images: int
    val_images: int
    labels: int
    instances: int
    base_frames: int
    propagated_frames: int
    rejected_propagations: int
    empty_labels: int
    report_json: str


def smart_float(value: object, default: float = 0.0) -> float:
    '''Надёжно парсит float из значения CSV.'''

    if value is None:
        return default

    text = str(value).strip()
    if not text:
        return default

    text = text.replace('\u00a0', '').replace(' ', '').replace(',', '.')
    match = re.search(r'-?\d+(?:\.\d+)?', text)
    if match is None:
        return default

    try:
        return float(match.group(0))
    except ValueError:
        return default


def mkdir(path: Path) -> Path:
    '''Создаёт папку и возвращает путь.'''

    path.mkdir(parents=True, exist_ok=True)
    return path


def clip_xyxy(box: list[float] | tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    '''Обрезает bbox по границам изображения.'''

    x1, y1, x2, y2 = map(float, box)

    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))

    return x1, y1, x2, y2


def yolo_line(box: list[float] | tuple[float, float, float, float], width: int, height: int, cls: int = 0) -> str | None:
    '''Преобразует bbox XYXY в строку YOLO label.'''

    x1, y1, x2, y2 = clip_xyxy(box, width, height)

    if x2 <= x1 or y2 <= y1:
        return None

    box_width = x2 - x1
    box_height = y2 - y1

    if box_width < 3 or box_height < 3:
        return None

    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    norm_width = box_width / width
    norm_height = box_height / height

    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < norm_width <= 1 and 0 < norm_height <= 1):
        return None

    return f'{cls} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}'


def is_validation_sample(video_key: str, timestamp_ms: float, val_ratio: float) -> bool:
    '''Детерминированно относит timestamp к train или val.

    Этот режим оставлен для совместимости, но для максимально близкого
    повторения датасета конкурентов лучше использовать split по frame_id.
    '''

    if val_ratio <= 0:
        return False
    if val_ratio >= 1:
        return True

    key = f'{video_key}:{int(round(timestamp_ms))}'.encode('utf-8')
    bucket = int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF

    return bucket < val_ratio


def is_validation_frame_id(frame_id: int, val_ratio: float) -> bool:
    '''Делает split по номеру сгенерированного кадра.

    При `val_ratio=0.2` каждый пятый frame_id попадает в validation.
    Это ближе к фактически наблюдаемому датасету конкурентов: один и тот же
    timestamp может иметь часть propagated-кадров в train и часть в val.
    Такой split менее честный из-за leakage между соседними кадрами, но
    воспроизводит их стабильную локальную validation-картину.
    '''

    if val_ratio <= 0:
        return False
    if val_ratio >= 1:
        return True

    stride = max(1, int(round(1.0 / val_ratio)))
    return frame_id % stride == 0


def template_track(
    prev_frame: object,
    next_frame: object,
    box: list[float] | tuple[float, float, float, float],
    search_pad: int = 80,
) -> tuple[list[float], float]:
    '''Переносит bbox на соседний кадр через простой template matching.

    Это намеренно близко к пайплайну конкурентов:
    без backward-check, edge-density и дополнительных строгих фильтров.
    '''

    x1, y1, x2, y2 = map(int, box)
    height, width = prev_frame.shape[:2]
    x1, y1, x2, y2 = map(int, clip_xyxy([x1, y1, x2, y2], width, height))

    template = cv2.cvtColor(prev_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

    if template.size == 0 or template.shape[0] < 8 or template.shape[1] < 8:
        return list(box), 0.0

    sx1 = max(0, x1 - search_pad)
    sy1 = max(0, y1 - search_pad)
    sx2 = min(width, x2 + search_pad)
    sy2 = min(height, y2 + search_pad)

    search = cv2.cvtColor(next_frame[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2GRAY)

    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return list(box), 0.0

    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_value, _, max_location = cv2.minMaxLoc(result)

    dx, dy = max_location

    new_box = [
        float(sx1 + dx),
        float(sy1 + dy),
        float(sx1 + dx + (x2 - x1)),
        float(sy1 + dy + (y2 - y1)),
    ]

    return new_box, float(max_value)


def read_frame_at_ms_from_cap(cap: cv2.VideoCapture, timestamp_ms: float) -> object | None:
    '''Читает кадр по timestamp в миллисекундах.'''

    cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
    ok, frame = cap.read()

    return frame if ok else None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''Нормализует названия колонок CSV.'''

    rename_map: dict[str, str] = {}

    for column in df.columns:
        clean = str(column).strip()
        lowered = clean.lower()
        rename_map[column] = COLUMN_ALIASES.get(lowered, clean)

    return df.rename(columns=rename_map)


def has_required_columns(df: pd.DataFrame) -> bool:
    '''Проверяет наличие нужных колонок для детекции.'''

    required = {'frame_timestamp', 'x_min', 'y_min', 'x_max', 'y_max'}
    return required.issubset(set(df.columns))


def find_video_for_csv(csv_path: Path) -> Path | None:
    '''Находит видео рядом с CSV.'''

    candidates = [
        csv_path.with_suffix('.mp4'),
        csv_path.with_suffix('.avi'),
        csv_path.with_suffix('.mov'),
        csv_path.with_suffix('.mkv'),
        csv_path.parent / f'{csv_path.parent.name}.mp4',
        csv_path.parent / f'{csv_path.parent.name}.avi',
        csv_path.parent / f'{csv_path.stem}.mp4',
        csv_path.parent / f'{csv_path.stem}.avi',
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    videos = [
        item
        for item in csv_path.parent.iterdir()
        if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if len(videos) == 1:
        return videos[0]

    return None


def discover_csv_files(data_dir: Path) -> list[Path]:
    '''Ищет CSV как в пайплайне конкурентов: только папки первого уровня.'''

    return sorted(data_dir.glob('*/*.csv'))


def guess_data_dir() -> Path:
    '''Автоматически выбирает папку с данными.'''

    for candidate in DEFAULT_DATA_DIR_CANDIDATES:
        if candidate.exists() and discover_csv_files(candidate):
            return candidate

    return Path('data/Данные')


def build_dataset(
    data_dir: Path,
    out_dir: Path,
    propagate: int = 8,
    val_ratio: float = 0.2,
    match_threshold: float = 0.42,
    search_pad: int = 80,
    clean_output: bool = True,
    jpeg_quality: int = 92,
    split_mode: str = 'frame_id_mod',
) -> YoloDatasetBuildResult:
    '''Собирает YOLO-датасет по конкурентскому full-frame pipeline.'''

    if clean_output and out_dir.exists():
        shutil.rmtree(out_dir)

    images_train = mkdir(out_dir / 'images' / 'train')
    images_val = mkdir(out_dir / 'images' / 'val')
    labels_train = mkdir(out_dir / 'labels' / 'train')
    labels_val = mkdir(out_dir / 'labels' / 'val')

    csv_files = discover_csv_files(data_dir)

    if not csv_files:
        print(f'[WARN] No CSV files found in {data_dir.resolve()} by pattern */*.csv')

    frame_id = 0
    train_images = 0
    val_images = 0
    label_files = 0
    instances = 0
    base_frames = 0
    propagated_frames = 0
    rejected_propagations = 0
    empty_labels = 0
    used_csvs: list[str] = []
    skipped_csvs: list[str] = []

    for csv_path in csv_files:
        try:
            df = normalize_columns(pd.read_csv(csv_path))
        except Exception as exc:
            print(f'[WARN] cannot read CSV {csv_path}: {exc}')
            skipped_csvs.append(str(csv_path))
            continue

        if not has_required_columns(df):
            print(f'[WARN] skip CSV without required bbox columns: {csv_path}')
            skipped_csvs.append(str(csv_path))
            continue

        video_path = find_video_for_csv(csv_path)

        if video_path is None:
            print(f'[WARN] no video for {csv_path}')
            skipped_csvs.append(str(csv_path))
            continue

        used_csvs.append(str(csv_path))

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f'[WARN] cannot open video: {video_path}')
            continue

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)

        try:
            for timestamp, group in df.groupby('frame_timestamp'):
                timestamp_ms = smart_float(timestamp)
                base_frame = read_frame_at_ms_from_cap(cap, timestamp_ms)

                if base_frame is None:
                    print(f'[WARN] cannot read frame at {timestamp_ms:.0f} ms from {video_path}')
                    continue

                boxes: list[list[float]] = []

                for _, row in group.iterrows():
                    box = [
                        smart_float(row.get('x_min')),
                        smart_float(row.get('y_min')),
                        smart_float(row.get('x_max')),
                        smart_float(row.get('y_max')),
                    ]
                    boxes.append(box)

                boxes = filter_valid_boxes(boxes, base_frame)

                if not boxes:
                    continue

                frame_variants: list[tuple[object, list[list[float]], int, float]] = [(base_frame, boxes, 0, 1.0)]

                if propagate > 0:
                    for direction in (-1, 1):
                        prev_frame = base_frame
                        prev_boxes = boxes

                        for step in range(1, propagate + 1):
                            neighbor_ts = max(0.0, timestamp_ms + direction * step * 1000.0 / fps)
                            frame = read_frame_at_ms_from_cap(cap, neighbor_ts)

                            if frame is None:
                                rejected_propagations += 1
                                break

                            tracked: list[list[float]] = []
                            scores: list[float] = []

                            for box in prev_boxes:
                                new_box, score = template_track(
                                    prev_frame=prev_frame,
                                    next_frame=frame,
                                    box=box,
                                    search_pad=search_pad,
                                )
                                tracked.append(new_box)
                                scores.append(score)

                            tracked = filter_valid_boxes(tracked, frame)
                            quality = min(scores) if scores else 0.0

                            if scores and tracked and quality >= match_threshold:
                                frame_variants.append((frame, tracked, direction * step, quality))
                                prev_frame = frame
                                prev_boxes = tracked
                            else:
                                rejected_propagations += 1
                                break

                for frame, variant_boxes, offset, quality in frame_variants:
                    height, width = frame.shape[:2]

                    if split_mode == 'timestamp_hash':
                        split_val = is_validation_sample(video_path.stem, timestamp_ms, val_ratio)
                    elif split_mode == 'frame_id_mod':
                        split_val = is_validation_frame_id(frame_id, val_ratio)
                    else:
                        raise ValueError(f'Unknown split_mode: {split_mode}')

                    image_dir = images_val if split_val else images_train
                    label_dir = labels_val if split_val else labels_train

                    name = f'{video_path.stem}_{int(round(timestamp_ms)):08d}_{offset:+03d}_{frame_id:06d}.jpg'

                    labels = [
                        label
                        for box in variant_boxes
                        if (label := yolo_line(box, width, height)) is not None
                    ]

                    labels = deduplicate_yolo_labels(labels)

                    if not labels:
                        empty_labels += 1
                        continue

                    cv2.imwrite(
                        str(image_dir / name),
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                    )

                    (label_dir / name.replace('.jpg', '.txt')).write_text(
                        '\n'.join(labels) + '\n',
                        encoding='utf-8',
                    )

                    if split_val:
                        val_images += 1
                    else:
                        train_images += 1

                    label_files += 1
                    instances += len(labels)
                    base_frames += int(offset == 0)
                    propagated_frames += int(offset != 0)
                    frame_id += 1

        finally:
            cap.release()

    data_yaml = out_dir / 'data.yaml'
    data_yaml.write_text(
        '\n'.join(
            [
                f'path: {out_dir.resolve().as_posix()}',
                'train: images/train',
                'val: images/val',
                'names:',
                '  0: price_tag',
                '',
            ]
        ),
        encoding='utf-8',
    )

    report = {
        'data_dir': str(data_dir.resolve()),
        'out_dir': str(out_dir.resolve()),
        'data_yaml': str(data_yaml.resolve()),
        'used_csvs': used_csvs,
        'skipped_csvs': skipped_csvs,
        'train_images': train_images,
        'val_images': val_images,
        'labels': label_files,
        'instances': instances,
        'base_frames': base_frames,
        'propagated_frames': propagated_frames,
        'rejected_propagations': rejected_propagations,
        'empty_labels': empty_labels,
        'params': {
            'propagate': propagate,
            'val_ratio': val_ratio,
            'match_threshold': match_threshold,
            'search_pad': search_pad,
            'clean_output': clean_output,
            'jpeg_quality': jpeg_quality,
            'split_mode': split_mode,
        },
    }

    report_json = out_dir / 'dataset_build_report.json'
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    result = YoloDatasetBuildResult(
        output_dir=str(out_dir),
        data_yaml=str(data_yaml),
        train_images=train_images,
        val_images=val_images,
        labels=label_files,
        instances=instances,
        base_frames=base_frames,
        propagated_frames=propagated_frames,
        rejected_propagations=rejected_propagations,
        empty_labels=empty_labels,
        report_json=str(report_json),
    )

    print_build_summary(result)
    return result


def build_yolo_dataset(
    data_dir: Path | None = None,
    output_dir: Path = Path('runs/datasets/lenta_yolo_full_prop10'),
    val_fraction: float = 0.2,
    seed: int = 42,
    propagate_frames: int = 8,
    template_match_threshold: float = 0.42,
    template_search_pad: int = 80,
    clean_output: bool = True,
    jpeg_quality: int = 92,
    split_mode: str = 'frame_id_mod',
    **_ignored: object,
) -> YoloDatasetBuildResult:
    '''Совместимый wrapper для старого API `ml.training.build_yolo_dataset`.

    Остальные параметры старого tiled/strict pipeline намеренно игнорируются,
    чтобы сборка шла по конкурентскому full-frame pipeline.
    '''

    del seed

    actual_data_dir = Path(data_dir) if data_dir is not None else guess_data_dir()

    return build_dataset(
        data_dir=actual_data_dir,
        out_dir=Path(output_dir),
        propagate=propagate_frames,
        val_ratio=val_fraction,
        match_threshold=template_match_threshold,
        search_pad=template_search_pad,
        clean_output=clean_output,
        jpeg_quality=jpeg_quality,
        split_mode=split_mode,
    )


def train_yolo_detector(
    data_yaml: Path,
    model: str = 'yolo26n.pt',
    epochs: int = 120,
    imgsz: int = 1280,
    batch: int = 4,
    device: str = '0',
    project: str = 'runs/detect',
    name: str = 'price_tag_yolo',
) -> object:
    '''Запускает Ultralytics YOLO train с умеренными параметрами.'''

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('ultralytics is required to train YOLO') from exc

    detector = YOLO(model)

    return detector.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=4,
        amp=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.05,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        patience=30,
        close_mosaic=10,
        mosaic=0.15,
        mixup=0.0,
        copy_paste=0.0,
        cutmix=0.0,
        erasing=0.0,
        degrees=1,
        translate=0.03,
        scale=0.20,
        shear=0.2,
        perspective=0.0001,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.005,
        hsv_s=0.2,
        hsv_v=0.2,
        seed=42,
        project=project,
        name=name,
    )


def filter_valid_boxes(boxes: list[list[float]], frame: object) -> list[list[float]]:
    '''Убирает некорректные bbox.'''

    height, width = frame.shape[:2]
    valid: list[list[float]] = []

    for box in boxes:
        x1, y1, x2, y2 = clip_xyxy(box, width, height)

        if x2 <= x1 or y2 <= y1:
            continue

        if x2 - x1 < 3 or y2 - y1 < 3:
            continue

        valid.append([x1, y1, x2, y2])

    return valid


def deduplicate_yolo_labels(labels: list[str]) -> list[str]:
    '''Удаляет полностью одинаковые YOLO-строки.'''

    result: list[str] = []
    seen: set[str] = set()

    for label in labels:
        if label in seen:
            continue

        seen.add(label)
        result.append(label)

    return result


def print_build_summary(result: YoloDatasetBuildResult) -> None:
    '''Печатает краткую статистику сборки.'''

    print('[DONE] YOLO dataset built')
    print(f'  output_dir: {result.output_dir}')
    print(f'  data_yaml:  {result.data_yaml}')
    print(f'  train_images: {result.train_images}')
    print(f'  val_images:   {result.val_images}')
    print(f'  label_files:  {result.labels}')
    print(f'  instances:    {result.instances}')
    print(f'  base_frames:        {result.base_frames}')
    print(f'  propagated_frames:  {result.propagated_frames}')
    print(f'  rejected_prop:      {result.rejected_propagations}')
    print(f'  empty_labels:       {result.empty_labels}')
    print(f'  report_json: {result.report_json}')

    if result.train_images == 0:
        print('[WARN] train_images=0. Check --data-dir and CSV/video layout.')

    if result.val_images == 0:
        print('[WARN] val_images=0. Increase --val-ratio or check split.')

    if result.empty_labels > 0:
        print('[WARN] empty labels were skipped; check CSV bbox values if this is unexpected.')


def parse_args() -> argparse.Namespace:
    '''Парсит CLI-аргументы.'''

    parser = argparse.ArgumentParser(
        description='Build YOLO dataset from provided Lenta CSV labels using competitor-style full-frame propagation.'
    )

    parser.add_argument('--data-dir', default=None, help='Папка с подпапками video/csv. Если не указана, используется авто-поиск.')
    parser.add_argument('--out-dir', default='runs/datasets/lenta_yolo_full_prop10', help='Папка результата YOLO dataset.')
    parser.add_argument('--propagate', type=int, default=8, help='Сколько соседних кадров добавлять в каждую сторону.')
    parser.add_argument('--val-ratio', '--val-fraction', dest='val_ratio', type=float, default=0.2)
    parser.add_argument('--match-threshold', '--template-match-threshold', dest='match_threshold', type=float, default=0.42)
    parser.add_argument('--search-pad', '--template-search-pad', dest='search_pad', type=int, default=80)
    parser.add_argument('--jpeg-quality', type=int, default=92)
    parser.add_argument('--split-mode', choices=['frame_id_mod', 'timestamp_hash'], default='frame_id_mod', help='frame_id_mod ближе к датасету конкурентов; timestamp_hash честнее разделяет timestamp-группы.')
    parser.add_argument('--no-clean', action='store_true', help='Не удалять старую папку out-dir перед сборкой.')

    # Совместимость со старым CLI. Эти параметры принимаются, но не используются.
    parser.add_argument('--no-tiled', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--tile-size', type=int, default=640, help=argparse.SUPPRESS)
    parser.add_argument('--tile-stride', type=int, default=512, help=argparse.SUPPRESS)
    parser.add_argument('--min-box-visibility', type=float, default=0.9, help=argparse.SUPPRESS)
    parser.add_argument('--centered-tiles-per-box', type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument('--background-tiles-per-frame', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--include-background-tiles', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--propagate-val', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--no-hash-split', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, default=42, help=argparse.SUPPRESS)

    return parser.parse_args()


def main() -> None:
    '''CLI entrypoint.'''

    args = parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else guess_data_dir()

    build_dataset(
        data_dir=data_dir,
        out_dir=Path(args.out_dir),
        propagate=args.propagate,
        val_ratio=args.val_ratio,
        match_threshold=args.match_threshold,
        search_pad=args.search_pad,
        clean_output=not args.no_clean,
        jpeg_quality=args.jpeg_quality,
        split_mode=args.split_mode,
    )


if __name__ == '__main__':
    main()
