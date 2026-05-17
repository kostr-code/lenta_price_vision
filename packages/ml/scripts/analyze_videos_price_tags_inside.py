from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
PRESET_SAMPLE_FPS = {"fast": 1.0, "balanced": 2.0, "accuracy": 4.0}


@dataclass(frozen=True)
class Box:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int | None = None


@dataclass(frozen=True)
class CropCandidate:
    video: str
    video_stem: str
    track_key: str
    track_id: int | None
    frame_index: int
    timestamp_ms: float
    crop_index: int
    price_confidence: float
    bbox: tuple[int, int, int, int]
    area: int
    sharpness: float
    score: float
    crop_path: Path
    base_name: str


@dataclass
class TrackState:
    video: str
    video_stem: str
    track_key: str
    track_id: int | None
    detections_seen: int = 0
    first_frame_index: int | None = None
    last_frame_index: int | None = None
    first_timestamp_ms: float | None = None
    last_timestamp_ms: float | None = None
    top_crops: list[CropCandidate] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect price tags in videos, keep top-K track-level crops, then run an "
            "inside-price-tag model over selected crops and save structured results."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(r"F:\lenta_price_vision\packages\ml\src\ml\data\Unlabeled"),
        help="Video file or folder with videos.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"F:\lenta_price_vision\packages\ml\outputs\video_price_inside_analysis"),
        help="Base output folder. A per-run subfolder is created inside it by default.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run folder name inside --out-dir. Defaults to a timestamped name.",
    )
    parser.add_argument(
        "--no-run-subdir",
        action="store_true",
        help="Write directly into --out-dir instead of creating a per-run subfolder.",
    )
    parser.add_argument(
        "--price-model",
        type=Path,
        default=Path(r"F:\lenta_price_vision\models\best.pt"),
        help="Outer price-tag detector weights.",
    )
    parser.add_argument(
        "--inside-model",
        type=Path,
        default=Path(
            r"F:\lenta_price_vision\packages\ml\runs\inside"
            r"\inside_price_tag_yolo_60_finetune-2\weights\best.pt"
        ),
        help="Inside-price-tag detector weights.",
    )
    parser.add_argument(
        "--tracker-config",
        type=Path,
        default=Path(r"F:\lenta_price_vision\packages\ml\configs\bytetrack_price.yaml"),
        help="ByteTrack config for --price-detector-mode track.",
    )
    parser.add_argument(
        "--price-detector-mode",
        choices=["track", "full", "tiled"],
        default="track",
        help=(
            "track = YOLO + ByteTrack and top-K per track; "
            "full = YOLO predict on full frame; tiled = tiled full-frame detection."
        ),
    )
    parser.add_argument("--price-conf", type=float, default=0.25)
    parser.add_argument("--inside-conf", type=float, default=0.25)
    parser.add_argument("--price-imgsz", type=int, default=1280)
    parser.add_argument("--inside-imgsz", type=int, default=960)
    parser.add_argument("--device", default=None, help='Device for YOLO, for example "0" or "cpu".')
    parser.add_argument(
        "--preset",
        choices=["custom", "fast", "balanced", "accuracy"],
        default="custom",
        help="Sampling preset. Used only when --sample-fps is not set.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth frame. Ignored when --sample-fps or non-custom --preset is set.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        help="Process roughly this many frames per second from each video.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Optional total processed-frame cap.")
    parser.add_argument(
        "--skip-blurry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip frames whose Laplacian sharpness is below --min-laplacian-var.",
    )
    parser.add_argument(
        "--min-laplacian-var",
        type=float,
        default=35.0,
        help="Minimum frame sharpness when --skip-blurry is enabled.",
    )
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.04,
        help="Extra crop padding as a fraction of the detected price-tag bbox size.",
    )
    parser.add_argument(
        "--top-k-crops-per-track",
        type=int,
        default=5,
        help="Keep only top-K crops per track by area * sharpness * confidence. Use 0 for all.",
    )
    parser.add_argument(
        "--min-track-hits",
        type=int,
        default=2,
        help="Skip tracked price tags seen fewer than this many times. Ignored in full/tiled modes.",
    )
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--tile-stride", type=int, default=512)
    parser.add_argument("--tile-nms-iou", type=float, default=0.40)
    parser.add_argument(
        "--rotate-crops-ccw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rotate crops 90 degrees counter-clockwise before inside analysis.",
    )
    parser.add_argument(
        "--inside-dedupe-iou",
        type=float,
        default=0.75,
        help="Suppress lower-confidence inside boxes when IoU is above this value. Use 0 to disable.",
    )
    parser.add_argument(
        "--inside-class-aware-dedupe",
        action="store_true",
        help="When deduping inside boxes, compare only boxes of the same class.",
    )
    parser.add_argument(
        "--save-frame-preview",
        action="store_true",
        help="Save full video frames with detected price-tag boxes.",
    )
    parser.add_argument(
        "--save-track-debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-track JSON summaries.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete output folder before writing new results.",
    )
    return parser.parse_args()


def import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("opencv-python is required to process videos.") from exc
    return cv2


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_.") or "item"


def normalize_names(names: object) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def iter_videos(source: Path) -> list[Path]:
    source = source.resolve()
    if source.is_file():
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise SystemExit(f"Not a supported video file: {source}")
        return [source]
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def requested_sample_fps(args: argparse.Namespace) -> float | None:
    if args.sample_fps is not None:
        return args.sample_fps
    if args.preset != "custom":
        return PRESET_SAMPLE_FPS[args.preset]
    return None


def frame_step_for_video(video_fps: float, frame_stride: int, sample_fps: float | None) -> int:
    if sample_fps is not None and sample_fps > 0 and video_fps > 0:
        return max(1, int(round(video_fps / sample_fps)))
    return max(1, frame_stride)


def box_area_xyxy(box: tuple[int, int, int, int] | tuple[float, float, float, float]) -> int:
    x1, y1, x2, y2 = box
    return int(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def expand_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = box_width * padding_ratio
    pad_y = box_height * padding_ratio
    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(width, int(round(x2 + pad_x)))
    bottom = min(height, int(round(y2 + pad_y)))
    return left, top, right, bottom


def laplacian_sharpness(image: Image.Image, cv2: Any) -> float:
    if image.size[0] <= 0 or image.size[1] <= 0:
        return 0.0
    array = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_quality_score(area: int, sharpness: float, detector_confidence: float) -> float:
    return float(area) * float(sharpness) * max(0.25, float(detector_confidence))


def box_iou(left: Box, right: Box) -> float:
    inter_x1 = max(left.x1, right.x1)
    inter_y1 = max(left.y1, right.y1)
    inter_x2 = min(left.x2, right.x2)
    inter_y2 = min(left.y2, right.y2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter_area <= 0:
        return 0.0
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - inter_area
    return inter_area / union if union > 0 else 0.0


def dedupe_boxes(boxes: list[Box], iou_threshold: float, class_aware: bool) -> list[Box]:
    if iou_threshold <= 0:
        return boxes
    kept: list[Box] = []
    for candidate in sorted(boxes, key=lambda item: item.confidence, reverse=True):
        should_keep = True
        for kept_box in kept:
            if class_aware and candidate.class_id != kept_box.class_id:
                continue
            if box_iou(candidate, kept_box) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            kept.append(candidate)
    return sorted(kept, key=lambda item: (item.y1, item.x1, -item.confidence))


def result_boxes(result: Any, class_names: dict[int, str]) -> list[Box]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    detections: list[Box] = []
    for box in boxes:
        class_id = int(box.cls.item()) if box.cls is not None else 0
        confidence = float(box.conf.item()) if box.conf is not None else 0.0
        track_id = None
        if getattr(box, "id", None) is not None:
            track_id = int(box.id.item())
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        detections.append(
            Box(
                class_id=class_id,
                class_name=class_names.get(class_id, str(class_id)),
                confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                track_id=track_id,
            )
        )
    return detections


def tile_origins(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    origins = list(range(0, max(1, length - tile_size + 1), max(1, stride)))
    last_origin = length - tile_size
    if origins[-1] != last_origin:
        origins.append(last_origin)
    return origins


def predict(model: YOLO, image: Image.Image, **kwargs: Any) -> Any:
    return model.predict(source=np.asarray(image), **kwargs)[0]


def detect_tiled(
    model: YOLO,
    image: Image.Image,
    class_names: dict[int, str],
    predict_kwargs: dict[str, Any],
    tile_size: int,
    tile_stride: int,
    nms_iou: float,
) -> list[Box]:
    width, height = image.size
    boxes: list[Box] = []

    for top in tile_origins(height, tile_size, tile_stride):
        for left in tile_origins(width, tile_size, tile_stride):
            right = min(width, left + tile_size)
            bottom = min(height, top + tile_size)
            tile = image.crop((left, top, right, bottom))
            result = predict(model, tile, **predict_kwargs)
            for box in result_boxes(result, class_names):
                boxes.append(
                    Box(
                        class_id=box.class_id,
                        class_name=box.class_name,
                        confidence=box.confidence,
                        x1=box.x1 + left,
                        y1=box.y1 + top,
                        x2=box.x2 + left,
                        y2=box.y2 + top,
                    )
                )

    return dedupe_boxes(boxes, iou_threshold=nms_iou, class_aware=False)


def detect_price_boxes(
    model: YOLO,
    image: Image.Image,
    class_names: dict[int, str],
    args: argparse.Namespace,
    predict_kwargs: dict[str, Any],
    tracker_config: Path,
    persist_tracks: bool,
) -> list[Box]:
    if args.price_detector_mode == "track":
        result = model.track(
            source=np.asarray(image),
            persist=persist_tracks,
            tracker=str(tracker_config),
            **predict_kwargs,
        )[0]
        return result_boxes(result, class_names)

    if args.price_detector_mode == "tiled":
        return detect_tiled(
            model=model,
            image=image,
            class_names=class_names,
            predict_kwargs=predict_kwargs,
            tile_size=args.tile_size,
            tile_stride=args.tile_stride,
            nms_iou=args.tile_nms_iou,
        )

    result = predict(model, image, **predict_kwargs)
    return result_boxes(result, class_names)


def add_crop_candidate(track: TrackState, candidate: CropCandidate, top_k: int) -> None:
    track.detections_seen += 1
    if track.first_frame_index is None:
        track.first_frame_index = candidate.frame_index
        track.first_timestamp_ms = candidate.timestamp_ms
    track.last_frame_index = candidate.frame_index
    track.last_timestamp_ms = candidate.timestamp_ms
    track.top_crops.append(candidate)
    track.top_crops.sort(key=lambda item: item.score, reverse=True)
    if top_k > 0:
        del track.top_crops[top_k:]


def yolo_line(box: Box, width: int, height: int) -> str:
    x1 = max(0.0, min(float(width), box.x1))
    y1 = max(0.0, min(float(height), box.y1))
    x2 = max(0.0, min(float(width), box.x2))
    y2 = max(0.0, min(float(height), box.y2))
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    norm_width = box_width / width
    norm_height = box_height / height
    return f"{box.class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"


def color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = [
        (230, 57, 70),
        (29, 53, 87),
        (42, 157, 143),
        (244, 162, 97),
        (131, 56, 236),
        (255, 183, 3),
        (33, 158, 188),
        (214, 40, 40),
        (106, 153, 78),
        (251, 86, 7),
        (67, 97, 238),
        (0, 0, 0),
    ]
    return palette[class_id % len(palette)]


def draw_boxes(image: Image.Image, boxes: list[Box], include_conf: bool = True) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    line_width = max(2, min(output.size) // 180)

    for box in boxes:
        color = color_for_class(box.class_id)
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=color, width=line_width)
        label = box.class_name
        if box.track_id is not None:
            label = f"{label} id{box.track_id}"
        if include_conf:
            label = f"{label} {box.confidence:.2f}"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_x = max(0, int(box.x1))
        text_y = max(0, int(box.y1) - text_height - 4)
        if text_y <= 0:
            text_y = min(output.height - text_height - 2, int(box.y1) + 4)
        draw.rectangle([text_x, text_y, text_x + text_width + 6, text_y + text_height + 4], fill=color)
        draw.text((text_x + 3, text_y + 2), label, fill=(255, 255, 255), font=font)

    return output


def write_data_yaml(output_dir: Path, class_names: dict[int, str]) -> None:
    lines = [
        f"path: {output_dir.as_posix()}",
        "train: inside_images",
        "val: inside_images",
        "names:",
    ]
    for class_id in sorted(class_names):
        lines.append(f"  {class_id}: {class_names[class_id]}")
    (output_dir / "inside_data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_dirs(out_dir: Path) -> dict[str, Path]:
    dirs = {
        "candidate_crops": out_dir / "price_crop_candidates",
        "crops": out_dir / "price_crops",
        "inside_images": out_dir / "inside_images",
        "inside_labels": out_dir / "inside_labels",
        "inside_annotated": out_dir / "inside_annotated",
        "inside_single": out_dir / "inside_annotated_single",
        "frame_preview": out_dir / "frame_price_preview",
        "tracks": out_dir / "tracks",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def resolve_run_output_dir(base_out_dir: Path, run_name: str | None, no_run_subdir: bool) -> Path:
    base_out_dir = base_out_dir.resolve()
    if no_run_subdir:
        return base_out_dir

    if run_name:
        safe_run_name = safe_name(run_name)
    else:
        safe_run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    return base_out_dir / safe_run_name


def write_track_debug(track: TrackState, selected: list[CropCandidate], output_dir: Path) -> None:
    track_dir = output_dir / safe_name(track.video_stem) / safe_name(track.track_key)
    track_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "video": track.video,
        "track_key": track.track_key,
        "track_id": track.track_id,
        "detections_seen": track.detections_seen,
        "first_frame_index": track.first_frame_index,
        "last_frame_index": track.last_frame_index,
        "first_timestamp_ms": track.first_timestamp_ms,
        "last_timestamp_ms": track.last_timestamp_ms,
        "selected_crops": [
            {
                "frame_index": candidate.frame_index,
                "timestamp_ms": candidate.timestamp_ms,
                "price_confidence": candidate.price_confidence,
                "bbox": candidate.bbox,
                "area": candidate.area,
                "sharpness": candidate.sharpness,
                "score": candidate.score,
                "crop_path": str(candidate.crop_path),
            }
            for candidate in selected
        ],
    }
    (track_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process_candidate_inside(
    candidate: CropCandidate,
    selected_rank: int,
    track: TrackState,
    dirs: dict[str, Path],
    inside_model: YOLO,
    inside_names: dict[int, str],
    inside_predict_kwargs: dict[str, Any],
    args: argparse.Namespace,
    price_writer: csv.DictWriter,
    inside_writer: csv.DictWriter,
) -> int:
    selected_base = (
        f"{candidate.video_stem}_{safe_name(candidate.track_key)}"
        f"_rank{selected_rank:02d}_f{candidate.frame_index:06d}"
        f"_t{int(round(candidate.timestamp_ms)):09d}"
        f"_score{int(round(candidate.score))}"
    )
    crop_path = dirs["crops"] / f"{selected_base}.jpg"
    inside_image_path = dirs["inside_images"] / f"{selected_base}.jpg"
    inside_label_path = dirs["inside_labels"] / f"{selected_base}.txt"
    inside_annotated_path = dirs["inside_annotated"] / f"{selected_base}.jpg"

    shutil.copy2(candidate.crop_path, crop_path)

    with Image.open(candidate.crop_path) as crop_image:
        crop_image = crop_image.convert("RGB")
        inside_image = crop_image.rotate(90, expand=True) if args.rotate_crops_ccw else crop_image.copy()

    inside_image.save(inside_image_path, quality=95, subsampling=0)

    inside_result = predict(inside_model, inside_image, **inside_predict_kwargs)
    inside_boxes = result_boxes(inside_result, inside_names)
    inside_boxes = dedupe_boxes(
        inside_boxes,
        iou_threshold=args.inside_dedupe_iou,
        class_aware=args.inside_class_aware_dedupe,
    )

    inside_width, inside_height = inside_image.size
    label_lines = [yolo_line(inside_box, inside_width, inside_height) for inside_box in inside_boxes]
    inside_label_path.write_text(
        "\n".join(label_lines) + ("\n" if label_lines else ""),
        encoding="utf-8",
    )
    draw_boxes(inside_image, inside_boxes).save(inside_annotated_path, quality=95, subsampling=0)

    left, top, right, bottom = candidate.bbox
    price_writer.writerow(
        {
            "video": candidate.video,
            "track_key": candidate.track_key,
            "track_id": "" if candidate.track_id is None else candidate.track_id,
            "track_seen": track.detections_seen,
            "selected_rank": selected_rank,
            "frame_index": candidate.frame_index,
            "timestamp_ms": f"{candidate.timestamp_ms:.1f}",
            "crop_index": candidate.crop_index,
            "price_confidence": f"{candidate.price_confidence:.6f}",
            "crop_area": candidate.area,
            "crop_sharpness": f"{candidate.sharpness:.6f}",
            "crop_score": f"{candidate.score:.6f}",
            "x1": left,
            "y1": top,
            "x2": right,
            "y2": bottom,
            "candidate_crop_file": str(candidate.crop_path),
            "crop_file": str(crop_path),
            "inside_image": str(inside_image_path),
            "inside_label_file": str(inside_label_path),
            "inside_annotated_file": str(inside_annotated_path),
            "inside_label_count": len(inside_boxes),
        }
    )

    for inside_index, inside_box in enumerate(inside_boxes):
        single_path = (
            dirs["inside_single"]
            / f"{selected_base}_inside{inside_index:02d}"
            f"_{safe_name(inside_box.class_name)}_conf{inside_box.confidence:.3f}.jpg"
        )
        draw_boxes(inside_image, [inside_box]).save(single_path, quality=95, subsampling=0)
        inside_writer.writerow(
            {
                "video": candidate.video,
                "track_key": candidate.track_key,
                "track_id": "" if candidate.track_id is None else candidate.track_id,
                "selected_rank": selected_rank,
                "frame_index": candidate.frame_index,
                "timestamp_ms": f"{candidate.timestamp_ms:.1f}",
                "crop_index": candidate.crop_index,
                "inside_index": inside_index,
                "class_id": inside_box.class_id,
                "class_name": inside_box.class_name,
                "confidence": f"{inside_box.confidence:.6f}",
                "x1": f"{inside_box.x1:.0f}",
                "y1": f"{inside_box.y1:.0f}",
                "x2": f"{inside_box.x2:.0f}",
                "y2": f"{inside_box.y2:.0f}",
                "crop_score": f"{candidate.score:.6f}",
                "candidate_crop_file": str(candidate.crop_path),
                "crop_file": str(crop_path),
                "inside_image": str(inside_image_path),
                "inside_label_file": str(inside_label_path),
                "inside_annotated_file": str(inside_annotated_path),
                "inside_single_file": str(single_path),
            }
        )

    return len(inside_boxes)


def main() -> None:
    args = parse_args()
    cv2 = import_cv2()

    if not args.price_model.exists():
        raise SystemExit(f"Price model not found: {args.price_model}")
    if not args.inside_model.exists():
        raise SystemExit(f"Inside model not found: {args.inside_model}")
    if args.price_detector_mode == "track" and not args.tracker_config.exists():
        raise SystemExit(f"Tracker config not found: {args.tracker_config}")

    videos = iter_videos(args.source)
    if not videos:
        raise SystemExit(f"No videos found in: {args.source}")

    out_dir = resolve_run_output_dir(args.out_dir, args.run_name, args.no_run_subdir)
    if args.clear_output and out_dir.exists():
        shutil.rmtree(out_dir)
    dirs = make_dirs(out_dir)

    price_model = YOLO(str(args.price_model))
    inside_model = YOLO(str(args.inside_model))
    price_names = normalize_names(price_model.names)
    inside_names = normalize_names(inside_model.names)

    price_predict_kwargs: dict[str, Any] = {
        "conf": args.price_conf,
        "imgsz": args.price_imgsz,
        "verbose": False,
        "save": False,
    }
    inside_predict_kwargs: dict[str, Any] = {
        "conf": args.inside_conf,
        "imgsz": args.inside_imgsz,
        "verbose": False,
        "save": False,
    }
    if args.device:
        price_predict_kwargs["device"] = args.device
        inside_predict_kwargs["device"] = args.device

    price_manifest_path = out_dir / "price_tags.csv"
    inside_manifest_path = out_dir / "inside_predictions.csv"
    candidate_manifest_path = out_dir / "price_candidates.csv"
    track_summary_path = out_dir / "track_summary.csv"
    summary_path = out_dir / "summary.json"
    write_data_yaml(out_dir, inside_names)

    total_frames_read = 0
    total_frames_processed = 0
    total_frames_skipped_blurry = 0
    total_candidate_crops = 0
    total_selected_crops = 0
    total_inside_labels = 0
    processed_frame_limit_hit = False
    sample_fps = requested_sample_fps(args)
    effective_min_track_hits = args.min_track_hits if args.price_detector_mode == "track" else 1

    with (
        price_manifest_path.open("w", newline="", encoding="utf-8") as price_file,
        inside_manifest_path.open("w", newline="", encoding="utf-8") as inside_file,
        candidate_manifest_path.open("w", newline="", encoding="utf-8") as candidate_file,
        track_summary_path.open("w", newline="", encoding="utf-8") as track_summary_file,
    ):
        price_writer = csv.DictWriter(
            price_file,
            fieldnames=[
                "video",
                "track_key",
                "track_id",
                "track_seen",
                "selected_rank",
                "frame_index",
                "timestamp_ms",
                "crop_index",
                "price_confidence",
                "crop_area",
                "crop_sharpness",
                "crop_score",
                "x1",
                "y1",
                "x2",
                "y2",
                "candidate_crop_file",
                "crop_file",
                "inside_image",
                "inside_label_file",
                "inside_annotated_file",
                "inside_label_count",
            ],
        )
        inside_writer = csv.DictWriter(
            inside_file,
            fieldnames=[
                "video",
                "track_key",
                "track_id",
                "selected_rank",
                "frame_index",
                "timestamp_ms",
                "crop_index",
                "inside_index",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "crop_score",
                "candidate_crop_file",
                "crop_file",
                "inside_image",
                "inside_label_file",
                "inside_annotated_file",
                "inside_single_file",
            ],
        )
        candidate_writer = csv.DictWriter(
            candidate_file,
            fieldnames=[
                "video",
                "track_key",
                "track_id",
                "frame_index",
                "timestamp_ms",
                "crop_index",
                "price_confidence",
                "crop_area",
                "crop_sharpness",
                "crop_score",
                "x1",
                "y1",
                "x2",
                "y2",
                "candidate_crop_file",
            ],
        )
        track_summary_writer = csv.DictWriter(
            track_summary_file,
            fieldnames=[
                "video",
                "track_key",
                "track_id",
                "detections_seen",
                "selected_count",
                "first_frame_index",
                "last_frame_index",
                "first_timestamp_ms",
                "last_timestamp_ms",
                "best_score",
                "best_crop_file",
                "skipped_reason",
            ],
        )
        price_writer.writeheader()
        inside_writer.writeheader()
        candidate_writer.writeheader()
        track_summary_writer.writeheader()

        for video_path in videos:
            video_stem = safe_name(video_path.stem)
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"[WARN] Could not open video: {video_path}")
                continue

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            step = frame_step_for_video(fps, args.frame_stride, sample_fps)
            frame_index = -1
            video_processed_frames = 0
            video_skipped_blurry = 0
            video_candidate_crops = 0
            tracks: dict[str, TrackState] = {}
            untracked_counter = 0

            print(
                f"[VIDEO] {video_path.name} fps={fps:.2f} step={step} "
                f"mode={args.price_detector_mode}"
            )

            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_index += 1
                total_frames_read += 1

                if frame_index % step != 0:
                    continue
                if args.max_frames is not None and total_frames_processed >= args.max_frames:
                    processed_frame_limit_hit = True
                    break

                timestamp_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_image = Image.fromarray(frame_rgb)

                frame_sharpness = laplacian_sharpness(frame_image, cv2)
                if args.skip_blurry and frame_sharpness < args.min_laplacian_var:
                    total_frames_skipped_blurry += 1
                    video_skipped_blurry += 1
                    continue

                frame_width, frame_height = frame_image.size
                price_boxes = detect_price_boxes(
                    model=price_model,
                    image=frame_image,
                    class_names=price_names,
                    args=args,
                    predict_kwargs=price_predict_kwargs,
                    tracker_config=args.tracker_config,
                    persist_tracks=video_processed_frames > 0,
                )

                total_frames_processed += 1
                video_processed_frames += 1

                if args.save_frame_preview:
                    frame_preview = draw_boxes(frame_image, price_boxes)
                    frame_preview.save(
                        dirs["frame_preview"] / f"{video_stem}_f{frame_index:06d}.jpg",
                        quality=92,
                        subsampling=0,
                    )

                for crop_index, price_box in enumerate(price_boxes):
                    left, top, right, bottom = expand_box(
                        (price_box.x1, price_box.y1, price_box.x2, price_box.y2),
                        frame_width,
                        frame_height,
                        args.crop_padding,
                    )
                    if right <= left or bottom <= top:
                        continue

                    if price_box.track_id is None:
                        untracked_counter += 1
                        track_key = f"{video_stem}_untracked_{untracked_counter:06d}"
                    else:
                        track_key = f"{video_stem}_track_{price_box.track_id:04d}"

                    crop = frame_image.crop((left, top, right, bottom))
                    area = box_area_xyxy((left, top, right, bottom))
                    crop_sharpness = laplacian_sharpness(crop, cv2)
                    crop_score_value = crop_quality_score(area, crop_sharpness, price_box.confidence)
                    base_name = (
                        f"{video_stem}_{safe_name(track_key)}_f{frame_index:06d}"
                        f"_t{int(round(timestamp_ms)):09d}_crop{crop_index:02d}"
                        f"_pconf{price_box.confidence:.3f}"
                    )
                    candidate_crop_path = dirs["candidate_crops"] / f"{base_name}.jpg"
                    crop.save(candidate_crop_path, quality=95, subsampling=0)

                    candidate = CropCandidate(
                        video=str(video_path),
                        video_stem=video_stem,
                        track_key=track_key,
                        track_id=price_box.track_id,
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        crop_index=crop_index,
                        price_confidence=price_box.confidence,
                        bbox=(left, top, right, bottom),
                        area=area,
                        sharpness=crop_sharpness,
                        score=crop_score_value,
                        crop_path=candidate_crop_path,
                        base_name=base_name,
                    )

                    track = tracks.setdefault(
                        track_key,
                        TrackState(
                            video=str(video_path),
                            video_stem=video_stem,
                            track_key=track_key,
                            track_id=price_box.track_id,
                        ),
                    )
                    add_crop_candidate(track, candidate, top_k=args.top_k_crops_per_track)

                    total_candidate_crops += 1
                    video_candidate_crops += 1
                    candidate_writer.writerow(
                        {
                            "video": str(video_path),
                            "track_key": track_key,
                            "track_id": "" if price_box.track_id is None else price_box.track_id,
                            "frame_index": frame_index,
                            "timestamp_ms": f"{timestamp_ms:.1f}",
                            "crop_index": crop_index,
                            "price_confidence": f"{price_box.confidence:.6f}",
                            "crop_area": area,
                            "crop_sharpness": f"{crop_sharpness:.6f}",
                            "crop_score": f"{crop_score_value:.6f}",
                            "x1": left,
                            "y1": top,
                            "x2": right,
                            "y2": bottom,
                            "candidate_crop_file": str(candidate_crop_path),
                        }
                    )

                if video_processed_frames % 25 == 0:
                    print(
                        f"  frames={video_processed_frames} "
                        f"candidate_crops={video_candidate_crops} tracks={len(tracks)}"
                    )

            cap.release()

            video_selected_crops = 0
            video_inside_labels = 0
            for track in sorted(tracks.values(), key=lambda item: item.track_key):
                selected = list(track.top_crops)
                skipped_reason = ""
                if track.detections_seen < effective_min_track_hits:
                    selected = []
                    skipped_reason = f"min_track_hits<{effective_min_track_hits}"

                best = track.top_crops[0] if track.top_crops else None
                track_summary_writer.writerow(
                    {
                        "video": track.video,
                        "track_key": track.track_key,
                        "track_id": "" if track.track_id is None else track.track_id,
                        "detections_seen": track.detections_seen,
                        "selected_count": len(selected),
                        "first_frame_index": track.first_frame_index,
                        "last_frame_index": track.last_frame_index,
                        "first_timestamp_ms": (
                            "" if track.first_timestamp_ms is None else f"{track.first_timestamp_ms:.1f}"
                        ),
                        "last_timestamp_ms": (
                            "" if track.last_timestamp_ms is None else f"{track.last_timestamp_ms:.1f}"
                        ),
                        "best_score": "" if best is None else f"{best.score:.6f}",
                        "best_crop_file": "" if best is None else str(best.crop_path),
                        "skipped_reason": skipped_reason,
                    }
                )

                if args.save_track_debug:
                    write_track_debug(track, selected, dirs["tracks"])

                for selected_rank, candidate in enumerate(selected):
                    labels_count = process_candidate_inside(
                        candidate=candidate,
                        selected_rank=selected_rank,
                        track=track,
                        dirs=dirs,
                        inside_model=inside_model,
                        inside_names=inside_names,
                        inside_predict_kwargs=inside_predict_kwargs,
                        args=args,
                        price_writer=price_writer,
                        inside_writer=inside_writer,
                    )
                    total_selected_crops += 1
                    total_inside_labels += labels_count
                    video_selected_crops += 1
                    video_inside_labels += labels_count

            print(
                f"[DONE] {video_path.name}: frames={video_processed_frames}, "
                f"skipped_blurry={video_skipped_blurry}, candidates={video_candidate_crops}, "
                f"tracks={len(tracks)}, selected_crops={video_selected_crops}, "
                f"inside_labels={video_inside_labels}"
            )

            if processed_frame_limit_hit:
                break

    summary = {
        "source": str(args.source.resolve()),
        "output_dir": str(out_dir),
        "base_out_dir": str(args.out_dir.resolve()),
        "run_name": out_dir.name,
        "no_run_subdir": args.no_run_subdir,
        "price_model": str(args.price_model.resolve()),
        "inside_model": str(args.inside_model.resolve()),
        "tracker_config": str(args.tracker_config.resolve()) if args.tracker_config.exists() else "",
        "videos": [str(path) for path in videos],
        "price_detector_mode": args.price_detector_mode,
        "preset": args.preset,
        "price_conf": args.price_conf,
        "inside_conf": args.inside_conf,
        "price_imgsz": args.price_imgsz,
        "inside_imgsz": args.inside_imgsz,
        "device": args.device,
        "frame_stride": args.frame_stride,
        "sample_fps": sample_fps,
        "skip_blurry": args.skip_blurry,
        "min_laplacian_var": args.min_laplacian_var,
        "crop_padding": args.crop_padding,
        "top_k_crops_per_track": args.top_k_crops_per_track,
        "min_track_hits": effective_min_track_hits,
        "rotate_crops_ccw": args.rotate_crops_ccw,
        "inside_dedupe_iou": args.inside_dedupe_iou,
        "inside_class_aware_dedupe": args.inside_class_aware_dedupe,
        "frames_read": total_frames_read,
        "frames_processed": total_frames_processed,
        "frames_skipped_blurry": total_frames_skipped_blurry,
        "candidate_price_crops": total_candidate_crops,
        "selected_price_crops": total_selected_crops,
        "inside_labels": total_inside_labels,
        "candidate_manifest": str(candidate_manifest_path),
        "price_manifest": str(price_manifest_path),
        "inside_manifest": str(inside_manifest_path),
        "track_summary": str(track_summary_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[ALL DONE]")
    print(f"  output_dir: {out_dir}")
    print(f"  frames_processed: {total_frames_processed}")
    print(f"  candidate_price_crops: {total_candidate_crops}")
    print(f"  selected_price_crops: {total_selected_crops}")
    print(f"  inside_labels: {total_inside_labels}")
    print(f"  candidate_manifest: {candidate_manifest_path}")
    print(f"  price_manifest: {price_manifest_path}")
    print(f"  inside_manifest: {inside_manifest_path}")
    print(f"  track_summary: {track_summary_path}")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
