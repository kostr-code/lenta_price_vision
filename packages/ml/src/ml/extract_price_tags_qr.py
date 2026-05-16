from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from .media import import_cv2, import_numpy, laplacian_sharpness, video_metadata

ML_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ML_ROOT.parent.parent

DEFAULT_MODEL = REPO_ROOT / "models" / "best.pt"
DEFAULT_QR_MODEL = ML_ROOT / "runs" / "detect" / "qr_code_yolo26n" / "weights" / "best.pt"
DEFAULT_VIDEO_DIR = ML_ROOT / "data" / "Unlabeled"
DEFAULT_TRACKER = ML_ROOT / "src" / "ml" / "configs" / "bytetrack_price.yaml"
DEFAULT_OUT = REPO_ROOT / "runs" / "price_tag_qr_crops_yolo_qr"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


@dataclass(frozen=True)
class DetectionCandidate:
    score: float
    frame_idx: int
    timestamp_ms: float
    conf: float
    cls_id: int
    cls_name: str
    bbox: tuple[int, int, int, int]
    crop_size: tuple[int, int]
    sharpness: float
    track_age: int


@dataclass
class TrackBuffer:
    track_id: int
    length: int = 0
    best: DetectionCandidate | None = None

    def update(self, candidate: DetectionCandidate) -> None:
        self.length += 1
        if self.best is None or candidate.score > self.best.score:
            self.best = candidate


@dataclass(frozen=True)
class QRDetectionCandidate:
    bbox: tuple[int, int, int, int]
    padded_bbox: tuple[int, int, int, int]
    conf: float
    cls_id: int
    cls_name: str
    score: float
    method: str = "yolo"

    def to_json(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "padded_bbox": list(self.padded_bbox),
            "conf": round(self.conf, 6),
            "cls_id": self.cls_id,
            "cls_name": self.cls_name,
            "score": round(self.score, 6),
            "method": self.method,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan shelf videos with the main YOLO model, save the best price-tag crop per "
            "track and the best QR-zone crop inside each price tag."
        )
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--qr-model", type=Path, default=DEFAULT_QR_MODEL)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tracker", type=Path, default=DEFAULT_TRACKER)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--pad", type=int, default=8)
    parser.add_argument("--qr-pad", type=int, default=4)
    parser.add_argument("--qr-imgsz", type=int, default=640)
    parser.add_argument("--qr-conf", type=float, default=0.15)
    parser.add_argument("--qr-iou", type=float, default=0.5)
    parser.add_argument(
        "--qr-device",
        default="",
        help="Empty value means reuse --device.",
    )
    parser.add_argument("--min-track-len", type=int, default=2)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means scan full videos.")
    parser.add_argument("--limit-videos", type=int, default=0, help="0 means scan all videos.")
    return parser.parse_args()


def list_videos(video_dir: Path, limit: int = 0) -> list[Path]:
    videos = sorted(
        path
        for path in video_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if limit > 0:
        return videos[:limit]
    return videos


def rotate_bbox_ccw(x1: int, y1: int, x2: int, y2: int, orig_w: int) -> tuple[int, int, int, int]:
    return y1, orig_w - x2, y2, orig_w - x1


def expand_xyxy(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    pad: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, int(x1) - pad),
        max(0, int(y1) - pad),
        min(width, int(x2) + pad),
        min(height, int(y2) + pad),
    )


def crop_xyxy(image: Any, bbox: tuple[int, int, int, int]) -> Any | None:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if crop is None or not getattr(crop, "size", 0):
        return None
    return crop.copy()


def score_price_tag(crop: Any, conf: float) -> tuple[float, float]:
    np = import_numpy()
    if crop is None or not getattr(crop, "size", 0):
        return 0.0, 0.0
    sharpness = laplacian_sharpness(crop)
    height, width = crop.shape[:2]
    gray = import_cv2().cvtColor(crop, import_cv2().COLOR_BGR2GRAY)
    overexposed = float(np.mean(gray > 245))
    exposure_score = max(0.05, 1.0 - overexposed * 4.0)
    size_score = min((width * height) / (180 * 120), 3.0)
    score = sharpness * (0.5 + conf) * exposure_score * (0.5 + size_score)
    return float(score), float(sharpness)


def class_name_for(result: Any, cls_id: int) -> str:
    names = getattr(result, "names", {}) or {}
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    try:
        return str(names[cls_id])
    except Exception:
        return str(cls_id)


def collect_best_tracks(
    model: YOLO,
    video_path: Path,
    tracker: Path,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    pad: int,
    max_frames: int,
) -> dict[int, TrackBuffer]:
    cv2 = import_cv2()
    meta = video_metadata(video_path)
    fps = float(meta["fps"] or 0.0)
    buffers: dict[int, TrackBuffer] = {}

    for frame_idx, result in enumerate(
        model.track(
            source=str(video_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            tracker=str(tracker),
            stream=True,
            persist=True,
            verbose=False,
        )
    ):
        if max_frames > 0 and frame_idx >= max_frames:
            break

        raw = result.orig_img
        rotated = cv2.rotate(raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
        orig_h, orig_w = raw.shape[:2]
        height, width = rotated.shape[:2]

        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.id is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy() if boxes.cls is not None else [0] * len(xyxy)

        for box, track_id, conf_value, cls_value in zip(
            xyxy, track_ids, confs, classes, strict=False
        ):
            x1, y1, x2, y2 = map(int, box)
            bbox = rotate_bbox_ccw(x1, y1, x2, y2, orig_w)
            bbox = expand_xyxy(bbox, width=width, height=height, pad=pad)
            crop = crop_xyxy(rotated, bbox)
            if crop is None:
                continue

            det_score, sharpness = score_price_tag(crop, float(conf_value))
            track_id_int = int(track_id)
            cls_id = int(cls_value)
            age = buffers[track_id_int].length + 1 if track_id_int in buffers else 1
            candidate = DetectionCandidate(
                score=det_score,
                frame_idx=frame_idx,
                timestamp_ms=frame_idx / fps * 1000 if fps > 0 else 0.0,
                conf=float(conf_value),
                cls_id=cls_id,
                cls_name=class_name_for(result, cls_id),
                bbox=bbox,
                crop_size=(int(crop.shape[1]), int(crop.shape[0])),
                sharpness=sharpness,
                track_age=age,
            )
            buffers.setdefault(track_id_int, TrackBuffer(track_id_int)).update(candidate)

        if frame_idx and frame_idx % 200 == 0:
            print(f"  pass1 frame={frame_idx} tracks={len(buffers)}")

    return buffers


def valid_tracks(
    buffers: dict[int, TrackBuffer],
    min_track_len: int,
    min_score: float,
) -> list[tuple[int, DetectionCandidate, int]]:
    selected: list[tuple[int, DetectionCandidate, int]] = []
    for track_id, buffer in buffers.items():
        if buffer.best is None:
            continue
        if buffer.length < min_track_len:
            continue
        if buffer.best.score < min_score:
            continue
        selected.append((track_id, buffer.best, buffer.length))
    return sorted(selected, key=lambda item: (item[1].frame_idx, item[0]))


def is_plausible_lenta_qr(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> bool:
    """Price tags usually place the QR in the upper-right half."""
    x1, y1, x2, y2 = bbox
    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)
    if box_w < 28 or box_h < 28:
        return False

    center_x = (x1 + x2) / 2 / max(1, width)
    center_y = (y1 + y2) / 2 / max(1, height)
    area_ratio = (box_w * box_h) / max(1, width * height)
    aspect = box_w / max(1, box_h)

    return (
        center_x >= 0.45
        and center_y <= 0.78
        and 0.015 <= area_ratio <= 0.55
        and 0.45 <= aspect <= 2.2
    )


def qr_layout_bonus(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> float:
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2 / max(1, width)
    center_y = (y1 + y2) / 2 / max(1, height)
    right_bonus = max(0.0, min(1.0, (center_x - 0.35) / 0.45))
    top_bonus = max(0.0, min(1.0, (0.85 - center_y) / 0.65))
    return 0.15 * right_bonus + 0.10 * top_bonus


def find_best_qr_crop_yolo(
    qr_model: YOLO,
    price_tag: Any,
    qr_pad: int,
    qr_imgsz: int,
    qr_conf: float,
    qr_iou: float,
    qr_device: str,
) -> tuple[Any | None, QRDetectionCandidate | None, list[QRDetectionCandidate]]:
    height, width = price_tag.shape[:2]
    results = qr_model.predict(
        source=price_tag,
        imgsz=qr_imgsz,
        conf=qr_conf,
        iou=qr_iou,
        device=qr_device,
        verbose=False,
    )
    if not results:
        return None, None, []

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None, None, []

    detections: list[QRDetectionCandidate] = []
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy() if boxes.cls is not None else [0] * len(xyxy)

    for box, conf_value, cls_value in zip(xyxy, confs, classes, strict=False):
        x1, y1, x2, y2 = map(int, box)
        bbox = expand_xyxy((x1, y1, x2, y2), width=width, height=height, pad=0)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        padded_bbox = expand_xyxy(bbox, width=width, height=height, pad=qr_pad)
        cls_id = int(cls_value)
        conf_float = float(conf_value)
        area_ratio = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / max(1, width * height)
        layout = qr_layout_bonus(bbox, width=width, height=height)
        plausible_bonus = 0.2 if is_plausible_lenta_qr(bbox, width=width, height=height) else 0.0
        score = conf_float + layout + plausible_bonus + min(area_ratio, 0.4) * 0.05
        detections.append(
            QRDetectionCandidate(
                bbox=bbox,
                padded_bbox=padded_bbox,
                conf=conf_float,
                cls_id=cls_id,
                cls_name=class_name_for(result, cls_id),
                score=float(score),
                method="yolo:qr_code_yolo26n",
            )
        )

    if not detections:
        return None, None, []

    best = max(detections, key=lambda item: item.score)
    qr_crop = crop_xyxy(price_tag, best.padded_bbox)
    return qr_crop, best, sorted(detections, key=lambda item: item.score, reverse=True)


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_best_tracks(
    video_path: Path,
    video_out: Path,
    qr_model: YOLO,
    buffers: dict[int, TrackBuffer],
    min_track_len: int,
    min_score: float,
    qr_pad: int,
    qr_imgsz: int,
    qr_conf: float,
    qr_iou: float,
    qr_device: str,
) -> list[dict[str, Any]]:
    cv2 = import_cv2()
    selected = valid_tracks(buffers, min_track_len=min_track_len, min_score=min_score)
    if not selected:
        print("  no valid tracks")
        return []

    frame_to_tracks: dict[int, list[tuple[int, DetectionCandidate, int]]] = defaultdict(list)
    for track_id, candidate, track_len in selected:
        frame_to_tracks[candidate.frame_idx].append((track_id, candidate, track_len))

    video_out.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    rows: list[dict[str, Any]] = []
    try:
        for frame_idx in sorted(frame_to_tracks):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, raw = capture.read()
            if not ok:
                continue
            rotated = cv2.rotate(raw, cv2.ROTATE_90_COUNTERCLOCKWISE)

            for track_id, candidate, track_len in frame_to_tracks[frame_idx]:
                price_tag = crop_xyxy(rotated, candidate.bbox)
                if price_tag is None:
                    continue

                item_name = f"track_{track_id:04d}_frame_{frame_idx:06d}"
                item_dir = video_out / item_name
                item_dir.mkdir(parents=True, exist_ok=True)

                price_tag_path = item_dir / "price_tag.jpg"
                qr_path = item_dir / "qr.jpg"
                candidates_path = item_dir / "qr_candidates.json"
                meta_path = item_dir / "meta.json"

                cv2.imwrite(str(price_tag_path), price_tag, [cv2.IMWRITE_JPEG_QUALITY, 95])

                qr_crop, qr_candidate, qr_candidates = find_best_qr_crop_yolo(
                    qr_model,
                    price_tag,
                    qr_pad=qr_pad,
                    qr_imgsz=qr_imgsz,
                    qr_conf=qr_conf,
                    qr_iou=qr_iou,
                    qr_device=qr_device,
                )
                if qr_crop is not None:
                    cv2.imwrite(str(qr_path), qr_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

                write_json(candidates_path, [candidate.to_json() for candidate in qr_candidates])

                row = {
                    "video": str(video_path),
                    "video_stem": video_path.stem,
                    "track_id": track_id,
                    "track_length": track_len,
                    "frame_idx": candidate.frame_idx,
                    "timestamp_ms": round(candidate.timestamp_ms),
                    "yolo_conf": round(candidate.conf, 4),
                    "class_id": candidate.cls_id,
                    "class_name": candidate.cls_name,
                    "score": round(candidate.score, 3),
                    "sharpness": round(candidate.sharpness, 3),
                    "price_tag_bbox": list(candidate.bbox),
                    "price_tag_size": [int(price_tag.shape[1]), int(price_tag.shape[0])],
                    "price_tag_path": str(price_tag_path),
                    "qr_found": qr_crop is not None,
                    "qr_method": qr_candidate.method if qr_candidate is not None else "",
                    "qr_score": round(qr_candidate.score, 3) if qr_candidate is not None else "",
                    "qr_conf": round(qr_candidate.conf, 4) if qr_candidate is not None else "",
                    "qr_class_id": qr_candidate.cls_id if qr_candidate is not None else "",
                    "qr_class_name": qr_candidate.cls_name if qr_candidate is not None else "",
                    "qr_bbox": list(qr_candidate.bbox) if qr_candidate is not None else [],
                    "qr_padded_bbox": (
                        list(qr_candidate.padded_bbox) if qr_candidate is not None else []
                    ),
                    "qr_path": str(qr_path) if qr_crop is not None else "",
                    "qr_candidate": qr_candidate.to_json() if qr_candidate is not None else {},
                    "item_dir": str(item_dir),
                }
                write_json(meta_path, row)
                rows.append(row)
                print(
                    f"  saved track={track_id:04d} frame={frame_idx:06d} "
                    f"qr={row['qr_found']} conf={row['qr_conf'] or '-'}"
                )
    finally:
        capture.release()

    return rows


def write_manifest(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "manifest.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = out_dir / "manifest.csv"
    fieldnames = [
        "video",
        "video_stem",
        "track_id",
        "track_length",
        "frame_idx",
        "timestamp_ms",
        "yolo_conf",
        "class_id",
        "class_name",
        "score",
        "sharpness",
        "price_tag_bbox",
        "price_tag_size",
        "price_tag_path",
        "qr_found",
        "qr_method",
        "qr_score",
        "qr_conf",
        "qr_class_id",
        "qr_class_name",
        "qr_bbox",
        "qr_padded_bbox",
        "qr_path",
        "item_dir",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    videos = list_videos(args.video_dir, limit=args.limit_videos)
    if not videos:
        raise FileNotFoundError(f"No videos found in {args.video_dir}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.qr_model.exists():
        raise FileNotFoundError(f"QR model not found: {args.qr_model}")
    if not args.tracker.exists():
        raise FileNotFoundError(f"Tracker config not found: {args.tracker}")

    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    qr_model = YOLO(str(args.qr_model))
    qr_device = args.qr_device or args.device
    all_rows: list[dict[str, Any]] = []

    print(f"model: {args.model}")
    print(f"qr_model: {args.qr_model}")
    print(f"videos: {len(videos)} from {args.video_dir}")
    print(f"out: {args.out}")

    for index, video_path in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {video_path.name}")
        buffers = collect_best_tracks(
            model=model,
            video_path=video_path,
            tracker=args.tracker,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            pad=args.pad,
            max_frames=args.max_frames,
        )
        rows = save_best_tracks(
            video_path=video_path,
            video_out=args.out / video_path.stem,
            qr_model=qr_model,
            buffers=buffers,
            min_track_len=args.min_track_len,
            min_score=args.min_score,
            qr_pad=args.qr_pad,
            qr_imgsz=args.qr_imgsz,
            qr_conf=args.qr_conf,
            qr_iou=args.qr_iou,
            qr_device=qr_device,
        )
        all_rows.extend(rows)
        write_manifest(args.out, all_rows)
        print(f"  video rows={len(rows)} total={len(all_rows)}")

    write_manifest(args.out, all_rows)
    return all_rows


def main() -> None:
    args = parse_args()
    rows = run(args)
    qr_count = sum(1 for row in rows if row["qr_found"])
    print(f"done: price_tags={len(rows)} qr_crops={qr_count} out={args.out}")


if __name__ == "__main__":
    main()
