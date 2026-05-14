from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x_min + self.width / 2.0, self.y_min + self.height / 2.0)

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return (
            int(round(self.x_min)),
            int(round(self.y_min)),
            int(round(self.x_max)),
            int(round(self.y_max)),
        )

    def to_record_values(self) -> dict[str, str]:
        return {
            "x_min": format_number(self.x_min),
            "y_min": format_number(self.y_min),
            "x_max": format_number(self.x_max),
            "y_max": format_number(self.y_max),
        }


@dataclass(frozen=True)
class VideoFrame:
    image: Any
    index: int
    timestamp_ms: int
    sharpness: float


def format_number(value: float) -> str:
    rounded = round(float(value), 1)
    if rounded.is_integer():
        return str(int(rounded))
    return str(rounded).replace(".", ",")


def import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError("OpenCV is required for video processing") from exc
    return cv2


def import_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError("NumPy is required for image processing") from exc
    return np


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    return BBox(
        x_min=max(0.0, min(float(width - 1), bbox.x_min)),
        y_min=max(0.0, min(float(height - 1), bbox.y_min)),
        x_max=max(0.0, min(float(width), bbox.x_max)),
        y_max=max(0.0, min(float(height), bbox.y_max)),
    )


def expand_bbox(bbox: BBox, image_width: int, image_height: int, pad_px: int) -> BBox:
    return clamp_bbox(
        BBox(
            bbox.x_min - pad_px,
            bbox.y_min - pad_px,
            bbox.x_max + pad_px,
            bbox.y_max + pad_px,
        ),
        image_width,
        image_height,
    )


def expand_price_tag_crop(
    bbox: BBox,
    image_width: int,
    image_height: int,
    side_pad: float = 0.18,
    top_pad: float = 0.10,
    bottom_pad: float = 1.45,
) -> BBox:
    width = max(1.0, bbox.width)
    height = max(1.0, bbox.height)
    return clamp_bbox(
        BBox(
            bbox.x_min - side_pad * width,
            bbox.y_min - top_pad * height,
            bbox.x_max + side_pad * width,
            bbox.y_max + bottom_pad * height,
        ),
        image_width,
        image_height,
    )


def bbox_iou(left: BBox, right: BBox) -> float:
    x_min = max(left.x_min, right.x_min)
    y_min = max(left.y_min, right.y_min)
    x_max = min(left.x_max, right.x_max)
    y_max = min(left.y_max, right.y_max)
    inter = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    union = left.area + right.area - inter
    if union <= 0:
        return 0.0
    return inter / union


def center_distance(left: BBox, right: BBox) -> float:
    ax, ay = left.center
    bx, by = right.center
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def crop_image(image: Any, bbox: BBox, pad_px: int = 0) -> Any:
    height, width = image.shape[:2]
    region = expand_bbox(bbox, width, height, pad_px)
    x_min, y_min, x_max, y_max = region.as_int_tuple()
    return image[y_min:y_max, x_min:x_max]


def laplacian_sharpness(image: Any) -> float:
    cv2 = import_cv2()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def enhance_crop(image: Any, upscale: float = 2.0) -> Any:
    cv2 = import_cv2()
    if image is None or image.size == 0:
        return image
    crop = image
    if upscale and upscale != 1.0:
        crop = cv2.resize(
            crop,
            None,
            fx=upscale,
            fy=upscale,
            interpolation=cv2.INTER_CUBIC,
        )
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    crop = cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(crop, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(crop, 1.45, blur, -0.45, 0)


def iter_sampled_frames(
    video_path: Path,
    sample_fps: float = 2.0,
    min_sharpness: float = 0.0,
    max_frames: int = 0,
) -> Iterator[VideoFrame]:
    cv2 = import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_step = 1
    if source_fps > 0 and sample_fps > 0:
        frame_step = max(1, int(round(source_fps / sample_fps)))

    emitted = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_step != 0:
                frame_index += 1
                continue
            timestamp = int(capture.get(cv2.CAP_PROP_POS_MSEC))
            sharpness = laplacian_sharpness(frame)
            if sharpness >= min_sharpness:
                yield VideoFrame(
                    image=frame,
                    index=frame_index,
                    timestamp_ms=timestamp,
                    sharpness=sharpness,
                )
                emitted += 1
                if max_frames > 0 and emitted >= max_frames:
                    break
            frame_index += 1
    finally:
        capture.release()


def video_metadata(video_path: Path) -> dict[str, float | int | str]:
    cv2 = import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    duration_ms = int(frame_count / fps * 1000) if fps > 0 else 0
    return {
        "path": str(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_ms": duration_ms,
    }


def load_image(image_path: Path) -> Any:
    cv2 = import_cv2()
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")
    return image


def image_metadata(image_path: Path) -> dict[str, float | int | str]:
    image = load_image(image_path)
    height, width = image.shape[:2]
    channels = int(image.shape[2]) if len(image.shape) >= 3 else 1
    return {
        "path": str(image_path),
        "width": int(width),
        "height": int(height),
        "channels": channels,
    }
