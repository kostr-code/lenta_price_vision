from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .media import BBox, import_cv2, import_numpy


@dataclass(frozen=True)
class RailRoiConfig:
    enabled: bool = False
    update_every_good_frames: int = 50
    vertical_margin_ratio: float = 0.12
    min_roi_height_ratio: float = 0.18
    max_roi_height_ratio: float = 0.45
    min_score: float = 0.08


@dataclass(frozen=True)
class RailRoi:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    score: float
    source: str

    @property
    def is_full_frame(self) -> bool:
        return self.x_min == 0 and self.y_min == 0 and self.source == "full_frame"

    def crop(self, image: Any) -> Any:
        return image[self.y_min : self.y_max, self.x_min : self.x_max]

    def translate_bbox(self, bbox: BBox) -> BBox:
        return BBox(
            bbox.x_min + self.x_min,
            bbox.y_min + self.y_min,
            bbox.x_max + self.x_min,
            bbox.y_max + self.y_min,
        )

    def to_debug(self) -> dict[str, int | float | str | bool]:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "score": round(self.score, 5),
            "source": self.source,
            "full_frame": self.is_full_frame,
        }


class RailRoiDetector:
    """Finds a horizontal shelf rail band and keeps detections inside it."""

    def __init__(self, config: RailRoiConfig | None = None) -> None:
        self.config = config or RailRoiConfig()
        self._last_good: RailRoi | None = None
        self._last_update_frame = -1

    def detect(self, image: Any, frame_order: int) -> RailRoi:
        height, width = image.shape[:2]
        if not self.config.enabled or width <= 0 or height <= 0:
            return full_frame_roi(width, height)

        should_update = (
            self._last_good is None
            or frame_order - self._last_update_frame >= self.config.update_every_good_frames
        )
        if not should_update:
            return self._last_good or full_frame_roi(width, height)

        detected = self._detect_current(image)
        self._last_update_frame = frame_order
        if detected.score >= self.config.min_score:
            self._last_good = detected
            return detected
        if self._last_good is not None:
            return RailRoi(
                self._last_good.x_min,
                self._last_good.y_min,
                self._last_good.x_max,
                self._last_good.y_max,
                detected.score,
                "cached_low_score",
            )
        return full_frame_roi(width, height)

    def _detect_current(self, image: Any) -> RailRoi:
        cv2 = import_cv2()
        np = import_numpy()
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if width > 1280:
            scale = 1280.0 / width
            resized_height = max(1, int(round(height * scale)))
            gray = cv2.resize(gray, (1280, resized_height), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0

        gray_float = gray.astype(np.float32) / 255.0
        horizontal_edges = np.abs(cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3))
        brightness = gray_float
        row_score = horizontal_edges.mean(axis=1) + 0.2 * brightness.mean(axis=1)
        row_score = cv2.GaussianBlur(row_score.reshape(-1, 1), (1, 31), 0).reshape(-1)
        peak_index = int(np.argmax(row_score))
        baseline = float(np.percentile(row_score, 50)) + 1e-6
        peak_score = float(row_score[peak_index])
        normalized_score = max(0.0, min(1.0, (peak_score - baseline) / (baseline * 6.0)))

        peak_y = int(round(peak_index / scale))
        roi_height = clamp_int(
            int(round(height * self.config.vertical_margin_ratio * 2.0)),
            int(round(height * self.config.min_roi_height_ratio)),
            int(round(height * self.config.max_roi_height_ratio)),
        )
        y_min = max(0, peak_y - roi_height // 2)
        y_max = min(height, y_min + roi_height)
        y_min = max(0, y_max - roi_height)
        return RailRoi(0, y_min, width, y_max, normalized_score, "rail_roi")


def full_frame_roi(width: int, height: int) -> RailRoi:
    return RailRoi(0, 0, max(0, width), max(0, height), 0.0, "full_frame")


def clamp_int(value: int, low: int, high: int) -> int:
    if high < low:
        return low
    return max(low, min(high, value))
