from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .media import BBox, bbox_iou, clamp_bbox, import_cv2, import_numpy


@dataclass(frozen=True)
class PriceTagCandidate:
    bbox: BBox
    confidence: float
    source: str
    label: str = "price_tag"


@dataclass(frozen=True)
class CandidateFinderConfig:
    yolo_weights: str | None = None
    yolo_conf: float = 0.23
    detector_imgsz: int = 1600
    tiled_yolo: bool = False
    tile_size: int = 640
    tile_stride: int = 512
    max_tiles_per_frame: int = 64
    max_detections_per_frame: int = 80
    max_color_fallback_detections: int = 32
    fallback_min_area: int = 1_400
    fallback_max_area_ratio: float = 0.18
    fallback_min_width: int = 45
    fallback_min_height: int = 25
    fallback_min_aspect: float = 0.25
    fallback_max_aspect: float = 8.0
    nms_iou: float = 0.42


class PriceTagCandidateFinder:
    """Finds likely shelf price tags using YOLO, QR seeds, and color geometry."""

    def __init__(self, config: CandidateFinderConfig | None = None) -> None:
        self.config = config or CandidateFinderConfig()
        self._yolo_model: Any | None = None
        self._yolo_load_error: str | None = None

    @property
    def yolo_status(self) -> dict[str, str | bool]:
        return {
            "enabled": bool(self.config.yolo_weights),
            "loaded": self._yolo_model is not None,
            "error": self._yolo_load_error or "",
        }

    def find(self, image: Any) -> list[PriceTagCandidate]:
        candidates: list[PriceTagCandidate] = []
        candidates.extend(self._find_with_yolo(image))
        candidates.extend(self._find_with_qr_seeds(image))
        candidates.extend(self._find_with_color_geometry(image))
        return self._deduplicate(candidates)

    def _find_with_yolo(self, image: Any) -> list[PriceTagCandidate]:
        model = self._load_yolo()
        if model is None:
            return []
        if self.config.tiled_yolo:
            return self._find_with_tiled_yolo(image, model)
        detections: list[PriceTagCandidate] = []
        try:
            results = model.predict(
                image,
                conf=self.config.yolo_conf,
                imgsz=self.config.detector_imgsz,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - depends on local model
            self._yolo_load_error = str(exc)
            return []
        height, width = image.shape[:2]
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                coords = box.xyxy[0].detach().cpu().tolist()
                confidence = float(box.conf[0].detach().cpu().item())
                bbox = clamp_bbox(BBox(*map(float, coords)), width, height)
                detections.append(PriceTagCandidate(bbox, confidence, "yolo"))
        return detections

    def _find_with_tiled_yolo(self, image: Any, model: Any) -> list[PriceTagCandidate]:
        detections: list[PriceTagCandidate] = []
        height, width = image.shape[:2]
        tiles = list(
            iter_tiles(
                width=width,
                height=height,
                tile_size=self.config.tile_size,
                stride=self.config.tile_stride,
                max_tiles=self.config.max_tiles_per_frame,
            )
        )
        for tile_index, tile in enumerate(tiles):
            x_min, y_min, x_max, y_max = tile
            tile_image = image[y_min:y_max, x_min:x_max]
            if tile_image.size == 0:
                continue
            try:
                results = model.predict(
                    tile_image,
                    conf=self.config.yolo_conf,
                    imgsz=self.config.detector_imgsz,
                    verbose=False,
                )
            except Exception as exc:  # pragma: no cover - depends on local model
                self._yolo_load_error = str(exc)
                return detections
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    coords = box.xyxy[0].detach().cpu().tolist()
                    confidence = float(box.conf[0].detach().cpu().item())
                    bbox = BBox(
                        float(coords[0]) + x_min,
                        float(coords[1]) + y_min,
                        float(coords[2]) + x_min,
                        float(coords[3]) + y_min,
                    )
                    detections.append(
                        PriceTagCandidate(
                            clamp_bbox(bbox, width, height),
                            confidence,
                            f"tiled_yolo:{tile_index}",
                        )
                    )
        return self._deduplicate(detections)

    def _load_yolo(self) -> Any | None:
        weights = self.config.yolo_weights
        if not weights:
            return None
        if self._yolo_model is not None:
            return self._yolo_model
        weights_path = Path(weights)
        if not weights_path.exists():
            self._yolo_load_error = f"YOLO weights not found: {weights_path}"
            return None
        try:
            from ultralytics import YOLO  # type: ignore

            self._yolo_model = YOLO(str(weights_path))
            self._yolo_load_error = None
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            self._yolo_load_error = str(exc)
            return None
        return self._yolo_model

    def _find_with_qr_seeds(self, image: Any) -> list[PriceTagCandidate]:
        cv2 = import_cv2()
        detections: list[PriceTagCandidate] = []
        detector = cv2.QRCodeDetector()
        height, width = image.shape[:2]

        points_list: list[Any] = []
        try:
            ok, _decoded, points, _straight = detector.detectAndDecodeMulti(image)
            if ok and points is not None:
                points_list.extend(points)
        except Exception:
            points = detector.detect(image)[1]
            if points is not None:
                points_list.append(points)

        for points in points_list:
            if points is None:
                continue
            xs = [float(point[0]) for point in points.reshape(-1, 2)]
            ys = [float(point[1]) for point in points.reshape(-1, 2)]
            qr_box = BBox(min(xs), min(ys), max(xs), max(ys))
            seed_w = max(1.0, qr_box.width)
            seed_h = max(1.0, qr_box.height)
            bbox = BBox(
                qr_box.x_min - 2.2 * seed_w,
                qr_box.y_min - 1.5 * seed_h,
                qr_box.x_max + 2.8 * seed_w,
                qr_box.y_max + 1.7 * seed_h,
            )
            detections.append(PriceTagCandidate(clamp_bbox(bbox, width, height), 0.76, "qr_seed"))
        return detections

    def _find_with_color_geometry(self, image: Any) -> list[PriceTagCandidate]:
        cv2 = import_cv2()
        np = import_numpy()
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        masks = [
            ("red", self._red_mask(hsv, cv2, np)),
            ("yellow", cv2.inRange(hsv, (15, 60, 80), (42, 255, 255))),
            ("green", cv2.inRange(hsv, (38, 35, 45), (92, 255, 255))),
            ("white", cv2.inRange(hsv, (0, 0, 165), (180, 80, 255))),
        ]

        detections: list[PriceTagCandidate] = []
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        max_area = width * height * self.config.fallback_max_area_ratio
        for color, mask in masks:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            contours, _hierarchy = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = float(w * h)
                aspect = float(w) / max(1.0, float(h))
                if not self._passes_geometry(area, aspect, w, h, max_area):
                    continue
                fill_ratio = cv2.contourArea(contour) / max(1.0, area)
                if fill_ratio < 0.18:
                    continue
                bbox = clamp_bbox(BBox(x, y, x + w, y + h), width, height)
                confidence = min(0.68, 0.25 + fill_ratio * 0.55)
                detections.append(PriceTagCandidate(bbox, confidence, f"color_{color}", color))
        return detections

    def _red_mask(self, hsv: Any, cv2: Any, np: Any) -> Any:
        lower = cv2.inRange(hsv, (0, 65, 55), (12, 255, 255))
        upper = cv2.inRange(hsv, (168, 65, 55), (180, 255, 255))
        return cv2.bitwise_or(lower, upper).astype(np.uint8)

    def _passes_geometry(
        self,
        area: float,
        aspect: float,
        width: int,
        height: int,
        max_area: float,
    ) -> bool:
        return (
            area >= self.config.fallback_min_area
            and area <= max_area
            and width >= self.config.fallback_min_width
            and height >= self.config.fallback_min_height
            and self.config.fallback_min_aspect <= aspect <= self.config.fallback_max_aspect
        )

    def _deduplicate(
        self,
        candidates: list[PriceTagCandidate],
    ) -> list[PriceTagCandidate]:
        ordered = sorted(candidates, key=lambda item: item.confidence, reverse=True)
        kept: list[PriceTagCandidate] = []
        for candidate in ordered:
            if any(bbox_iou(candidate.bbox, other.bbox) > self.config.nms_iou for other in kept):
                continue
            kept.append(candidate)
            if len(kept) >= self._dedup_limit(candidates):
                break
        return kept

    def _dedup_limit(self, candidates: list[PriceTagCandidate]) -> int:
        limit = self.config.max_detections_per_frame
        if candidates and all(candidate.source.startswith("color_") for candidate in candidates):
            return min(limit, self.config.max_color_fallback_detections)
        return limit


def iter_tiles(
    width: int,
    height: int,
    tile_size: int,
    stride: int,
    max_tiles: int,
) -> list[tuple[int, int, int, int]]:
    if width <= 0 or height <= 0:
        return []
    tile_size = max(64, tile_size)
    stride = max(32, stride)
    xs = axis_starts(width, tile_size, stride)
    ys = axis_starts(height, tile_size, stride)
    tiles: list[tuple[int, int, int, int]] = []
    for y_min in ys:
        for x_min in xs:
            x_max = min(width, x_min + tile_size)
            y_max = min(height, y_min + tile_size)
            tiles.append((x_min, y_min, x_max, y_max))
            if max_tiles > 0 and len(tiles) >= max_tiles:
                return tiles
    return tiles


def axis_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts
