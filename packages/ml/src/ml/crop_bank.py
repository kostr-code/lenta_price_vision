from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .media import BBox, import_cv2, laplacian_sharpness


@dataclass(frozen=True)
class CropQuality:
    sharpness: float
    area: float
    detector_confidence: float
    qr_decoded: bool
    near_border: bool
    phash: str

    @property
    def score(self) -> float:
        sharpness_score = min(1.0, self.sharpness / 500.0)
        area_score = min(1.0, self.area / 80_000.0)
        qr_bonus = 1.0 if self.qr_decoded else 0.0
        border_score = 0.0 if self.near_border else 1.0
        return (
            0.30 * sharpness_score
            + 0.25 * area_score
            + 0.20 * self.detector_confidence
            + 0.20 * qr_bonus
            + 0.05 * border_score
        )


class CropDeduplicator:
    """Small pHash gate to avoid repeating OCR on nearly identical crops."""

    def __init__(self, max_hamming_distance: int = 2) -> None:
        self.max_hamming_distance = max_hamming_distance
        self._seen: dict[str, float] = {}

    def should_process(self, quality: CropQuality) -> bool:
        if not quality.phash:
            return True
        for known_hash, known_score in self._seen.items():
            if (
                phash_distance(quality.phash, known_hash) <= self.max_hamming_distance
                and known_score >= quality.score
            ):
                return False
        self._seen[quality.phash] = quality.score
        return True


def estimate_crop_quality(
    crop: Any,
    bbox: BBox,
    frame_width: int,
    frame_height: int,
    detector_confidence: float,
    qr_decoded: bool,
) -> CropQuality:
    sharpness = laplacian_sharpness(crop) if crop is not None and crop.size else 0.0
    return CropQuality(
        sharpness=sharpness,
        area=bbox.area,
        detector_confidence=detector_confidence,
        qr_decoded=qr_decoded,
        near_border=is_near_border(bbox, frame_width, frame_height),
        phash=image_phash(crop),
    )


def is_near_border(
    bbox: BBox, frame_width: int, frame_height: int, margin_ratio: float = 0.015
) -> bool:
    margin_x = frame_width * margin_ratio
    margin_y = frame_height * margin_ratio
    return (
        bbox.x_min <= margin_x
        or bbox.y_min <= margin_y
        or bbox.x_max >= frame_width - margin_x
        or bbox.y_max >= frame_height - margin_y
    )


def image_phash(image: Any, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    if image is None or not getattr(image, "size", 0):
        return ""
    cv2 = import_cv2()
    img_size = hash_size * highfreq_factor
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized.astype("float32"))
    low_freq = dct[:hash_size, :hash_size]
    median = float((low_freq.flatten()[1:]).mean())
    bits = (low_freq > median).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return f"{value:0{hash_size * hash_size // 4}x}"


def phash_distance(left: str, right: str) -> int:
    if not left or not right:
        return 64
    return (int(left, 16) ^ int(right, 16)).bit_count()
