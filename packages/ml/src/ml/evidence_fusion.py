from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .media import BBox, bbox_iou, center_distance
from .schema import (
    ABSENT_VALUE,
    canonical_text,
    merge_record_values,
    normalize_value,
    record_completeness,
)


@dataclass(frozen=True)
class PriceTagObservation:
    record: dict[str, str]
    bbox: BBox
    frame_timestamp: int
    frame_index: int
    confidence: float
    sharpness: float = 0.0
    source: str = ""

    @property
    def score(self) -> float:
        return self.confidence * 2.0 + record_completeness(self.record) + self.sharpness / 500.0


@dataclass
class PriceTagTrack:
    observations: list[PriceTagObservation] = field(default_factory=list)
    fused_record: dict[str, str] = field(default_factory=dict)
    last_frame_index: int = 0

    def add(self, observation: PriceTagObservation) -> None:
        self.observations.append(observation)
        if not self.fused_record:
            self.fused_record = dict(observation.record)
        else:
            self.fused_record = merge_record_values(self.fused_record, observation.record)
        self.last_frame_index = observation.frame_index

    @property
    def best_observation(self) -> PriceTagObservation:
        return max(self.observations, key=lambda item: item.score)

    def to_record(self) -> dict[str, str]:
        best = self.best_observation
        record = dict(self.fused_record)
        record.update(best.bbox.to_record_values())
        record["frame_timestamp"] = str(best.frame_timestamp)
        return record


@dataclass(frozen=True)
class FusionConfig:
    tracker_iou: float = 0.12
    tracker_center_threshold: float = 250.0
    max_lost: int = 5
    min_track_observations: int = 1


class EvidenceFusionTracker:
    """Temporal fusion for repeated views of the same physical price tag."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()
        self.tracks: list[PriceTagTrack] = []

    def update(self, observations: list[PriceTagObservation], frame_index: int) -> None:
        for observation in observations:
            track = self._find_track(observation, frame_index)
            if track is None:
                track = PriceTagTrack()
                self.tracks.append(track)
            track.add(observation)

    def finalize(self) -> list[dict[str, str]]:
        active = [
            track
            for track in self.tracks
            if len(track.observations) >= self.config.min_track_observations
        ]
        return [track.to_record() for track in sorted(active, key=self._track_sort_key)]

    def _find_track(
        self,
        observation: PriceTagObservation,
        frame_index: int,
    ) -> PriceTagTrack | None:
        best_track: PriceTagTrack | None = None
        best_score = 0.0
        for track in self.tracks:
            if frame_index - track.last_frame_index > self.config.max_lost:
                continue
            score = self._match_score(track, observation)
            if score > best_score:
                best_score = score
                best_track = track
        return best_track if best_score >= 1.0 else None

    def _match_score(self, track: PriceTagTrack, observation: PriceTagObservation) -> float:
        best = track.best_observation
        score = 0.0
        if same_non_empty(
            track.fused_record.get("qr_code_barcode"),
            observation.record.get("qr_code_barcode"),
        ):
            score += 3.5
        if same_non_empty(track.fused_record.get("barcode"), observation.record.get("barcode")):
            score += 3.0
        if same_product_price(track.fused_record, observation.record):
            score += 1.8

        iou = bbox_iou(best.bbox, observation.bbox)
        if iou >= self.config.tracker_iou:
            score += 1.2 + iou

        distance = center_distance(best.bbox, observation.bbox)
        if distance <= self.config.tracker_center_threshold:
            score += 1.0 - (distance / max(1.0, self.config.tracker_center_threshold)) * 0.5
        return score

    def _track_sort_key(self, track: PriceTagTrack) -> tuple[int, float, float]:
        best = track.best_observation
        return (best.frame_timestamp, best.bbox.y_min, best.bbox.x_min)


def same_non_empty(left: Any, right: Any) -> bool:
    a = normalize_value(left)
    b = normalize_value(right)
    return bool(a and b and a != ABSENT_VALUE and b != ABSENT_VALUE and a == b)


def same_product_price(left: dict[str, str], right: dict[str, str]) -> bool:
    left_name = canonical_text(left.get("product_name"))
    right_name = canonical_text(right.get("product_name"))
    if not left_name or not right_name:
        return False
    if left_name != right_name:
        return False
    for column in ("price_default", "price_card", "price_discount"):
        if same_non_empty(left.get(column), right.get(column)):
            return True
    return False
