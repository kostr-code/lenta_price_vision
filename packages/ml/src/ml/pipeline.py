from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .candidates import CandidateFinderConfig, PriceTagCandidate, PriceTagCandidateFinder
from .crop_bank import CropDeduplicator, CropQuality, estimate_crop_quality
from .evidence_fusion import EvidenceFusionTracker, FusionConfig, PriceTagObservation
from .field_derivation import derive_fields
from .field_extractor import ExtractionInput, PriceTagFieldExtractor
from .media import (
    BBox,
    bbox_iou,
    clamp_bbox,
    crop_image,
    enhance_crop,
    expand_price_tag_crop,
    image_metadata,
    import_cv2,
    iter_sampled_frames,
    laplacian_sharpness,
    load_image,
    video_metadata,
)
from .qr_tools import QRDecoder
from .rail_roi import RailRoi, RailRoiConfig, RailRoiDetector
from .schema import (
    KEY_FIELDS_FOR_QUALITY,
    OUTPUT_COLUMNS,
    comparable_field_match,
    normalize_record,
    read_records_csv,
    record_completeness,
    write_records_csv,
)
from .text_reader import TextReader, TextReaderConfig

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PACKAGE_DIR / "data"
TRACKER_APP_ONLY_KEYS = {"min_track_hits", "stable_min_track_hits"}


@dataclass(frozen=True)
class PipelineConfig:
    mode: str = "cpu_safe"
    sample_fps: float = 2.0
    yolo_weights: str | None = None
    yolo_conf: float = 0.15
    detector_iou: float = 0.5
    detector_imgsz: int = 1280
    detector_device: str | None = None
    tiled_yolo: bool = False
    tile_size: int = 640
    tile_stride: int = 512
    max_tiles_per_frame: int = 64
    enable_detector_fallbacks: bool = True
    fallback_when_tracked: bool = False
    rail_roi_enabled: bool = False
    rail_roi_update_every_good_frames: int = 50
    rail_roi_vertical_margin_ratio: float = 0.12
    rail_roi_min_height_ratio: float = 0.18
    rail_roi_max_height_ratio: float = 0.45
    rail_roi_full_frame_fallback: bool = True
    min_sharpness: float = 40.0
    max_frames: int = 0
    max_detections_per_frame: int = 80
    enable_ocr: bool = True
    enable_qr: bool = True
    qr_scales: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
    qr_try_orientations: bool = True
    qr_try_preprocessing: bool = True
    defer_ocr: bool = True
    zoned_ocr: bool = False
    top_k_crops_per_track: int = 5
    max_crops_per_track: int = 10
    prefer_paddle: bool = True
    use_tesseract_fallback: bool = True
    ocr_lang: str = "ru"
    use_gpu: bool = False
    crop_pad_px: int = 8
    crop_expand_side_pad: float = 0.18
    crop_expand_top_pad: float = 0.10
    crop_expand_bottom_pad: float = 1.45
    tracking_backend: str = "bytetrack"
    tracker_config: str | None = None
    tracking_fallback_to_fusion: bool = False
    tracker_iou: float = 0.12
    tracker_center_threshold: float = 250.0
    max_lost: int = 5
    min_track_observations: int = 1
    crop_phash_dedup: bool = True
    phash_max_distance: int = 2
    derive_qr_fields_when_missing: bool = False
    save_crops: bool = False
    save_debug_json: bool = True
    save_tracking_video: bool = False

    @classmethod
    def from_mode(cls, mode: str, **overrides: Any) -> PipelineConfig:
        normalized = mode.lower().replace("-", "_")
        values: dict[str, Any] = {"mode": normalized}
        if normalized == "fast":
            values.update({"enable_ocr": False, "sample_fps": 1.5, "min_sharpness": 10.0})
        elif normalized in {"accurate", "quality"}:
            values.update(
                {
                    "enable_ocr": True,
                    "enable_qr": True,
                    "sample_fps": 2.0,
                    "tiled_yolo": True,
                    "rail_roi_enabled": True,
                    "prefer_paddle": True,
                }
            )
        else:
            values.update(
                {
                    "mode": "cpu_safe",
                    "enable_ocr": True,
                    "enable_qr": True,
                    "prefer_paddle": False,
                }
            )
        values.update({key: value for key, value in overrides.items() if value is not None})
        if not values.get("yolo_weights"):
            values["yolo_weights"] = os.getenv("ML_YOLO_WEIGHTS") or default_yolo_weights()
        return cls(**values)


@dataclass(frozen=True)
class PipelineRunResult:
    video_path: str
    output_csv: str
    debug_json: str | None
    debug_tracks_json: str | None
    debug_detections_json: str | None
    tracking_video: str | None
    rows: int
    frames_seen: int
    detections_seen: int
    metadata: dict[str, float | int | str]
    status: dict[str, Any]


@dataclass(frozen=True)
class DeferredCropCandidate:
    track_key: str
    track_id: int | None
    frame_order: int
    frame_index: int
    timestamp_ms: int
    candidate_order: int
    source: str
    bbox: BBox
    expanded_bbox: BBox
    frame_width: int
    frame_height: int
    confidence: float
    quality: CropQuality
    enhanced_crop: Any
    color_hint: str
    crop_paths: dict[str, str]


class RetailShelfPipeline:
    """End-to-end video-to-CSV recognizer for shelf price tags."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig.from_mode("cpu_safe")
        self.finder = PriceTagCandidateFinder(
            CandidateFinderConfig(
                yolo_weights=self.config.yolo_weights,
                yolo_conf=self.config.yolo_conf,
                detector_iou=self.config.detector_iou,
                detector_imgsz=self.config.detector_imgsz,
                tiled_yolo=self.config.tiled_yolo,
                tile_size=self.config.tile_size,
                tile_stride=self.config.tile_stride,
                max_tiles_per_frame=self.config.max_tiles_per_frame,
                max_detections_per_frame=self.config.max_detections_per_frame,
            )
        )
        self.qr_decoder = QRDecoder(
            scales=tuple(self.config.qr_scales),
            try_orientations=self.config.qr_try_orientations,
            try_preprocessing=self.config.qr_try_preprocessing,
        )
        self.rail_roi = RailRoiDetector(
            RailRoiConfig(
                enabled=self.config.rail_roi_enabled,
                update_every_good_frames=self.config.rail_roi_update_every_good_frames,
                vertical_margin_ratio=self.config.rail_roi_vertical_margin_ratio,
                min_roi_height_ratio=self.config.rail_roi_min_height_ratio,
                max_roi_height_ratio=self.config.rail_roi_max_height_ratio,
            )
        )
        self.text_reader = TextReader(
            TextReaderConfig(
                enabled=self.config.enable_ocr,
                prefer_paddle=self.config.prefer_paddle,
                use_tesseract_fallback=self.config.use_tesseract_fallback,
                language=self.config.ocr_lang,
                use_gpu=self.config.use_gpu,
                zoned=self.config.zoned_ocr,
            )
        )
        self.extractor = PriceTagFieldExtractor()

    def run_video(self, video_path: Path, output_dir: Path) -> PipelineRunResult:
        if self._uses_bytetrack():
            try:
                return self._run_video_with_bytetrack(video_path, output_dir)
            except Exception:
                if not self.config.tracking_fallback_to_fusion:
                    raise
        return self._run_video_with_fusion(video_path, output_dir)

    def _run_video_with_fusion(self, video_path: Path, output_dir: Path) -> PipelineRunResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        crops_dir = output_dir / "debug_crops"
        if self.config.save_crops:
            crops_dir.mkdir(parents=True, exist_ok=True)

        tracker = EvidenceFusionTracker(self._fusion_config())
        debug: dict[str, Any] = {
            "video": str(video_path),
            "config": asdict(self.config),
            "frames": [],
        }
        detection_events: list[dict[str, Any]] = []
        frames_seen = 0
        detections_seen = 0
        crop_deduplicators: dict[str, CropDeduplicator] = {}
        deferred_crops: dict[str, list[DeferredCropCandidate]] = {}

        for frame_order, frame in enumerate(
            iter_sampled_frames(
                video_path,
                sample_fps=self.config.sample_fps,
                min_sharpness=self.config.min_sharpness,
                max_frames=self.config.max_frames,
            )
        ):
            frames_seen += 1
            frame_observations: list[PriceTagObservation] = []
            rail_roi = self.rail_roi.detect(frame.image, frame_order)
            candidates, detection_roi = self._find_candidates_in_roi(frame.image, rail_roi)
            detections_seen += len(candidates)
            debug_detections: list[dict[str, Any]] = []
            for candidate_order, candidate in enumerate(candidates):
                observation, debug_item = self._process_candidate(
                    video_path=video_path,
                    crops_dir=crops_dir,
                    crop_deduplicators=crop_deduplicators,
                    frame_image=frame.image,
                    frame_order=frame_order,
                    frame_index=frame.index,
                    timestamp_ms=frame.timestamp_ms,
                    candidate=candidate,
                    candidate_order=candidate_order,
                    track_id=None,
                    deferred_crops=deferred_crops,
                )
                frame_observations.append(observation)
                debug_detections.append(debug_item)
                detection_events.append(debug_item)

            tracker.update(frame_observations, frame_order)
            debug["frames"].append(
                {
                    "frame_index": frame.index,
                    "timestamp_ms": frame.timestamp_ms,
                    "sharpness": round(frame.sharpness, 2),
                    "rail_roi": detection_roi.to_debug(),
                    "detections": len(candidates),
                    "sources": [candidate.source for candidate in candidates],
                    "items": debug_detections,
                }
            )

        deferred_reads = self._process_deferred_crops(video_path, tracker, deferred_crops)
        detection_events.extend(deferred_reads)
        records = tracker.finalize()
        output_csv = output_dir / f"{video_path.stem}_recognized.csv"
        write_records_csv(records, output_csv)

        debug_path: Path | None = None
        debug_tracks_path: Path | None = None
        debug_detections_path: Path | None = None
        if self.config.save_debug_json:
            tracks_debug = tracker.debug_tracks()
            debug.update(
                {
                    "rows": len(records),
                    "frames_seen": frames_seen,
                    "detections_seen": detections_seen,
                    "finder": self.finder.yolo_status,
                    "ocr": self.text_reader.status,
                    "tracks": tracks_debug,
                    "deferred_reads": deferred_reads,
                }
            )
            debug_path = output_dir / f"{video_path.stem}_debug.json"
            debug_path.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
            debug_tracks_path = output_dir / "debug_tracks.json"
            debug_tracks_path.write_text(
                json.dumps(tracks_debug, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            debug_detections_path = output_dir / "debug_detections.json"
            debug_detections_path.write_text(
                json.dumps(detection_events, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return PipelineRunResult(
            video_path=str(video_path),
            output_csv=str(output_csv),
            debug_json=str(debug_path) if debug_path else None,
            debug_tracks_json=str(debug_tracks_path) if debug_tracks_path else None,
            debug_detections_json=str(debug_detections_path) if debug_detections_path else None,
            tracking_video=None,
            rows=len(records),
            frames_seen=frames_seen,
            detections_seen=detections_seen,
            metadata=video_metadata(video_path),
            status={"finder": self.finder.yolo_status, "ocr": self.text_reader.status},
        )

    def _run_video_with_bytetrack(self, video_path: Path, output_dir: Path) -> PipelineRunResult:
        model = self.finder.load_yolo_model()
        if model is None:
            raise RuntimeError(self.finder.yolo_status["error"] or "YOLO model is not available")

        output_dir.mkdir(parents=True, exist_ok=True)
        crops_dir = output_dir / "debug_crops"
        if self.config.save_crops:
            crops_dir.mkdir(parents=True, exist_ok=True)

        metadata = video_metadata(video_path)
        fps = float(metadata.get("fps") or 0.0)
        vid_stride = sample_stride(fps, self.config.sample_fps)
        tracker = EvidenceFusionTracker(self._fusion_config())
        tracker_config_argument = self._tracker_config_argument(output_dir)
        debug: dict[str, Any] = {
            "video": str(video_path),
            "config": asdict(self.config),
            "tracking_backend": "bytetrack",
            "tracker_config": tracker_config_argument,
            "min_track_observations": self._effective_min_track_observations(),
            "vid_stride": vid_stride,
            "frames": [],
        }
        detection_events: list[dict[str, Any]] = []
        crop_deduplicators: dict[str, CropDeduplicator] = {}
        deferred_crops: dict[str, list[DeferredCropCandidate]] = {}
        frames_seen = 0
        detections_seen = 0
        tracking_video_path = (
            output_dir / f"{video_path.stem}_yolo_bytetrack.mp4"
            if self.config.save_tracking_video
            else None
        )
        tracking_video_writer: Any | None = None
        tracking_video_fps = max(1.0, fps / max(1, vid_stride)) if fps > 0 else 1.0

        track_kwargs: dict[str, Any] = {}
        if self.config.detector_device:
            track_kwargs["device"] = self.config.detector_device

        results = model.track(
            source=str(video_path),
            stream=True,
            persist=True,
            conf=self.config.yolo_conf,
            iou=self.config.detector_iou,
            imgsz=self.config.detector_imgsz,
            tracker=tracker_config_argument,
            vid_stride=vid_stride,
            verbose=False,
            **track_kwargs,
        )

        for frame_order, result in enumerate(results):
            if self.config.max_frames > 0 and frames_seen >= self.config.max_frames:
                break

            tracking_video_writer = write_tracking_frame(
                writer=tracking_video_writer,
                output_path=tracking_video_path,
                result=result,
                fps=tracking_video_fps,
            )

            frame_image = result.orig_img
            frame_index = frame_order * vid_stride
            timestamp_ms = int(round(frame_index / fps * 1000.0)) if fps > 0 else 0
            sharpness = laplacian_sharpness(frame_image)

            if sharpness < self.config.min_sharpness:
                debug["frames"].append(
                    {
                        "frame_index": frame_index,
                        "timestamp_ms": timestamp_ms,
                        "sharpness": round(sharpness, 2),
                        "detections": 0,
                        "skipped": "blurry",
                    }
                )
                continue

            frames_seen += 1
            tracked_candidates = self._candidates_from_track_result(result)
            fallback_candidates = self._fallback_candidates(frame_image, tracked_candidates)
            candidates = tracked_candidates + fallback_candidates
            detections_seen += len(candidates)

            frame_observations: list[PriceTagObservation] = []
            debug_detections: list[dict[str, Any]] = []
            for candidate_order, (candidate, track_id) in enumerate(candidates):
                observation, debug_item = self._process_candidate(
                    video_path=video_path,
                    crops_dir=crops_dir,
                    crop_deduplicators=crop_deduplicators,
                    frame_image=frame_image,
                    frame_order=frame_order,
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    candidate=candidate,
                    candidate_order=candidate_order,
                    track_id=track_id,
                    deferred_crops=deferred_crops,
                )
                frame_observations.append(observation)
                debug_detections.append(debug_item)
                detection_events.append(debug_item)

            tracker.update(frame_observations, frame_order)
            debug["frames"].append(
                {
                    "frame_index": frame_index,
                    "timestamp_ms": timestamp_ms,
                    "sharpness": round(sharpness, 2),
                    "detections": len(candidates),
                    "tracked_detections": len(tracked_candidates),
                    "fallback_detections": len(fallback_candidates),
                    "items": debug_detections,
                }
            )

        if tracking_video_writer is not None:
            tracking_video_writer.release()

        deferred_reads = self._process_deferred_crops(video_path, tracker, deferred_crops)
        detection_events.extend(deferred_reads)
        records = tracker.finalize()
        output_csv = output_dir / f"{video_path.stem}_recognized.csv"
        write_records_csv(records, output_csv)

        debug_path: Path | None = None
        debug_tracks_path: Path | None = None
        debug_detections_path: Path | None = None
        if self.config.save_debug_json:
            tracks_debug = tracker.debug_tracks()
            debug.update(
                {
                    "rows": len(records),
                    "frames_seen": frames_seen,
                    "detections_seen": detections_seen,
                    "finder": self.finder.yolo_status,
                    "ocr": self.text_reader.status,
                    "tracks": tracks_debug,
                    "deferred_reads": deferred_reads,
                }
            )
            debug_path = output_dir / f"{video_path.stem}_debug.json"
            debug_path.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
            debug_tracks_path = output_dir / "debug_tracks.json"
            debug_tracks_path.write_text(
                json.dumps(tracks_debug, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            debug_detections_path = output_dir / "debug_detections.json"
            debug_detections_path.write_text(
                json.dumps(detection_events, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return PipelineRunResult(
            video_path=str(video_path),
            output_csv=str(output_csv),
            debug_json=str(debug_path) if debug_path else None,
            debug_tracks_json=str(debug_tracks_path) if debug_tracks_path else None,
            debug_detections_json=str(debug_detections_path) if debug_detections_path else None,
            tracking_video=(
                str(tracking_video_path)
                if tracking_video_path is not None and tracking_video_path.exists()
                else None
            ),
            rows=len(records),
            frames_seen=frames_seen,
            detections_seen=detections_seen,
            metadata=metadata,
            status={"finder": self.finder.yolo_status, "ocr": self.text_reader.status},
        )

    def run_image(self, image_path: Path, output_dir: Path) -> PipelineRunResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        crops_dir = output_dir / "debug_crops"
        if self.config.save_crops:
            crops_dir.mkdir(parents=True, exist_ok=True)

        tracker = EvidenceFusionTracker(self._fusion_config())
        debug: dict[str, Any] = {
            "image": str(image_path),
            "config": asdict(self.config),
            "frames": [],
        }
        detection_events: list[dict[str, Any]] = []
        crop_deduplicators: dict[str, CropDeduplicator] = {}
        deferred_crops: dict[str, list[DeferredCropCandidate]] = {}

        image = load_image(image_path)
        sharpness = laplacian_sharpness(image)
        candidates = self.finder.find(image)
        detections_seen = len(candidates)
        frame_observations: list[PriceTagObservation] = []
        debug_detections: list[dict[str, Any]] = []

        for candidate_order, candidate in enumerate(candidates):
            observation, debug_item = self._process_candidate(
                video_path=image_path,
                crops_dir=crops_dir,
                crop_deduplicators=crop_deduplicators,
                frame_image=image,
                frame_order=0,
                frame_index=0,
                timestamp_ms=0,
                candidate=candidate,
                candidate_order=candidate_order,
                track_id=None,
                deferred_crops=deferred_crops,
            )
            frame_observations.append(observation)
            debug_detections.append(debug_item)
            detection_events.append(debug_item)

        tracker.update(frame_observations, 0)
        debug["frames"].append(
            {
                "frame_index": 0,
                "timestamp_ms": 0,
                "sharpness": round(sharpness, 2),
                "detections": detections_seen,
                "sources": [candidate.source for candidate in candidates],
                "items": debug_detections,
            }
        )

        deferred_reads = self._process_deferred_crops(image_path, tracker, deferred_crops)
        detection_events.extend(deferred_reads)
        records = tracker.finalize()
        output_csv = output_dir / f"{image_path.stem}_recognized.csv"
        write_records_csv(records, output_csv)

        debug_path: Path | None = None
        debug_tracks_path: Path | None = None
        debug_detections_path: Path | None = None
        if self.config.save_debug_json:
            tracks_debug = tracker.debug_tracks()
            debug.update(
                {
                    "rows": len(records),
                    "frames_seen": 1,
                    "detections_seen": detections_seen,
                    "finder": self.finder.yolo_status,
                    "ocr": self.text_reader.status,
                    "tracks": tracks_debug,
                    "deferred_reads": deferred_reads,
                }
            )
            debug_path = output_dir / f"{image_path.stem}_debug.json"
            debug_path.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
            debug_tracks_path = output_dir / "debug_tracks.json"
            debug_tracks_path.write_text(
                json.dumps(tracks_debug, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            debug_detections_path = output_dir / "debug_detections.json"
            debug_detections_path.write_text(
                json.dumps(detection_events, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return PipelineRunResult(
            video_path=str(image_path),
            output_csv=str(output_csv),
            debug_json=str(debug_path) if debug_path else None,
            debug_tracks_json=str(debug_tracks_path) if debug_tracks_path else None,
            debug_detections_json=str(debug_detections_path) if debug_detections_path else None,
            tracking_video=None,
            rows=len(records),
            frames_seen=1,
            detections_seen=detections_seen,
            metadata=image_metadata(image_path),
            status={"finder": self.finder.yolo_status, "ocr": self.text_reader.status},
        )

    def _process_candidate(
        self,
        video_path: Path,
        crops_dir: Path,
        crop_deduplicators: dict[str, CropDeduplicator],
        frame_image: Any,
        frame_order: int,
        frame_index: int,
        timestamp_ms: int,
        candidate: PriceTagCandidate,
        candidate_order: int,
        track_id: int | None,
        deferred_crops: dict[str, list[DeferredCropCandidate]],
    ) -> tuple[PriceTagObservation, dict[str, Any]]:
        frame_height, frame_width = frame_image.shape[:2]
        expanded_bbox = expand_price_tag_crop(
            candidate.bbox,
            frame_width,
            frame_height,
            side_pad=self.config.crop_expand_side_pad,
            top_pad=self.config.crop_expand_top_pad,
            bottom_pad=self.config.crop_expand_bottom_pad,
        )
        raw_crop = crop_image(frame_image, candidate.bbox, self.config.crop_pad_px)
        expanded_crop = crop_image(frame_image, expanded_bbox)
        enhanced = enhance_crop(expanded_crop)
        quick_quality = estimate_crop_quality(
            enhanced,
            expanded_bbox,
            frame_width,
            frame_height,
            candidate.confidence,
            qr_decoded=False,
        )
        dedup_key = f"track_{track_id}" if track_id is not None else "untracked"
        deduplicator = crop_deduplicators.setdefault(
            dedup_key,
            CropDeduplicator(self.config.phash_max_distance),
        )
        should_read = not self.config.crop_phash_dedup or deduplicator.should_process(
            quick_quality
        )
        defer_reading = self.config.defer_ocr and track_id is not None
        qr_decodes = []
        text_lines = []
        if should_read and not defer_reading:
            qr_decodes = self.qr_decoder.decode(enhanced) if self.config.enable_qr else []
            text_lines = self.text_reader.read(enhanced) if self.config.enable_ocr else []
        crop_quality = estimate_crop_quality(
            enhanced,
            expanded_bbox,
            frame_width,
            frame_height,
            candidate.confidence,
            qr_decoded=bool(qr_decodes),
        )
        record = self.extractor.extract(
            ExtractionInput(
                filename=video_path.name,
                text_lines=text_lines,
                qr_decodes=qr_decodes,
                color_hint=self._color_hint(candidate.label, candidate.source),
                crop=enhanced,
            )
        )
        record = derive_fields(
            record,
            derive_qr_fields_when_missing=self.config.derive_qr_fields_when_missing,
        )
        record.update(candidate.bbox.to_record_values())
        record["frame_timestamp"] = str(timestamp_ms)
        crop_paths = self._save_debug_crops(
            crops_dir=crops_dir,
            frame_order=frame_order,
            candidate_order=candidate_order,
            track_id=track_id,
            raw_crop=raw_crop,
            expanded_crop=expanded_crop,
        )
        if defer_reading and should_read:
            self._add_deferred_crop(
                deferred_crops,
                DeferredCropCandidate(
                    track_key=dedup_key,
                    track_id=track_id,
                    frame_order=frame_order,
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    candidate_order=candidate_order,
                    source=candidate.source,
                    bbox=candidate.bbox,
                    expanded_bbox=expanded_bbox,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    confidence=candidate.confidence,
                    quality=quick_quality,
                    enhanced_crop=enhanced,
                    color_hint=self._color_hint(candidate.label, candidate.source),
                    crop_paths=crop_paths,
                ),
            )
        observation = PriceTagObservation(
            record=record,
            bbox=candidate.bbox,
            frame_timestamp=timestamp_ms,
            frame_index=frame_order,
            confidence=candidate.confidence,
            sharpness=crop_quality.sharpness,
            source=candidate.source,
            track_id=track_id,
            crop_path=crop_paths.get("expanded"),
            expanded_bbox=expanded_bbox,
        )
        debug_item = {
            "frame_order": frame_order,
            "frame_index": frame_index,
            "timestamp_ms": timestamp_ms,
            "candidate": candidate_order,
            "track_id": track_id,
            "source": candidate.source,
            "bbox": list(candidate.bbox.as_int_tuple()),
            "expanded_bbox": list(expanded_bbox.as_int_tuple()),
            "confidence": round(candidate.confidence, 4),
            "crop_paths": crop_paths,
            "crop_quality": {
                "sharpness": round(crop_quality.sharpness, 3),
                "area": round(crop_quality.area, 1),
                "score": round(crop_quality.score, 4),
                "phash": crop_quality.phash,
                "near_border": crop_quality.near_border,
                "dedup_skipped_ocr": not should_read,
                "deferred_ocr": defer_reading and should_read,
            },
            "qr_decoded": bool(qr_decodes),
            "qr_sources": [decode.source for decode in qr_decodes],
            "ocr_lines": len(text_lines),
            "filled_fields": record_completeness(record),
        }
        return observation, debug_item

    def _save_debug_crops(
        self,
        crops_dir: Path,
        frame_order: int,
        candidate_order: int,
        track_id: int | None,
        raw_crop: Any,
        expanded_crop: Any,
    ) -> dict[str, str]:
        if not self.config.save_crops:
            return {}
        try:
            from .media import import_cv2

            cv2 = import_cv2()
            track_dir = crops_dir / (f"track_{track_id}" if track_id is not None else "untracked")
            track_dir.mkdir(parents=True, exist_ok=True)
            stem = f"frame_{frame_order:05d}_{candidate_order:03d}"
            raw_path = track_dir / f"{stem}_raw.jpg"
            expanded_path = track_dir / f"{stem}_expanded.jpg"
            cv2.imwrite(str(raw_path), raw_crop)
            cv2.imwrite(str(expanded_path), expanded_crop)
            return {"raw": str(raw_path), "expanded": str(expanded_path)}
        except Exception:
            return {}

    def _add_deferred_crop(
        self,
        banks: dict[str, list[DeferredCropCandidate]],
        crop: DeferredCropCandidate,
    ) -> None:
        bank = banks.setdefault(crop.track_key, [])
        bank.append(crop)
        bank.sort(key=lambda item: item.quality.score, reverse=True)
        del bank[max(1, self.config.max_crops_per_track) :]

    def _process_deferred_crops(
        self,
        video_path: Path,
        tracker: EvidenceFusionTracker,
        banks: dict[str, list[DeferredCropCandidate]],
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        top_k = max(1, self.config.top_k_crops_per_track)

        for track_key, crops in sorted(banks.items()):
            selected = sorted(crops, key=lambda item: item.quality.score, reverse=True)[:top_k]
            for rank, crop in enumerate(selected, start=1):
                qr_decodes = (
                    self.qr_decoder.decode(crop.enhanced_crop) if self.config.enable_qr else []
                )
                text_lines = (
                    self.text_reader.read(crop.enhanced_crop) if self.config.enable_ocr else []
                )
                crop_quality = estimate_crop_quality(
                    crop.enhanced_crop,
                    crop.expanded_bbox,
                    crop.frame_width,
                    crop.frame_height,
                    crop.confidence,
                    qr_decoded=bool(qr_decodes),
                )
                record = self.extractor.extract(
                    ExtractionInput(
                        filename=video_path.name,
                        text_lines=text_lines,
                        qr_decodes=qr_decodes,
                        color_hint=crop.color_hint,
                        crop=crop.enhanced_crop,
                    )
                )
                record = derive_fields(
                    record,
                    derive_qr_fields_when_missing=self.config.derive_qr_fields_when_missing,
                )
                record.update(crop.bbox.to_record_values())
                record["frame_timestamp"] = str(crop.timestamp_ms)
                observation = PriceTagObservation(
                    record=record,
                    bbox=crop.bbox,
                    frame_timestamp=crop.timestamp_ms,
                    frame_index=crop.frame_order,
                    confidence=crop.confidence,
                    sharpness=crop_quality.sharpness,
                    source=f"{crop.source}:deferred_ocr",
                    track_id=crop.track_id,
                    crop_path=crop.crop_paths.get("expanded"),
                    expanded_bbox=crop.expanded_bbox,
                )
                tracker.update([observation], crop.frame_order)
                events.append(
                    {
                        "type": "deferred_ocr",
                        "track_key": track_key,
                        "track_id": crop.track_id,
                        "rank": rank,
                        "frame_order": crop.frame_order,
                        "frame_index": crop.frame_index,
                        "timestamp_ms": crop.timestamp_ms,
                        "candidate": crop.candidate_order,
                        "source": crop.source,
                        "bbox": list(crop.bbox.as_int_tuple()),
                        "expanded_bbox": list(crop.expanded_bbox.as_int_tuple()),
                        "confidence": round(crop.confidence, 4),
                        "crop_paths": crop.crop_paths,
                        "crop_quality": {
                            "sharpness": round(crop_quality.sharpness, 3),
                            "area": round(crop_quality.area, 1),
                            "score": round(crop_quality.score, 4),
                            "phash": crop_quality.phash,
                            "near_border": crop_quality.near_border,
                        },
                        "qr_decoded": bool(qr_decodes),
                        "qr_sources": [decode.source for decode in qr_decodes],
                        "ocr_lines": len(text_lines),
                        "ocr_text": [line.text for line in text_lines[:12]],
                        "filled_fields": record_completeness(record),
                    }
                )
        return events

    def _uses_bytetrack(self) -> bool:
        backend = self.config.tracking_backend.lower().replace("-", "_")
        return backend in {"bytetrack", "byte_track", "ultralytics"} and bool(
            self.config.yolo_weights
        )

    def _fusion_config(self) -> FusionConfig:
        return FusionConfig(
            tracker_iou=self.config.tracker_iou,
            tracker_center_threshold=self.config.tracker_center_threshold,
            max_lost=self.config.max_lost,
            min_track_observations=self._effective_min_track_observations(),
        )

    def _effective_min_track_observations(self) -> int:
        minimum = max(1, int(self.config.min_track_observations))
        tracker_path = self._resolved_tracker_config_path()
        if tracker_path is None:
            return minimum

        tracker_values = read_yaml_mapping(tracker_path)
        for key in ("min_track_hits", "stable_min_track_hits"):
            try:
                minimum = max(minimum, int(tracker_values.get(key)))
            except (TypeError, ValueError):
                continue
        return minimum

    def _tracker_config_argument(self, output_dir: Path | None = None) -> str:
        tracker_path = self._resolved_tracker_config_path()
        if tracker_path is None:
            return "bytetrack.yaml"
        if output_dir is not None:
            runtime_path = write_runtime_tracker_config(tracker_path, output_dir)
            if runtime_path is not None:
                return str(runtime_path)
        return str(tracker_path)

    def _resolved_tracker_config_path(self) -> Path | None:
        if self.config.tracker_config:
            resolved = resolve_config_path(self.config.tracker_config)
            path = Path(resolved)
            return path if path.exists() else None
        package_config = PACKAGE_DIR / "configs" / "bytetrack_price.yaml"
        if package_config.exists():
            return package_config
        return None

    def _candidates_from_track_result(
        self,
        result: Any,
    ) -> list[tuple[PriceTagCandidate, int | None]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        frame_height, frame_width = result.orig_img.shape[:2]
        ids = getattr(boxes, "id", None)
        ids_list = ids.detach().cpu().tolist() if ids is not None else []
        candidates: list[tuple[PriceTagCandidate, int | None]] = []

        for index, box in enumerate(boxes):
            coords = box.xyxy[0].detach().cpu().tolist()
            confidence = float(box.conf[0].detach().cpu().item())
            bbox = BBox(*map(float, coords))
            track_id = int(ids_list[index]) if index < len(ids_list) else None
            candidates.append(
                (
                    PriceTagCandidate(
                        bbox=clamp_bbox(bbox, frame_width, frame_height),
                        confidence=confidence,
                        source="bytetrack:yolo",
                    ),
                    track_id,
                )
            )
        return deduplicate_tracked_candidates(candidates)

    def _fallback_candidates(
        self,
        image: Any,
        tracked_candidates: list[tuple[PriceTagCandidate, int | None]],
    ) -> list[tuple[PriceTagCandidate, int | None]]:
        if not self.config.enable_detector_fallbacks:
            return []
        if tracked_candidates and not self.config.fallback_when_tracked:
            return []

        tracked_boxes = [candidate.bbox for candidate, _track_id in tracked_candidates]
        fallbacks = []
        for candidate in self.finder.find_fallbacks(image):
            if any(bbox_iou(candidate.bbox, bbox) > 0.35 for bbox in tracked_boxes):
                continue
            fallbacks.append(
                (
                    PriceTagCandidate(
                        bbox=candidate.bbox,
                        confidence=candidate.confidence,
                        source=f"{candidate.source}:fallback",
                        label=candidate.label,
                    ),
                    None,
                )
            )
        return fallbacks

    def _find_candidates_in_roi(
        self,
        image: Any,
        rail_roi: RailRoi,
    ) -> tuple[list[PriceTagCandidate], RailRoi]:
        if rail_roi.is_full_frame:
            return self.finder.find(image), rail_roi

        roi_image = rail_roi.crop(image)
        roi_candidates = [
            PriceTagCandidate(
                bbox=rail_roi.translate_bbox(candidate.bbox),
                confidence=candidate.confidence,
                source=f"{candidate.source}:rail_roi",
                label=candidate.label,
            )
            for candidate in self.finder.find(roi_image)
        ]
        if roi_candidates or not self.config.rail_roi_full_frame_fallback:
            return roi_candidates, rail_roi
        height, width = image.shape[:2]
        return self.finder.find(image), RailRoi(
            0,
            0,
            width,
            height,
            rail_roi.score,
            "rail_roi_fallback_full_frame",
        )

    def _color_hint(self, label: str, source: str) -> str:
        for color in ("red", "yellow", "green", "white"):
            if label == color or source.endswith(color):
                return color
        return ""


@dataclass(frozen=True)
class LabeledSequence:
    name: str
    directory: Path
    video_path: Path
    csv_path: Path


def default_yolo_weights() -> str | None:
    candidates = [
        Path.cwd() / "models" / "best.pt",
        Path.cwd() / "models" / "price_tag_yolo.pt",
        PACKAGE_DIR.parents[3] / "models" / "best.pt",
        PACKAGE_DIR.parents[3] / "models" / "price_tag_yolo.pt",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def write_tracking_frame(
    writer: Any | None,
    output_path: Path | None,
    result: Any,
    fps: float,
) -> Any | None:
    if output_path is None:
        return writer

    try:
        frame = result.plot()
        if frame is None:
            frame = result.orig_img
        height, width = frame.shape[:2]
        cv2 = import_cv2()
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, max(1.0, fps), (width, height))
            if not writer.isOpened():
                return None
        writer.write(frame)
    except Exception:
        return writer
    return writer


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_runtime_tracker_config(tracker_path: Path, output_dir: Path) -> Path | None:
    data = read_yaml_mapping(tracker_path)
    if not data or not any(key in data for key in TRACKER_APP_ONLY_KEYS):
        return None

    runtime_data = {
        key: value for key, value in data.items() if key not in TRACKER_APP_ONLY_KEYS
    }
    runtime_path = output_dir / "_bytetrack_runtime.yaml"
    try:
        import yaml  # type: ignore

        runtime_path.write_text(
            yaml.safe_dump(runtime_data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except Exception:
        return None
    return runtime_path


def resolve_config_path(value: str) -> str:
    path = Path(value)
    candidates = [
        path,
        Path.cwd() / path,
        PACKAGE_DIR / path,
        PACKAGE_DIR / "configs" / path.name,
        PACKAGE_DIR.parents[3] / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return value


def sample_stride(fps: float, sample_fps: float) -> int:
    if fps <= 0 or sample_fps <= 0:
        return 1
    return max(1, int(round(fps / sample_fps)))


def deduplicate_tracked_candidates(
    candidates: list[tuple[PriceTagCandidate, int | None]],
    iou_threshold: float = 0.75,
) -> list[tuple[PriceTagCandidate, int | None]]:
    ordered = sorted(candidates, key=lambda item: item[0].confidence, reverse=True)
    kept: list[tuple[PriceTagCandidate, int | None]] = []
    for candidate, track_id in ordered:
        if any(bbox_iou(candidate.bbox, other.bbox) > iou_threshold for other, _ in kept):
            continue
        kept.append((candidate, track_id))
    return kept


def discover_labeled_sequences(data_dir: Path = DEFAULT_DATA_DIR) -> list[LabeledSequence]:
    sequences: list[LabeledSequence] = []
    if not data_dir.exists():
        return sequences
    for directory in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        csv_files = sorted(directory.glob("*.csv"))
        video_files = sorted(directory.glob("*.mp4"))
        if not csv_files or not video_files:
            continue
        csv_path = csv_files[0]
        same_stem = [video for video in video_files if video.stem == csv_path.stem]
        video_path = (
            same_stem[0] if same_stem else max(video_files, key=lambda path: path.stat().st_size)
        )
        sequences.append(
            LabeledSequence(
                name=directory.name,
                directory=directory,
                video_path=video_path,
                csv_path=csv_path,
            )
        )
    return sequences


def evaluate_prediction_csv(
    ground_truth_csv: Path,
    predicted_csv: Path,
    iou_threshold: float = 0.15,
) -> dict[str, Any]:
    truth = read_records_csv(ground_truth_csv)
    predicted = read_records_csv(predicted_csv)
    matches: list[dict[str, Any]] = []
    used_predictions: set[int] = set()

    for truth_row in truth:
        truth_box = bbox_from_record(truth_row)
        best_index = -1
        best_iou = 0.0
        for index, pred_row in enumerate(predicted):
            if index in used_predictions:
                continue
            pred_box = bbox_from_record(pred_row)
            iou = bbox_iou(truth_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index < 0 or best_iou < iou_threshold:
            matches.append({"matched": False, "iou": best_iou, "field_accuracy": 0.0})
            continue
        used_predictions.add(best_index)
        pred_row = predicted[best_index]
        compared = [
            comparable_field_match(truth_row.get(field), pred_row.get(field))
            for field in KEY_FIELDS_FOR_QUALITY
        ]
        accuracy = sum(compared) / max(1, len(compared))
        matches.append({"matched": True, "iou": best_iou, "field_accuracy": accuracy})

    quality_hits = [item for item in matches if item["matched"] and item["field_accuracy"] >= 0.8]
    return {
        "ground_truth_rows": len(truth),
        "predicted_rows": len(predicted),
        "matched_rows": sum(1 for item in matches if item["matched"]),
        "quality_rows_at_80pct": len(quality_hits),
        "proxy_score": len(quality_hits) / max(1, len(truth)),
        "duplicate_prediction_rows": max(0, len(predicted) - len(used_predictions)),
        "avg_iou": average(item["iou"] for item in matches if item["matched"]),
        "avg_field_accuracy": average(
            item["field_accuracy"] for item in matches if item["matched"]
        ),
    }


def run_public_evaluation(
    data_dir: Path,
    output_dir: Path,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = RetailShelfPipeline(config or PipelineConfig.from_mode("cpu_safe"))
    sequence_reports: list[dict[str, Any]] = []
    for sequence in discover_labeled_sequences(data_dir):
        sequence_output = output_dir / sequence.name
        run = pipeline.run_video(sequence.video_path, sequence_output)
        metrics = evaluate_prediction_csv(sequence.csv_path, Path(run.output_csv))
        sequence_reports.append(
            {
                "name": sequence.name,
                "video": str(sequence.video_path),
                "ground_truth_csv": str(sequence.csv_path),
                "prediction_csv": run.output_csv,
                "metrics": metrics,
            }
        )
    report = {
        "columns": OUTPUT_COLUMNS,
        "sequences": sequence_reports,
        "mean_proxy_score": average(item["metrics"]["proxy_score"] for item in sequence_reports),
    }
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def bbox_from_record(row: dict[str, str]) -> BBox:
    return BBox(
        parse_coord(row.get("x_min")),
        parse_coord(row.get("y_min")),
        parse_coord(row.get("x_max")),
        parse_coord(row.get("y_max")),
    )


def parse_coord(value: Any) -> float:
    text = str(value or "").strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return 0.0


def average(values: Any) -> float:
    collected = [float(value) for value in values]
    if not collected:
        return 0.0
    return sum(collected) / len(collected)


def summarize_csv(path: Path) -> dict[str, Any]:
    rows = read_records_csv(path)
    return {
        "path": str(path),
        "rows": len(rows),
        "columns": OUTPUT_COLUMNS,
        "avg_completeness": average(record_completeness(normalize_record(row)) for row in rows),
    }
