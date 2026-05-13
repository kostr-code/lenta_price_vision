from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .candidates import CandidateFinderConfig, PriceTagCandidate, PriceTagCandidateFinder
from .crop_bank import CropDeduplicator, estimate_crop_quality
from .evidence_fusion import EvidenceFusionTracker, FusionConfig, PriceTagObservation
from .field_derivation import derive_fields
from .field_extractor import ExtractionInput, PriceTagFieldExtractor
from .media import BBox, bbox_iou, crop_image, enhance_crop, iter_sampled_frames, video_metadata
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


@dataclass(frozen=True)
class PipelineConfig:
    mode: str = "cpu_safe"
    sample_fps: float = 2.0
    yolo_weights: str | None = None
    yolo_conf: float = 0.23
    detector_imgsz: int = 1600
    tiled_yolo: bool = False
    tile_size: int = 640
    tile_stride: int = 512
    max_tiles_per_frame: int = 64
    rail_roi_enabled: bool = False
    rail_roi_update_every_good_frames: int = 50
    rail_roi_vertical_margin_ratio: float = 0.12
    rail_roi_min_height_ratio: float = 0.18
    rail_roi_max_height_ratio: float = 0.45
    rail_roi_full_frame_fallback: bool = True
    min_sharpness: float = 18.0
    max_frames: int = 0
    max_detections_per_frame: int = 80
    enable_ocr: bool = True
    enable_qr: bool = True
    prefer_paddle: bool = True
    ocr_lang: str = "ru"
    use_gpu: bool = False
    crop_pad_px: int = 8
    tracker_iou: float = 0.12
    tracker_center_threshold: float = 250.0
    max_lost: int = 5
    min_track_observations: int = 1
    crop_phash_dedup: bool = True
    phash_max_distance: int = 2
    derive_qr_fields_when_missing: bool = False
    save_crops: bool = False
    save_debug_json: bool = True

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
                }
            )
        else:
            values.update({"mode": "cpu_safe", "enable_ocr": True, "enable_qr": True})
        values.update({key: value for key, value in overrides.items() if value is not None})
        if not values.get("yolo_weights"):
            values["yolo_weights"] = os.getenv("ML_YOLO_WEIGHTS") or default_yolo_weights()
        return cls(**values)


@dataclass(frozen=True)
class PipelineRunResult:
    video_path: str
    output_csv: str
    debug_json: str | None
    rows: int
    frames_seen: int
    detections_seen: int
    metadata: dict[str, float | int | str]
    status: dict[str, Any]


class RetailShelfPipeline:
    """End-to-end video-to-CSV recognizer for shelf price tags."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig.from_mode("cpu_safe")
        self.finder = PriceTagCandidateFinder(
            CandidateFinderConfig(
                yolo_weights=self.config.yolo_weights,
                yolo_conf=self.config.yolo_conf,
                detector_imgsz=self.config.detector_imgsz,
                tiled_yolo=self.config.tiled_yolo,
                tile_size=self.config.tile_size,
                tile_stride=self.config.tile_stride,
                max_tiles_per_frame=self.config.max_tiles_per_frame,
                max_detections_per_frame=self.config.max_detections_per_frame,
            )
        )
        self.qr_decoder = QRDecoder()
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
                language=self.config.ocr_lang,
                use_gpu=self.config.use_gpu,
            )
        )
        self.extractor = PriceTagFieldExtractor()

    def run_video(self, video_path: Path, output_dir: Path) -> PipelineRunResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        crops_dir = output_dir / "crops"
        if self.config.save_crops:
            crops_dir.mkdir(parents=True, exist_ok=True)

        tracker = EvidenceFusionTracker(
            FusionConfig(
                tracker_iou=self.config.tracker_iou,
                tracker_center_threshold=self.config.tracker_center_threshold,
                max_lost=self.config.max_lost,
                min_track_observations=self.config.min_track_observations,
            )
        )
        debug: dict[str, Any] = {
            "video": str(video_path),
            "config": asdict(self.config),
            "frames": [],
        }
        frames_seen = 0
        detections_seen = 0
        crop_deduplicator = CropDeduplicator(self.config.phash_max_distance)

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
            frame_height, frame_width = frame.image.shape[:2]
            debug_detections: list[dict[str, Any]] = []
            for candidate_order, candidate in enumerate(candidates):
                crop = crop_image(frame.image, candidate.bbox, self.config.crop_pad_px)
                enhanced = enhance_crop(crop)
                quick_quality = estimate_crop_quality(
                    enhanced,
                    candidate.bbox,
                    frame_width,
                    frame_height,
                    candidate.confidence,
                    qr_decoded=False,
                )
                should_read = not self.config.crop_phash_dedup or crop_deduplicator.should_process(
                    quick_quality
                )
                qr_decodes = (
                    self.qr_decoder.decode(enhanced)
                    if self.config.enable_qr and should_read
                    else []
                )
                text_lines = (
                    self.text_reader.read(enhanced)
                    if self.config.enable_ocr and should_read
                    else []
                )
                crop_quality = estimate_crop_quality(
                    enhanced,
                    candidate.bbox,
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
                record["frame_timestamp"] = str(frame.timestamp_ms)
                frame_observations.append(
                    PriceTagObservation(
                        record=record,
                        bbox=candidate.bbox,
                        frame_timestamp=frame.timestamp_ms,
                        frame_index=frame_order,
                        confidence=candidate.confidence,
                        sharpness=crop_quality.sharpness,
                        source=candidate.source,
                    )
                )
                debug_detections.append(
                    {
                        "candidate": candidate_order,
                        "source": candidate.source,
                        "bbox": list(candidate.bbox.as_int_tuple()),
                        "confidence": round(candidate.confidence, 4),
                        "crop_quality": {
                            "sharpness": round(crop_quality.sharpness, 3),
                            "area": round(crop_quality.area, 1),
                            "score": round(crop_quality.score, 4),
                            "phash": crop_quality.phash,
                            "near_border": crop_quality.near_border,
                            "dedup_skipped_ocr": not should_read,
                        },
                        "qr_decoded": bool(qr_decodes),
                        "qr_sources": [decode.source for decode in qr_decodes],
                        "ocr_lines": len(text_lines),
                        "filled_fields": record_completeness(record),
                    }
                )
                if self.config.save_crops:
                    self._save_crop(crops_dir, frame_order, candidate_order, enhanced)

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

        records = tracker.finalize()
        output_csv = output_dir / f"{video_path.stem}_recognized.csv"
        write_records_csv(records, output_csv)

        debug_path: Path | None = None
        if self.config.save_debug_json:
            debug.update(
                {
                    "rows": len(records),
                    "frames_seen": frames_seen,
                    "detections_seen": detections_seen,
                    "finder": self.finder.yolo_status,
                    "ocr": self.text_reader.status,
                    "tracks": tracker.debug_tracks(),
                }
            )
            debug_path = output_dir / f"{video_path.stem}_debug.json"
            debug_path.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")

        return PipelineRunResult(
            video_path=str(video_path),
            output_csv=str(output_csv),
            debug_json=str(debug_path) if debug_path else None,
            rows=len(records),
            frames_seen=frames_seen,
            detections_seen=detections_seen,
            metadata=video_metadata(video_path),
            status={"finder": self.finder.yolo_status, "ocr": self.text_reader.status},
        )

    def _save_crop(
        self,
        crops_dir: Path,
        frame_order: int,
        candidate_order: int,
        image: Any,
    ) -> None:
        try:
            from .media import import_cv2

            cv2 = import_cv2()
            crop_path = crops_dir / f"frame_{frame_order:05d}_{candidate_order:03d}.jpg"
            cv2.imwrite(str(crop_path), image)
        except Exception:
            return

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
        Path.cwd() / "models" / "price_tag_yolo.pt",
        PACKAGE_DIR.parents[3] / "models" / "price_tag_yolo.pt",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


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
