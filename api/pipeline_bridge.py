"""
api/pipeline_bridge.py — Wraps pipeline modules for API use.

Two entry points:
  load_models(settings)          — call once on startup
  process_single_crop(crop_bgr)  — image endpoint: one crop → 29-field dict
  process_video_file(...)        — video endpoint: full ByteTrack pipeline
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import structlog

from pipeline.parsers import (
    ABSENT_VALUE,
    OUTPUT_COLUMNS,
    make_empty_row,
    normalize_text,
    parse_fields,
)
from pipeline.quality import estimate_crop_quality
from pipeline.video import cut_crop_bbox, rotate_frame
from pipeline.vlm import Qwen25VLProvider, VLMConfig, VLMProvider
from pipeline.fragments import (
    FragmentMap,
    FragmentProvider,
    heuristic_provider,
    load_yolo2_provider,
)

if TYPE_CHECKING:
    from api.config import MLSettings

# ── Optional imports ──────────────────────────────────────────────────────────

_HAS_OCR = False
try:
    from pipeline.ocr import enhance_crop, load_ocr, ocr_zoned

    _HAS_OCR = True
except ImportError:
    pass

_HAS_QR = False
try:
    from pipeline.qr import (
        decode_barcode_linear,
        decode_qr,
        load_wechat,
        parse_qr_payload,
    )

    _HAS_QR = True
except ImportError:
    pass

_HAS_DETECTOR = False
try:
    from pipeline.detector import (
        Detection,
        load_detector,
        detect_price_tags,
        track_price_tags,
    )

    _HAS_DETECTOR = True
except ImportError:
    pass

try:
    from pipeline.sr import enhance_clahe, enhance_sharpen

    BARCODE_ENHANCE_STEPS: list = [enhance_clahe, enhance_sharpen]
except ImportError:
    BARCODE_ENHANCE_STEPS = []

QR_ENHANCE_STEPS: list = []

# ── ByteTrack config ──

_DEFAULT_TRACKER = str(
    Path(__file__).resolve().parent.parent / "train" / "bytetrack_price.yaml"
)

# ── Module-level singletons (loaded once on startup) ──

_vlm_provider: VLMProvider | None = None
_fragment_provider: FragmentProvider = heuristic_provider
_ocr_model: Any = None
_wechat: Any = None


def load_models(settings: "MLSettings") -> None:
    """Load all models into module-level singletons. Call once at server startup."""
    global _vlm_provider, _fragment_provider, _ocr_model, _wechat
    log = structlog.get_logger("bridge")

    # VLM
    log.info("models.load_vlm", model=settings.vlm_model_id, four_bit=settings.vlm_4bit)
    _vlm_provider = Qwen25VLProvider(
        VLMConfig(
            model_id=settings.vlm_model_id,
            load_in_4bit=settings.vlm_4bit,
            device_map=settings.vlm_device_map,
        )
    )
    _vlm_provider.load()
    log.info("models.vlm_ready")

    # YOLO Stage 2 (fragment provider)
    if settings.weights_inside and Path(settings.weights_inside).exists():
        log.info("models.load_yolo2", weights=settings.weights_inside)
        _fragment_provider = load_yolo2_provider(settings.weights_inside)
        log.info("models.yolo2_ready")
    else:
        log.info("models.heuristic_fragments")

    # OCR
    if _HAS_OCR:
        log.info("models.load_ocr")
        _ocr_model = load_ocr(use_gpu=True)
        log.info("models.ocr_ready")

    # WeChat QR
    if _HAS_QR:
        try:
            _wechat = load_wechat()
            log.info("models.wechat_ready")
        except Exception as exc:
            log.warning("models.wechat_failed", error=str(exc))


def model_status() -> dict[str, bool]:
    return {
        "vlm": _vlm_provider is not None,
        "ocr": _ocr_model is not None,
        "yolo2": not isinstance(_fragment_provider, type(heuristic_provider)),
        "wechat": _wechat is not None,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────


def _decode_price_tag(
    crop: np.ndarray,
    fragment_provider: FragmentProvider,
) -> tuple[list[dict], str]:
    if not _HAS_QR:
        return [], ""

    fragments: FragmentMap = fragment_provider(crop)

    qr_img = fragments.get("qr_code")
    texts: list[str] = []
    if qr_img is not None:
        texts = decode_qr(qr_img, wechat=_wechat, enhance_steps=QR_ENHANCE_STEPS)
    if not texts:
        texts = decode_qr(crop, wechat=_wechat, enhance_steps=QR_ENHANCE_STEPS)
    qr_payloads = [parse_qr_payload(t) for t in texts if t]

    barcode_str = ""
    barcode_img = fragments.get("barcode")
    if barcode_img is not None:
        codes = decode_barcode_linear(barcode_img, enhance_steps=BARCODE_ENHANCE_STEPS)
        barcode_str = codes[0] if codes else ""

    return qr_payloads, barcode_str


def _run_ocr(crop_bgr: np.ndarray, qr_payloads: list[dict]) -> dict[str, str]:
    if not _HAS_OCR or _ocr_model is None:
        return {}
    try:
        enhanced = enhance_crop(crop_bgr)
        lines = ocr_zoned(_ocr_model, enhanced)
        return parse_fields(lines, qr_payloads, crop_bgr)
    except Exception as exc:
        structlog.get_logger("bridge").warning("ocr.failed", error=str(exc))
        return {}


def _merge_vlm_and_ocr(vlm: dict[str, str], ocr: dict[str, str]) -> dict[str, str]:
    merged = dict(vlm)
    for k, v in ocr.items():
        if k not in merged or not merged[k]:
            merged[k] = v
    return merged


def _rank_crop(crop: np.ndarray) -> float:
    return estimate_crop_quality(crop)


def _fill_output_row(
    out_row: dict[str, str],
    combined: dict[str, str],
    qr_payloads: list[dict],
    barcode_str: str,
    quality: float,
) -> dict[str, str]:
    for col in OUTPUT_COLUMNS:
        if col in combined and combined[col]:
            out_row[col] = normalize_text(combined[col])

    for payload in qr_payloads:
        for csv_col, val in payload.items():
            if csv_col in out_row and not out_row[csv_col]:
                out_row[csv_col] = val

    if barcode_str and not out_row.get("barcode"):
        out_row["barcode"] = barcode_str

    for col in (
        "price_discount",
        "discount_amount",
        "code",
        "additional_info",
        "special_symbols",
    ):
        if not out_row[col]:
            out_row[col] = ABSENT_VALUE

    out_row["_quality"] = f"{quality:.3f}"
    return out_row


# ── Public API ────────────────────────────────────────────────────────────────


def process_single_crop(
    crop_bgr: np.ndarray,
    use_vlm: bool = True,
    use_ocr: bool = True,
) -> dict[str, str]:
    """
    Run full pipeline on a single price-tag crop.

    Used by the image endpoint. Returns a 29-field dict.
    """
    log = structlog.get_logger("bridge")
    log.info("crop.start", h=crop_bgr.shape[0], w=crop_bgr.shape[1])

    quality = _rank_crop(crop_bgr)
    qr_payloads, barcode_str = _decode_price_tag(crop_bgr, _fragment_provider)

    if qr_payloads:
        log.info("crop.qr_ok", fields=list(qr_payloads[0].keys()))
    if barcode_str:
        log.info("crop.barcode", value=barcode_str)

    vlm_result: dict[str, str] = {}
    if use_vlm and _vlm_provider is not None:
        vlm_result = _vlm_provider.extract_fields(crop_bgr)
        log.info("crop.vlm_done", fields_found=sum(1 for v in vlm_result.values() if v))

    ocr_fields: dict[str, str] = {}
    if use_ocr:
        ocr_fields = _run_ocr(crop_bgr, qr_payloads)
        log.info("crop.ocr_done", fields_found=sum(1 for v in ocr_fields.values() if v))

    combined = _merge_vlm_and_ocr(vlm_result, ocr_fields)
    out_row = make_empty_row()
    _fill_output_row(out_row, combined, qr_payloads, barcode_str, quality)

    log.info("crop.done", quality=f"{quality:.3f}")
    return out_row


@dataclass
class _CropCandidate:
    crop: np.ndarray
    score: float
    ts_ms: float
    det: Any


def process_video_file(
    video_path: str,
    run_id: str,
    runs_dir: str,
    quality_threshold: float = 0.2,
    track_top_k: int = 3,
    use_vlm: bool = True,
    use_ocr: bool = True,
    conf_det: float = 0.25,
    device: str | None = None,
    tracker: str = _DEFAULT_TRACKER,
) -> list[dict[str, str]]:
    """
    Run full unlabeled video pipeline: ByteTrack → top-K crops → VLM + OCR.

    Returns list of output rows (one per unique track). Caller is responsible
    for writing CSV and debug files to runs_dir/run_id/.
    """
    log = structlog.get_logger("bridge").bind(run_id=run_id)

    if not _HAS_DETECTOR:
        raise RuntimeError("ultralytics not installed — cannot process video")

    video_name = Path(video_path).name
    log.info("video.start", path=video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    log.info("video.opened", frames=total_frames)

    track_candidates: dict[int, list[_CropCandidate]] = defaultdict(list)
    frame_idx = 0

    while True:
        ok, raw = cap.read()
        if not ok:
            break
        frame = rotate_frame(raw)
        ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        detections = track_price_tags(
            frame, conf=conf_det, device=device, tracker=tracker
        )

        for det in detections:
            if det.track_id is None:
                continue
            crop = cut_crop_bbox(frame, det.x1, det.y1, det.x2, det.y2)
            if crop is None or crop.size == 0:
                continue
            score = _rank_crop(crop)
            bucket = track_candidates[det.track_id]
            bucket.append(_CropCandidate(crop=crop, score=score, ts_ms=ts_ms, det=det))
            if len(bucket) > track_top_k:
                bucket.sort(key=lambda c: c.score, reverse=True)
                del bucket[track_top_k:]

        frame_idx += 1
        if frame_idx % 100 == 0:
            log.info(
                "video.progress",
                frame=frame_idx,
                total=total_frames,
                tracks=len(track_candidates),
            )

    cap.release()
    log.info("video.tracking_done", unique_tracks=len(track_candidates))

    output_rows: list[dict[str, str]] = []

    for track_id, candidates in sorted(track_candidates.items()):
        best = max(candidates, key=lambda c: c.score)

        if best.score < quality_threshold:
            log.info(
                "track.skip",
                track_id=track_id,
                score=f"{best.score:.3f}",
                reason="low_quality",
            )
            continue

        log.info(
            "track.process",
            track_id=track_id,
            score=f"{best.score:.3f}",
            ts=f"{best.ts_ms:.0f}ms",
        )

        qr_payloads, barcode_str = _decode_price_tag(best.crop, _fragment_provider)
        if qr_payloads:
            log.info(
                "track.qr_ok", track_id=track_id, fields=list(qr_payloads[0].keys())
            )

        vlm_result: dict[str, str] = {}
        if use_vlm and _vlm_provider is not None:
            vlm_result = _vlm_provider.extract_fields(best.crop)
            log.info(
                "track.vlm_done",
                track_id=track_id,
                fields=sum(1 for v in vlm_result.values() if v),
            )

        ocr_fields: dict[str, str] = {}
        if use_ocr:
            ocr_fields = _run_ocr(best.crop, qr_payloads)
            log.info(
                "track.ocr_done",
                track_id=track_id,
                fields=sum(1 for v in ocr_fields.values() if v),
            )

        combined = _merge_vlm_and_ocr(vlm_result, ocr_fields)
        out_row = make_empty_row()
        out_row["filename"] = video_name
        out_row["frame_timestamp"] = str(int(best.ts_ms))
        out_row["x_min"] = str(best.det.x1)
        out_row["y_min"] = str(best.det.y1)
        out_row["x_max"] = str(best.det.x2)
        out_row["y_max"] = str(best.det.y2)

        _fill_output_row(out_row, combined, qr_payloads, barcode_str, best.score)
        out_row["_track_id"] = str(track_id)
        output_rows.append(out_row)

    log.info("video.done", rows=len(output_rows))
    return output_rows


def save_run_files(
    run_id: str,
    rows: list[dict[str, str]],
    runs_dir: str,
) -> dict[str, Path]:
    """Write output.csv and debug.json to runs_dir/run_id/. Returns file paths."""
    import pandas as pd

    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cols = OUTPUT_COLUMNS + ["_quality", "_track_id"]
    out_csv = run_dir / "output.csv"
    pd.DataFrame(
        rows, columns=[c for c in cols if c in (rows[0] if rows else {})]
    ).to_csv(out_csv, index=False)

    debug_path = run_dir / "debug.json"
    debug_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    return {"output.csv": out_csv, "debug.json": debug_path}
