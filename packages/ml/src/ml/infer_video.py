from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .pipeline import PipelineConfig, RetailShelfPipeline, resolve_config_path
from .schema import read_records_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run video-to-CSV price tag inference.")
    parser.add_argument("video", type=Path, help="Input .mp4 video")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        default="cpu_safe",
        choices=["fast", "cpu_safe", "accurate", "quality"],
    )
    parser.add_argument("--weights", default=None, help="YOLO weights path")
    parser.add_argument("--sample-fps", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tiled-yolo", action="store_true")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--tile-stride", type=int, default=None)
    parser.add_argument("--max-tiles-per-frame", type=int, default=None)
    parser.add_argument("--tracking-backend", default=None, choices=["bytetrack", "fusion"])
    parser.add_argument("--tracker-config", default=None)
    parser.add_argument("--no-defer-ocr", action="store_true")
    parser.add_argument("--top-k-crops-per-track", type=int, default=None)
    parser.add_argument("--zoned-ocr", action="store_true")
    parser.add_argument("--derive-qr-fields", action="store_true")
    parser.add_argument("--disable-ocr", action="store_true")
    parser.add_argument("--disable-qr", action="store_true")
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--no-debug-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_values = config_overrides_from_file(args.config)
    config_values.update(config_overrides_from_args(args))
    config = PipelineConfig.from_mode(args.mode, **config_values)
    result = RetailShelfPipeline(config).run_video(args.video, args.output_dir)
    rows = read_records_csv(Path(result.output_csv))
    print(
        json.dumps(
            {
                **asdict(result),
                "rows_preview": rows[:3],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def config_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {
        "yolo_weights": args.weights,
        "sample_fps": args.sample_fps,
        "max_frames": args.max_frames,
        "yolo_conf": args.conf,
        "detector_iou": args.iou,
        "detector_imgsz": args.imgsz,
        "detector_device": args.device,
        "tiled_yolo": True if args.tiled_yolo else None,
        "tile_size": args.tile_size,
        "tile_stride": args.tile_stride,
        "max_tiles_per_frame": args.max_tiles_per_frame,
        "tracking_backend": args.tracking_backend,
        "tracker_config": args.tracker_config,
        "defer_ocr": False if args.no_defer_ocr else None,
        "top_k_crops_per_track": args.top_k_crops_per_track,
        "zoned_ocr": True if args.zoned_ocr else None,
        "derive_qr_fields_when_missing": True if args.derive_qr_fields else None,
        "enable_ocr": False if args.disable_ocr else None,
        "enable_qr": False if args.disable_qr else None,
        "save_crops": True if args.save_crops else None,
        "save_debug_json": False if args.no_debug_json else None,
    }
    return {key: value for key, value in values.items() if value is not None}


def config_overrides_from_file(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = read_yaml(path)
    values: dict[str, Any] = {}

    detector = data.get("detector", {})
    weights = detector.get("weights")
    values.update(
        {
            "yolo_weights": resolve_config_path(str(weights)) if weights else None,
            "detector_imgsz": detector.get("imgsz"),
            "yolo_conf": detector.get("conf"),
            "detector_iou": detector.get("iou"),
            "detector_device": normalize_device(detector.get("device")),
            "tiled_yolo": detector.get("tiled"),
            "tile_size": detector.get("tile_size"),
            "tile_stride": detector.get("tile_stride"),
            "max_tiles_per_frame": detector.get("max_tiles_per_frame"),
            "enable_detector_fallbacks": detector.get("enable_fallbacks"),
            "fallback_when_tracked": detector.get("fallback_when_tracked"),
        }
    )

    sampling = data.get("sampling", {})
    values.update(
        {
            "sample_fps": sampling.get("sample_fps"),
            "min_sharpness": sampling.get("min_laplacian_var"),
        }
    )

    tracking = data.get("tracking", {})
    values.update(
        {
            "tracking_backend": "bytetrack" if tracking.get("enabled", True) else "fusion",
            "tracker_config": tracking.get("tracker_config") or tracking.get("tracker"),
            "min_track_observations": tracking.get("min_track_hits"),
        }
    )

    crop = data.get("crop", {})
    values.update(
        {
            "crop_expand_side_pad": crop.get("expand_side_pad"),
            "crop_expand_top_pad": crop.get("expand_top_pad"),
            "crop_expand_bottom_pad": crop.get("expand_bottom_pad"),
            "save_crops": crop.get("save_debug_crops"),
            "max_crops_per_track": crop.get("max_crops_per_track"),
        }
    )

    qr = data.get("qr", {})
    ocr = data.get("ocr", {})
    output = data.get("output", {})
    values.update(
        {
            "enable_qr": qr.get("enabled"),
            "qr_scales": tuple(qr.get("try_scales") or (1.0, 1.5, 2.0, 3.0)),
            "qr_try_orientations": qr.get("try_orientations"),
            "qr_try_preprocessing": qr.get("try_preprocessing"),
            "enable_ocr": ocr.get("enabled"),
            "defer_ocr": ocr.get("defer"),
            "top_k_crops_per_track": ocr.get("top_k_crops_per_track"),
            "zoned_ocr": ocr.get("zoned"),
            "prefer_paddle": ocr.get("use_paddleocr"),
            "use_tesseract_fallback": ocr.get("use_tesseract_fallback"),
            "derive_qr_fields_when_missing": data.get("parser", {}).get("derive_qr_fields"),
            "save_debug_json": output.get("write_debug_json"),
        }
    )
    return {key: value for key, value in values.items() if value is not None}


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read --config YAML files") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def normalize_device(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


if __name__ == "__main__":
    main()
