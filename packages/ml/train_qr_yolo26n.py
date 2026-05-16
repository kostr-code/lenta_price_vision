from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "packages" / "ml" / "data" / "qr_code_yolo_dataset" / "data.yaml"
DEFAULT_PROJECT = ROOT / "packages" / "ml" / "runs" / "detect"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOLO26n QR-code detector.")
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="YOLO weights/model name. Use a local .pt path if yolo26n.pt is not available.",
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset data.yaml path.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="Use 0/cuda:0 for GPU or cpu.")
    parser.add_argument("--workers", type=int, default=0, help="0 is safer on Windows.")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--name", default="qr_code_yolo26n")
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM/disk if possible.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Run: uv sync --project packages\\ml"
        ) from exc

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        cache=args.cache,
        single_cls=True,
        close_mosaic=20,
        cos_lr=True,
        plots=True,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
