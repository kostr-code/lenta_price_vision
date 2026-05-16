from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .field_derivation import derive_fields
from .field_extractor import ExtractionInput, PriceTagFieldExtractor
from .media import import_cv2, laplacian_sharpness
from .qr_tools import DEFAULT_PREFERRED_CANDIDATE_METHODS, QRDecode, QRDecoder
from .schema import OUTPUT_COLUMNS, normalize_record, record_completeness
from .text_reader import (
    TextReader,
    TextReaderConfig,
    build_ocr_variants,
    enhance_text_image,
    suppress_code_artifacts,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def process_crops(
    input_dir: Path,
    output_dir: Path,
    limit: int = 0,
    enable_ocr: bool = True,
    write_debug_images: bool = True,
    derive_qr_fields_when_missing: bool = False,
    adnet_weights_path: str | None = None,
    adnet_repo_dir: str | None = None,
    adnet_device: str = "cpu",
    qr_prefer_methods: tuple[str, ...] = DEFAULT_PREFERRED_CANDIDATE_METHODS,
) -> dict[str, Any]:
    cv2 = import_cv2()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = discover_images(input_dir)
    if limit > 0:
        image_paths = image_paths[:limit]

    qr_decoder = QRDecoder(
        enable_adnet=bool(adnet_weights_path),
        adnet_weights_path=adnet_weights_path,
        adnet_repo_dir=adnet_repo_dir,
        adnet_device=adnet_device,
        preferred_candidate_methods=qr_prefer_methods,
    )
    text_reader = TextReader(TextReaderConfig(enabled=enable_ocr))
    extractor = PriceTagFieldExtractor()
    records: list[dict[str, str]] = []
    jsonl_rows: list[dict[str, Any]] = []

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None or image.size == 0:
            continue
        crop_dir = output_dir / safe_stem(image_path, index)
        if write_debug_images:
            crop_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_dir / "raw_crop.png"), image)
            cv2.imwrite(str(crop_dir / "text_enhanced.png"), enhance_text_image(image))
            cv2.imwrite(
                str(crop_dir / "text_masked_enhanced.png"),
                enhance_text_image(suppress_code_artifacts(image)),
            )
            for variant_name, variant in build_ocr_variants(image, max_variants=6, zoned=True):
                cv2.imwrite(str(crop_dir / f"ocr_{variant_name}.png"), variant)

        qr_result = qr_decoder.decode_with_diagnostics(
            image,
            debug_dir=(crop_dir / "qr") if write_debug_images else None,
        )
        text_lines = text_reader.read(image) if enable_ocr else []
        record = extractor.extract(
            ExtractionInput(
                filename=crop_filename(image_path, input_dir),
                text_lines=text_lines,
                qr_decodes=qr_result.decodes,
                crop=image,
            )
        )
        record = derive_fields(
            record,
            derive_qr_fields_when_missing=derive_qr_fields_when_missing,
        )
        records.append(normalize_record(record))

        detail = {
            "file": str(image_path),
            "output_dir": str(crop_dir) if write_debug_images else "",
            "sharpness": round(laplacian_sharpness(image), 3),
            "qr": {
                "success": qr_result.success,
                "decodes": [decode_to_json(decode) for decode in qr_result.decodes],
                "candidates": [candidate.to_json() for candidate in qr_result.candidates],
                "variants_tried": qr_result.variants_tried,
                "reconstructed_tried": qr_result.reconstructed_tried,
                "adnet_used": qr_result.adnet_used,
                "adnet_status": qr_result.adnet_status,
            },
            "ocr": {
                "enabled": enable_ocr,
                "lines": [asdict(line) for line in text_lines],
                "status": text_reader.status,
            },
            "record": normalize_record(record),
            "filled_fields": record_completeness(record),
        }
        jsonl_rows.append(detail)
        if write_debug_images:
            (crop_dir / "result.json").write_text(
                json.dumps(detail, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    write_records_csv(output_dir / "recognized_from_crops.csv", records)
    write_jsonl(output_dir / "results.jsonl", jsonl_rows)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "images_seen": len(image_paths),
        "rows": len(records),
        "qr_success": sum(1 for row in jsonl_rows if row["qr"]["success"]),
        "ocr_enabled": enable_ocr,
        "ocr_status": text_reader.status,
        "adnet_enabled": bool(adnet_weights_path),
        "adnet_status": qr_decoder.adnet_restorer.status,
        "qr_prefer_methods": list(qr_prefer_methods),
        "csv": str(output_dir / "recognized_from_crops.csv"),
        "jsonl": str(output_dir / "results.jsonl"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def discover_images(input_dir: Path) -> list[Path]:
    if input_dir.is_file() and input_dir.suffix.lower() in IMAGE_SUFFIXES:
        return [input_dir]
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def crop_filename(image_path: Path, input_dir: Path) -> str:
    if input_dir.is_file():
        return image_path.name
    return str(image_path.relative_to(input_dir))


def decode_to_json(decode: QRDecode) -> dict[str, Any]:
    return {"raw": decode.raw, "fields": decode.fields, "source": decode.source}


def safe_stem(path: Path, index: int) -> str:
    stem = path.stem
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in stem)
    return f"{index:05d}_{safe[:96]}"


def write_records_csv(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(normalize_record(record))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enhance crop OCR and reconstruct/decode QR codes."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("runs") / "crops",
        help="Directory with price-tag crops, searched recursively.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "crop_enhancement",
        help="Directory for enhanced images, QR debug artifacts and summaries.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum images to process.")
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Skip OCR and only process QR/images.",
    )
    parser.add_argument(
        "--no-debug-images",
        action="store_true",
        help="Write only CSV/JSONL summaries without per-crop PNG artifacts.",
    )
    parser.add_argument(
        "--derive-qr-fields",
        action="store_true",
        help="Derive QR fields from OCR values when QR payload is absent.",
    )
    parser.add_argument(
        "--adnet-weights",
        type=str,
        default=None,
        help="Optional EG-Restormer/ADNet .pth weights path for QR restoration fallback.",
    )
    parser.add_argument(
        "--adnet-repo",
        type=str,
        default=None,
        help="Optional ADNet repo directory. Defaults to experiments/adnet_repo or ADNET_REPO_DIR.",
    )
    parser.add_argument(
        "--adnet-device",
        type=str,
        default="cpu",
        help="Torch device for ADNet fallback, for example cpu or cuda:0.",
    )
    parser.add_argument(
        "--qr-prefer-method",
        type=str,
        default=",".join(DEFAULT_PREFERRED_CANDIDATE_METHODS),
        help=(
            "Comma-separated candidate method priority, e.g. "
            "opencv_detect:otsu:1x,opencv_detect:gray:1x."
        ),
    )
    return parser


def parse_preferred_methods(value: str) -> tuple[str, ...]:
    methods = tuple(method.strip() for method in value.split(",") if method.strip())
    return methods or DEFAULT_PREFERRED_CANDIDATE_METHODS


def main() -> None:
    args = build_parser().parse_args()
    summary = process_crops(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        enable_ocr=not args.no_ocr,
        write_debug_images=not args.no_debug_images,
        derive_qr_fields_when_missing=args.derive_qr_fields,
        adnet_weights_path=args.adnet_weights,
        adnet_repo_dir=args.adnet_repo,
        adnet_device=args.adnet_device,
        qr_prefer_methods=parse_preferred_methods(args.qr_prefer_method),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
