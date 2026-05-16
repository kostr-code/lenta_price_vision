from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse

from .adnet_restore import ADNetConfig, ADNetRestorer
from .media import bbox_iou
from .schema import ABSENT_VALUE, QR_FIELD_ALIASES, normalize_value

DEFAULT_PREFERRED_CANDIDATE_METHODS = (
    "opencv_detect:otsu:1x",
    "opencv_detect:gray:1x",
    "opencv_detect:clahe:1x",
    "opencv_detect:adaptive:1x",
    "opencv_detect:otsu",
    "opencv_detect:gray",
)


@dataclass(frozen=True)
class QRDecode:
    raw: str
    fields: dict[str, str]
    source: str


@dataclass(frozen=True)
class QRZoneCandidate:
    bbox: tuple[int, int, int, int]
    method: str
    score: float = 0.0
    points: tuple[tuple[float, float], ...] = ()

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QRDecodeResult:
    decodes: list[QRDecode]
    candidates: list[QRZoneCandidate]
    variants_tried: int
    reconstructed_tried: int
    adnet_used: bool = False
    adnet_status: dict[str, str | bool] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return bool(self.decodes)


class QRDecoder:
    """QR/barcode reader with QR-zone search, preprocessing and grid reconstruction.

    The normal pipeline still consumes ``decode(image)``. For crop debugging use
    ``decode_with_diagnostics(image, debug_dir=...)`` to also save candidates and
    reconstructed QR attempts.
    """

    def __init__(
        self,
        scales: tuple[float, ...] = (1.0, 2.0, 4.0),
        grid_sizes: tuple[int, ...] = (21, 25, 29, 33),
        max_candidates: int = 3,
        enable_grid_reconstruction: bool = True,
        enable_wechat: bool = True,
        enable_adnet: bool = False,
        adnet_weights_path: str | None = None,
        adnet_repo_dir: str | None = None,
        adnet_device: str = "cpu",
        preferred_candidate_methods: tuple[str, ...] = DEFAULT_PREFERRED_CANDIDATE_METHODS,
    ) -> None:
        self.scales = scales
        self.grid_sizes = grid_sizes
        self.max_candidates = max_candidates
        self.enable_grid_reconstruction = enable_grid_reconstruction
        self.enable_wechat = enable_wechat
        self.preferred_candidate_methods = preferred_candidate_methods
        self.adnet_restorer = ADNetRestorer(
            ADNetConfig(
                enabled=enable_adnet,
                weights_path=adnet_weights_path,
                repo_dir=adnet_repo_dir,
                device=adnet_device,
            )
        )
        self._opencv_detector: Any | None = None
        self._wechat_detector: Any | None = None
        self._wechat_loaded = False

    def decode(self, image: Any) -> list[QRDecode]:
        return self.decode_with_diagnostics(image).decodes

    def decode_with_diagnostics(
        self,
        image: Any,
        debug_dir: Path | None = None,
    ) -> QRDecodeResult:
        if image is None or not getattr(image, "size", 0):
            return self._result([], [], 0, 0)

        candidates = find_qr_zone_candidates(image, max_candidates=self.max_candidates)
        candidates = prioritize_qr_candidates(candidates, self.preferred_candidate_methods)
        payloads: list[tuple[str, str]] = []
        variants_tried = 0
        reconstructed_tried = 0

        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            self._write_debug_json(debug_dir / "qr_candidates.json", candidates)

        for label, candidate_image in self._candidate_images(image, candidates, debug_dir):
            for variant, suffix in preprocess_qr_variants(candidate_image, self.scales):
                variants_tried += 1
                payloads.extend(
                    (raw, f"{source}|{label}|{suffix}")
                    for raw, source in self._decode_once(variant)
                )
                if payloads:
                    return QRDecodeResult(
                        self._normalize_payloads(payloads),
                        candidates,
                        variants_tried,
                        reconstructed_tried,
                        adnet_status=self.adnet_restorer.status,
                    )

        if self.enable_grid_reconstruction:
            for label, candidate_image in self._candidate_images(image, candidates, debug_dir)[:2]:
                for grid_label, reconstructed in reconstruct_qr_variants(
                    candidate_image,
                    grid_sizes=self.grid_sizes,
                ):
                    reconstructed_tried += 1
                    if debug_dir is not None and reconstructed_tried <= 80:
                        self._save_debug_image(
                            debug_dir / f"reconstructed_{reconstructed_tried:03d}_{grid_label}.png",
                            reconstructed,
                        )
                    payloads.extend(
                        (raw, f"{source}|{label}|grid:{grid_label}")
                        for raw, source in self._decode_once(reconstructed)
                    )
                    if payloads:
                        return QRDecodeResult(
                            self._normalize_payloads(payloads),
                            candidates,
                            variants_tried,
                            reconstructed_tried,
                            adnet_status=self.adnet_restorer.status,
                        )

        adnet_used = False
        if self.adnet_restorer.config.enabled:
            for label, candidate_image in self._candidate_images(image, candidates, debug_dir)[:2]:
                restored = self.adnet_restorer.restore(candidate_image)
                if restored is None:
                    continue
                adnet_used = True
                if debug_dir is not None:
                    self._save_debug_image(
                        debug_dir / f"{safe_filename(label)}_adnet.png",
                        restored,
                    )
                for variant, suffix in preprocess_qr_variants(restored, self.scales):
                    variants_tried += 1
                    payloads.extend(
                        (raw, f"{source}|{label}|adnet|{suffix}")
                        for raw, source in self._decode_once(variant)
                    )
                    if payloads:
                        return QRDecodeResult(
                            self._normalize_payloads(payloads),
                            candidates,
                            variants_tried,
                            reconstructed_tried,
                            adnet_used=adnet_used,
                            adnet_status=self.adnet_restorer.status,
                        )

        return QRDecodeResult(
            self._normalize_payloads(payloads),
            candidates,
            variants_tried,
            reconstructed_tried,
            adnet_used=adnet_used,
            adnet_status=self.adnet_restorer.status,
        )

    def _result(
        self,
        decodes: list[QRDecode],
        candidates: list[QRZoneCandidate],
        variants_tried: int,
        reconstructed_tried: int,
    ) -> QRDecodeResult:
        return QRDecodeResult(
            decodes=decodes,
            candidates=candidates,
            variants_tried=variants_tried,
            reconstructed_tried=reconstructed_tried,
            adnet_status=self.adnet_restorer.status,
        )

    def _candidate_images(
        self,
        image: Any,
        candidates: list[QRZoneCandidate],
        debug_dir: Path | None,
    ) -> list[tuple[str, Any]]:
        images: list[tuple[str, Any]] = [] if candidates else [("full", image)]
        for index, candidate in enumerate(candidates, start=1):
            crop = crop_candidate(image, candidate)
            if crop is not None and getattr(crop, "size", 0):
                label = f"candidate_{index:02d}_{candidate.method}"
                images.append((label, crop))
                if debug_dir is not None:
                    self._save_debug_image(debug_dir / f"{safe_filename(label)}.png", crop)
            warped = warp_candidate(image, candidate)
            if warped is not None and getattr(warped, "size", 0):
                label = f"candidate_{index:02d}_{candidate.method}_warped"
                images.append((label, warped))
                if debug_dir is not None:
                    self._save_debug_image(debug_dir / f"{safe_filename(label)}.png", warped)
            _straight_payload, straight = opencv_straight_qr(image, candidate)
            if straight is not None and getattr(straight, "size", 0):
                label = f"candidate_{index:02d}_{candidate.method}_straight"
                images.append((label, straight))
                if debug_dir is not None:
                    self._save_debug_image(debug_dir / f"{safe_filename(label)}.png", straight)
        return images

    def _decode_once(self, image: Any) -> list[tuple[str, str]]:
        payloads: list[tuple[str, str]] = []
        payloads.extend(self._decode_with_zxingcpp(image))
        payloads.extend(self._decode_with_pyzbar(image))
        payloads.extend(self._decode_with_opencv(image))
        payloads.extend(self._decode_with_wechat(image))
        return payloads

    def _normalize_payloads(self, payloads: list[tuple[str, str]]) -> list[QRDecode]:
        seen: set[str] = set()
        decoded: list[QRDecode] = []
        for raw, source in payloads:
            value = normalize_value(raw)
            if not value or value in seen:
                continue
            seen.add(value)
            decoded.append(QRDecode(raw=value, fields=parse_qr_payload(value), source=source))
        return decoded

    def _decode_with_zxingcpp(self, image: Any) -> list[tuple[str, str]]:
        try:
            import zxingcpp  # type: ignore
        except ImportError:
            return []
        payloads: list[tuple[str, str]] = []
        seen: set[str] = set()
        format_groups = [
            None,
            (
                zxingcpp.BarcodeFormat.QRCode,
                zxingcpp.BarcodeFormat.DataMatrix,
                zxingcpp.BarcodeFormat.Aztec,
                zxingcpp.BarcodeFormat.PDF417,
                zxingcpp.BarcodeFormat.MicroQRCode,
            ),
            (
                zxingcpp.BarcodeFormat.EAN13,
                zxingcpp.BarcodeFormat.EAN8,
                zxingcpp.BarcodeFormat.Code128,
            ),
        ]
        binarizers = [
            zxingcpp.Binarizer.LocalAverage,
            zxingcpp.Binarizer.GlobalHistogram,
            zxingcpp.Binarizer.FixedThreshold,
        ]
        for formats in format_groups:
            for binarizer in binarizers:
                for is_pure in (False, True):
                    try:
                        kwargs = {
                            "try_rotate": True,
                            "try_downscale": False,
                            "try_invert": True,
                            "binarizer": binarizer,
                            "is_pure": is_pure,
                        }
                        if formats is not None:
                            kwargs["formats"] = formats
                        results = zxingcpp.read_barcodes(image, **kwargs)
                    except Exception:
                        continue
                    for item in results:
                        text = normalize_value(getattr(item, "text", ""))
                        if not text or text in seen:
                            continue
                        seen.add(text)
                        source = f"zxingcpp:{binarizer.name}:pure={int(is_pure)}"
                        payloads.append((text, source))
                    if payloads:
                        return payloads
        return payloads

    def _decode_with_pyzbar(self, image: Any) -> list[tuple[str, str]]:
        try:
            from pyzbar.pyzbar import decode  # type: ignore
        except ImportError:
            return []
        try:
            results = decode(image)
        except Exception:
            return []
        payloads: list[tuple[str, str]] = []
        for result in results:
            raw = getattr(result, "data", b"")
            if isinstance(raw, bytes):
                payloads.append((raw.decode("utf-8", errors="ignore"), "pyzbar"))
            elif raw:
                payloads.append((str(raw), "pyzbar"))
        return payloads

    def _decode_with_opencv(self, image: Any) -> list[tuple[str, str]]:
        try:
            detector = self._get_opencv_detector()
            payloads: list[tuple[str, str]] = []
            try:
                ok, decoded_info, _points, _straight = detector.detectAndDecodeMulti(image)
                if ok:
                    payloads.extend((text, "opencv_qr") for text in decoded_info if text)
            except Exception:
                text, _points, _straight = detector.detectAndDecode(image)
                if text:
                    payloads.append((text, "opencv_qr"))
            return payloads
        except Exception:
            return []

    def _decode_with_wechat(self, image: Any) -> list[tuple[str, str]]:
        detector = self._get_wechat_detector()
        if detector is None:
            return []
        try:
            from .media import import_cv2

            cv2 = import_cv2()
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            decoded, _points = detector.detectAndDecode(image)
        except Exception:
            return []
        return [(str(text), "wechat_qr") for text in decoded if text]

    def _get_opencv_detector(self) -> Any:
        if self._opencv_detector is None:
            from .media import import_cv2

            self._opencv_detector = import_cv2().QRCodeDetector()
        return self._opencv_detector

    def _get_wechat_detector(self) -> Any | None:
        if not self.enable_wechat:
            return None
        if self._wechat_loaded:
            return self._wechat_detector
        self._wechat_loaded = True
        try:
            from .media import import_cv2

            cv2 = import_cv2()
            if not hasattr(cv2, "wechat_qrcode_WeChatQRCode"):
                return None
            model_dir = find_wechat_model_dir()
            if model_dir is not None:
                self._wechat_detector = cv2.wechat_qrcode_WeChatQRCode(
                    str(model_dir / "detect.prototxt"),
                    str(model_dir / "detect.caffemodel"),
                    str(model_dir / "sr.prototxt"),
                    str(model_dir / "sr.caffemodel"),
                )
            else:
                self._wechat_detector = cv2.wechat_qrcode_WeChatQRCode()
        except Exception:
            self._wechat_detector = None
        return self._wechat_detector

    def _save_debug_image(self, path: Path, image: Any) -> None:
        try:
            from .media import import_cv2

            import_cv2().imwrite(str(path), image)
        except Exception:
            return

    def _write_debug_json(self, path: Path, candidates: list[QRZoneCandidate]) -> None:
        payload = [candidate.to_json() for candidate in candidates]
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def find_qr_zone_candidates(image: Any, max_candidates: int = 4) -> list[QRZoneCandidate]:
    candidates: list[QRZoneCandidate] = []
    candidates.extend(find_qr_zone_by_detector(image))
    candidates.extend(find_lenta_top_right_prior_candidates(image))
    if len(nms_candidates(candidates, max_candidates=max_candidates)) < max_candidates:
        candidates.extend(
            find_high_frequency_qr_candidates(image, max_candidates=max_candidates * 3)
        )
    return nms_candidates(candidates, max_candidates=max_candidates)


def prioritize_qr_candidates(
    candidates: list[QRZoneCandidate],
    preferred_methods: tuple[str, ...],
) -> list[QRZoneCandidate]:
    def priority(candidate: QRZoneCandidate) -> tuple[int, float]:
        for index, method in enumerate(preferred_methods):
            if candidate.method == method or candidate.method.startswith(method):
                return index, -candidate.score
        if candidate.method.startswith("opencv_detect"):
            return len(preferred_methods), -candidate.score
        if candidate.method.startswith("lenta_top_right_prior"):
            return len(preferred_methods) + 1, -candidate.score
        return len(preferred_methods) + 2, -candidate.score

    return sorted(candidates, key=priority)


def find_qr_zone_by_detector(image: Any) -> list[QRZoneCandidate]:
    try:
        from .media import import_cv2

        cv2 = import_cv2()
    except Exception:
        return []

    detector = cv2.QRCodeDetector()
    height, width = image.shape[:2]
    candidates: list[QRZoneCandidate] = []
    for roi_box in qr_search_boxes(width, height):
        x1, y1, x2, y2 = roi_box
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        for scale in (1, 2, 4):
            resized = cv2.resize(
                roi,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LANCZOS4,
            )
            for variant_name, variant in detector_variants(resized):
                point_sets = detect_point_sets(detector, variant)
                for points in point_sets:
                    full_points = tuple(
                        (float(point[0] / scale + x1), float(point[1] / scale + y1))
                        for point in points
                    )
                    if not valid_qr_points(full_points, width, height):
                        continue
                    bbox = padded_points_bbox(full_points, width, height, pad_ratio=0.22)
                    side = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                    score = 1000.0 + side
                    candidates.append(
                        QRZoneCandidate(
                            bbox=bbox,
                            method=f"opencv_detect:{variant_name}:{scale}x",
                            score=score,
                            points=full_points,
                        )
                    )
    return nms_candidates(candidates, max_candidates=6)


def find_lenta_top_right_prior_candidates(image: Any) -> list[QRZoneCandidate]:
    height, width = image.shape[:2]
    if height < 40 or width < 40:
        return []
    boxes = [
        (int(0.54 * width), 0, width, int(0.48 * height)),
        (int(0.58 * width), int(0.02 * height), int(0.98 * width), int(0.45 * height)),
        (int(0.50 * width), 0, width, int(0.58 * height)),
    ]
    candidates: list[QRZoneCandidate] = []
    for index, box in enumerate(boxes, start=1):
        bbox = clamp_bbox_tuple(box, width, height)
        if bbox[2] - bbox[0] < 24 or bbox[3] - bbox[1] < 24:
            continue
        candidates.append(
            QRZoneCandidate(
                bbox=bbox,
                method=f"lenta_top_right_prior_{index}",
                score=800.0 - index,
            )
        )
    return candidates


def detect_point_sets(detector: Any, image: Any) -> list[list[tuple[float, float]]]:
    point_sets: list[list[tuple[float, float]]] = []
    try:
        ok, points = detector.detectMulti(image)
        if ok and points is not None:
            for item in points:
                point_sets.append([(float(x), float(y)) for x, y in item.reshape(-1, 2)])
    except Exception:
        pass
    try:
        ok, points = detector.detect(image)
        if ok and points is not None:
            point_sets.append([(float(x), float(y)) for x, y in points.reshape(-1, 2)])
    except Exception:
        pass
    return point_sets


def detector_variants(image: Any) -> list[tuple[str, Any]]:
    from .media import import_cv2

    cv2 = import_cv2()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    variants: list[tuple[str, Any]] = [("gray", gray)]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants.append(("clahe", clahe.apply(gray)))
    variants.append(("otsu", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]))
    block_size = adaptive_block_size(gray)
    variants.append(
        (
            "adaptive",
            cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                5,
            ),
        )
    )
    return variants


def find_high_frequency_qr_candidates(
    image: Any,
    max_candidates: int = 12,
) -> list[QRZoneCandidate]:
    try:
        from .media import import_cv2, import_numpy

        cv2 = import_cv2()
        np = import_numpy()
    except Exception:
        return []

    height, width = image.shape[:2]
    if height < 32 or width < 32:
        return []
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    candidates: list[QRZoneCandidate] = []

    for roi_box in qr_search_boxes(width, height):
        x1, y1, x2, y2 = roi_box
        gray = gray_full[y1:y2, x1:x2]
        if gray.size == 0:
            continue
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        binary = cv2.adaptiveThreshold(
            clahe,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            adaptive_block_size(clahe),
            4,
        )
        edges = cv2.Canny(clahe, 45, 160)
        roi_h, roi_w = gray.shape[:2]
        min_side = max(32, int(min(roi_w, roi_h) * 0.24))
        max_side = max(min_side, int(min(roi_w, roi_h) * 0.82))
        sizes = sorted(
            {
                int(value)
                for value in np.linspace(min_side, max_side, num=7)
                if value >= 24
            }
        )
        for side in sizes:
            step = max(8, side // 3)
            if side >= roi_w or side >= roi_h:
                continue
            for yy in range(0, roi_h - side + 1, step):
                for xx in range(0, roi_w - side + 1, step):
                    window = binary[yy : yy + side, xx : xx + side]
                    edge_window = edges[yy : yy + side, xx : xx + side]
                    dark_ratio = float((window < 128).mean())
                    if not 0.14 <= dark_ratio <= 0.68:
                        continue
                    edge_density = float((edge_window > 0).mean())
                    if edge_density < 0.035:
                        continue
                    transitions_x = float((window[:, 1:] != window[:, :-1]).mean())
                    transitions_y = float((window[1:, :] != window[:-1, :]).mean())
                    balance_penalty = abs(transitions_x - transitions_y) * 0.65
                    text_penalty = 0.08 if transitions_x > transitions_y * 1.8 else 0.0
                    top_right_bonus = 0.08 if (x1 + xx + side / 2) > width * 0.52 else 0.0
                    size_bonus = 0.16 * (side / max(1, min(roi_w, roi_h)))
                    score = (
                        transitions_x
                        + transitions_y
                        + 2.2 * edge_density
                        + top_right_bonus
                        + size_bonus
                        - balance_penalty
                        - text_penalty
                    )
                    if score < 0.20:
                        continue
                    pad = max(4, int(side * 0.18))
                    bbox = clamp_bbox_tuple(
                        (x1 + xx - pad, y1 + yy - pad, x1 + xx + side + pad, y1 + yy + side + pad),
                        width,
                        height,
                    )
                    candidates.append(
                        QRZoneCandidate(bbox=bbox, method="high_frequency", score=score)
                    )

    return nms_candidates(candidates, max_candidates=max_candidates)


def preprocess_qr_variants(
    qr_image: Any,
    scales: tuple[float, ...] = (1.0, 2.0, 4.0),
    max_variants: int = 18,
) -> list[tuple[Any, str]]:
    if qr_image is None or not getattr(qr_image, "size", 0):
        return []
    from .media import import_cv2

    cv2 = import_cv2()
    gray = cv2.cvtColor(qr_image, cv2.COLOR_BGR2GRAY) if len(qr_image.shape) == 3 else qr_image
    variants: list[tuple[Any, str]] = []
    seen_shapes: set[tuple[str, tuple[int, int]]] = set()

    for scale in scales:
        for interpolation, interp_name in (
            (cv2.INTER_NEAREST, "nearest"),
            (cv2.INTER_LANCZOS4, "lanczos"),
        ):
            if scale == 1.0 and interpolation == cv2.INTER_LANCZOS4:
                continue
            resized = gray
            if scale != 1.0:
                resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interpolation)
            border = max(20, int(0.10 * min(resized.shape[:2])))
            bordered = cv2.copyMakeBorder(
                resized,
                border,
                border,
                border,
                border,
                cv2.BORDER_CONSTANT,
                value=255,
            )
            add_qr_variant(variants, seen_shapes, bordered, f"{scale:g}x_{interp_name}_border")
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(bordered)
            otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            add_qr_variant(variants, seen_shapes, otsu, f"{scale:g}x_{interp_name}_otsu")
            adaptive = cv2.adaptiveThreshold(
                clahe,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                adaptive_block_size(clahe),
                5,
            )
            add_qr_variant(variants, seen_shapes, adaptive, f"{scale:g}x_{interp_name}_adaptive")
            if len(variants) >= max_variants:
                return variants
    return variants


def add_qr_variant(
    variants: list[tuple[Any, str]],
    seen: set[tuple[str, tuple[int, int]]],
    image: Any,
    name: str,
) -> None:
    key = (name, tuple(image.shape[:2]))
    if key in seen:
        return
    variants.append((image, name))
    seen.add(key)


def reconstruct_qr_variants(
    qr_image: Any,
    grid_sizes: tuple[int, ...] = (21, 25, 29, 33),
    trims: tuple[float, ...] = (0.0, 0.04, 0.08, 0.12),
    module_size: int = 12,
) -> list[tuple[str, Any]]:
    if qr_image is None or not getattr(qr_image, "size", 0):
        return []
    variants: list[tuple[str, Any]] = []
    for trim in trims:
        trimmed = trim_image(qr_image, trim)
        if trimmed is None or not getattr(trimmed, "size", 0):
            continue
        for grid_size in grid_sizes:
            matrix = sample_qr_matrix(trimmed, grid_size)
            if matrix is None or not looks_like_qr_matrix(matrix):
                continue
            rendered = render_qr_matrix(matrix, module_size=module_size)
            variants.append((f"N{grid_size}_trim{int(trim * 100):02d}", rendered))
    return variants


def sample_qr_matrix(qr_image: Any, grid_size: int) -> Any | None:
    try:
        from .media import import_cv2, import_numpy

        cv2 = import_cv2()
        np = import_numpy()
    except Exception:
        return None
    if grid_size < 21:
        return None
    gray = cv2.cvtColor(qr_image, cv2.COLOR_BGR2GRAY) if len(qr_image.shape) == 3 else qr_image
    if min(gray.shape[:2]) < 12:
        return None
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    reduced = cv2.resize(gray, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    threshold = cv2.threshold(reduced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    matrix = (reduced < threshold).astype(np.uint8)
    return matrix


def render_qr_matrix(
    matrix: Any,
    module_size: int = 12,
    quiet_zone_modules: int = 4,
) -> Any:
    from .media import import_numpy

    np = import_numpy()
    modules = np.kron(matrix.astype(np.uint8), np.ones((module_size, module_size), dtype=np.uint8))
    image = np.where(modules > 0, 0, 255).astype(np.uint8)
    border = quiet_zone_modules * module_size
    return np.pad(image, ((border, border), (border, border)), constant_values=255)


def crop_candidate(image: Any, candidate: QRZoneCandidate) -> Any | None:
    x1, y1, x2, y2 = candidate.bbox
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()


def warp_candidate(image: Any, candidate: QRZoneCandidate, output_size: int = 256) -> Any | None:
    if len(candidate.points) != 4:
        return None
    try:
        from .media import import_cv2, import_numpy

        cv2 = import_cv2()
        np = import_numpy()
    except Exception:
        return None
    points = order_quad_points(np.array(candidate.points, dtype=np.float32))
    target = np.array(
        [
            [0.0, 0.0],
            [output_size - 1.0, 0.0],
            [output_size - 1.0, output_size - 1.0],
            [0.0, output_size - 1.0],
        ],
        dtype=np.float32,
    )
    try:
        transform = cv2.getPerspectiveTransform(points, target)
        return cv2.warpPerspective(image, transform, (output_size, output_size), borderValue=255)
    except Exception:
        return None


def opencv_straight_qr(image: Any, candidate: QRZoneCandidate) -> tuple[str, Any | None]:
    if len(candidate.points) != 4:
        return "", None
    try:
        from .media import import_cv2, import_numpy

        cv2 = import_cv2()
        np = import_numpy()
        detector = cv2.QRCodeDetector()
        points = np.array(candidate.points, dtype=np.float32).reshape(1, 4, 2)
        text, straight = detector.decode(image, points)
        return str(text or ""), straight
    except Exception:
        return "", None


def order_quad_points(points: Any) -> Any:
    from .media import import_numpy

    np = import_numpy()
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def qr_search_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes = [
        (int(0.52 * width), 0, width, int(0.52 * height)),
        (int(0.20 * width), 0, int(0.92 * width), int(0.60 * height)),
        (int(0.45 * width), 0, width, int(0.70 * height)),
        (0, 0, width, int(0.62 * height)),
    ]
    unique: list[tuple[int, int, int, int]] = []
    for box in boxes:
        clamped = clamp_bbox_tuple(box, width, height)
        is_large_enough = clamped[2] - clamped[0] >= 24 and clamped[3] - clamped[1] >= 24
        if is_large_enough and clamped not in unique:
            unique.append(clamped)
    return unique


def valid_qr_points(
    points: tuple[tuple[float, float], ...],
    width: int,
    height: int,
) -> bool:
    if len(points) != 4:
        return False
    if any(not (-0.05 * width <= x <= 1.05 * width) for x, _y in points):
        return False
    if any(not (-0.05 * height <= y <= 1.05 * height) for _x, y in points):
        return False
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    box_w = max(xs) - min(xs)
    box_h = max(ys) - min(ys)
    if box_w < 16 or box_h < 16:
        return False
    aspect = box_w / max(1.0, box_h)
    return 0.45 <= aspect <= 2.2


def padded_points_bbox(
    points: tuple[tuple[float, float], ...],
    width: int,
    height: int,
    pad_ratio: float,
) -> tuple[int, int, int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    pad = pad_ratio * max(x2 - x1, y2 - y1)
    return clamp_bbox_tuple(
        (int(x1 - pad), int(y1 - pad), int(x2 + pad), int(y2 + pad)),
        width,
        height,
    )


def clamp_bbox_tuple(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width - 1, int(x1))),
        max(0, min(height - 1, int(y1))),
        max(0, min(width, int(x2))),
        max(0, min(height, int(y2))),
    )


def nms_candidates(
    candidates: list[QRZoneCandidate],
    max_candidates: int,
    iou_threshold: float = 0.35,
) -> list[QRZoneCandidate]:
    from .media import BBox

    selected: list[QRZoneCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        candidate_box = BBox(*candidate.bbox)
        if any(bbox_iou(candidate_box, BBox(*item.bbox)) > iou_threshold for item in selected):
            continue
        selected.append(candidate)
        if len(selected) >= max_candidates:
            break
    return selected


def adaptive_block_size(image: Any) -> int:
    side = max(15, int(min(image.shape[:2]) * 0.12))
    if side % 2 == 0:
        side += 1
    return max(15, min(side, 61))


def trim_image(image: Any, trim_ratio: float) -> Any | None:
    if trim_ratio <= 0:
        return image
    height, width = image.shape[:2]
    dx = int(width * trim_ratio)
    dy = int(height * trim_ratio)
    if width - 2 * dx < 12 or height - 2 * dy < 12:
        return None
    return image[dy : height - dy, dx : width - dx]


def looks_like_qr_matrix(matrix: Any) -> bool:
    black_ratio = float(matrix.mean())
    return 0.16 <= black_ratio <= 0.72


def find_wechat_model_dir() -> Path | None:
    required = {"detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel"}
    roots = [
        Path.cwd() / "wechat_models",
        Path.cwd() / "experiments" / "wechat_models",
        Path(__file__).resolve().parents[3] / "experiments" / "wechat_models",
    ]
    for root in roots:
        if all((root / name).exists() for name in required):
            return root
    return None


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"


def parse_qr_payload(payload: str) -> dict[str, str]:
    raw_fields = _parse_raw_payload(payload)
    normalized = {column: ABSENT_VALUE for column in QR_FIELD_ALIASES}
    for output_column, aliases in QR_FIELD_ALIASES.items():
        for alias in aliases:
            value = raw_fields.get(alias.casefold())
            if value:
                normalized[output_column] = normalize_qr_value(value)
                break
    return normalized


def _parse_raw_payload(payload: str) -> dict[str, str]:
    text = payload.strip()
    parsed: dict[str, str] = {}

    if text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                for key, value in data.items():
                    parsed[str(key).casefold()] = normalize_value(value)
                return parsed
        except json.JSONDecodeError:
            pass

    query = urlparse(text).query
    if query:
        parsed.update({key.casefold(): value for key, value in parse_qsl(query)})

    if "=" in text:
        parsed.update(
            {key.casefold(): value for key, value in parse_qsl(text, keep_blank_values=True)}
        )

    for key, value in re.findall(r"([A-Za-z][A-Za-z0-9_]+)\s*[:=]\s*([^;,&\s]+)", text):
        parsed[key.casefold()] = value

    if not parsed and re.fullmatch(r"\d{8,14}", text):
        parsed["barcode"] = text
    return parsed


def normalize_qr_value(value: Any) -> str:
    text = normalize_value(value)
    if re.fullmatch(r"\d+[.,]\d+", text):
        return text.replace(",", ".")
    return text
