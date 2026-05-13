from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .schema import ABSENT_VALUE, OUTPUT_COLUMNS, empty_record, normalize_value
from .validators import normalize_field_value, validate_field

SOURCE_PRIORITY = {
    "qr": 1.0,
    "zxingcpp": 0.95,
    "pyzbar": 0.92,
    "opencv_qr": 0.88,
    "paddleocr": 0.78,
    "tesseract": 0.68,
    "regex": 0.62,
    "derived": 0.45,
    "yolo": 0.35,
    "qr_seed": 0.35,
}


@dataclass(frozen=True)
class FieldCandidate:
    field_name: str
    value: str
    source: str
    confidence: float
    frame_timestamp: int = 0
    validator_passed: bool = True


@dataclass(frozen=True)
class FieldVote:
    field_name: str
    value: str
    confidence: float
    sources: tuple[str, ...]
    votes: int
    validator_passed: bool


class FieldVoter:
    """Track-level field aggregation with validators and repeated-value voting."""

    def __init__(self) -> None:
        self._candidates: dict[str, list[FieldCandidate]] = defaultdict(list)

    def add_candidate(self, candidate: FieldCandidate) -> None:
        value = normalize_field_value(candidate.field_name, candidate.value)
        if not value or value == ABSENT_VALUE:
            if value == ABSENT_VALUE:
                self._candidates[candidate.field_name].append(
                    FieldCandidate(
                        field_name=candidate.field_name,
                        value=ABSENT_VALUE,
                        source=candidate.source,
                        confidence=0.05,
                        frame_timestamp=candidate.frame_timestamp,
                        validator_passed=True,
                    )
                )
            return
        validator_passed = candidate.validator_passed and validate_field(
            candidate.field_name, value
        )
        if not validator_passed and candidate.field_name in strict_fields():
            return
        self._candidates[candidate.field_name].append(
            FieldCandidate(
                field_name=candidate.field_name,
                value=value,
                source=candidate.source,
                confidence=max(0.0, min(1.0, candidate.confidence)),
                frame_timestamp=candidate.frame_timestamp,
                validator_passed=validator_passed,
            )
        )

    def add_record(
        self,
        record: dict[str, Any],
        source: str,
        confidence: float,
        frame_timestamp: int,
    ) -> None:
        for field_name in OUTPUT_COLUMNS:
            if field_name in {"filename", "frame_timestamp", "x_min", "y_min", "x_max", "y_max"}:
                continue
            value = normalize_value(record.get(field_name))
            if not value:
                continue
            inferred_source = infer_source(field_name, source)
            self.add_candidate(
                FieldCandidate(
                    field_name=field_name,
                    value=value,
                    source=inferred_source,
                    confidence=confidence * source_weight(inferred_source),
                    frame_timestamp=frame_timestamp,
                )
            )

    def vote(self, field_name: str) -> FieldVote | None:
        candidates = self._candidates.get(field_name, [])
        if not candidates:
            return None

        present = [item for item in candidates if item.value != ABSENT_VALUE]
        if not present:
            return FieldVote(
                field_name=field_name,
                value=ABSENT_VALUE,
                confidence=0.05,
                sources=tuple(sorted({item.source for item in candidates})),
                votes=len(candidates),
                validator_passed=True,
            )

        grouped: dict[str, list[FieldCandidate]] = defaultdict(list)
        for candidate in present:
            grouped[candidate.value].append(candidate)

        best_value = ""
        best_score = -1.0
        for value, group in grouped.items():
            score = sum(item.confidence for item in group)
            score += min(0.35, 0.08 * (len(group) - 1))
            if any(item.validator_passed for item in group):
                score += 0.15
            score += max(source_weight(item.source) for item in group)
            if score > best_score:
                best_value = value
                best_score = score

        winners = grouped[best_value]
        return FieldVote(
            field_name=field_name,
            value=best_value,
            confidence=min(1.0, best_score / max(1, len(winners))),
            sources=tuple(sorted({item.source for item in winners})),
            votes=len(winners),
            validator_passed=any(item.validator_passed for item in winners),
        )

    def to_record(self, base: dict[str, Any] | None = None) -> dict[str, str]:
        record = empty_record(normalize_value((base or {}).get("filename", "")))
        if base:
            for column in OUTPUT_COLUMNS:
                value = normalize_value(base.get(column))
                if value:
                    record[column] = value
        for field_name in OUTPUT_COLUMNS:
            if field_name in {"filename", "frame_timestamp", "x_min", "y_min", "x_max", "y_max"}:
                continue
            vote = self.vote(field_name)
            if vote is not None:
                record[field_name] = vote.value
        return record

    def debug_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for field_name, candidates in self._candidates.items():
            vote = self.vote(field_name)
            counts = Counter(candidate.value for candidate in candidates)
            summary[field_name] = {
                "winner": vote.value if vote else "",
                "winner_confidence": round(vote.confidence, 3) if vote else 0.0,
                "candidate_count": len(candidates),
                "top_values": counts.most_common(5),
            }
        return summary


def infer_source(field_name: str, fallback_source: str) -> str:
    if field_name.endswith("_qr") or field_name in {
        "qr_code_barcode",
        "wholesale_level_1_count",
        "wholesale_level_1_price",
        "wholesale_level_2_count",
        "wholesale_level_2_price",
        "action_price_qr",
        "action_code_qr",
    }:
        return "qr" if fallback_source not in {"derived"} else fallback_source
    return fallback_source


def source_weight(source: str) -> float:
    return SOURCE_PRIORITY.get(source, 0.5)


def strict_fields() -> set[str]:
    return {
        "price_default",
        "price_card",
        "price_discount",
        "barcode",
        "qr_code_barcode",
        "id_sku",
        "print_datetime",
        "discount_amount",
    }
