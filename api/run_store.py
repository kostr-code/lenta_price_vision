"""
api/run_store.py — in-memory store mapping run_id → RunResult.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class RunResult:
    run_id: str
    status: Literal["ok", "error"]
    rows: list[dict[str, str]]
    files: dict[str, Path] = field(default_factory=dict)
    error: str = ""

    @property
    def row_count(self) -> int:
        return len(self.rows)


_store: dict[str, RunResult] = {}


def new_run_id() -> str:
    return str(uuid.uuid4())


def save_run(result: RunResult) -> None:
    _store[result.run_id] = result


def get_run(run_id: str) -> RunResult | None:
    return _store.get(run_id)
