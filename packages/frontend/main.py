from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_path() -> None:
    src_path = Path(__file__).resolve().parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def run() -> None:
    _bootstrap_path()
    from frontend.main import run as app_run

    app_run()


if __name__ == "__main__":
    run()
