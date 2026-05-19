"""api/logging_setup.py — structlog + rich logging configuration."""

from __future__ import annotations

import logging
import sys

import structlog
from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)


def configure_logging(level: str = "INFO") -> None:
    """
    Configure structlog to use Rich for human-readable console output.

    Call once at application startup before any logging occurs.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[
            RichHandler(
                console=_console,
                rich_tracebacks=True,
                markup=True,
                show_path=False,
            )
        ],
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Quieten noisy third-party loggers
    for noisy in ("uvicorn.access", "ultralytics", "PIL", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
