"""Project-wide logging helpers."""

from __future__ import annotations

import logging
import os
import sys
from typing import Mapping, Optional

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _coerce_level(level: Optional[str | int]) -> int:
    if level is None:
        return logging.INFO
    if isinstance(level, int):
        return level
    name = str(level).upper()
    if name.isdigit():
        return int(name)
    return logging._nameToLevel.get(name, logging.INFO)


def _coerce_count(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def log_task_summary(
    logger: logging.Logger,
    task_name: str,
    *,
    processed_rows: int,
    skipped_rows: int = 0,
    failed_rows: int = 0,
    total_rows: Optional[int] = None,
    succeeded_rows: Optional[int] = None,
    success_label: str = "succeeded",
    skipped_label: str = "skipped",
    extra_counts: Optional[Mapping[str, int]] = None,
) -> None:
    """Log a consistent end-of-task row processing summary."""

    processed = _coerce_count(processed_rows)
    skipped = _coerce_count(skipped_rows)
    failed = _coerce_count(failed_rows)
    if succeeded_rows is None:
        succeeded = max(0, processed - failed)
    else:
        succeeded = _coerce_count(succeeded_rows)

    parts: list[str] = []
    if total_rows is not None:
        parts.append(f"{_coerce_count(total_rows)} total row(s)")
    parts.extend(
        [
            f"{processed} processed",
            f"{succeeded} {success_label}",
            f"{skipped} {skipped_label}",
            f"{failed} failed",
        ]
    )
    for label, count in (extra_counts or {}).items():
        count = _coerce_count(count)
        if count:
            parts.append(f"{count} {label}")

    logger.info("%s summary: %s", task_name, ", ".join(parts))


def setup_logging(
    level: Optional[str | int] = None,
    *,
    verbose: bool = False,
    quiet: bool = False,
    log_file: Optional[str] = None,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
) -> None:
    """Configure root logging with sensible defaults.

    Environment variables:
    - IMPERANDI_LOG_LEVEL
    - IMPERANDI_LOG_FORMAT
    - IMPERANDI_LOG_DATEFMT
    - IMPERANDI_LOG_FILE
    """

    if verbose:
        level = "DEBUG"
    elif quiet:
        level = "WARNING"
    elif level is None:
        level = os.getenv("IMPERANDI_LOG_LEVEL", DEFAULT_LOG_LEVEL)

    if fmt is None:
        fmt = os.getenv("IMPERANDI_LOG_FORMAT", DEFAULT_LOG_FORMAT)
    if datefmt is None:
        datefmt = os.getenv("IMPERANDI_LOG_DATEFMT", DEFAULT_DATE_FORMAT)
    if log_file is None:
        log_file = os.getenv("IMPERANDI_LOG_FILE")

    root = logging.getLogger()
    if getattr(root, "_imperandi_configured", False):
        root.setLevel(_coerce_level(level))
        return

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=_coerce_level(level),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )
    root._imperandi_configured = True
