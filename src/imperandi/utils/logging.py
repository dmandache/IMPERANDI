"""Project-wide logging helpers."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Mapping, Optional

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def log_script_namespace(
    logger, script_file: str, args: argparse.Namespace, **settings
) -> argparse.Namespace:
    """Log resolved settings without mutating arguments or exposing CLI handlers."""
    namespace = argparse.Namespace(
        **{k: v for k, v in {**vars(args), **settings}.items() if not k.startswith("_")}
    )
    logger.info("🚀 Running %s with namespace: %s", Path(script_file).name, namespace)
    return namespace


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
    resumed_rows: Optional[int] = None,
    failed_rows: int = 0,
    total_rows: Optional[int] = None,
    succeeded_rows: Optional[int] = None,
    success_label: str = "succeeded",
    skipped_label: str = "skipped",
    extra_counts: Optional[Mapping[str, int]] = None,
) -> None:
    """Log outcomes, separating reused successes from other skipped rows.

    ``skipped_rows`` includes ``resumed_rows`` (checkpoint or existing outputs).
    Processing counts and skip reasons are available at DEBUG level.
    """

    processed = _coerce_count(processed_rows)
    skipped = _coerce_count(skipped_rows)
    failed = _coerce_count(failed_rows)
    resumed = _coerce_count(resumed_rows)
    if succeeded_rows is None:
        succeeded = max(0, processed - failed)
    else:
        succeeded = _coerce_count(succeeded_rows)

    parts: list[str] = []
    if total_rows is not None:
        parts.append(f"{_coerce_count(total_rows)} total")
    parts.append(f"{succeeded} {success_label}")
    if resumed_rows is not None:
        parts.append(f"{resumed} resumed")
    parts.extend([f"{max(0, skipped - resumed)} {skipped_label}", f"{failed} failed"])

    logger.info("%s summary: %s", task_name, ", ".join(parts))

    details = [f"{processed} processed"]
    for label, count in (extra_counts or {}).items():
        count = _coerce_count(count)
        if count:
            details.append(f"{count} {label}")

    logger.debug("%s details: %s", task_name, ", ".join(details))


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
