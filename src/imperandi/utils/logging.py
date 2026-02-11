"""Project-wide logging helpers."""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _coerce_level(level: Optional[str | int]) -> int:
    """Perform level.

    Args:
        level (Optional[str | int]): Input value for level.

    Returns:
        int: Result of `_coerce_level`.
    """
    if level is None:
        return logging.INFO
    if isinstance(level, int):
        return level
    name = str(level).upper()
    if name.isdigit():
        return int(name)
    return logging._nameToLevel.get(name, logging.INFO)


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
