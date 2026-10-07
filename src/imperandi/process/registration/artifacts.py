"""Atomic publication of SimpleITK images and transforms."""

from __future__ import annotations

from pathlib import Path
import uuid

from .organ import backend


def write_artifact(image, path) -> str:
    """Write an image atomically and return its absolute path."""
    sitk = backend()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{uuid.uuid4().hex}.{destination.name}")
    try:
        sitk.WriteImage(image, str(temporary))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return str(destination.resolve())
