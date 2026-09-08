"""Atomic publication of SimpleITK images and transforms."""

from __future__ import annotations

from pathlib import Path
import uuid

from .organ import backend


def write_artifact(value, path, *, transform: bool = False) -> str:
    """Write an image or transform atomically and return its absolute path."""
    sitk = backend()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{uuid.uuid4().hex}.{destination.name}")
    try:
        writer = sitk.WriteTransform if transform else sitk.WriteImage
        writer(value, str(temporary))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return str(destination.resolve())
