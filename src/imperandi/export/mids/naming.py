"""Validated MIDS-style path and filename construction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .identity import validate_label
from .models import MidsExportError

NIFTI_SUFFIXES = (".nii", ".nii.gz")


def _first_present(row: pd.Series, columns: list[str]) -> Any | None:
    for column in columns:
        if column in row and not pd.isna(row[column]) and str(row[column]).strip():
            return row[column]
    return None


def normalized_modality(value: Any, modality_config: dict[str, Any]) -> tuple[str, str]:
    raw = str(value).strip().upper()
    aliases = {
        str(alias).upper(): str(target).upper()
        for alias, target in modality_config.get("aliases", {}).items()
    }
    raw = aliases.get(raw, raw)
    mapping = modality_config.get("mapping", {})
    if raw not in mapping:
        raise MidsExportError(
            f"Unsupported modality: {value!r}; strict profile supports CT and MR only."
        )
    entry = mapping[raw]
    if not isinstance(entry, dict):
        raise MidsExportError(f"modality.mapping.{raw} must be a mapping.")
    directory = str(entry.get("directory", ""))
    suffix = validate_label(entry.get("suffix", ""), field=f"{raw} suffix")
    if not re.fullmatch(r"mim-[a-z0-9]+", directory):
        raise MidsExportError(
            f"modality.mapping.{raw}.directory must match 'mim-<label>': {directory!r}"
        )
    return directory, suffix


def mapped_entity(row: pd.Series, config: dict[str, Any]) -> str | None:
    value = _first_present(row, list(config.get("columns", [])))
    if value is None:
        return None
    mapping = {
        str(key).strip().casefold(): val
        for key, val in config.get("mapping", {}).items()
    }
    label = mapping.get(str(value).strip().casefold())
    if label is None or str(label).strip() == "":
        return None
    return validate_label(label, field="filename entity")


def image_basename(
    row: pd.Series,
    *,
    participant: str,
    session: str,
    suffix: str,
    filename_config: dict[str, Any],
) -> str:
    entities: list[str] = [f"sub-{participant}", f"ses-{session}"]
    for entity, prefix in (
        ("body_part", "bp"),
        ("acquisition", "acq"),
        ("phase", "pc"),
        ("reconstruction", "rec"),
    ):
        label = mapped_entity(row, filename_config.get(entity, {}))
        if label:
            entities.append(f"{prefix}-{label}")
    repeat = _first_present(
        row, list(filename_config.get("repeat", {}).get("columns", []))
    )
    if repeat is not None:
        try:
            run = int(float(repeat))
        except (TypeError, ValueError):
            run = 0
        if run >= 1:
            entities.append(f"run-{run:02d}")
    return "_".join([*entities, suffix])


def destination_for_image(
    participant: str, session: str, directory: str, basename: str
) -> Path:
    return (
        Path(f"sub-{participant}") / f"ses-{session}" / directory / f"{basename}.nii.gz"
    )


def mask_label(column: str, patterns: list[str]) -> str:
    label = column
    for prefix in ("consensus_mask_", "mask_consensus_", "mask_"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
            break
    label = re.sub(r"[^A-Za-z0-9]+", "", label)
    return validate_label(label or "roi", field=f"mask label from {column}")


def nifti_extension(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(suffix) for suffix in NIFTI_SUFFIXES)
