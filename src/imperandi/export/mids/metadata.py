"""Allowlisted metadata projection for MIDS-style TSV files."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .models import MidsExportError

SAFE_FIELD = re.compile(r"^[a-z][a-z0-9_]*$")
DISALLOWED_SOURCE_FRAGMENTS = (
    "patientname",
    "patientbirth",
    "birthdate",
    "accession",
    "referringphysician",
    "physicianname",
    "patientid",
    "patient_key",
    "dicom_path",
    "series_path",
    "nifti_path",
    "filepath",
    "file_path",
    "uid",
    "_raw",
)
DISALLOWED_OUTPUT_FRAGMENTS = (
    "patient_name",
    "birth_date",
    "accession",
    "referring_physician",
    "raw_id",
    "source_path",
)


def _columns(rule: Any) -> list[str]:
    if isinstance(rule, str):
        return [rule]
    if isinstance(rule, list) and all(isinstance(value, str) for value in rule):
        return rule
    if isinstance(rule, dict):
        columns = rule.get("columns", [])
        if isinstance(columns, str):
            return [columns]
        if isinstance(columns, list) and all(
            isinstance(value, str) for value in columns
        ):
            return columns
    raise MidsExportError(
        "Metadata allowlist entries must be a column, list, or {columns: [...]} mapping."
    )


def validate_allowlist(config: dict[str, Any]) -> None:
    """Reject paths, identifiers, and common direct identifiers at configuration time."""
    for level in ("participants", "sessions", "scans"):
        rules = config.get(level, {})
        if not isinstance(rules, dict):
            raise MidsExportError(f"metadata.{level} must be a mapping.")
        for output, rule in rules.items():
            normalized_output = str(output).casefold()
            reserved = {
                "participants": {"participant_id"},
                "sessions": {"session_id"},
                "scans": {"filename", "source_image_id", "modality"},
            }[level]
            if normalized_output in reserved:
                raise MidsExportError(
                    f"metadata.{level}.{output} would overwrite a required export field."
                )
            if not SAFE_FIELD.fullmatch(str(output)):
                raise MidsExportError(f"Unsafe metadata output field: {output!r}")
            if any(
                fragment in normalized_output
                for fragment in DISALLOWED_OUTPUT_FRAGMENTS
            ):
                raise MidsExportError(
                    f"Disallowed identifying metadata output field: {output!r}"
                )
            for column in _columns(rule):
                normalized = column.replace(" ", "").casefold()
                if any(
                    fragment in normalized for fragment in DISALLOWED_SOURCE_FRAGMENTS
                ):
                    raise MidsExportError(
                        f"Disallowed source column {column!r} in metadata allowlist. "
                        "Use generated safe export identifiers for provenance."
                    )


def project(row: pd.Series, rules: dict[str, Any]) -> dict[str, Any]:
    """Project only explicitly allowlisted source fields."""
    projected: dict[str, Any] = {}
    for output, rule in rules.items():
        value: Any = pd.NA
        for column in _columns(rule):
            if column in row and not pd.isna(row[column]) and str(row[column]).strip():
                value = row[column]
                break
        projected[output] = value
    return projected


def tsv_value(value: Any) -> Any:
    """Normalize a scalar for a TSV without leaking multiline content."""
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        return ",".join(
            str(item).replace("\t", " ").replace("\n", " ") for item in value
        )
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")
