"""Human-readable registration labels shared by logs, QC, and errors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def _value(record: Mapping[str, Any], *columns: str) -> str | None:
    for column in columns:
        value = record.get(column)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        if text:
            return text
    return None


def group_context(record: Mapping[str, Any], visit_column: str) -> dict[str, str]:
    """Return the clinical attributes used to describe one registration group."""
    patient = _value(record, "patient_id", "patient_key") or "unknown"
    modality = (_value(record, "Modality", "modality") or "unknown").upper()
    if modality == "MRI":
        modality = "MR"
    return {
        "patient_id": patient,
        "date": _value(record, "date", "visit_date", "StudyDate") or "unknown",
        "visit_order": _value(record, "visit_order") or "unknown",
        "visit": _value(record, visit_column) or "unknown",
        "modality": modality,
    }


def group_label(record: Mapping[str, Any], visit_column: str) -> str:
    context = group_context(record, visit_column)
    return ", ".join(f"{key}={value}" for key, value in context.items())


def scan_label(record: Mapping[str, Any], _visit_column: str) -> str:
    """Describe a scan within its group without repeating group attributes."""
    position = _value(record, "registration_series_number") or "?"
    total = _value(record, "registration_group_size") or "?"
    fields = {
        "series": f"{position}/{total}",
        "phase": _value(record, "phase"),
        "sequence": _value(record, "mri_sequence"),
    }
    detail = ", ".join(f"{key}={value}" for key, value in fields.items() if value)
    return detail
