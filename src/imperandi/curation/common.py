"""Shared helpers for deterministic modality curation."""

from __future__ import annotations

import logging
import re
from datetime import datetime, time
from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TEXT_COLS_DEFAULT = [
    "SeriesDescription",
    "ProtocolName",
    "StudyDescription",
    "ImageType",
    "ScanningSequence",
    "SequenceVariant",
    "ScanOptions",
    "SequenceName",
]

_TEXT_COLUMNS: ContextVar[tuple[str, ...] | None] = ContextVar(
    "curation_text_columns", default=None
)


def resolve_text_columns(defaults: Sequence[str]) -> list[str]:
    """Resolve defaults within one curation call, without changing module globals."""
    configured = _TEXT_COLUMNS.get()
    return list(defaults if configured is None else configured)


def has_text_columns_override() -> bool:
    return _TEXT_COLUMNS.get() is not None


def phase_evidence_summary(row: pd.Series) -> str:
    """Compact provenance for wide QC cells; score remains a separate measure."""
    fields = ("volume_id", "phase_status", "phase_source", "phase_confidence",
              "mri_perfusion_source", "phase_reason")
    return "; ".join(
        f"{column}={row[column]}"
        for column in fields
        if column in row and not is_missing(row[column]) and str(row[column]).strip()
    )


@contextmanager
def manifest_text_columns(modality: str, config):
    """Keep annotation, scoring and export under the same metadata precedence."""
    from imperandi.curation.phase import validate_phase_curation

    columns = validate_phase_curation(config)["text_columns"].get(modality)
    token = _TEXT_COLUMNS.set(None if columns is None else tuple(columns))
    try:
        yield
    finally:
        _TEXT_COLUMNS.reset(token)


def with_manifest_text_columns(modality: str):
    """Scope a manifest override to an entire synchronous curation pipeline.

    Context-local state lets nested metadata helpers share the same ordered
    fields while keeping concurrent calls and later datasets independent.
    """

    def decorate(function):
        call_signature = signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            arguments = call_signature.bind(*args, **kwargs)
            with manifest_text_columns(modality, arguments.arguments.get("phase_curation")):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(Path(path).expanduser(), low_memory=False, **kwargs)


def is_missing(x) -> bool:
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def first_scalar(x):
    if isinstance(x, (list, tuple, set, np.ndarray)):
        values = sorted(x) if isinstance(x, set) else x
        for value in values:
            value = first_scalar(value)
            if not is_missing(value):
                return value
        return np.nan
    return x


def stable_text(x) -> str:
    if isinstance(x, (list, tuple, set, np.ndarray)):
        values = sorted(x) if isinstance(x, set) else x
        return "|".join(stable_text(v) for v in values)
    if is_missing(x):
        return ""
    return str(x)


def clean_text(x) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())


def safe_str(x) -> str:
    if isinstance(x, (list, tuple, set, np.ndarray)):
        return clean_text(" ".join(safe_str(v) for v in x if safe_str(v)))
    return clean_text(x) if pd.notna(x) else ""


def norm_label(x, default: str = "OTHER") -> str:
    x = first_scalar(x)
    if is_missing(x) or str(x).strip() == "":
        return default
    return str(x).strip().upper()


def safe_float(x) -> float:
    try:
        x = first_scalar(x)
        if x is None:
            return np.nan
        if is_missing(x) or str(x).strip() == "":
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def acquisition_time_seconds(value) -> float:
    """Parse valid DICOM/colon times, using the start of a volume's time list."""
    if isinstance(value, (list, tuple, set, np.ndarray)):
        values = [acquisition_time_seconds(item) for item in value]
        return min((item for item in values if pd.notna(item)), default=np.nan)
    if is_missing(value):
        return np.nan
    if isinstance(value, datetime):
        value = value.time()
    if isinstance(value, time):
        return (
            value.hour * 3600
            + value.minute * 60
            + value.second
            + value.microsecond / 1e6
        )
    match = re.fullmatch(r"(\d{2}):?(\d{2}):?(\d{2})(?:\.(\d+))?", str(value).strip())
    if match is None:
        return np.nan
    hour, minute, second, fraction = match.groups()
    if int(hour) > 23 or int(minute) > 59 or int(second) > 59:
        return np.nan
    return (
        int(hour) * 3600
        + int(minute) * 60
        + int(second)
        + (float("0." + fraction) if fraction else 0)
    )


def build_series_text(
    row: pd.Series,
    cols: Sequence[str] | None = None,
) -> str:
    return " | ".join(text for _, text in iter_series_text_columns(row, cols))


def iter_series_text_columns(row: pd.Series, cols: Sequence[str] | None = None):
    """Yield normalized, nonempty fields in their configured precedence order."""
    for col in resolve_text_columns(TEXT_COLS_DEFAULT) if cols is None else cols:
        if col in row.index:
            text = safe_str(row.get(col))
            if text:
                yield col, text


def first_text_column_value(row: pd.Series, evaluator, cols=None):
    """Evaluate fields separately so unrelated fields cannot form a new rule."""
    for _, text in iter_series_text_columns(row, cols):
        value = evaluator(text)
        if value is not None:
            return value
    return None


def first_text_column(row: pd.Series, has_evidence, cols=None):
    """Select one ordered evidence field for every stage of a decision."""
    return next(
        (
            (column, text)
            for column, text in iter_series_text_columns(row, cols)
            if has_evidence(text)
        ),
        None,
    )


def first_phase_text_column(
    row: pd.Series, cols=None, *, phase_rules=None, post=None, extra_patterns=()
):
    """Keep contrast/profile evidence even when it cannot resolve an exact phase."""
    from . import rules

    return first_text_column(
        row,
        lambda text: rules.has_phase_text_evidence(
            text,
            phase_rules=rules.PHASE_RULES if phase_rules is None else phase_rules,
            post=rules.RX_PHASE_POST_CONTRAST if post is None else post,
            extra_patterns=extra_patterns,
        ),
        cols,
    )


def get_exam_group_cols(
    df: pd.DataFrame,
    patient_col: str = "patient_key",
    study_col: str | None = "study_id",
    date_col: str = "date",
    exam_group_columns: Sequence[str] | None = None,
) -> list[str]:
    if exam_group_columns is not None:
        missing = [col for col in exam_group_columns if col not in df.columns]
        if missing:
            fallback = [col for col in ("patient_key", "date") if col in df.columns]
            logger.warning(
                "Best-candidate grouping columns are missing: %s; falling back to "
                "available patient_key/date columns: %s",
                missing,
                fallback,
            )
            return fallback
        return list(exam_group_columns)
    cols = [patient_col]
    if study_col is not None and study_col in df.columns:
        cols.append(study_col)
    if date_col in df.columns:
        cols.append(date_col)
    return [c for c in cols if c in df.columns]
