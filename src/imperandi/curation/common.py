"""Shared helpers for deterministic modality curation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

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


def build_series_text(
    row: pd.Series,
    cols: Sequence[str] | None = None,
) -> str:
    cols = list(cols or TEXT_COLS_DEFAULT)
    parts = [safe_str(row.get(c)) for c in cols if c in row.index]
    return " | ".join(part for part in parts if part)


def get_exam_group_cols(
    df: pd.DataFrame,
    patient_col: str = "patient_key",
    study_col: str | None = "study_id",
    date_col: str = "date",
) -> list[str]:
    cols = [patient_col]
    if study_col is not None and study_col in df.columns:
        cols.append(study_col)
    if date_col in df.columns:
        cols.append(date_col)
    return [c for c in cols if c in df.columns]
