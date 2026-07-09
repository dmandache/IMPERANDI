"""Shared helpers for deterministic modality curation."""

from __future__ import annotations

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


def safe_str(x) -> str:
    return str(x).strip().lower() if pd.notna(x) else ""


def norm_label(x, default: str = "OTHER") -> str:
    if pd.isna(x) or str(x).strip() == "":
        return default
    return str(x).strip().upper()


def safe_float(x) -> float:
    try:
        if x is None:
            return np.nan
        if isinstance(x, (list, tuple, set, np.ndarray)):
            return np.nan
        if pd.isna(x) or str(x).strip() == "":
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def build_series_text(
    row: pd.Series,
    cols: Sequence[str] | None = None,
) -> str:
    cols = list(cols or TEXT_COLS_DEFAULT)
    return " | ".join(
        safe_str(row.get(c)) for c in cols if c in row.index and safe_str(row.get(c))
    )


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
