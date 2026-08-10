from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

import pandas as pd

FILTER_ALL_COLUMNS = "__all_columns__"
SUPPORTED_IMAGE_PATH_SUFFIXES = (
    ".nii",
    ".nii.gz",
    ".img",
    ".hdr",
    ".mgh",
    ".mgz",
    ".nrrd",
    ".mha",
    ".mhd",
)


def is_empty_value(value) -> bool:
    """Return whether a scalar should be treated as missing by the viewer."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip() == ""


def load_dataframe(source, source_name: str | None = None) -> pd.DataFrame:
    """Load a supported dataframe from a path or uploaded byte content."""
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve()
        return _read_dataframe(path, path.suffix.lower())

    if source is None:
        raise ValueError("No dataframe source was provided.")

    if source_name is None:
        raise ValueError("A source name is required when loading dataframe bytes.")

    suffix = Path(source_name).suffix.lower()
    return _read_dataframe(BytesIO(bytes(source)), suffix)


def is_image_path_value(value) -> bool:
    """Return whether a value resembles a supported medical-image path."""
    if is_empty_value(value):
        return False

    text = str(value).strip()
    if text.startswith("file://"):
        text = text[7:]

    path = Path(text).expanduser()
    try:
        if path.exists():
            return path.is_file()
    except Exception:
        return False

    lower = text.lower()
    if any(lower.endswith(suffix) for suffix in SUPPORTED_IMAGE_PATH_SUFFIXES):
        return True

    return "/" in text or "\\" in text


def is_image_path_column(
    series: pd.Series,
    *,
    allow_empty: bool = True,
    min_valid_ratio: float = 0.8,
) -> bool:
    """Return whether enough non-empty values in a series look like paths."""
    non_empty = series[~series.apply(is_empty_value)]
    if non_empty.empty:
        return False

    valid_ratio = non_empty.apply(is_image_path_value).mean()
    if valid_ratio < min_valid_ratio:
        return False

    if not allow_empty and len(non_empty) != len(series):
        return False

    return True


def get_image_path_columns(
    df: pd.DataFrame,
    *,
    allow_empty: bool = True,
    min_valid_ratio: float = 0.8,
) -> list[str]:
    """Find dataframe columns whose values are predominantly image paths."""
    return [
        column
        for column in df.columns
        if is_image_path_column(
            df[column],
            allow_empty=allow_empty,
            min_valid_ratio=min_valid_ratio,
        )
    ]


def validate_image_path_column(
    df: pd.DataFrame,
    column: str,
    *,
    allow_empty: bool,
    label: str | None = None,
) -> None:
    """Validate a selected image-path column and raise a useful user error."""
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")

    series = df[column]
    column_label = label or column

    if allow_empty:
        values_to_check = series[~series.apply(is_empty_value)]
    else:
        empty_mask = series.apply(is_empty_value)
        if empty_mask.any():
            raise ValueError(f"{column_label} contains empty values.")
        values_to_check = series

    if values_to_check.empty:
        raise ValueError(f"{column_label} does not contain any usable paths.")

    invalid = values_to_check[~values_to_check.apply(is_image_path_value)]
    if invalid.empty:
        return

    sample = ", ".join(repr(value) for value in invalid.head(3).tolist())
    raise ValueError(
        f"{column_label} must contain file paths. Invalid example values: {sample}"
    )


def guess_ct_scan_col(
    columns: Iterable[str], preferred: str | None = None
) -> str | None:
    """Choose the most likely CT image path column from column names."""
    columns = list(columns)
    if preferred in columns:
        return preferred

    candidates = [
        "nifti_path",
        "ct_scan_path",
        "ct_path",
        "scan_path",
        "image_path",
    ]
    for candidate in candidates:
        if candidate in columns:
            return candidate

    for column in columns:
        lower = str(column).lower()
        if "nifti" in lower and "path" in lower:
            return column

    for column in columns:
        lower = str(column).lower()
        if lower.endswith("_path"):
            return column

    return columns[0] if columns else None


def guess_phase_col(columns: Iterable[str], preferred: str | None = None) -> str | None:
    """Choose a preferred or conventional contrast-phase column."""
    columns = list(columns)
    if preferred in columns:
        return preferred

    for candidate in ["phase", "totalseg_phase"]:
        if candidate in columns:
            return candidate

    return None


def guess_segmentation_cols(
    df: pd.DataFrame,
    preferred: str | Iterable[str] | None = None,
) -> list[str]:
    """Choose populated segmentation path columns for viewer overlays."""
    if preferred is None:
        columns = [c for c in df.columns if str(c).startswith("mask_")]
        if not columns:
            columns = [c for c in ["liver_path", "liver_tumor_path"] if c in df.columns]
    elif isinstance(preferred, str):
        columns = [preferred]
    else:
        columns = list(preferred)

    return [
        column
        for column in columns
        if column in df.columns
        and df[column].apply(lambda value: not is_empty_value(value)).any()
    ]


def filter_dataframe(
    df: pd.DataFrame,
    text: str | None = None,
    column: str | None = FILTER_ALL_COLUMNS,
    mode: str = "contains",
    query: str | None = None,
    case_sensitive: bool = False,
) -> pd.DataFrame:
    """Filter viewer rows with an optional pandas query and text match."""
    filtered = df

    if query and query.strip():
        filtered = filtered.query(query.strip(), engine="python")

    if text is None or str(text).strip() == "":
        return filtered.reset_index(drop=True)

    needle = str(text).strip()

    if column not in [None, FILTER_ALL_COLUMNS] and column not in filtered.columns:
        raise KeyError(f"Column not found: {column}")

    if column in [None, FILTER_ALL_COLUMNS]:
        mask = pd.Series(False, index=filtered.index)
        for current in filtered.columns:
            mask = mask | _match_series(
                filtered[current],
                needle=needle,
                mode=mode,
                case_sensitive=case_sensitive,
            )
    else:
        mask = _match_series(
            filtered[column],
            needle=needle,
            mode=mode,
            case_sensitive=case_sensitive,
        )

    return filtered.loc[mask].reset_index(drop=True)


def _match_series(
    series: pd.Series,
    needle: str,
    mode: str,
    case_sensitive: bool,
) -> pd.Series:
    values = series.fillna("").astype(str)

    if mode == "contains":
        return values.str.contains(needle, regex=False, case=case_sensitive)
    if mode == "exact":
        if not case_sensitive:
            values = values.str.lower()
            needle = needle.lower()
        return values == needle
    if mode == "regex":
        return values.str.contains(needle, regex=True, case=case_sensitive)

    raise ValueError(f"Unsupported filter mode: {mode}")


def _read_dataframe(source, suffix: str) -> pd.DataFrame:
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in [".tsv", ".tab"]:
        return pd.read_csv(source, sep="\t")
    if suffix == ".txt":
        return pd.read_csv(source, sep=None, engine="python")
    if suffix == ".json":
        return pd.read_json(source)
    if suffix in [".pkl", ".pickle"]:
        return pd.read_pickle(source)
    if suffix == ".parquet":
        return pd.read_parquet(source)

    raise ValueError(
        "Unsupported dataframe format. Expected one of: "
        ".csv, .tsv, .tab, .txt, .json, .pkl, .pickle, .parquet."
    )
