"""Consistent CSV and Parquet handling for pipeline artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from imperandi.config.models import TableFormat

logger = logging.getLogger(__name__)

# This is deliberately a product-level UX heuristic, not project configuration.
CSV_FILE_COUNT_WARNING_THRESHOLD = 100_000


def table_suffix(table_format: TableFormat | str) -> str:
    return ".parquet" if TableFormat(table_format) is TableFormat.PARQUET else ".csv"


def warn_if_csv_is_large(
    table_format: TableFormat | str,
    file_count: int,
    *,
    log: logging.Logger = logger,
) -> bool:
    """Recommend Parquet for a large inventory without changing user intent."""
    should_warn = (
        TableFormat(table_format) is TableFormat.CSV
        and file_count > CSV_FILE_COUNT_WARNING_THRESHOLD
    )
    if should_warn:
        log.warning(
            "%d DICOM files were discovered. Large CSV inventories may be slow "
            "and storage-intensive; Parquet is recommended. Continuing with CSV.",
            file_count,
        )
    return should_warn


def _is_structured(value: Any) -> bool:
    return isinstance(value, (list, tuple, set, dict, np.ndarray))


def _json_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    elif isinstance(value, set):
        value = sorted(value, key=str)
    elif isinstance(value, tuple):
        value = list(value)
    return json.dumps(value, ensure_ascii=False, default=str)


def _is_missing_scalar(value: Any) -> bool:
    if _is_structured(value):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _storage_frame_and_schema(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """JSON-encode structured columns so both formats tolerate mixed cells.

    DICOM metadata assembly can legitimately produce a scalar for one volume and
    a list for another in the same column. Arrow cannot represent that mixed
    shape directly, and CSV cannot round-trip it without an explicit schema.
    """
    out = df.copy()
    structured_columns = []
    for column in out.columns:
        non_missing = out[column].dropna()
        if any(_is_structured(value) for value in non_missing):
            structured_columns.append(column)
            out[column] = out[column].apply(
                lambda value: (
                    value if _is_missing_scalar(value) else _json_value(value)
                )
            )
    schema = {
        "version": 1,
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "json_columns": structured_columns,
    }
    return out, schema


def table_schema_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_suffix(path.suffix + ".schema.json")


def _decode_json_columns(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    schema_path = table_schema_path(path)
    if not schema_path.exists():
        return df
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    for column in schema.get("json_columns", []):
        if column not in df.columns:
            continue

        def decode(value: Any) -> Any:
            if not isinstance(value, str) or not value.strip():
                return value
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                # Accommodate sidecars produced before scalar cells were encoded.
                return value

        df[column] = df[column].apply(decode)
    return df


def _csv_dtype_hints(path: Path) -> dict[str, str]:
    schema_path = table_schema_path(path)
    if not schema_path.exists():
        return {}
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    hints = {column: "string" for column in schema.get("json_columns", [])}
    for column, dtype in schema.get("dtypes", {}).items():
        if dtype == "object" or dtype.startswith("string"):
            hints[column] = "string"
    return hints


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out, schema = _storage_frame_and_schema(df)
    if path.suffix.lower() == ".parquet":
        out.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        out.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported table extension: {path.suffix}")
    table_schema_path(path).write_text(
        json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return _decode_json_columns(pd.read_parquet(path), path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported table extension: {path.suffix}")

    hints = _csv_dtype_hints(path)
    return _decode_json_columns(
        pd.read_csv(path, dtype=hints if hints else object), path
    )
