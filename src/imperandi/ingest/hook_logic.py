"""Shared hook application helpers used by ingest parse and clean flows."""

from __future__ import annotations

import logging
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)

HookResolver = Callable[[dict], Any]


def _as_list(value: Any) -> list[Any]:
    """Return list-like representation."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _resolve_hook_safely(
    hook_config: dict,
    *,
    resolve_hook_fn: HookResolver,
    logger_obj: logging.Logger | None = None,
):
    """Resolve hook and return None when unavailable."""
    if not isinstance(hook_config, dict) or not hook_config:
        return None
    try:
        return resolve_hook_fn(hook_config)
    except Exception as exc:
        if logger_obj is not None:
            logger_obj.warning(
                "Skipping invalid hook config %s (%s)",
                hook_config,
                exc,
            )
        return None


def _resolve_standardization_hook(
    manifest: dict,
    *,
    hook_key: str,
    resolve_hook_fn: HookResolver,
    logger_obj: logging.Logger | None = None,
):
    """Resolve standardization hook by key with backward-compatible fallbacks."""
    hook_config = manifest.get("id_standardization") or {}
    if not isinstance(hook_config, dict):
        return None

    keyed_function = hook_config.get(hook_key)
    if keyed_function:
        normalized = {
            "hook_module": hook_config.get("hook_module"),
            "function": keyed_function,
        }
        return _resolve_hook_safely(
            normalized,
            resolve_hook_fn=resolve_hook_fn,
            logger_obj=logger_obj,
        )

    return _resolve_hook_safely(
        hook_config,
        resolve_hook_fn=resolve_hook_fn,
        logger_obj=logger_obj,
    )


def apply_id_standardization(
    df: pd.DataFrame,
    manifest: dict,
    *,
    column: str = "patient_key",
    keep_raw: bool = True,
    mark_failures: bool = True,
    resolve_hook_fn: HookResolver,
    logger_obj: logging.Logger | None = None,
) -> pd.DataFrame:
    """Apply id standardization hook for one column."""
    if column not in df.columns:
        return df

    source_column = column
    raw_column = f"{column}_raw"
    if keep_raw:
        if raw_column not in df.columns:
            df[raw_column] = df[column]
        source_column = raw_column

    hook = _resolve_standardization_hook(
        manifest,
        hook_key=column,
        resolve_hook_fn=resolve_hook_fn,
        logger_obj=logger_obj,
    )
    if not hook:
        return df

    df[column] = df[source_column].apply(hook)

    if not mark_failures:
        return df

    raw_ok = df[source_column].notna() & (df[source_column].astype(str).str.strip() != "")
    std_bad = df[column].isna() | (df[column].astype(str).str.strip() == "")
    failed = raw_ok & std_bad

    failure_column = f"{column}_std_failed"
    if failed.any():
        df[failure_column] = failed
        if logger_obj is not None:
            n_keys = int(df.loc[failed, source_column].nunique())
            logger_obj.warning(
                "[id_standardization] failed on unique raw keys=%s",
                n_keys,
            )

    return df


def _iter_derived_column_operations(
    manifest: dict,
) -> list[dict]:
    """Return normalized derived-column operations."""
    operations: list[dict] = []

    raw_ops = manifest.get("derived_columns", [])
    for entry in _as_list(raw_ops):
        if not isinstance(entry, dict):
            continue

        nested = entry.get("operations")
        if nested is None:
            operations.append(entry)
            continue

        defaults = {k: v for k, v in entry.items() if k != "operations"}
        for op in _as_list(nested):
            if not isinstance(op, dict):
                continue
            merged = dict(defaults)
            merged.update(op)
            operations.append(merged)

    return operations


def apply_derived_columns(
    df: pd.DataFrame,
    manifest: dict,
    *,
    resolve_hook_fn: HookResolver,
    logger_obj: logging.Logger | None = None,
) -> pd.DataFrame:
    """Apply one or many derived-column operations from manifest config."""
    operations = _iter_derived_column_operations(manifest)
    if not operations:
        return df

    for operation in operations:
        from_column = operation.get("from_column")
        if not from_column or from_column not in df.columns:
            continue

        hook = _resolve_hook_safely(
            operation,
            resolve_hook_fn=resolve_hook_fn,
            logger_obj=logger_obj,
        )
        if not hook:
            continue

        derived_values = df[from_column].apply(hook)
        derived_df = derived_values.apply(pd.Series)
        if derived_df.empty:
            continue

        join_mode = str(operation.get("join_mode", "missing_only")).strip().lower()
        if join_mode == "overwrite":
            drop_cols = [col for col in derived_df.columns if col in df.columns]
            if drop_cols:
                df = df.drop(columns=drop_cols)
            df = df.join(derived_df)
            continue

        if join_mode != "missing_only" and logger_obj is not None:
            logger_obj.warning(
                "Unknown join_mode '%s', defaulting to missing_only.",
                join_mode,
            )
        derived_df = derived_df.loc[:, ~derived_df.columns.isin(df.columns)]
        if not derived_df.empty:
            df = df.join(derived_df)

    return df
