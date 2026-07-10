from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

from imperandi.utils.manifest import resolve_hook


CLEAN_HOOK_OUTPUTS_ATTR = "_imperandi_clean_outputs"


def clean_hook(*, outputs: list[str]):
    """Declare which columns a clean-stage hook produces."""

    def decorator(func: Callable) -> Callable:
        setattr(func, CLEAN_HOOK_OUTPUTS_ATTR, tuple(outputs))
        return func

    return decorator


def get_clean_hook_outputs(func: Callable) -> list[str]:
    """Return the column list declared by ``@clean_hook``."""
    outputs = getattr(func, CLEAN_HOOK_OUTPUTS_ATTR, ())
    return list(outputs)


def apply_patient_key_standardization(
    df: pd.DataFrame,
    hook: Callable | None,
    *,
    logger: logging.Logger | None = None,
    log_prefix: str = "id_standardization",
) -> pd.DataFrame:
    """Rewrite ``patient_key`` while preserving raw values and failure flags."""
    if "patient_key" not in df.columns:
        return df

    df = df.copy()
    if "_patient_key_raw" not in df.columns:
        df["_patient_key_raw"] = df["patient_key"]

    if not hook:
        return df

    df["patient_key"] = df["_patient_key_raw"].apply(hook)

    raw_ok = df["_patient_key_raw"].notna() & (
        df["_patient_key_raw"].astype(str).str.strip() != ""
    )
    std_bad = df["patient_key"].isna() | (
        df["patient_key"].astype(str).str.strip() == ""
    )
    failed = raw_ok & std_bad

    if failed.any():
        df["patient_key_std_failed"] = failed
        if logger is not None:
            n_keys = int(df.loc[failed, "_patient_key_raw"].nunique())
            logger.warning(
                "[%s] failed on unique raw keys=%s",
                log_prefix,
                n_keys,
            )

    return df


def apply_id_standardization(
    df: pd.DataFrame,
    manifest: dict,
    *,
    hook_resolver: Callable[[dict], Callable | None] = resolve_hook,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Apply the manifest's parse-time ``id_standardization`` hook."""
    hook = hook_resolver(manifest.get("id_standardization") or {})
    return apply_patient_key_standardization(df, hook, logger=logger)


def apply_derived_columns(
    df: pd.DataFrame,
    manifest: dict,
    *,
    hook_resolver: Callable[[dict], Callable | None] = resolve_hook,
) -> pd.DataFrame:
    """Join manifest-derived columns onto ``df`` using the configured join mode."""
    derived_columns = manifest.get("derived_columns", [])
    if not derived_columns:
        return df

    for derived in derived_columns:
        from_column = derived.get("from_column")
        if not from_column or from_column not in df.columns:
            continue
        hook = hook_resolver(derived)
        if not hook:
            continue
        derived_values = df[from_column].apply(hook)
        derived_df = derived_values.apply(pd.Series)
        if derived_df.empty:
            continue
        join_mode = derived.get("join_mode", "missing_only")
        if join_mode == "overwrite":
            df = df.drop(
                columns=[col for col in derived_df.columns if col in df.columns]
            )
            df = df.join(derived_df)
        else:
            derived_df = derived_df.loc[:, ~derived_df.columns.isin(df.columns)]
            if not derived_df.empty:
                df = df.join(derived_df)
    return df
