"""Resolve independently produced annotation candidates without overwriting evidence."""

from __future__ import annotations

import pandas as pd


def _present(series: pd.Series) -> pd.Series:
    return (
        series.notna()
        & series.astype("string").str.strip().ne("")
        & ~series.astype("string").str.upper().isin(["OTHER", "UNKNOWN", "NONE"])
    )


def resolve_annotation(
    df: pd.DataFrame,
    *,
    candidates: list[str],
    target: str,
    source_target: str | None = None,
    conflict_target: str | None = None,
    disagreement: str = "flag",
) -> pd.DataFrame:
    """Resolve candidate columns in precedence order and retain disagreement flags."""
    out = df.copy()
    available = [column for column in candidates if column in out.columns]
    out[target] = pd.NA
    source_target = source_target or f"{target}_source"
    conflict_target = conflict_target or f"{target}_conflict"
    out[source_target] = pd.NA

    for column in available:
        mask = out[target].isna() & _present(out[column])
        out.loc[mask, target] = out.loc[mask, column]
        out.loc[mask, source_target] = column

    def has_conflict(row: pd.Series) -> bool:
        values = {
            str(row[column]).strip().upper()
            for column in available
            if pd.notna(row[column])
            and str(row[column]).strip().upper() not in {"", "OTHER", "UNKNOWN", "NONE"}
        }
        return len(values) > 1

    out[conflict_target] = out.apply(has_conflict, axis=1) if available else False
    if disagreement == "error" and out[conflict_target].any():
        raise ValueError(f"Conflicting candidates while resolving {target!r}")
    if disagreement == "ignore":
        out[conflict_target] = False
    return out
