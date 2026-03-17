from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from imperandi.utils.manifest import load_manifest


def normalize_cli_filters(filter_args: list[str] | None) -> dict[str, list[str]]:
    if not filter_args:
        return {}

    normalized: dict[str, list[str]] = {}
    for raw_filter in filter_args:
        if raw_filter.count("=") != 1:
            raise ValueError(
                "Invalid --filter value. Expected exactly one '=' in "
                f"{raw_filter!r}."
            )

        column, raw_values = raw_filter.split("=", 1)
        column = column.strip()
        if not column:
            raise ValueError(
                f"Invalid --filter value {raw_filter!r}: column name is empty."
            )

        values = [value.strip() for value in raw_values.split(",") if value.strip()]
        if not values:
            raise ValueError(
                f"Invalid --filter value {raw_filter!r}: at least one value is required."
            )
        normalized[column] = values

    return normalized


def load_manifest_stage_filters(
    manifest_arg: str | None,
    *,
    stage_key: str,
) -> dict[str, list[Any]]:
    if not manifest_arg:
        return {}

    manifest = load_manifest(
        manifest_arg,
        base_path=Path(__file__).resolve().parents[1],
    )
    stage_settings = manifest.get(stage_key)
    if stage_settings is None:
        return {}
    if not isinstance(stage_settings, dict):
        raise ValueError(
            f"Manifest {stage_key} settings must be an object under key '{stage_key}'."
        )

    raw_filters = stage_settings.get("filters")
    if raw_filters is None:
        return {}
    if not isinstance(raw_filters, dict):
        raise ValueError(f"Manifest {stage_key}.filters must be an object.")

    normalized: dict[str, list[Any]] = {}
    for column, values in raw_filters.items():
        column_name = str(column).strip()
        if not column_name:
            raise ValueError(
                f"Manifest {stage_key}.filters contains an empty column name."
            )
        if not isinstance(values, list):
            raise ValueError(
                f"Manifest {stage_key}.filters[{column_name!r}] must be a list."
            )
        if not values:
            raise ValueError(
                f"Manifest {stage_key}.filters[{column_name!r}] must not be empty."
            )
        normalized[column_name] = values

    return normalized


def resolve_row_filters(
    args: argparse.Namespace,
    *,
    stage_key: str,
    stage_label: str,
    logger: logging.Logger,
) -> dict[str, list[Any]]:
    if getattr(args, "skip_filter", False):
        logger.info("%s row filters skipped via --skip_filter", stage_label)
        return {}

    cli_filters = getattr(args, "filters", {}) or {}
    manifest_filters = load_manifest_stage_filters(
        getattr(args, "manifest", None),
        stage_key=stage_key,
    )
    effective_filters = dict(cli_filters)

    if manifest_filters:
        overlapping_columns = sorted(set(cli_filters) & set(manifest_filters))
        for column in overlapping_columns:
            logger.info(
                "Manifest %s filter overrides CLI filter for column '%s'",
                stage_key,
                column,
            )
        effective_filters.update(manifest_filters)

    if effective_filters:
        logger.info("%s row filters resolved: %s", stage_label, effective_filters)
    else:
        logger.info("%s row filters resolved: none", stage_label)
    return effective_filters


def apply_row_filters(
    df: pd.DataFrame,
    filters: dict[str, list[Any]],
    *,
    stage_label: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    if not filters:
        logger.info(
            "No %s row filters applied; keeping %d rows",
            stage_label.lower(),
            len(df),
        )
        return df

    missing_columns = [column for column in filters if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{stage_label} filter column(s) missing from input CSV: "
            + ", ".join(sorted(missing_columns))
        )

    filtered = df.copy()
    logger.info("Applying %s row filters to %d rows", stage_label.lower(), len(filtered))
    for column, allowed_values in filters.items():
        before_count = len(filtered)
        filtered = filtered[filtered[column].isin(allowed_values)]
        logger.info(
            "%s filter applied | column=%s | values=%s | rows=%d -> %d",
            stage_label,
            column,
            allowed_values,
            before_count,
            len(filtered),
        )
    return filtered
