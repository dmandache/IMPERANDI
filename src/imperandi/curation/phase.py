"""Manifest-driven contrast-phase resolution with ordered fallbacks."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any

import pandas as pd

SUPPORTED_PHASE_STRATEGIES = {"ontology", "rules", "totalsegmentator"}
DEFAULT_UNRESOLVED_LABELS = ["", "OTHER", "UNKNOWN", "UNCLASSIFIED", "NONE"]
DEFAULT_PHASE_CURATION = {
    "strategies": [{"type": "rules"}],
    "fallback": "OTHER",
    "unresolved_labels": DEFAULT_UNRESOLVED_LABELS,
}


def normalize_phase_label(value: Any) -> str | None:
    """Normalize a phase value to uppercase underscore form."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None

    text = str(value).strip()
    if not text:
        return None
    return re.sub(r"[^A-Z0-9]+", "_", text.upper()).strip("_") or None


def _validate_string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list of strings.")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field} must contain only non-empty strings.")
    return [item.strip() for item in value]


def _validate_mapping(value: Any, field: str, *, required: bool) -> dict[str, str]:
    if value is None and not required:
        return {}
    if not isinstance(value, Mapping) or (required and not value):
        qualifier = "a non-empty" if required else "a"
        raise ValueError(f"{field} must be {qualifier} mapping.")

    mapping: dict[str, str] = {}
    for source, target in value.items():
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{field} keys must be non-empty strings.")
        normalized_target = normalize_phase_label(target)
        if normalized_target is None:
            raise ValueError(f"{field} values must be non-empty phase labels.")
        mapping[source.strip().casefold()] = normalized_target
    return mapping


def validate_phase_curation(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and normalize the manifest ``phase_curation`` section."""
    if config is None:
        config = DEFAULT_PHASE_CURATION
    if not isinstance(config, Mapping):
        raise ValueError("phase_curation must be a mapping.")

    strategies = config.get("strategies")
    if not isinstance(strategies, list) or not strategies:
        raise ValueError("phase_curation.strategies must be a non-empty list.")

    normalized_strategies: list[dict[str, Any]] = []
    seen_types: set[str] = set()
    for index, raw_strategy in enumerate(strategies):
        field = f"phase_curation.strategies[{index}]"
        if not isinstance(raw_strategy, Mapping):
            raise ValueError(f"{field} must be a mapping.")

        strategy = copy.deepcopy(dict(raw_strategy))
        strategy_type = str(strategy.get("type", "")).strip().lower()
        if strategy_type not in SUPPORTED_PHASE_STRATEGIES:
            raise ValueError(
                f"{field}.type must be one of {sorted(SUPPORTED_PHASE_STRATEGIES)}."
            )
        if strategy_type in seen_types:
            raise ValueError(
                f"phase_curation strategy type {strategy_type!r} may appear only once."
            )
        seen_types.add(strategy_type)
        strategy["type"] = strategy_type

        name = strategy.get("name", strategy_type)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{field}.name must be a non-empty string.")
        strategy["name"] = name.strip()

        if strategy_type == "ontology":
            strategy["columns"] = _validate_string_list(
                strategy.get("columns"), f"{field}.columns"
            )
            strategy["mapping"] = _validate_mapping(
                strategy.get("mapping"), f"{field}.mapping", required=True
            )
            confidence = strategy.get("confidence", "high")
            if not isinstance(confidence, (str, int, float)):
                raise ValueError(f"{field}.confidence must be a scalar value.")
            strategy["confidence"] = confidence

        elif strategy_type == "rules":
            strategy["mapping"] = _validate_mapping(
                strategy.get("mapping"), f"{field}.mapping", required=False
            )

        else:
            column = strategy.get("column", "totalseg_phase")
            if not isinstance(column, str) or not column.strip():
                raise ValueError(f"{field}.column must be a non-empty string.")
            strategy["column"] = column.strip()
            strategy["modalities"] = [
                normalize_phase_label(value)
                for value in _validate_string_list(
                    strategy.get("modalities", ["CT"]),
                    f"{field}.modalities",
                )
            ]
            strategy["mapping"] = _validate_mapping(
                strategy.get("mapping"), f"{field}.mapping", required=False
            )
            confidence_columns = strategy.get(
                "confidence_columns",
                ["totalseg_probability", "totalseg_confidence"],
            )
            strategy["confidence_columns"] = _validate_string_list(
                confidence_columns, f"{field}.confidence_columns"
            )

        normalized_strategies.append(strategy)

    fallback = config.get("fallback", "OTHER")
    if fallback is not None:
        fallback = normalize_phase_label(fallback)
        if fallback is None:
            raise ValueError("phase_curation.fallback must be a phase label or null.")

    unresolved = config.get("unresolved_labels", DEFAULT_UNRESOLVED_LABELS)
    if not isinstance(unresolved, (list, tuple, set)):
        raise ValueError("phase_curation.unresolved_labels must be a list.")
    unresolved_labels = {
        label
        for value in unresolved
        if (label := normalize_phase_label(value)) is not None
    }
    unresolved_labels.add("")

    return {
        "strategies": normalized_strategies,
        "fallback": fallback,
        "unresolved_labels": unresolved_labels,
    }


def phase_curation_input_columns(config: Mapping[str, Any] | None) -> set[str]:
    """Return source columns referenced by manifest-defined strategies."""
    normalized = validate_phase_curation(config)
    columns: set[str] = set()
    for strategy in normalized["strategies"]:
        if strategy["type"] == "ontology":
            columns.update(strategy["columns"])
        elif strategy["type"] == "totalsegmentator":
            columns.add(strategy["column"])
            columns.update(strategy["confidence_columns"])
            if strategy.get("modalities"):
                columns.add("Modality")
    return columns


def get_phase_strategy(
    config: Mapping[str, Any] | None, strategy_type: str
) -> dict[str, Any] | None:
    """Return one normalized strategy by type, if configured."""
    normalized = validate_phase_curation(config)
    wanted = strategy_type.strip().lower()
    return next(
        (
            strategy
            for strategy in normalized["strategies"]
            if strategy["type"] == wanted
        ),
        None,
    )


def _iter_values(value: Any):
    if isinstance(value, (list, tuple, set)):
        values = sorted(value, key=str) if isinstance(value, set) else value
        for item in values:
            yield from _iter_values(item)
        return
    yield value


def _map_value(
    value: Any,
    mapping: Mapping[str, str],
    *,
    allow_unmapped: bool,
) -> tuple[str | None, Any]:
    for candidate in _iter_values(value):
        try:
            if candidate is None or pd.isna(candidate):
                continue
        except (TypeError, ValueError):
            continue
        raw = str(candidate).strip()
        if not raw:
            continue
        mapped = mapping.get(raw.casefold())
        if mapped is not None:
            return mapped, candidate
        if allow_unmapped:
            return normalize_phase_label(candidate), candidate
    return None, None


def _first_populated(row: Mapping[str, Any], columns: list[str]) -> Any:
    for column in columns:
        for candidate in _iter_values(row.get(column)):
            try:
                missing = candidate is None or pd.isna(candidate)
            except (TypeError, ValueError):
                missing = True
            if not missing and str(candidate).strip():
                return candidate
    return None


def _strategy_applies(row: Mapping[str, Any], strategy: Mapping[str, Any]) -> bool:
    modalities = strategy.get("modalities")
    if not modalities or "Modality" not in row:
        return True

    found = {
        label
        for value in _iter_values(row.get("Modality"))
        if (label := normalize_phase_label(value)) is not None
    }
    return bool(found.intersection(modalities))


def _resolve_strategy(
    row: Mapping[str, Any], strategy: Mapping[str, Any]
) -> tuple[str | None, Any, str | None]:
    strategy_type = strategy["type"]
    mapping = strategy.get("mapping", {})

    if not _strategy_applies(row, strategy):
        return None, None, None

    if strategy_type == "ontology":
        for column in strategy["columns"]:
            phase, raw = _map_value(row.get(column), mapping, allow_unmapped=False)
            if phase is not None:
                return (
                    phase,
                    strategy["confidence"],
                    f"ontology matched {column}={raw!r}",
                )
        return None, None, None

    if strategy_type == "rules":
        phase, _ = _map_value(
            row.get("rule_phase"), mapping, allow_unmapped=not mapping
        )
        return phase, row.get("rule_phase_confidence"), row.get("rule_phase_reason")

    column = strategy["column"]
    phase, raw = _map_value(row.get(column), mapping, allow_unmapped=not mapping)
    confidence = _first_populated(row, strategy["confidence_columns"])
    reason = f"TotalSegmentator mapped {column}={raw!r}" if phase else None
    return phase, confidence, reason


def apply_phase_curation(
    df: pd.DataFrame,
    config: Mapping[str, Any] | None,
    *,
    stop_before: str | None = None,
    apply_fallback: bool = True,
) -> pd.DataFrame:
    """Resolve a canonical phase using configured strategy precedence."""
    normalized = validate_phase_curation(config)
    source = df.copy()
    out = df.copy()
    out["phase"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out["phase_source"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out["phase_confidence"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out["phase_reason"] = pd.Series(pd.NA, index=out.index, dtype="object")

    unresolved = [True] * len(out)
    output_positions = {
        column: out.columns.get_loc(column)
        for column in (
            "phase",
            "phase_source",
            "phase_confidence",
            "phase_reason",
        )
    }
    stop_type = stop_before.strip().lower() if stop_before else None
    for strategy in normalized["strategies"]:
        if strategy["type"] == stop_type:
            break
        for row_position, is_unresolved in enumerate(unresolved):
            if not is_unresolved:
                continue
            phase, confidence, reason = _resolve_strategy(
                source.iloc[row_position], strategy
            )
            if phase is None or phase in normalized["unresolved_labels"]:
                continue
            out.iat[row_position, output_positions["phase"]] = phase
            out.iat[row_position, output_positions["phase_source"]] = strategy["name"]
            out.iat[row_position, output_positions["phase_confidence"]] = confidence
            out.iat[row_position, output_positions["phase_reason"]] = reason
            unresolved[row_position] = False

    if apply_fallback and normalized["fallback"] is not None:
        for row_position, is_unresolved in enumerate(unresolved):
            if not is_unresolved:
                continue
            out.iat[row_position, output_positions["phase"]] = normalized["fallback"]
            out.iat[row_position, output_positions["phase_source"]] = "fallback"
            out.iat[row_position, output_positions["phase_reason"]] = (
                "no phase strategy resolved"
            )

    return out


def phase_needs_strategy(
    df: pd.DataFrame,
    config: Mapping[str, Any] | None,
    strategy_type: str,
) -> pd.Series:
    """Return eligible rows unresolved before ``strategy_type`` is reached."""
    normalized = validate_phase_curation(config)
    wanted = strategy_type.strip().lower()
    strategy = next(
        (item for item in normalized["strategies"] if item["type"] == wanted),
        None,
    )
    if strategy is None:
        return pd.Series(False, index=df.index)

    prior = apply_phase_curation(
        df,
        normalized,
        stop_before=wanted,
        apply_fallback=False,
    )
    eligible = [
        _strategy_applies(df.iloc[position], strategy) for position in range(len(df))
    ]
    return pd.Series(
        [
            bool(pd.isna(prior["phase"].iloc[position])) and eligible[position]
            for position in range(len(df))
        ],
        index=df.index,
    )


__all__ = [
    "DEFAULT_PHASE_CURATION",
    "SUPPORTED_PHASE_STRATEGIES",
    "apply_phase_curation",
    "get_phase_strategy",
    "normalize_phase_label",
    "phase_curation_input_columns",
    "phase_needs_strategy",
    "validate_phase_curation",
]
