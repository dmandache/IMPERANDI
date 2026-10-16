"""Shared manifest predicates for cleaning filters and phase assignments."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

import numpy as np
import pandas as pd

SUPPORTED_OPERATORS = {
    "eq",
    "ne",
    "in",
    "not_in",
    "contains",
    "icontains",
    "regex",
    "lt",
    "lte",
    "gt",
    "gte",
    "is_null",
    "not_null",
}
_COLLECTION_TYPES = (list, tuple, set, np.ndarray)


def validate_rule_group(
    group: Mapping,
    field: str,
    *,
    default_logic: str | None = None,
    strict_values: bool = False,
) -> dict:
    """Validate the common ``logic``/``rules`` structure without changing inputs.

    Cleaning requires explicit logic for multiple rules; phase assignments default
    to AND and additionally require finite scalar values and nonempty patterns.
    """
    rules = group.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError(f"{field} must define a non-empty 'rules' list.")
    logic = group.get("logic")
    if logic is None:
        if len(rules) > 1 and default_logic is None:
            raise ValueError(
                f"{field} must define logic='and' or 'or' for multiple rules."
            )
        logic = default_logic or "and"
    if not isinstance(logic, str) or logic not in {"and", "or"}:
        raise ValueError(f"{field}.logic must be 'and' or 'or'.")
    normalized = []
    for index, raw_rule in enumerate(rules):
        rule_field = f"{field}.rules[{index}]"
        if not isinstance(raw_rule, Mapping):
            raise ValueError(f"{rule_field} must be a mapping.")
        rule = dict(raw_rule)
        column = rule.get("column")
        if not isinstance(column, str) or not column.strip():
            raise ValueError(f"{rule_field}.column must be a non-empty string.")
        rule["column"] = column.strip()
        op = rule.get("op")
        if not isinstance(op, str) or op not in SUPPORTED_OPERATORS:
            raise ValueError(f"{rule_field}: Unsupported filter operator: {op!r}.")
        if op not in {"is_null", "not_null"} and "value" not in rule:
            raise ValueError(f"{rule_field}: operator {op!r} requires a 'value'.")
        target = rule.get("value")
        if op in {"in", "not_in"} and not isinstance(target, list):
            raise ValueError(f"{rule_field}: operator {op!r} requires a list 'value'.")
        if strict_values:
            targets = target if op in {"in", "not_in"} else [target]
            if op not in {"is_null", "not_null"} and any(
                not isinstance(item, (str, int, float, bool))
                or (isinstance(item, float) and not math.isfinite(item))
                for item in targets
            ):
                raise ValueError(f"{rule_field}.value must contain scalar values.")
            if op in {"contains", "icontains", "regex"} and (
                not isinstance(target, str) or not target
            ):
                raise ValueError(f"{rule_field}.value must be a non-empty string.")
            if op in {"lt", "lte", "gt", "gte"} and (
                isinstance(target, bool) or not isinstance(target, (int, float))
            ):
                raise ValueError(f"{rule_field}.value must be numeric.")
        if op == "regex":
            try:
                re.compile(str(target))
            except re.error as exc:
                raise ValueError(f"{rule_field}.value is an invalid regex.") from exc
        normalized.append(rule)
    return {"rules": normalized, "logic": logic}


def _scalar_rule_mask(series: pd.Series, rule: Mapping) -> pd.Series:
    op = rule["op"]
    value = rule.get("value")
    non_null = series.notna()
    if op == "eq":
        return non_null & (series == value).fillna(False)
    if op == "ne":
        return non_null & (series != value).fillna(False)
    if op == "in":
        return non_null & series.isin(value)
    if op == "not_in":
        return non_null & ~series.isin(value)
    if op in {"contains", "icontains", "regex"}:
        return series.astype("string").str.contains(
            str(value), regex=op == "regex", case=op != "icontains", na=False
        )
    if op in {"lt", "lte", "gt", "gte"}:
        numeric = pd.to_numeric(series, errors="coerce")
        comparison = {
            "lt": numeric.lt,
            "lte": numeric.le,
            "gt": numeric.gt,
            "gte": numeric.ge,
        }[op]
        return np.isfinite(numeric).fillna(False) & comparison(value).fillna(False)
    if op == "is_null":
        return series.isna()
    if op == "not_null":
        return non_null
    raise ValueError(f"Unsupported filter operator: {op!r}")


def _flatten(value):
    if isinstance(value, _COLLECTION_TYPES):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


def rule_mask(df: pd.DataFrame, rule: Mapping) -> pd.Series:
    """Evaluate one predicate, preserving row positions and duplicate indices.

    Populated list items use any-match semantics, except NE/NOT IN require all
    items to pass. Empty/all-null lists count as null. Absent columns never match.
    """
    if rule["column"] not in df:
        return pd.Series(False, index=df.index)
    series = df[rule["column"]]
    if not any(isinstance(value, _COLLECTION_TYPES) for value in series):
        return _scalar_rule_mask(series, rule).fillna(False).astype(bool)
    flattened = (
        series.reset_index(drop=True).map(lambda value: list(_flatten(value))).explode()
    )
    op = rule["op"]
    if op in {"ne", "not_in"}:
        flattened = flattened.loc[flattened.notna()]
    mask = _scalar_rule_mask(flattened, rule).fillna(False)
    grouped = mask.groupby(level=0)
    combined = grouped.all() if op in {"ne", "not_in", "is_null"} else grouped.any()
    combined = combined.reindex(range(len(df)), fill_value=False)
    return pd.Series(combined.to_numpy(dtype=bool), index=df.index)


def rules_mask(df: pd.DataFrame, rules: list[Mapping], logic: str) -> pd.Series:
    """Combine validated predicates using the manifest's AND/OR logic."""
    mask = rule_mask(df, rules[0])
    for rule in rules[1:]:
        current = rule_mask(df, rule)
        mask = mask & current if logic == "and" else mask | current
    return mask
