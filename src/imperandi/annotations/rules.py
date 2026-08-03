"""Small declarative rule engine for site-specific annotations and exclusions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuleCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str
    operator: Literal[
        "eq",
        "normalized_eq",
        "contains",
        "regex",
        "in",
        "exists",
        "lt",
        "lte",
        "gt",
        "gte",
    ]
    value: Any = None


class RuleWhen(BaseModel):
    model_config = ConfigDict(extra="forbid")
    any: list[RuleCondition] = Field(default_factory=list)
    all: list[RuleCondition] = Field(default_factory=list)

    @model_validator(mode="after")
    def has_conditions(self):
        if not self.any and not self.all:
            raise ValueError("A rule must define at least one any/all condition")
        return self


class AnnotationRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    action: Literal["set", "exclude", "qc"] = "set"
    target: str | None = None
    value: Any = None
    reason: str | None = None
    evidence: Literal["explicit", "inferred"] = "explicit"
    priority: int = 0
    when: RuleWhen

    @model_validator(mode="after")
    def validate_action(self):
        if self.action == "set" and not self.target:
            raise ValueError("set rules require target")
        if self.action in {"exclude", "qc"} and not self.reason:
            raise ValueError(f"{self.action} rules require reason")
        return self


class RulePack(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    rules: list[AnnotationRule]

    @model_validator(mode="after")
    def unique_rule_ids(self):
        ids = [rule.id for rule in self.rules]
        if len(set(ids)) != len(ids):
            raise ValueError("Rule IDs must be unique")
        return self


def _normalize(value) -> str:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        values = value.tolist() if isinstance(value, np.ndarray) else list(value)
        return " | ".join(_normalize(item) for item in values)
    return re.sub(r"\s+", " ", str(value).strip().lower()) if pd.notna(value) else ""


def _values(value) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _condition_mask(df: pd.DataFrame, condition: RuleCondition) -> pd.Series:
    if condition.column not in df.columns:
        return pd.Series(False, index=df.index)
    values = df[condition.column]
    op = condition.operator
    if op == "exists":
        return values.notna()
    if op == "eq":
        return values.apply(lambda value: condition.value in _values(value))
    if op == "normalized_eq":
        return values.apply(_normalize).eq(_normalize(condition.value))
    if op == "contains":
        return values.astype("string").str.contains(
            str(condition.value), regex=False, case=False, na=False
        )
    if op == "regex":
        return values.astype("string").str.contains(
            str(condition.value), regex=True, case=False, na=False
        )
    if op == "in":
        allowed = condition.value
        if not isinstance(allowed, (list, tuple, set)):
            raise TypeError("Rule operator 'in' requires a list-like value")
        allowed = set(allowed)
        return values.apply(
            lambda value: any(item in allowed for item in _values(value))
        )
    numeric = values.apply(
        lambda value: (
            pd.to_numeric(_values(value)[0], errors="coerce")
            if len(_values(value)) == 1
            else np.nan
        )
    )
    if op == "lt":
        return numeric < condition.value
    if op == "lte":
        return numeric <= condition.value
    if op == "gt":
        return numeric > condition.value
    if op == "gte":
        return numeric >= condition.value
    raise ValueError(f"Unsupported rule operator: {op}")


def _rule_mask(df: pd.DataFrame, rule: AnnotationRule) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for condition in rule.when.all:
        mask &= _condition_mask(df, condition)
    if rule.when.any:
        any_mask = pd.Series(False, index=df.index)
        for condition in rule.when.any:
            any_mask |= _condition_mask(df, condition)
        mask &= any_mask
    return mask


def load_rule_pack(path: str | Path) -> RulePack:
    with Path(path).open("r", encoding="utf-8") as handle:
        return RulePack.model_validate(yaml.safe_load(handle) or {})


def apply_rule_pack(df: pd.DataFrame, pack: RulePack) -> pd.DataFrame:
    out = df.copy()
    if "eligible" not in out.columns:
        out["eligible"] = True
    ordered = sorted(pack.rules, key=lambda rule: rule.priority, reverse=True)
    set_priorities: dict[str, pd.Series] = {}
    exclusion_priority = pd.Series(float("-inf"), index=out.index)
    qc_priority = pd.Series(float("-inf"), index=out.index)
    for rule in ordered:
        mask = _rule_mask(out, rule)
        if not mask.any():
            continue
        if rule.action == "exclude":
            apply_mask = mask & exclusion_priority.lt(rule.priority)
            out.loc[apply_mask, "eligible"] = False
            out.loc[apply_mask, "exclusion_reason"] = rule.reason
            out.loc[apply_mask, "exclusion_rule_id"] = rule.id
            exclusion_priority.loc[apply_mask] = rule.priority
        elif rule.action == "qc":
            apply_mask = mask & qc_priority.lt(rule.priority)
            out.loc[apply_mask, "qc_rule"] = rule.reason
            out.loc[apply_mask, "qc_rule_id"] = rule.id
            qc_priority.loc[apply_mask] = rule.priority
        else:
            assert rule.target is not None
            if rule.target not in out.columns:
                out[rule.target] = pd.NA
            priority = set_priorities.setdefault(
                rule.target, pd.Series(float("-inf"), index=out.index)
            )
            apply_mask = mask & priority.lt(rule.priority)
            tie_conflict = (
                mask
                & priority.eq(rule.priority)
                & out[rule.target].notna()
                & out[rule.target].ne(rule.value)
            )
            if tie_conflict.any():
                raise ValueError(
                    f"Rules conflict at equal priority for {rule.target!r}: {rule.id}"
                )
            out.loc[apply_mask, rule.target] = rule.value
            out.loc[apply_mask, f"{rule.target}_rule_id"] = rule.id
            out.loc[apply_mask, f"{rule.target}_evidence"] = rule.evidence
            priority.loc[apply_mask] = rule.priority
    return out


def apply_rule_packs(df: pd.DataFrame, references: list[str]) -> pd.DataFrame:
    rules = []
    for reference in references:
        if reference.startswith("builtin:"):
            # Built-in CT/MR rules are executed by imperandi.curation.
            continue
        rules.extend(load_rule_pack(reference).rules)
    return apply_rule_pack(df, RulePack(rules=rules))
