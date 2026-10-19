"""CT contextual rules and Python extension execution."""

from dataclasses import replace

import pandas as pd

from .. import metadata_helpers as common
from ..native_inference import infer_pre_post_native_acquisitions
from ..phase_resolution import apply_phase_curation
from ..rule_context import custom_contexts
from ..rule_engine import (
    FunctionRule,
    RuleContext,
    RuleResult,
    RuleRunner,
    execution_order,
    validate_rule_inputs,
)
from . import ct_rules


def run_ct_context_rules(out, *, group_columns, policy, trace=None):
    from .ct_curation import detect_ct_features

    out = out.copy()
    runner = RuleRunner(
        out,
        modality="CT",
        precedence=policy.precedence("CT"),
        trace=trace,
        unresolved_labels=policy.phase_curation["unresolved_labels"],
    )
    text_columns = tuple(common.resolve_text_columns(common.TEXT_COLS_DEFAULT))
    context = RuleContext(out.copy(), "CT", text_columns, tuple(group_columns))
    for idx, row in out.iterrows():
        proposal = RuleResult(
            idx, row.ct_phase, row.ct_phase_reason, row.ct_phase_confidence
        )
        runner.record(
            idx,
            "metadata",
            proposal,
            "accepted",
            "initial metadata evidence",
            replace(context, rows=out.loc[[idx]].copy(), scope="row"),
        )
        if row.ct_phase != "OTHER":
            runner.winners[idx] = "metadata"

    def pre_post(context):
        eligible = context.rows.loc[
            context.rows["phase_applicability_reason"].isna()
            & ~context.rows.apply(
                lambda row: detect_ct_features(row)["is_derived_low_value"], axis=1
            )
        ]
        for idx, reason in infer_pre_post_native_acquisitions(
            eligible,
            text_columns=context.text_columns,
            group_columns=list(context.group_columns),
            phase_column="ct_phase",
            phase_rules=ct_rules.PHASE_RULES,
        ):
            if context.rows.loc[idx, "ct_phase"] == "OTHER":
                yield RuleResult(idx, "NATIVE", reason)

    options = policy.options("CT")
    group_columns = list(group_columns)
    for column in options.get("context_columns", ()):
        if column not in out:
            raise ValueError(f"Missing curation context column: {column}")
        if column not in group_columns:
            group_columns.append(column)
    registrations = {
        r.rule.name: r for r in policy.registrations if r.rule.modality == "CT"
    }
    active = (
        []
        if "pre_post_native" in options.get("disabled", ())
        else [FunctionRule("pre_post_native", "CT", pre_post)]
    )
    for registration in registrations.values():
        validate_rule_inputs(registration.rule, out)
    active.extend(r.rule for r in registrations.values())
    ordered = execution_order(active, policy.precedence("CT"))
    config = policy.phase_curation
    preceding = []
    for strategy in config["strategies"]:
        if strategy["type"] == "rules":
            break
        preceding.append(strategy)
    authority = (
        apply_phase_curation(out, {**config, "strategies": preceding})
        if preceding and registrations
        else pd.DataFrame(index=out.index)
    )
    authoritative = (
        frozenset(authority.index[authority.phase_status.eq("RESOLVED")])
        if "phase_status" in authority
        else frozenset()
    )
    groups = (
        out.groupby(
            [out[c].map(common.stable_text) for c in group_columns], dropna=False
        ).groups.values()
        if group_columns
        else [out.index]
    )
    for indices in groups:
        for rule in ordered:
            rows = out.loc[indices].copy()
            protected = authoritative.intersection(rows.index)
            if protected:
                rows.loc[list(protected), "ct_phase"] = authority.loc[
                    list(protected), "phase"
                ]
            context = RuleContext(
                rows,
                "CT",
                text_columns,
                tuple(group_columns),
                authoritative_rows=protected,
            )
            registration = registrations.get(rule.name)
            contexts = (
                custom_contexts(context, registration.context)
                if registration
                else [context]
            )
            for scoped in contexts:
                runner.apply(
                    rule.name,
                    rule.evaluate(scoped),
                    scoped,
                    custom=registration is not None,
                )
    return out
