"""MRI contextual phase inference as named rules with explicit dependencies.

The numerical and protocol algorithms remain in mri_curation for compatibility.
This module owns their orchestration and converts their outputs into proposals.
"""

from __future__ import annotations

import pandas as pd

from .. import metadata_helpers as common
from ..native_inference import infer_pre_post_native_acquisitions
from ..phase_resolution import apply_phase_curation
from ..policy import CurationPolicy
from ..rule_engine import (
    FunctionRule,
    RuleContext,
    RuleResult,
    RuleRunner,
    execution_order,
    validate_rule_inputs,
)
from ..rule_context import custom_contexts


def builtin_rules():
    from . import mri_curation as mc
    from . import mri_rules as rules

    def acquisition_phase(context):
        shared = mc._share_acquisition_phase_evidence(context.rows)
        for idx, row in shared.loc[
            shared.mri_perfusion_source.eq("acquisition_phase")
        ].iterrows():
            yield RuleResult(
                idx,
                row.mri_perfusion_label,
                row.mri_perfusion_reason,
                row.mri_perfusion_confidence,
                "acquisition_phase",
            )

    def profile(function, source):
        def evaluate(context):
            for idx, phase, reason in function(context.rows):
                yield RuleResult(
                    idx,
                    phase,
                    reason,
                    "unknown" if phase == "OTHER" else "inferred",
                    source,
                )

        return evaluate

    def native(context):
        for idx, reason in mc._infer_missing_native_fallbacks(context.rows):
            yield RuleResult(idx, "NATIVE", reason, "fallback", "exam_context")

    def generic(context):
        for (
            idx,
            phase,
            reason,
            source,
            order,
        ) in mc.infer_generic_dynamic_phases_from_exam_context(context.rows):
            yield RuleResult(
                idx,
                phase,
                reason,
                "unknown" if phase == "OTHER" else "inferred",
                source,
                order,
            )

    def pre_post(context):
        eligible = context.rows.loc[
            context.rows["mri_sequence"].eq("T1")
            & ~context.rows["is_derived_low_value"].fillna(False)
        ]
        compatibility = [
            c
            for c in ("study_id", "StudyInstanceUID", "dixon_component", "plane")
            if c in eligible
        ]
        for idx, reason in infer_pre_post_native_acquisitions(
            eligible,
            text_columns=context.text_columns,
            group_columns=compatibility,
            phase_column="mri_perfusion_label",
            native=rules.RX_PHASE_NATIVE,
            post=rules.RX_PHASE_POST_CONTRAST,
            phase_rules=rules.PHASE_RULES,
            extra_patterns=(rules.RX_PHASE_GENERIC_DYNAMIC,),
        ):
            row = context.rows.loc[idx]
            if row.mri_perfusion_source == "dicom_phase_conflict":
                continue
            if row.mri_perfusion_label != "OTHER" and not (
                row.mri_perfusion_label == "NATIVE"
                and row.mri_perfusion_source
                in {"volume_order", "acquisition_order_dixon_component"}
            ):
                continue
            yield RuleResult(idx, "NATIVE", reason, "inferred", "group_pre_post_order")

    # These dependencies preserve the established inference sequence. Disabling
    # an optional pass removes it from the built-in dependency chain below.
    return [
        FunctionRule("acquisition_phase", "MR", acquisition_phase),
        FunctionRule(
            "art_port_late",
            "MR",
            profile(
                mc.infer_art_port_late_phases_by_acquisition_order,
                "acquisition_order_art_port_late",
            ),
        ),
        FunctionRule(
            "art_port",
            "MR",
            profile(
                mc.infer_art_port_phases_by_acquisition_order,
                "acquisition_order_art_port",
            ),
        ),
        FunctionRule(
            "mask_multiart",
            "MR",
            profile(
                mc.infer_mask_multiart_phases_by_acquisition_order,
                "acquisition_order_mask_multiart",
            ),
        ),
        FunctionRule("native_context", "MR", native),
        FunctionRule("generic_dynamic", "MR", generic),
        FunctionRule("pre_post_native", "MR", pre_post),
    ]


def run_mri_context_rules(df, *, exam_group_cols, policy: CurationPolicy, trace=None):
    from dataclasses import replace
    from . import mri_curation as mc

    out = df.copy()
    text_columns = tuple(mc._resolve_text_cols())
    runner = RuleRunner(
        out,
        modality="MR",
        precedence=policy.precedence("MR"),
        trace=trace,
        unresolved_labels=policy.phase_curation["unresolved_labels"],
    )
    initial = RuleContext(out.copy(), "MR", text_columns, (), "row")
    # Direct metadata has already been evaluated. Keep legacy weak volume-order
    # evidence revisable by contextual rules, while protecting explicit labels.
    for idx, row in out.iterrows():
        proposal = RuleResult(
            idx,
            row.mri_perfusion_label,
            row.mri_perfusion_reason,
            row.mri_perfusion_confidence,
            row.mri_perfusion_source,
        )
        runner.record(
            idx,
            "metadata",
            proposal,
            "accepted",
            "initial metadata evidence",
            replace(initial, rows=out.loc[[idx]].copy()),
        )
        if row.mri_perfusion_label != "OTHER" and row.mri_perfusion_source in {
            "explicit_text",
            "dicom_phase",
            "explicit_duration",
        }:
            runner.winners[idx] = "metadata"

    config = policy.phase_curation
    preceding = []
    for strategy in config["strategies"]:
        if strategy["type"] == "rules":
            break
        preceding.append(strategy)
    authority = (
        apply_phase_curation(out, {**config, "strategies": preceding})
        if preceding
        else pd.DataFrame(index=out.index)
    )
    authoritative = (
        frozenset(authority.index[authority.phase_status.eq("RESOLVED")])
        if "phase_status" in authority
        else frozenset()
    )
    registrations = {
        r.rule.name: r for r in policy.registrations if r.rule.modality == "MR"
    }
    options = policy.options("MR")
    active = []
    previous = "metadata"
    for rule in builtin_rules():
        if rule.name not in options.get("disabled", ()):
            active.append(replace(rule, requires=(previous,)))
            previous = rule.name
    for registration in registrations.values():
        validate_rule_inputs(registration.rule, out)
    active.extend(r.rule for r in registrations.values())
    ordered = execution_order(active, policy.precedence("MR"))

    group_cols = [c for c in (exam_group_cols or []) if c in out]
    for col in options.get("context_columns", ()):
        if col not in out:
            raise ValueError(f"Missing curation context column: {col}")
        if col not in group_cols:
            group_cols.append(col)
    groups = (
        out.groupby(
            [out[c].map(common.stable_text) for c in group_cols], dropna=False
        ).groups.values()
        if group_cols
        else [out.index]
    )

    def context_rows(indices):
        rows = out.loc[indices].copy()
        selected = rows.index.intersection(list(authoritative))
        if len(selected):
            rows.loc[selected, "mri_perfusion_label"] = authority.loc[selected, "phase"]
            rows.loc[selected, "mri_perfusion_source"] = "strategy_context"
            rows.loc[selected, "mri_perfusion_reason"] = authority.loc[
                selected, "phase_reason"
            ]
            rows.loc[selected, "mri_perfusion_confidence"] = authority.loc[
                selected, "phase_confidence"
            ]
        # Accepted Python proposals can anchor the subsequent built-in chronology.
        for idx in rows.index:
            if runner.winners.get(idx) in registrations and idx not in authoritative:
                rows.loc[idx, "mri_perfusion_source"] = "strategy_context"
        return RuleContext(
            rows,
            "MR",
            text_columns,
            tuple(group_cols),
            authoritative_rows=authoritative.intersection(rows.index),
        )

    for indices in groups:
        # Same initialization as the legacy pipeline, including acquisition sharing
        # before delay estimation. Subsequent native changes recompute the delays.
        shared = (
            mc._share_acquisition_phase_evidence(context_rows(indices).rows)
            if "acquisition_phase" not in options.get("disabled", ())
            else context_rows(indices).rows
        )
        out.loc[indices, "time_delta"] = mc._dynamic_time_deltas(shared)
        for rule in ordered:
            context = context_rows(indices)
            registration = registrations.get(rule.name)
            contexts = (
                custom_contexts(
                    context, registration.context, protocol_key=mc._generic_dynamic_key
                )
                if registration
                else [context]
            )
            changed = []
            for scoped in contexts:
                changed.extend(
                    runner.apply(
                        rule.name,
                        rule.evaluate(scoped),
                        scoped,
                        custom=registration is not None,
                    )
                )
            if changed and (rule.name == "native_context" or registration):
                out.loc[indices, "time_delta"] = mc._dynamic_time_deltas(
                    context_rows(indices).rows
                )
    return out
