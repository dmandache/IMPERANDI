"""Named phase rules, explicit context, and deterministic decision arbitration.

Rules propose results; only the runner writes phase columns. Execution dependencies
and evidence precedence are separate: dependencies order evaluation, precedence
decides whether a proposal can replace an earlier result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Protocol

import pandas as pd


@dataclass(frozen=True)
class RuleResult:
    """A phase proposal for an index in ``RuleContext.rows``.

    ``phase=None`` records an abstention. ``OTHER`` records unresolved evidence,
    which never displaces a resolved winning decision. Provisional metadata
    hints without a winning rule may be revised by contextual inference.
    Source defaults to the rule's name.
    """

    row_id: Hashable
    phase: str | None
    reason: str
    confidence: str | float = "inferred"
    source: str | None = None
    order_source: str | None = None


@dataclass(frozen=True)
class RuleContext:
    """An isolated dataframe snapshot for one rule invocation.

    The index is the row identity; volume IDs need not be unique. Authoritative
    strategy labels are already overlaid in ``rows``. Custom rules must return
    proposals rather than editing this snapshot. Missing context columns are
    errors, never a reason to silently widen the group.
    """

    rows: pd.DataFrame
    modality: str
    text_columns: tuple[str, ...]
    group_columns: tuple[str, ...]
    scope: str = "exam"
    authoritative_rows: frozenset[Hashable] = frozenset()


class Rule(Protocol):
    """Python extension contract. Bump version when behavior/settings change.

    ``requires`` names rules that must execute first, even if their evidence has
    lower precedence. ``input_columns`` declares external inputs for caching.
    """

    name: str
    modality: str
    version: str
    input_columns: tuple[str, ...]
    requires: tuple[str, ...]

    def evaluate(self, context: RuleContext) -> Iterable[RuleResult]: ...


@dataclass(frozen=True)
class FunctionRule:
    """Adapt an existing inference function without an inheritance hierarchy."""

    name: str
    modality: str
    evaluator: Callable[[RuleContext], Iterable[RuleResult]]
    requires: tuple[str, ...] = ()
    input_columns: tuple[str, ...] = ()
    version: str = "1"

    def evaluate(self, context: RuleContext) -> Iterable[RuleResult]:
        return self.evaluator(context)


def validate_rule_inputs(rule: Rule, rows: pd.DataFrame) -> None:
    missing = set(rule.input_columns) - set(rows.columns)
    if missing:
        raise ValueError(f"Rule {rule.name!r} needs input columns: {sorted(missing)}")


def execution_order(rules: Iterable[Rule], precedence: tuple[str, ...]) -> list[Rule]:
    """Stable topological order; prefer evidence precedence among ready rules."""
    remaining = {rule.name: rule for rule in rules}
    completed = {"metadata"}
    ordered = []
    while remaining:
        ready = next(
            (
                name
                for name in precedence
                if name in remaining
                and set(remaining[name].requires).issubset(completed)
            ),
            None,
        )
        if ready is None:
            raise ValueError(
                "Curation rule dependencies are missing, disabled, or cyclic: "
                + ", ".join(remaining)
            )
        ordered.append(remaining.pop(ready))
        completed.add(ready)
    return ordered


class RuleRunner:
    """Apply proposals centrally and retain the reason for every disposition."""

    def __init__(
        self,
        rows: pd.DataFrame,
        *,
        modality: str,
        precedence: tuple[str, ...],
        trace: list[dict] | None = None,
        unresolved_labels: Iterable[str] = (),
    ):
        if not rows.index.is_unique:
            raise ValueError("Rule evaluation requires a unique dataframe index.")
        self.rows = rows
        self.modality = modality
        self.precedence = {name: rank for rank, name in enumerate(precedence)}
        self.trace = trace
        self.unresolved_labels = {"OTHER", *unresolved_labels}
        self.prefix = "mri_perfusion" if modality == "MR" else "ct_phase"
        self.label_column = "mri_perfusion_label" if modality == "MR" else "ct_phase"
        self.winners: dict[Hashable, str] = {}
        self._winning_trace: dict[Hashable, int] = {}

    def record(self, row_id, rule, proposal, disposition, detail, context):
        if self.trace is None:
            return
        row = self.rows.loc[row_id]
        self.trace.append(
            {
                "row_id": row_id,
                "volume_id": row.get("volume_id"),
                "modality": self.modality,
                "rule": rule,
                "phase": proposal.phase if proposal else None,
                "source": (proposal.source or rule) if proposal else rule,
                "reason": proposal.reason if proposal else detail,
                "confidence": proposal.confidence if proposal else None,
                "disposition": disposition,
                "detail": detail,
                "scope": context.scope,
                "context_row_ids": tuple(context.rows.index),
                "text_column": row.get("phase_text_column", ""),
                "text_value": row.get("phase_text_value", ""),
            }
        )
        if disposition == "accepted":
            previous = self._winning_trace.get(row_id)
            if previous is not None:
                self.trace[previous]["disposition"] = "superseded"
                self.trace[previous]["detail"] = f"replaced by {rule}"
            self._winning_trace[row_id] = len(self.trace) - 1

    def apply(
        self,
        rule: str,
        proposals: Iterable[RuleResult],
        context: RuleContext,
        *,
        custom: bool = False,
    ):
        from .phase_resolution import normalize_phase_label

        accepted = []
        seen = set()
        for proposal in proposals:
            if not isinstance(proposal, RuleResult):
                raise TypeError(f"Rule {rule!r} must return RuleResult objects.")
            if not isinstance(proposal.reason, str) or not proposal.reason.strip():
                raise ValueError(f"Rule {rule!r} must explain its proposal.")
            idx = proposal.row_id
            if idx not in context.rows.index:
                raise ValueError(
                    f"Rule {rule!r} proposed a row outside its context: {idx!r}"
                )
            if idx in seen:
                raise ValueError(f"Rule {rule!r} proposed the same row twice: {idx!r}")
            seen.add(idx)
            if proposal.phase is None:
                self.record(
                    idx, rule, proposal, "abstained", "rule could not decide", context
                )
                continue
            if not isinstance(proposal.phase, str) or not proposal.phase.strip():
                raise ValueError(f"Rule {rule!r} returned an invalid phase.")
            phase = normalize_phase_label(proposal.phase)
            if phase is None:
                raise ValueError(f"Rule {rule!r} returned an invalid phase.")
            if custom and phase in self.unresolved_labels:
                self.record(
                    idx,
                    rule,
                    proposal,
                    "abstained",
                    "unresolved proposal preserves existing evidence",
                    context,
                )
                continue
            winner = self.winners.get(idx)
            rejection = None
            if idx in context.authoritative_rows:
                rejection = (
                    "higher-priority phase strategy supplies authoritative context"
                )
            elif rule != "metadata" and pd.notna(
                self.rows.loc[idx].get("phase_applicability_reason")
            ):
                rejection = "phase is not applicable to this acquisition"
            elif (
                self.modality == "MR"
                and self.rows.loc[idx, "mri_perfusion_source"] == "dicom_phase_conflict"
            ):
                rejection = "conflicting structured DICOM phase evidence"
            elif phase == "OTHER" and winner:
                rejection = "unresolved evidence cannot replace a resolved phase"
            elif winner and self.precedence[rule] >= self.precedence[winner]:
                rejection = f"lower precedence than {winner}"
            if rejection:
                self.record(idx, rule, proposal, "rejected", rejection, context)
                continue
            self.rows.loc[idx, self.label_column] = phase
            self.rows.loc[idx, f"{self.prefix}_reason"] = proposal.reason
            confidence_column = f"{self.prefix}_confidence"
            if custom and not isinstance(proposal.confidence, str):
                # Pandas 3 infers string dtype for built-in confidence labels.
                # Numeric extension confidence must coexist with those labels.
                self.rows[confidence_column] = self.rows[confidence_column].astype(
                    object
                )
            self.rows.loc[idx, confidence_column] = proposal.confidence
            if self.modality == "MR":
                self.rows.loc[idx, "mri_perfusion_source"] = proposal.source or rule
                if proposal.order_source is not None:
                    self.rows.loc[idx, "mri_dynamic_order_source"] = (
                        proposal.order_source
                    )
            if phase != "OTHER":
                self.winners[idx] = rule
            accepted.append(idx)
            self.record(idx, rule, proposal, "accepted", "proposal applied", context)
        for idx in context.rows.index.difference(list(seen), sort=False):
            self.record(
                idx,
                rule,
                None,
                "abstained",
                "no proposal under the current context",
                context,
            )
        return accepted
