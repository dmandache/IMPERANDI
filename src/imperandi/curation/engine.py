"""Public object API over the existing modality-specific curation entry points."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .policy import CurationPolicy


@dataclass(frozen=True)
class CurationResult:
    curated: pd.DataFrame
    selected_long: pd.DataFrame
    trace: pd.DataFrame
    by_modality: dict
    policy_fingerprint: str

    def explain(self, volume_id=None, *, row_id=None) -> pd.DataFrame:
        """Return the decision trace for one input row.

        row_id is the zero-based input position. Use it to disambiguate repeated
        volume IDs. The trace contains proposals and abstentions, not invented
        counterfactual labels for rules that did not propose a decision.
        """
        if (volume_id is None) == (row_id is None):
            raise ValueError("Supply exactly one of volume_id or row_id.")
        if self.trace.empty:
            raise KeyError("No curation decisions are available.")
        mask = (
            self.trace.row_id.eq(row_id)
            if row_id is not None
            else self.trace.volume_id.eq(volume_id)
        )
        explanation = self.trace.loc[mask]
        if explanation.empty:
            raise KeyError(
                f"No curation decisions for {row_id if row_id is not None else volume_id!r}."
            )
        if explanation.row_id.nunique() != 1:
            raise ValueError("Volume ID is ambiguous; use row_id (input position).")
        return explanation.copy().reset_index(drop=True)


@dataclass(frozen=True)
class CurationEngine:
    policy: CurationPolicy = field(default_factory=CurationPolicy.from_manifest)

    def curate(
        self,
        df: pd.DataFrame,
        *,
        patient_col="patient_key",
        study_col="study_id",
        date_col="date",
    ) -> CurationResult:
        from . import curate_by_modality

        # Internal row identities are positional and remain unique even if callers
        # concatenate frames with duplicate indices or duplicate volume IDs.
        data = df.reset_index(drop=True).copy()
        trace: list[dict] = []
        results = curate_by_modality(
            data,
            patient_col=patient_col,
            study_col=study_col,
            date_col=date_col,
            curation_policy=self.policy,
            rule_trace=trace,
        )
        for modality, key in (("CT", "ct"), ("MR", "mri")):
            if results[key] is None:
                continue
            for idx, row in results[key]["curated"].iterrows():
                trace.append(
                    {
                        "row_id": idx,
                        "volume_id": row.get("volume_id"),
                        "modality": modality,
                        "rule": "phase_resolution",
                        "phase": row.get("phase"),
                        "source": row.get("phase_source"),
                        "reason": row.get("phase_reason"),
                        "confidence": row.get("phase_confidence"),
                        "disposition": "final",
                        "detail": f"{row.get('phase_status')}: {row.get('phase_source')}",
                        "scope": "row",
                        "context_row_ids": (idx,),
                        "text_column": row.get("phase_text_column", ""),
                        "text_value": row.get("phase_text_value", ""),
                    }
                )
        return CurationResult(
            curated=results["curated_all"],
            selected_long=results["selected_long_all"],
            trace=pd.DataFrame(trace),
            by_modality=results,
            policy_fingerprint=self.policy.fingerprint,
        )
