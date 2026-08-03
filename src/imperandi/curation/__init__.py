"""Modality-specific metadata curation router."""

from __future__ import annotations

from collections.abc import Collection

import pandas as pd

from imperandi.curation.ct import curate_ct
from imperandi.curation.mri import curate_mri

SUPPORTED_CURATORS = frozenset({"builtin:liver_ct", "builtin:liver_mri"})


def _modality_label(value) -> str:
    if isinstance(value, (list, tuple, set)):
        labels = {_modality_label(v) for v in value}
        labels.discard("")
        if len(labels) == 1:
            return labels.pop()
        return "|".join(sorted(labels))
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def split_by_modality(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "Modality" not in df.columns:
        return (
            pd.DataFrame(columns=df.columns),
            pd.DataFrame(columns=df.columns),
            df.copy(),
        )

    modality = df["Modality"].apply(_modality_label)
    ct = df[modality.eq("CT")].copy()
    mr = df[modality.isin(["MR", "MRI"])].copy()
    other = df[~modality.isin(["CT", "MR", "MRI"])].copy()
    return ct, mr, other


def curate_by_modality(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    study_col: str | None = "study_id",
    date_col: str = "date",
    contextual_strategies: Collection[str] | None = None,
    curators: Collection[str] | None = None,
) -> dict[str, object]:
    ct_df, mri_df, other_df = split_by_modality(df)
    enabled_curators = SUPPORTED_CURATORS if curators is None else frozenset(curators)
    unknown_curators = enabled_curators - SUPPORTED_CURATORS
    if unknown_curators:
        raise ValueError(f"Unknown modality curators: {sorted(unknown_curators)}")

    ct_results = (
        curate_ct(
            ct_df, patient_col=patient_col, study_col=study_col, date_col=date_col
        )
        if not ct_df.empty and "builtin:liver_ct" in enabled_curators
        else (
            {"curated": ct_df.copy(), "selected_long": ct_df.iloc[0:0].copy()}
            if not ct_df.empty
            else None
        )
    )
    mri_results = (
        curate_mri(
            mri_df,
            patient_col=patient_col,
            study_col=study_col,
            date_col=date_col,
            contextual_strategies=contextual_strategies,
        )
        if not mri_df.empty and "builtin:liver_mri" in enabled_curators
        else (
            {"curated": mri_df.copy(), "selected_long": mri_df.iloc[0:0].copy()}
            if not mri_df.empty
            else None
        )
    )

    curated_parts = []
    selected_parts = []

    if ct_results is not None:
        curated_parts.append(ct_results["curated"].assign(curation_modality="CT"))
        selected_parts.append(
            ct_results["selected_long"].assign(curation_modality="CT")
        )

    if mri_results is not None:
        curated_parts.append(mri_results["curated"].assign(curation_modality="MR"))
        selected_parts.append(
            mri_results["selected_long"].assign(curation_modality="MR")
        )

    curated_all = (
        pd.concat(curated_parts, ignore_index=True, sort=False)
        if curated_parts
        else pd.DataFrame()
    )
    selected_long_all = (
        pd.concat(selected_parts, ignore_index=True, sort=False)
        if selected_parts
        else pd.DataFrame()
    )

    return {
        "ct": ct_results,
        "mri": mri_results,
        "other": other_df,
        "curated_all": curated_all,
        "selected_long_all": selected_long_all,
    }


__all__ = ["curate_by_modality", "curate_ct", "curate_mri", "split_by_modality"]
