"""Modality-specific metadata curation router."""

from __future__ import annotations

import pandas as pd

from imperandi.curation.ct import curate_ct
from imperandi.curation.mri import curate_mri


def split_by_modality(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "Modality" not in df.columns:
        return df.copy(), pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)

    modality = df["Modality"].fillna("").astype(str).str.upper()
    ct = df[modality.eq("CT")].copy()
    mr = df[modality.isin(["MR", "MRI"])].copy()
    other = df[~modality.isin(["CT", "MR", "MRI"])].copy()
    return ct, mr, other


def curate_by_modality(
    df: pd.DataFrame,
    patient_col: str = "patient_key",
    study_col: str | None = "study_id",
    date_col: str = "date",
) -> dict[str, object]:
    ct_df, mri_df, other_df = split_by_modality(df)

    ct_results = (
        curate_ct(ct_df, patient_col=patient_col, study_col=study_col, date_col=date_col)
        if not ct_df.empty
        else None
    )
    mri_results = (
        curate_mri(mri_df, patient_col=patient_col, study_col=study_col, date_col=date_col)
        if not mri_df.empty
        else None
    )

    curated_parts = []
    selected_parts = []

    if ct_results is not None:
        curated_parts.append(ct_results["curated"].assign(curation_modality="CT"))
        selected_parts.append(ct_results["selected_long"].assign(curation_modality="CT"))

    if mri_results is not None:
        curated_parts.append(mri_results["curated"].assign(curation_modality="MR"))
        selected_parts.append(mri_results["selected_long"].assign(curation_modality="MR"))

    curated_all = pd.concat(curated_parts, ignore_index=True, sort=False) if curated_parts else pd.DataFrame()
    selected_long_all = pd.concat(selected_parts, ignore_index=True, sort=False) if selected_parts else pd.DataFrame()

    return {
        "ct": ct_results,
        "mri": mri_results,
        "other": other_df,
        "curated_all": curated_all,
        "selected_long_all": selected_long_all,
    }


__all__ = ["curate_by_modality", "split_by_modality", "curate_ct", "curate_mri"]
