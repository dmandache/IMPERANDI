"""Cohort validation, stable identities, grouping, and reference ordering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import SUPPORTED_MODALITIES
from .labels import group_label, scan_label
from .qc import QC_FIELDS

OBSOLETE_ARTIFACT_COLUMNS = (
    "registration_report_path",
    "reg_reference_to_scan_path",
    "reg_scan_to_reference_path",
    "reg_tumor_common_path",
    "reg_tumor_coverage_common_path",
    "reg_tumor_coverage_native_path",
    "reg_tumor_probability_common_path",
    "reg_tumor_probability_native_path",
    "reg_nifti_path",
    "reg_organ_path",
)


def stable_id(value) -> str:
    """Return a filesystem-safe, deterministic identifier for structured values."""
    payload = json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def normalize_label(value) -> str:
    return "" if pd.isna(value) else str(value).strip().upper()


def normalize_modalities(table: pd.DataFrame) -> pd.Series:
    """Normalize supported modality labels without changing source columns."""
    return table.Modality.map(normalize_label).replace({"MRI": "MR"})


def reference_rank(row, priorities) -> tuple[int, str]:
    """Rank a reference candidate using configured selectors and a stable tie-break."""
    rank = next(
        (
            index
            for index, selector in enumerate(priorities)
            if all(
                normalize_label(row.get(key)) == normalize_label(value)
                for key, value in selector.items()
            )
        ),
        len(priorities),
    )
    return rank, str(row["registration_scan_id"])


def prepare_cohort(table: pd.DataFrame, config) -> pd.DataFrame:
    """Validate identities and initialize registration output columns."""
    df = table.drop(columns=OBSOLETE_ARTIFACT_COLUMNS, errors="ignore").reset_index(
        drop=True
    )

    # Source columns remain authoritative across repeated registration runs.
    for column in (config.organ_column, config.tumor_column):
        source = f"source_{column}"
        if source not in df:
            if column == config.organ_column and column not in df:
                continue
            df[source] = df[column] if column in df else None
        df[column] = df[source].astype(object)

    required = [
        "patient_key",
        config.visit_column,
        "Modality",
        "nifti_path",
        config.organ_column,
    ]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing registration columns: {sorted(missing)}")
    for column in ["patient_key", config.visit_column, "Modality", "nifti_path"]:
        if df[column].isna().any() or df[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing registration identity: {column}")

    modalities = normalize_modalities(df)
    if not modalities.isin(SUPPORTED_MODALITIES).all():
        raise ValueError("Registration supports CT and MR only")

    identities = [
        [
            str(row.patient_key),
            str(row[config.visit_column]),
            modalities[index],
            str(Path(row.nifti_path).expanduser().resolve()),
        ]
        for index, row in df.iterrows()
    ]
    scan_ids = [stable_id(identity) for identity in identities]
    if len(scan_ids) != len(set(scan_ids)):
        raise ValueError("Duplicate scan identity in registration input")

    df["registration_scan_id"] = scan_ids
    df["registration_group_id"] = [
        stable_id(
            (str(row.patient_key), str(row[config.visit_column]), modalities[index])
        )
        for index, row in df.iterrows()
    ]
    df["registration_group_label"] = [
        group_label(row, config.visit_column) for _, row in df.iterrows()
    ]
    df["registration_series_number"] = None
    df["registration_group_size"] = None
    for _, group in df.groupby("registration_group_id", sort=False):
        modality = modalities.loc[group.index[0]]
        priorities = config.reference_priority.get(modality, [])
        ordered = sorted(
            group.index,
            key=lambda index: reference_rank(df.loc[index], priorities),
        )
        for position, index in enumerate(ordered, start=1):
            df.at[index, "registration_series_number"] = position
            df.at[index, "registration_group_size"] = len(group)
    df["registration_scan_label"] = [scan_label(row) for _, row in df.iterrows()]

    derived_columns = [
        "registration_qc_path",
        "registration_log_path",
        *QC_FIELDS,
        "registration_reference_id",
        "registration_reference_label",
        "registration_status",
        "consensus_status",
        "reg_tumor_native_path",
        "reg_organ_native_path",
    ]
    for column in derived_columns:
        df[column] = pd.Series([None] * len(df), dtype=object)

    return df
