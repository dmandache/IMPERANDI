"""Explicit, auditable patient identity resolution."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from dataclasses import dataclass

import pandas as pd

from imperandi.config.models import IdentityConfig
from imperandi.io.tables import read_table


@dataclass
class IdentityResult:
    cohort: pd.DataFrame
    sensitive: pd.DataFrame
    qc_flags: pd.DataFrame


def _normalize(value, config) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    if config.strip:
        text = text.strip()
    if config.collapse_whitespace:
        text = re.sub(r"\s+", " ", text)
    if config.case == "upper":
        text = text.upper()
    elif config.case == "lower":
        text = text.lower()
    return text


def _first_available(row: pd.Series, columns: list[str], normalizer) -> tuple[str, str]:
    for column in columns:
        if column in row.index:
            value = normalizer(row.get(column))
            if value:
                return value, column
    return "", ""


def _source_identity_frame(df: pd.DataFrame, config: IdentityConfig) -> pd.DataFrame:
    def norm(value):
        return _normalize(value, config.normalization)

    records = []
    for idx, row in df.iterrows():
        raw_id, source_column = _first_available(
            row, config.source.patient_id_columns, norm
        )
        method = "dicom"
        confidence = "high"
        if not raw_id:
            raw_id, source_column = _first_available(
                row, config.source.fallback.columns, norm
            )
            method = "fallback"
            confidence = "low"

        namespaces = []
        for column in config.source.namespace_columns:
            if column in row.index:
                value = norm(row.get(column))
                if value:
                    namespaces.append(f"{column}={value}")
        namespace = "|".join(namespaces)
        source_key = (
            "|".join(part for part in [namespace, raw_id] if part) if raw_id else ""
        )
        records.append(
            {
                "_row_index": idx,
                "dicom_patient_id": raw_id or pd.NA,
                "source_patient_key": source_key or pd.NA,
                "patient_id_source_column": source_column or pd.NA,
                "patient_id_method": method if raw_id else "missing",
                "identity_confidence": confidence if raw_id else "none",
            }
        )
    columns = [
        "_row_index",
        "dicom_patient_id",
        "source_patient_key",
        "patient_id_source_column",
        "patient_id_method",
        "identity_confidence",
    ]
    return pd.DataFrame(records, columns=columns).set_index("_row_index")


def _hmac_patient_id(source_key: str, config) -> str:
    secret = os.getenv(config.secret_env)
    if not secret:
        raise RuntimeError(
            f"Environment variable {config.secret_env!r} is required for HMAC "
            "patient identity generation"
        )
    message = f"{config.namespace}|{source_key}".encode()
    digest = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return f"{config.prefix}{digest[: config.length]}"


def _normalized_crosswalk(config: IdentityConfig) -> pd.DataFrame:
    canonical = config.canonical
    assert canonical.crosswalk is not None
    crosswalk = read_table(canonical.crosswalk)
    required = {*canonical.crosswalk_keys, canonical.crosswalk_value}
    missing = required - set(crosswalk.columns)
    if missing:
        raise ValueError(
            f"Identity crosswalk is missing required columns: {sorted(missing)}"
        )

    def norm(value):
        return _normalize(value, config.normalization)

    for column in canonical.crosswalk_keys:
        crosswalk[column] = crosswalk[column].apply(norm)
    incomplete_keys = crosswalk[canonical.crosswalk_keys].eq("").any(axis=1)
    if incomplete_keys.any():
        raise ValueError(
            "Identity crosswalk contains empty key values in "
            f"{int(incomplete_keys.sum())} row(s)"
        )

    def canonical_value(value):
        if value is None or pd.isna(value):
            return pd.NA
        text = str(value).strip()
        return text or pd.NA

    value_column = canonical.crosswalk_value
    crosswalk[value_column] = crosswalk[value_column].apply(canonical_value)
    if crosswalk[value_column].isna().any():
        raise ValueError("Identity crosswalk contains empty canonical patient IDs")
    duplicates = crosswalk.groupby(canonical.crosswalk_keys, dropna=False)[
        value_column
    ].nunique()
    if (duplicates > 1).any():
        raise ValueError(
            "Identity crosswalk maps a source identity to multiple patients"
        )
    return crosswalk.drop_duplicates(canonical.crosswalk_keys)


def validate_identity_crosswalk(config: IdentityConfig) -> None:
    """Validate crosswalk keys and canonical IDs without resolving a cohort."""
    if config.canonical.crosswalk is not None:
        _normalized_crosswalk(config)


def _apply_crosswalk(
    out: pd.DataFrame, identity: pd.DataFrame, config: IdentityConfig
) -> pd.Series:
    canonical = config.canonical
    crosswalk = _normalized_crosswalk(config)
    lookup = out.copy()
    lookup["dicom_patient_id"] = identity["dicom_patient_id"]

    def norm(value):
        return _normalize(value, config.normalization)

    for column in canonical.crosswalk_keys:
        if column not in lookup.columns:
            lookup[column] = ""
        lookup[column] = lookup[column].apply(norm)
    lookup["_identity_order"] = range(len(lookup))
    merged = lookup.merge(
        crosswalk[[*canonical.crosswalk_keys, canonical.crosswalk_value]],
        on=canonical.crosswalk_keys,
        how="left",
        sort=False,
    ).sort_values("_identity_order")
    merged.index = out.index
    return merged[canonical.crosswalk_value]


def _validate_identity_mapping(
    identity: pd.DataFrame, config: IdentityConfig
) -> pd.DataFrame:
    flags = []
    valid = identity.dropna(subset=["source_patient_key", "patient_id"])
    source_counts = valid.groupby("source_patient_key")["patient_id"].nunique()
    for source_key in source_counts[source_counts > 1].index:
        flags.append(
            {
                "qc_code": "IDENTITY_SOURCE_COLLISION",
                "source_patient_key": source_key,
                "severity": (
                    "error"
                    if config.validation.source_collision == "error"
                    else "warning"
                ),
            }
        )
    multiple_source_policy = config.validation.multiple_source_ids_per_patient
    if multiple_source_policy != "allow":
        canonical_counts = valid.groupby("patient_id")["source_patient_key"].nunique()
        for patient_id in canonical_counts[canonical_counts > 1].index:
            flags.append(
                {
                    "qc_code": "IDENTITY_MULTIPLE_SOURCE_IDS",
                    "patient_id": patient_id,
                    "severity": (
                        "error" if multiple_source_policy == "error" else "warning"
                    ),
                }
            )
    result = pd.DataFrame(flags)
    if (
        config.validation.source_collision == "error"
        and not result.empty
        and result["qc_code"].eq("IDENTITY_SOURCE_COLLISION").any()
    ):
        raise ValueError("A source identity resolved to multiple canonical patients")
    if (
        multiple_source_policy == "error"
        and not result.empty
        and result["qc_code"].eq("IDENTITY_MULTIPLE_SOURCE_IDS").any()
    ):
        raise ValueError(
            "Multiple source identities resolved to one canonical patient while "
            "identity policy is 'error'"
        )
    return result


def resolve_patient_identities(
    df: pd.DataFrame, config: IdentityConfig
) -> IdentityResult:
    """Resolve canonical IDs and return separately controlled sensitive fields."""
    out = df.copy()
    identity = _source_identity_frame(out, config)
    canonical = config.canonical

    if canonical.strategy == "source":
        identity["patient_id"] = identity["source_patient_key"]
    else:
        identity["patient_id"] = pd.NA

    if "crosswalk" in canonical.strategy:
        mapped = _apply_crosswalk(out, identity, config)
        identity["patient_id"] = mapped.values
        mapped_mask = identity["patient_id"].notna()
        identity.loc[mapped_mask, "patient_id_method"] = "crosswalk"
        identity.loc[mapped_mask, "identity_confidence"] = "high"

    if "hmac" in canonical.strategy:
        missing = identity["patient_id"].isna() & identity["source_patient_key"].notna()
        assert canonical.hmac is not None
        identity.loc[missing, "patient_id"] = identity.loc[
            missing, "source_patient_key"
        ].apply(lambda value: _hmac_patient_id(str(value), canonical.hmac))
        identity.loc[missing, "patient_id_method"] = "hmac"
        identity.loc[missing, "identity_confidence"] = "high"

    identity["patient_id"] = identity["patient_id"].astype("string")

    missing_identity = identity["patient_id"].isna()
    policy = config.source.fallback.on_missing
    if missing_identity.any() and policy == "error":
        raise ValueError(
            "Could not resolve patient identity for "
            f"{int(missing_identity.sum())} row(s)"
        )

    identity["identity_algorithm_version"] = 1
    out["patient_id"] = identity["patient_id"]
    out["patient_id_method"] = identity["patient_id_method"]
    out["identity_confidence"] = identity["identity_confidence"]
    out["identity_algorithm_version"] = identity["identity_algorithm_version"]

    sensitive_policy = config.sensitive_fields.persist_raw_identifiers
    if sensitive_policy == "never":
        sensitive_cols = ["patient_id"]
    else:
        sensitive_cols = [
            column
            for column in [
                "patient_id",
                "dicom_patient_id",
                "source_patient_key",
                "patient_id_source_column",
            ]
            if column in identity.columns
        ]
    sensitive = identity[sensitive_cols].drop_duplicates().reset_index(drop=True)

    if sensitive_policy == "cohort":
        out["dicom_patient_id"] = identity["dicom_patient_id"]
        out["source_patient_key"] = identity["source_patient_key"]

    raw_columns = {"patient_key", "_patient_key_raw"}
    if sensitive_policy != "cohort":
        raw_columns.update(config.source.patient_id_columns)
        raw_columns.update(config.source.fallback.columns)
        raw_columns.discard("patient_id")
    # Legacy parser aliases are never public v2 cohort fields. Under `cohort`,
    # the equivalent raw values use the explicit names above.
    out = out.drop(columns=list(raw_columns), errors="ignore")

    qc_flags = _validate_identity_mapping(identity, config)
    if missing_identity.any():
        missing_flags = pd.DataFrame(
            {
                "source_row": identity.index[missing_identity],
                "qc_code": "IDENTITY_MISSING",
                "severity": "error",
            }
        )
        qc_flags = pd.concat([qc_flags, missing_flags], ignore_index=True, sort=False)
    return IdentityResult(cohort=out, sensitive=sensitive, qc_flags=qc_flags)
