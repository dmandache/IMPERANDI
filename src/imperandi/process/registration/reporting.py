"""Tabular QC, error records, and persistent series-level logs."""

from collections.abc import Mapping
from typing import Any
import json
from dataclasses import asdict
import logging
from pathlib import Path

import pandas as pd

from .config import REGISTRATION_STAGES

ANATOMICAL_QC_FIELDS = [
    "registration_organ_qc",
    "registration_completeness",
    "registration_organ_volume_mm3",
    "registration_organ_bbox_index",
    "registration_boundary_contact",
    "registration_largest_component_fraction",
    "registration_organ_volume_ratio",
    "registration_confidence",
    "registration_dice_full",
    "registration_dice_common_fov",
    "registration_common_fov_fraction",
    "registration_consensus_contributors",
    "registration_consensus_excluded",
    "registration_consensus_exclusion_reasons",
    "registration_consensus_support_policy",
]

logger = logging.getLogger(__name__)
ERROR_COLUMNS = [
    "patient_id",
    "date",
    "visit_order",
    "visit",
    "modality",
    "registration_group_label",
    "registration_scan_label",
    "registration_scan_id",
    "stage",
    "error",
]
QC_FIELDS = [
    *ANATOMICAL_QC_FIELDS,
    "registration_skip_reason",
    "tumor_consensus_input_status",
    "registration_selected_stage",
    "registration_dice_selected",
    *[f"registration_dice_{stage}" for stage in REGISTRATION_STAGES],
    *[f"registration_tumor_dice_{stage}" for stage in REGISTRATION_STAGES],
    "registration_stage_details",
    "registration_warnings",
    "registration_started_at",
    "registration_elapsed_seconds",
]


def _value(record: Mapping[str, Any], *columns: str) -> str | None:
    for column in columns:
        value = record.get(column)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        if text:
            return text
    return None


def group_context(record: Mapping[str, Any], visit_column: str) -> dict[str, str]:
    """Return the clinical attributes used to describe one registration group."""
    patient = _value(record, "patient_id", "patient_key") or "unknown"
    modality = (_value(record, "Modality", "modality") or "unknown").upper()
    if modality == "MRI":
        modality = "MR"
    return {
        "patient_id": patient,
        "date": _value(record, "date", "visit_date", "StudyDate") or "unknown",
        "visit_order": _value(record, "visit_order") or "unknown",
        "visit": _value(record, visit_column) or "unknown",
        "modality": modality,
    }


def group_label(record: Mapping[str, Any], visit_column: str) -> str:
    context = group_context(record, visit_column)
    return ", ".join(f"{key}={value}" for key, value in context.items())


def scan_label(record: Mapping[str, Any]) -> str:
    """Describe a scan within its group without repeating group attributes."""
    position = _value(record, "registration_series_number") or "?"
    total = _value(record, "registration_group_size") or "?"
    fields = {
        "series": f"{position}/{total}",
        "phase": _value(record, "phase"),
        "sequence": _value(record, "mri_sequence"),
    }
    detail = ", ".join(f"{key}={value}" for key, value in fields.items() if value)
    return detail


def _decode_stages(value):
    if not isinstance(value, str) or not value:
        return {}
    decoded = json.loads(value)
    return {stage: decoded[stage] for stage in REGISTRATION_STAGES if stage in decoded}


def _optional(value):
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


def _stage_summary(stages):
    parts = []
    for stage in REGISTRATION_STAGES:
        detail = stages.get(stage, {})
        dice = _optional(detail.get("dice"))
        dice_text = "n/a" if dice is None else f"{float(dice):.4f}"
        parts.append(f"{stage}={dice_text} ({detail.get('status', 'not_run')})")
    return ", ".join(parts)


def build_error_record(row, config, *, stage, error):
    """Build the shared error-table row for any registration failure."""
    return {
        **group_context(row, config.visit_column),
        "registration_group_label": row.get("registration_group_label"),
        "registration_scan_label": row.get("registration_scan_label"),
        "registration_scan_id": row.get("registration_scan_id"),
        "stage": stage,
        "error": str(error),
    }


def record_stages(df, index, report):
    for key in ("dice_full", "dice_common_fov", "common_fov_fraction"):
        df.at[index, f"registration_{key}"] = report.get("overlap", {}).get(key)
    df.at[index, "registration_organ_volume_ratio"] = report.get("organ_volume_ratio")
    df.at[index, "registration_confidence"] = report.get("confidence")
    df.at[index, "registration_selected_stage"] = report.get("stage")
    df.at[index, "registration_dice_selected"] = report.get("dice_after")
    stages = report.get("stages", {})
    for stage in REGISTRATION_STAGES:
        df.at[index, f"registration_dice_{stage}"] = stages.get(stage, {}).get("dice")
    df.at[index, "registration_stage_details"] = json.dumps(stages, sort_keys=True)
    df.at[index, "registration_warnings"] = json.dumps(report.get("warnings", []))


def record_organ_qc(df, index, quality):
    """Keep anatomical QC serialization consistent between tables and logs."""
    df.at[index, "registration_organ_qc"] = json.dumps(asdict(quality))
    df.at[index, "registration_completeness"] = quality.status
    df.at[index, "registration_organ_volume_mm3"] = quality.volume_mm3
    df.at[index, "registration_organ_bbox_index"] = json.dumps(quality.bbox_index)
    df.at[index, "registration_boundary_contact"] = json.dumps(quality.boundary_contact)
    df.at[index, "registration_largest_component_fraction"] = (
        quality.largest_component_fraction
    )


def build_qc(table, errors, config):
    """One row per scan, including failures and deliberately unexecuted stages."""
    columns = list(
        dict.fromkeys(
            [
                "patient_key",
                "patient_id",
                "date",
                "visit_order",
                config.visit_column,
                "study_id",
                "series_id",
                "Modality",
                "phase",
                "mri_sequence",
                "nifti_path",
                f"source_{config.organ_column}",
                f"source_{config.tumor_column}",
                config.organ_column,
                config.tumor_column,
                "registration_scan_id",
                "registration_scan_label",
                "registration_series_number",
                "registration_group_size",
                "registration_group_id",
                "registration_group_label",
                "registration_reference_id",
                "registration_reference_label",
                "registration_status",
                "consensus_status",
                *QC_FIELDS,
                "reg_organ_native_path",
                "reg_tumor_native_path",
                "registration_qc_path",
                "registration_log_path",
            ]
        )
    )
    result = table.reindex(columns=columns).copy()
    result["consensus_method"] = config.method
    messages = {}
    for row in errors.to_dict("records"):
        messages.setdefault(row["registration_scan_id"], []).append(
            f"{row['stage']}: {row['error']}"
        )
    result["errors"] = result.registration_scan_id.map(
        lambda key: " | ".join(messages.get(key, []))
    )
    stage_details = result.registration_stage_details.map(_decode_stages)
    for stage in REGISTRATION_STAGES:
        result[f"{stage}_status"] = stage_details.map(
            lambda details: details.get(stage, {}).get("status", "not_run")
        )
    return result.rename(
        columns={
            **{
                f"registration_dice_{stage}": f"dice_{stage}"
                for stage in (*REGISTRATION_STAGES, "selected")
            },
            **{
                f"registration_tumor_dice_{stage}": f"tumor_dice_{stage}"
                for stage in REGISTRATION_STAGES
            },
            "registration_selected_stage": "selected_stage",
        }
    )


def publish_group_log(df, indices, errors, config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "registration.jsonl"
    df.loc[indices, "registration_log_path"] = str(log_path.resolve())
    qc = build_qc(df.loc[indices], pd.DataFrame(errors), config)
    rows = qc.to_dict("records")
    first = rows[0]
    reference_label = first["registration_reference_label"]
    events = [
        {
            "event": "group_context",
            **group_context(first, config.visit_column),
            "series_count": len(rows),
            "registration_reference_label": _optional(reference_label),
            "consensus_method": first["consensus_method"],
        }
    ]
    for row in rows:
        context = {
            key: row[key]
            for key in [
                "registration_scan_label",
                "selected_stage",
                "dice_selected",
                "registration_elapsed_seconds",
            ]
        }
        context = {key: _optional(value) for key, value in context.items()}
        stages = _decode_stages(row["registration_stage_details"])
        logger.info(
            "Registration stages: %s; %s",
            context["registration_scan_label"],
            _stage_summary(stages),
        )
        events.append(
            {
                **context,
                "event": "scan_result",
                "stages": stages,
                "registration_status": row["registration_status"],
                "consensus_status": row["consensus_status"],
                "has_errors": bool(row["errors"]),
                "anatomical_qc": {
                    key: _optional(row.get(key)) for key in ANATOMICAL_QC_FIELDS
                },
                "tumor_consensus_input_status": _optional(
                    row.get("tumor_consensus_input_status")
                ),
            }
        )
    temporary = log_path.with_suffix(".tmp")
    temporary.write_text(
        "".join(json.dumps(event, default=str) + "\n" for event in events),
        encoding="utf-8",
    )
    temporary.replace(log_path)
