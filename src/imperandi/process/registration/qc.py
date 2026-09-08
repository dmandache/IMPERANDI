"""Tabular QC, error records, and persistent series-level logs."""

import json
import logging
from pathlib import Path

import pandas as pd

from imperandi.utils.run_state import atomic_write_csv
from .config import REGISTRATION_STAGES
from .labels import group_context

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
    "registration_selected_stage",
    "registration_dice_selected",
    *[f"registration_dice_{stage}" for stage in REGISTRATION_STAGES],
    "registration_stage_details",
    "registration_warnings",
    "registration_started_at",
    "registration_elapsed_seconds",
]


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
    df.at[index, "registration_selected_stage"] = report.get("stage")
    df.at[index, "registration_dice_selected"] = report.get("dice_after")
    stages = report.get("stages", {})
    for stage in REGISTRATION_STAGES:
        df.at[index, f"registration_dice_{stage}"] = stages.get(stage, {}).get("dice")
    df.at[index, "registration_stage_details"] = json.dumps(stages, sort_keys=True)
    df.at[index, "registration_warnings"] = json.dumps(report.get("warnings", []))


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
                "reg_reference_to_scan_path",
                "reg_scan_to_reference_path",
                "reg_organ_native_path",
                "reg_tumor_native_path",
                "registration_report_path",
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
            "registration_selected_stage": "selected_stage",
        }
    )


def publish_group_qc(df, indices, errors, config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    qc_path, log_path = directory / "qc.csv", directory / "registration.jsonl"
    df.loc[indices, "registration_qc_path"] = str(qc_path.resolve())
    df.loc[indices, "registration_log_path"] = str(log_path.resolve())
    qc = build_qc(df.loc[indices], pd.DataFrame(errors), config)
    atomic_write_csv(qc, qc_path, index=False)
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
            }
        )
    temporary = log_path.with_suffix(".tmp")
    temporary.write_text(
        "".join(json.dumps(event, default=str) + "\n" for event in events),
        encoding="utf-8",
    )
    temporary.replace(log_path)
