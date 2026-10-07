"""Tabular QC and persistent, scan-addressable stage logs."""

import json
import logging
from pathlib import Path

import pandas as pd

from imperandi.utils.run_state import atomic_write_csv

logger = logging.getLogger(__name__)
STAGES = ("baseline", "pca", "rigid", "affine")
QC_FIELDS = [
    "registration_selected_stage",
    "registration_dice_selected",
    *[f"registration_dice_{stage}" for stage in STAGES],
    "registration_stage_details",
    "registration_warnings",
    "registration_started_at",
    "registration_elapsed_seconds",
]


def record_stages(df, index, report):
    df.at[index, "registration_selected_stage"] = report.get("stage")
    df.at[index, "registration_dice_selected"] = report.get("dice_after")
    stages = report.get("stages", {})
    for stage in STAGES:
        df.at[index, f"registration_dice_{stage}"] = stages.get(stage, {}).get("dice")
    df.at[index, "registration_stage_details"] = json.dumps(stages, sort_keys=True)
    df.at[index, "registration_warnings"] = json.dumps(report.get("warnings", []))


def build_qc(table, errors, config):
    """One row per scan, including failures and deliberately unexecuted stages."""
    columns = list(
        dict.fromkeys(
            [
                "patient_key",
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
                "registration_group_id",
                "registration_reference_id",
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
    for stage in STAGES:
        result[f"{stage}_status"] = result.registration_stage_details.map(
            lambda value: (
                json.loads(value).get(stage, {}).get("status", "not_run")
                if isinstance(value, str) and value
                else "not_run"
            )
        )
    return result.rename(
        columns={
            **{
                f"registration_dice_{stage}": f"dice_{stage}"
                for stage in (*STAGES, "selected")
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
    events = []
    for row in qc.to_dict("records"):
        context = {
            key: row[key]
            for key in [
                "registration_group_id",
                "registration_scan_id",
                "registration_reference_id",
                "nifti_path",
                "registration_started_at",
                "selected_stage",
                "dice_selected",
                "consensus_method",
                "registration_elapsed_seconds",
                f"source_{config.organ_column}",
                f"source_{config.tumor_column}",
                "reg_reference_to_scan_path",
                "reg_scan_to_reference_path",
                "reg_organ_native_path",
                "reg_tumor_native_path",
            ]
        }
        context = {
            key: None if pd.isna(value) else value for key, value in context.items()
        }
        details = row["registration_stage_details"]
        stages = json.loads(details) if isinstance(details, str) and details else {}
        for stage, detail in stages.items():
            events.append({**context, "event": "organ_stage", "stage": stage, **detail})
            logger.info(
                "group=%s scan=%s reference=%s stage=%s dice=%s status=%s",
                context["registration_group_id"],
                context["registration_scan_id"],
                context["registration_reference_id"],
                stage,
                detail.get("dice"),
                detail.get("status"),
            )
        events.append(
            {
                **context,
                "event": "scan_result",
                "registration_status": row["registration_status"],
                "consensus_status": row["consensus_status"],
                "errors": row["errors"],
            }
        )
    temporary = log_path.with_suffix(".tmp")
    temporary.write_text(
        "".join(json.dumps(event, default=str) + "\n" for event in events),
        encoding="utf-8",
    )
    temporary.replace(log_path)
