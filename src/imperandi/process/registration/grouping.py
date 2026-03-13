from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Iterable

import pandas as pd

from imperandi.process import _registration_common as reg_common


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupingKeys:
    patient: str = "patient_key"
    visit: str = "visit_order"
    phase: str = "phase"


@dataclass(frozen=True)
class IntraPairTask:
    patient_key: str
    task_kind: str
    reference_source_idx: int
    moving_source_idx: int
    reference_row: dict[str, Any]
    moving_row: dict[str, Any]


def _sort_frame_for_reference(df: pd.DataFrame, *, keys: GroupingKeys) -> pd.DataFrame:
    working = df.copy()
    working["_phase_norm"] = working.apply(
        lambda row: reg_common.infer_phase_from_row(row.to_dict()),
        axis=1,
    )
    working["_portal_rank"] = working["_phase_norm"].apply(
        lambda value: 0 if value == "portal" else 1
    )
    if keys.visit in working.columns:
        working["_sort_visit"] = reg_common.to_numeric_sort_series(
            working[keys.visit],
            missing=float("inf"),
        )
    else:
        working["_sort_visit"] = float("inf")
    if "date" in working.columns:
        dates = pd.to_datetime(working["date"], errors="coerce")
        working["_sort_date_missing"] = dates.isna().astype(int)
        working["_sort_date"] = dates
    else:
        working["_sort_date_missing"] = 1
        working["_sort_date"] = pd.NaT
    if "followup_months" in working.columns:
        working["_sort_followup"] = reg_common.to_numeric_sort_series(
            working["followup_months"],
            missing=float("inf"),
        )
    else:
        working["_sort_followup"] = float("inf")
    working["_sort_source_idx"] = reg_common.to_numeric_sort_series(
        working["_source_idx"],
        missing=float("inf"),
    )
    return working.sort_values(
        by=[
            "_portal_rank",
            "_sort_visit",
            "_sort_date_missing",
            "_sort_date",
            "_sort_followup",
            "_sort_source_idx",
        ],
        kind="stable",
    )


def _valid_rows(
    df: pd.DataFrame,
    *,
    pending_source_indices: set[int] | None,
    mask_column: str,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    working = df.copy()
    working["_has_required_inputs"] = working.apply(
        lambda row: (
            pending_source_indices is None
            or int(row["_source_idx"]) in pending_source_indices
        )
        and reg_common._is_existing_path(row.get("nifti_path"))
        and reg_common._is_existing_path(row.get(mask_column)),
        axis=1,
    )
    return working[working["_has_required_inputs"]].copy()


def _choose_reference_row(group_df: pd.DataFrame, *, keys: GroupingKeys) -> dict[str, Any] | None:
    if group_df.empty:
        return None
    ordered = _sort_frame_for_reference(group_df, keys=keys)
    if ordered.empty:
        return None
    reference = ordered.iloc[0].to_dict()
    logger.debug(
        (
            "Selected intra reference row source_idx=%s patient=%s visit=%s "
            "phase=%s from %d candidate rows."
        ),
        int(reference.get("_source_idx", -1)),
        reference.get(keys.patient),
        reference.get(keys.visit),
        reg_common.infer_phase_from_row(reference),
        len(ordered),
    )
    return reference


def _infer_auto_mode(df: pd.DataFrame, *, keys: GroupingKeys) -> str:
    if keys.visit in df.columns:
        visits = (
            df[keys.visit]
            .dropna()
            .astype(str)
            .map(str.strip)
            .replace("", pd.NA)
            .dropna()
            .nunique()
        )
        if int(visits) > 1:
            logger.debug(
                "Auto intra mode resolved to longitudinal from %d distinct visits.",
                int(visits),
            )
            return "longitudinal"
        logger.debug(
            "Auto intra mode resolved to multiphasic from %d distinct visits.",
            int(visits),
        )
    else:
        logger.debug(
            "Auto intra mode resolved to multiphasic because visit column '%s' is absent.",
            keys.visit,
        )
    return "multiphasic"


def _build_multiphasic_tasks(
    patient_df: pd.DataFrame,
    *,
    keys: GroupingKeys,
    pending_source_indices: set[int],
    mask_column: str,
) -> tuple[list[IntraPairTask], set[int]]:
    tasks: list[IntraPairTask] = []
    anchor_indices: set[int] = set()
    if keys.visit in patient_df.columns:
        grouped: Iterable[tuple[Any, pd.DataFrame]] = patient_df.groupby(
            keys.visit, sort=False, dropna=False
        )
    else:
        grouped = [("single_visit", patient_df)]
    for _, visit_df in grouped:
        visit_value = visit_df.iloc[0].get(keys.visit) if not visit_df.empty else None
        valid_visit = _valid_rows(
            visit_df,
            pending_source_indices=None,
            mask_column=mask_column,
        )
        ref = _choose_reference_row(valid_visit, keys=keys)
        if ref is None:
            logger.debug(
                "Skipping multiphasic visit=%s because no valid reference row was found.",
                visit_value,
            )
            continue
        ref_idx = int(ref["_source_idx"])
        anchor_indices.add(ref_idx)
        for _, row in valid_visit.iterrows():
            moving = row.to_dict()
            moving_idx = int(moving["_source_idx"])
            if moving_idx == ref_idx or moving_idx not in pending_source_indices:
                continue
            tasks.append(
                IntraPairTask(
                    patient_key=str(ref.get(keys.patient, "")),
                    task_kind="multiphasic",
                    reference_source_idx=ref_idx,
                    moving_source_idx=moving_idx,
                    reference_row=dict(ref),
                    moving_row=moving,
                )
            )
    return tasks, anchor_indices


def _build_longitudinal_tasks(
    patient_df: pd.DataFrame,
    *,
    keys: GroupingKeys,
    pending_source_indices: set[int],
    mask_column: str,
) -> tuple[list[IntraPairTask], set[int]]:
    valid_df = _valid_rows(
        patient_df,
        pending_source_indices=None,
        mask_column=mask_column,
    )
    ref = _choose_reference_row(valid_df, keys=keys)
    if ref is None:
        logger.debug("No valid longitudinal reference row found for patient group.")
        return [], set()
    ref_idx = int(ref["_source_idx"])
    tasks: list[IntraPairTask] = []
    for _, row in valid_df.iterrows():
        moving = row.to_dict()
        moving_idx = int(moving["_source_idx"])
        if moving_idx == ref_idx or moving_idx not in pending_source_indices:
            continue
        tasks.append(
            IntraPairTask(
                patient_key=str(ref.get(keys.patient, "")),
                task_kind="longitudinal",
                reference_source_idx=ref_idx,
                moving_source_idx=moving_idx,
                reference_row=dict(ref),
                moving_row=moving,
            )
        )
    return tasks, {ref_idx}


def build_intra_patient_tasks(
    patient_df: pd.DataFrame,
    *,
    keys: GroupingKeys | None = None,
    pending_source_indices: set[int],
    mask_column: str,
    mode: str = "auto",
) -> tuple[list[IntraPairTask], set[int], str]:
    """Build intra-patient registration tasks and anchor source indices.

    Returns:
        (tasks, anchor_source_indices, resolved_mode)
    """

    if keys is None:
        keys = GroupingKeys()
    resolved_mode = str(mode or "auto").strip().lower()
    if resolved_mode not in {"auto", "multiphasic", "longitudinal"}:
        resolved_mode = "auto"
    if resolved_mode == "auto":
        resolved_mode = _infer_auto_mode(patient_df, keys=keys)

    if resolved_mode == "multiphasic":
        tasks, anchors = _build_multiphasic_tasks(
            patient_df,
            keys=keys,
            pending_source_indices=pending_source_indices,
            mask_column=mask_column,
        )
        return tasks, anchors, resolved_mode

    tasks, anchors = _build_longitudinal_tasks(
        patient_df,
        keys=keys,
        pending_source_indices=pending_source_indices,
        mask_column=mask_column,
    )
    return tasks, anchors, resolved_mode
