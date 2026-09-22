"""CT metadata curation and diagnostic candidate selection.

This module deliberately contains CT-specific clinical/technical heuristics.
`imperandi.ingest.clean` should orchestrate this module, not duplicate these rules.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from imperandi.curation.common import (
    build_series_text,
    first_text_column_value,
    get_exam_group_cols,
    norm_label,
    safe_float,
    safe_str,
    stable_text,
)
from imperandi.curation.phase import apply_phase_curation, validate_phase_curation
from imperandi.curation.rules import (
    RX_IMAGE_ORIGINAL,
    RX_IMAGE_PRIMARY,
    match_phase,
    match_plane,
)
from . import rules


def detect_ct_phase(row: pd.Series) -> tuple[str, str, str]:
    match = first_text_column_value(
        row, lambda text: match_phase(text, rules.PHASE_RULES)
    )
    if match is not None:
        label, description = match
        return label, f"matched CT {description} keyword", "high"
    return "OTHER", "no CT phase keyword matched", "low"


def detect_ct_features(row: pd.Series) -> dict:
    text = build_series_text(row)
    image_type = safe_str(row.get("ImageType"))
    plane = first_text_column_value(row, match_plane)
    rows = safe_float(row.get("Rows"))
    cols = safe_float(row.get("Columns"))
    n_slices = safe_float(
        row.get(
            "n_rows_in_volume",
            row.get(
                "n_sop_instances_in_volume",
                row.get("n_rows_in_series", row.get("n_files", np.nan)),
            ),
        )
    )

    return {
        "is_localizer": bool(re.search(rules.RX_CT_LOCALIZER, text)),
        "is_axial": plane == "AXIAL"
        or (plane is None and pd.notna(rows) and pd.notna(cols) and rows == cols),
        "is_original": bool(re.search(RX_IMAGE_ORIGINAL, image_type))
        and bool(re.search(RX_IMAGE_PRIMARY, image_type)),
        "is_derived_low_value": bool(re.search(rules.RX_CT_DERIVED_LOW_VALUE, text)),
        "rows": rows,
        "cols": cols,
        "n_slices": n_slices,
        "slice_thickness": safe_float(row.get("SliceThickness")),
    }


def score_ct(row: pd.Series) -> float:
    phase = norm_label(row.get("phase", row.get("ct_phase")))
    f = detect_ct_features(row)

    score = float(rules.CT_PHASE_PRIORITY.get(phase, 0))
    score += 40 if f["is_axial"] else 0
    score += 30 if f["is_original"] else 0

    if pd.notna(f["rows"]) and pd.notna(f["cols"]):
        score += 20 if f["rows"] == 512 and f["cols"] == 512 else -10

    if pd.notna(f["n_slices"]):
        if f["n_slices"] >= 80:
            score += 20
        elif f["n_slices"] < 20:
            score -= 50

    if pd.notna(f["slice_thickness"]):
        if f["slice_thickness"] <= 3:
            score += 10
        elif f["slice_thickness"] > 7:
            score -= 10

    score -= 500 if f["is_localizer"] else 0
    score -= 200 if f["is_derived_low_value"] else 0
    return float(score)


def annotate_ct(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.date

    result = out.apply(detect_ct_phase, axis=1)
    out[["ct_phase", "ct_phase_reason", "ct_phase_confidence"]] = pd.DataFrame(
        result.tolist(), index=out.index
    )
    out["rule_phase"] = out["ct_phase"]
    out["rule_phase_reason"] = out["ct_phase_reason"]
    out["rule_phase_confidence"] = out["ct_phase_confidence"]
    out["ct_selection_score"] = out.apply(score_ct, axis=1)
    out["selection_slot"] = out["ct_phase"].map(
        lambda x: f"CT_{norm_label(x)}" if norm_label(x) != "OTHER" else "CT_OTHER"
    )
    out["selection_score"] = out["ct_selection_score"]
    out["selection_modality"] = "CT"
    return out


def select_ct_per_exam(
    curated: pd.DataFrame,
    patient_col: str = "patient_key",
    study_col: str | None = "study_id",
    date_col: str = "date",
    exam_group_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = curated.copy()
    exam_cols = get_exam_group_cols(
        data, patient_col, study_col, date_col, exam_group_columns
    )
    candidates = data[data["selection_score"].fillna(-9999) > -500].copy()

    if candidates.empty:
        return candidates, pd.DataFrame(columns=[*exam_cols])

    candidates["_row_order"] = np.arange(len(candidates))
    exam_key_cols = []
    for col in exam_cols:
        key_col = f"_exam_key_{col}"
        candidates[key_col] = candidates[col].apply(stable_text)
        exam_key_cols.append(key_col)

    sort_cols = [*exam_key_cols, "selection_slot", "selection_score", "_row_order"]
    selected_long = (
        candidates.sort_values(
            sort_cols,
            ascending=[True] * len(exam_key_cols) + [True, False, True],
            na_position="last",
        )
        .groupby([*exam_key_cols, "selection_slot"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    def _display(row: pd.Series) -> str:
        desc = row.get("SeriesDescription", "")
        if safe_str(desc) == "":
            desc = row.get("ProtocolName", build_series_text(row))
        return f"{desc} [score={row.get('selection_score'):.1f}]"

    selected_long["selected_candidate"] = selected_long.apply(_display, axis=1)
    exam_lookup = selected_long[[*exam_key_cols, *exam_cols]].drop_duplicates(
        exam_key_cols
    )
    selected_wide = (
        selected_long[[*exam_key_cols, "selection_slot", "selected_candidate"]]
        .pivot_table(
            index=exam_key_cols,
            columns="selection_slot",
            values="selected_candidate",
            aggfunc="first",
        )
        .reset_index()
    )
    selected_wide.columns.name = None
    selected_wide = exam_lookup.merge(selected_wide, on=exam_key_cols, how="right")
    selected_wide = selected_wide.drop(columns=exam_key_cols, errors="ignore")
    selected_long = selected_long.drop(columns=exam_key_cols, errors="ignore")
    return selected_long, selected_wide


def curate_ct(
    df: pd.DataFrame,
    patient_col: str = "patient_key",
    study_col: str | None = "study_id",
    date_col: str = "date",
    phase_curation: dict | None = None,
) -> dict[str, pd.DataFrame]:
    curated = annotate_ct(df, date_col=date_col)
    curated = apply_phase_curation(curated, phase_curation)
    curated["ct_selection_score"] = curated.apply(score_ct, axis=1)
    curated["selection_slot"] = curated["phase"].map(
        lambda x: f"CT_{norm_label(x)}" if norm_label(x) != "OTHER" else "CT_OTHER"
    )
    curated["selection_score"] = curated["ct_selection_score"]
    selected_long, selected_wide = select_ct_per_exam(
        curated,
        patient_col=patient_col,
        study_col=study_col,
        date_col=date_col,
        exam_group_columns=validate_phase_curation(phase_curation)[
            "best_candidate_group_columns"
        ],
    )
    return {
        "curated": curated,
        "candidates": curated[curated["selection_score"].fillna(-9999) > 0].copy(),
        "selected_long": selected_long,
        "selected_wide": selected_wide,
    }
