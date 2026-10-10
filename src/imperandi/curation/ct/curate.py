"""CT metadata curation and diagnostic candidate selection.

This module deliberately contains CT-specific clinical/technical heuristics.
`imperandi.ingest.clean` should orchestrate this module, not duplicate these rules.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from imperandi.curation import common, rules as shared_rules
from imperandi.curation.common import (
    build_series_text,
    first_text_column_value,
    get_exam_group_cols,
    norm_label,
    TEXT_COLS_DEFAULT,
    safe_float,
    safe_str,
    stable_text,
    resolve_text_columns,
    has_text_columns_override,
    with_manifest_text_columns,
)
from imperandi.curation.contrast import infer_pre_post_native_acquisitions
from imperandi.curation.phase import apply_phase_curation, validate_phase_curation
from imperandi.curation.rules import (
    RX_IMAGE_ORIGINAL,
    RX_IMAGE_PRIMARY,
    match_phase,
    match_plane,
)
from . import rules


def _phase_text_column(row: pd.Series):
    return common.first_phase_text_column(row, phase_rules=rules.PHASE_RULES)


def detect_ct_phase(row: pd.Series) -> tuple[str, str, str]:
    not_applicable = row.get("phase_applicability_reason")
    if (
        not isinstance(not_applicable, str)
        or not not_applicable.strip()
        or not_applicable.strip().upper() == "DERIVED"
    ):
        not_applicable = detect_ct_phase_applicability(row)
    if not_applicable:
        return (
            "OTHER",
            f"phase not applicable: {not_applicable}",
            "not_applicable",
        )

    evidence = _phase_text_column(row)
    match = match_phase(evidence[1], rules.PHASE_RULES) if evidence else None
    if match is not None:
        label, description = match
        return label, f"matched CT {description} evidence in {evidence[0]}={evidence[1]!r}", "high"
    return "OTHER", "no CT phase keyword matched", "low"


def detect_ct_features(row: pd.Series) -> dict:
    acquisition_type = first_text_column_value(
        row,
        lambda text: (
            "LOCALIZER"
            if re.search(rules.RX_CT_LOCALIZER, text)
            else (
                "BOLUS_MONITORING"
                if re.search(rules.RX_CT_BOLUS_MONITORING, text)
                else (
                    "ACQUISITION"
                    if shared_rules.has_phase_text_evidence(
                        text, phase_rules=rules.PHASE_RULES
                    )
                    or match_plane(text) is not None
                    else None
                )
            )
        ),
    )
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
        "is_localizer": acquisition_type == "LOCALIZER",
        "is_bolus_monitoring": acquisition_type == "BOLUS_MONITORING",
        "is_axial": plane == "AXIAL"
        or (plane is None and pd.notna(rows) and pd.notna(cols) and rows == cols),
        "is_original": bool(re.search(RX_IMAGE_ORIGINAL, image_type))
        and bool(re.search(RX_IMAGE_PRIMARY, image_type)),
        # Derived-product markers affect selection, not phase applicability.
        "is_derived_low_value": bool(
            first_text_column_value(
                row,
                lambda text: (
                    True if re.search(rules.RX_CT_DERIVED_LOW_VALUE, text) else None
                ),
            )
        ),
        "rows": rows,
        "cols": cols,
        "n_slices": n_slices,
        "slice_thickness": safe_float(row.get("SliceThickness")),
    }


def detect_ct_phase_applicability(row: pd.Series) -> str | None:
    """Return why liver perfusion phase is not applicable to this CT row."""
    features = detect_ct_features(row)
    if features["is_localizer"]:
        return "LOCALIZER"
    if features["is_bolus_monitoring"]:
        return "BOLUS_MONITORING"
    return None


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


def annotate_ct(
    df: pd.DataFrame,
    date_col: str = "date",
    exam_group_columns: list[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.date

    out["phase_text_column"] = out.apply(
        lambda row: evidence[0] if (evidence := _phase_text_column(row)) else "",
        axis=1,
    )
    out["phase_text_value"] = out.apply(
        lambda row: evidence[1] if (evidence := _phase_text_column(row)) else "",
        axis=1,
    )
    out["phase_applicability_reason"] = out.apply(detect_ct_phase_applicability, axis=1)
    result = out.apply(detect_ct_phase, axis=1)
    out[["ct_phase", "ct_phase_reason", "ct_phase_confidence"]] = pd.DataFrame(
        result.tolist(), index=out.index
    )
    eligible = out.loc[
        out["phase_applicability_reason"].isna()
        & ~out.apply(
            lambda row: detect_ct_features(row)["is_derived_low_value"], axis=1
        )
    ]
    group_columns = get_exam_group_cols(
        out, date_col=date_col, exam_group_columns=exam_group_columns
    )
    group_columns = list(
        dict.fromkeys(
            [
                *group_columns,
                *(
                    c
                    for c in ("study_id", "StudyInstanceUID", "acquisition_plane")
                    if c in out
                ),
            ]
        )
    )
    for idx, reason in infer_pre_post_native_acquisitions(
        eligible,
        text_columns=resolve_text_columns(TEXT_COLS_DEFAULT),
        group_columns=group_columns,
        phase_column="ct_phase",
        phase_rules=rules.PHASE_RULES,
    ):
        if out.loc[idx, "ct_phase"] == "OTHER":
            out.loc[idx, "ct_phase"] = "NATIVE"
            out.loc[idx, "ct_phase_reason"] = reason
            out.loc[idx, "ct_phase_confidence"] = "inferred"
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
    # A reconstruction can retain its acquisition phase without being eligible
    # for the default diagnostic cohort.
    if not candidates.empty:
        candidates = candidates.loc[
            ~candidates.apply(
                lambda row: detect_ct_features(row)["is_derived_low_value"], axis=1
            )
        ].copy()
    if "phase_status" in candidates.columns:
        candidates = candidates.loc[
            ~candidates["phase_status"].eq("NOT_APPLICABLE")
        ].copy()

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
        if has_text_columns_override():
            desc = build_series_text(row)
        else:
            desc = row.get("SeriesDescription", "")
            if safe_str(desc) == "":
                desc = row.get("ProtocolName", build_series_text(row))
        return f"{desc} [score={row.get('selection_score'):.1f}] {common.phase_evidence_summary(row)}"

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


@with_manifest_text_columns("CT")
def curate_ct(
    df: pd.DataFrame,
    patient_col: str = "patient_key",
    study_col: str | None = "study_id",
    date_col: str = "date",
    phase_curation: dict | None = None,
) -> dict[str, pd.DataFrame]:
    exam_cols = get_exam_group_cols(
        df,
        patient_col,
        study_col,
        date_col,
        validate_phase_curation(phase_curation)["best_candidate_group_columns"],
    )
    curated = annotate_ct(df, date_col=date_col, exam_group_columns=exam_cols)
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
