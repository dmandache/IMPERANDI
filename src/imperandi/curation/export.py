"""Export of the best CT/MR candidates for each exam."""

import logging
from pathlib import Path

import pandas as pd

from imperandi.curation import split_by_modality
from imperandi.curation.ct.curate import annotate_ct, score_ct, select_ct_per_exam
from imperandi.curation.mri.curate import (
    annotate_mri,
    add_scores,
    select_best_candidates,
)
from imperandi.curation.common import get_exam_group_cols, norm_label
from imperandi.curation.phase import validate_phase_curation
from imperandi.utils.run_state import atomic_write_csv

logger = logging.getLogger(__name__)


def add_selected_csv_argument(parser) -> None:
    parser.add_argument(
        "--selected_csv_path",
        default=None,
        help=(
            "Save selected_long (best series per exam) to this CSV and "
            "selected_wide QC to <stem>_qc.csv beside it. "
            "Default: <input_stem>_selected.csv beside the input CSV."
        ),
    )


def selected_qc_path(path: Path) -> Path:
    """Return the companion path for the selected_wide QC table."""
    return path.with_name(f"{path.stem}_qc.csv")


def selected_output_path(
    configured, output_path, *, input_path=None, protected_paths=()
) -> Path | None:
    """Default to <input_stem>_selected.csv and reject output collisions."""
    if configured is None:
        if input_path is None:
            return None
        source = Path(input_path).expanduser()
        path = source.with_name(f"{source.stem}_selected.csv")
    else:
        path = Path(configured).expanduser()
    if path.suffix.lower() != ".csv":
        raise ValueError("selected_csv_path must end in .csv.")
    protected = {
        Path(value).resolve()
        for value in (input_path, output_path, *protected_paths)
        if value
    }
    if any(
        candidate.resolve() in protected
        for candidate in (path, selected_qc_path(path))
    ):
        raise ValueError(
            "selected_csv_path and its QC path must differ from input "
            "and main/error outputs."
        )
    return path


def save_selected_candidates(df: pd.DataFrame, path: Path, config) -> int:
    """Save selected_long and companion selected_wide QC from the final cohort."""
    normalized = validate_phase_curation(config)
    exam_columns = normalized["exam_group_columns"]
    ct, mr, _ = split_by_modality(df)
    long_parts = []
    wide_parts = []
    for modality, data in (("CT", ct), ("MR", mr)):
        if data.empty:
            continue
        if not get_exam_group_cols(data, exam_group_columns=exam_columns):
            logger.warning(
                "No exam grouping columns for %s; %d row(s) cannot be selected. "
                "Writing empty selection and QC tables.",
                modality,
                len(data),
            )
            continue
        # Annotation supplies selection features even for a phase-only input.
        annotated = (
            annotate_ct(data)
            if modality == "CT"
            else annotate_mri(data, phase_curation=normalized)
        )
        for column in ("phase", "phase_source", "phase_confidence", "phase_reason"):
            if column in data:
                annotated[column] = data[column]
        if modality == "CT":
            annotated["selection_score"] = annotated.apply(score_ct, axis=1)
            annotated["ct_selection_score"] = annotated["selection_score"]
            annotated["selection_slot"] = annotated["phase"].map(
                lambda value: f"CT_{norm_label(value)}"
            )
            selected_long, selected_wide = select_ct_per_exam(
                annotated, exam_group_columns=exam_columns
            )
        else:
            annotated = add_scores(annotated)
            selected_long, selected_wide = select_best_candidates(
                annotated, exam_group_columns=exam_columns
            )
        long_parts.append(selected_long.assign(curation_modality=modality))
        wide_parts.append(selected_wide.assign(curation_modality=modality))
    selected_long = (
        pd.concat(long_parts, ignore_index=True, sort=False)
        if long_parts
        else df.iloc[:0].copy()
    )
    selected_wide = (
        pd.concat(wide_parts, ignore_index=True, sort=False)
        if wide_parts
        else pd.DataFrame(
            columns=[
                *get_exam_group_cols(df, exam_group_columns=exam_columns),
                "curation_modality",
            ]
        )
    )
    selected_long = selected_long.drop(
        columns=[
            c
            for c in selected_long
            if c == "_source_idx" or (c.startswith("_") and c not in df.columns)
        ]
    )
    atomic_write_csv(selected_long, path, index=False)
    qc_path = selected_qc_path(path)
    atomic_write_csv(selected_wide, qc_path, index=False)
    logger.info("Saved %d exam/modality rows for QC -> %s", len(selected_wide), qc_path)
    return len(selected_long)
