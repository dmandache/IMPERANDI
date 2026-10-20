"""
MRI metadata annotation, scoring, and diagnostic candidate selection.

Core functionality:
1. Classify MRI sequence family: T1, T2, DWI, LOCALIZER, KEY_IMAGES, OTHER.
2. Classify T1 perfusion/contrast phase: NATIVE, ARTERIAL,
   PORTAL_VENOUS, DELAYED, HEPATOBILIARY, OTHER.
3. Score diagnostic candidates.
4. Select one best candidate per exam per sequence, and one best T1 per phase.

Expected input: one row per MRI volume/series candidate, ideally volume-level.
Important columns when available:
    patient_key, study_id, series_id, volume_id, date, time,
    SeriesDescription, ProtocolName, StudyDescription, ImageType,
    SliceThickness, PixelSpacing, n_rows_in_volume

If volume_order_in_series / n_volumes_in_series are missing and volume_id exists,
they are inferred from patient/study/series grouping.
"""

from __future__ import annotations

from ast import literal_eval
import re
import logging
from typing import Sequence

import numpy as np
import pandas as pd

from imperandi.curation import metadata_helpers as common, shared_rules
from imperandi.utils.dicom_timing import scalar_values
from imperandi.curation.native_inference import infer_pre_post_native_acquisitions
from imperandi.curation.metadata_helpers import (
    clean_text as clean_text,
    get_exam_group_cols,
    is_missing as _is_missing,
    read_csv,
    safe_str,
)
from imperandi.curation.phase_resolution import (
    apply_phase_curation,
    phase_selection_bonus,
    validate_phase_curation,
)
from imperandi.curation.shared_rules import SEP_CHARS, match_phase, match_plane

from . import mri_rules as rules

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

TEXT_COLS_DEFAULT = [
    "SeriesDescription",
    # "ProtocolName",
    # "StudyDescription",
    # "ImageType",
    # "ScanningSequence",
    # "SequenceVariant",
    # "ScanOptions",
    # "SequenceName",
]

DIXON_TEXT_COLS = [col for col in TEXT_COLS_DEFAULT if col != "ImageType"]

ART_PORT_LATE_CONTEXT_PENDING = "art_port_late_context_pending"
ART_PORT_CONTEXT_PENDING = "art_port_context_pending"
MASK_MULTIART_CONTEXT_PENDING = "mask_multiart_context_pending"
GENERIC_DYNAMIC_CONTEXT_PENDING = "generic_dynamic_context_pending"


def _first_numeric(value) -> float:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        parsed = [_first_numeric(v) for v in value]
        parsed = [v for v in parsed if pd.notna(v)]
        return min(parsed) if parsed else np.nan
    return safe_float(value)


def _stable_text(value) -> str:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        parts = [_stable_text(v) for v in value]
        if isinstance(value, set):
            parts = sorted(parts)
        return "|".join(parts)
    if _is_missing(value):
        return ""
    return str(value)


def _display_str(x) -> str:
    if isinstance(x, (list, tuple, set, np.ndarray)):
        values = sorted(x, key=str) if isinstance(x, set) else x
        parts = [_display_str(v) for v in values]
        return " / ".join(part for part in parts if part)
    if _is_missing(x):
        return ""
    return re.sub(r"\s+", " ", str(x).strip())


def norm_label(x, default: str = "OTHER") -> str:
    if isinstance(x, (list, tuple, set, np.ndarray)):
        x = next((v for v in x if not _is_missing(v)), "")
    if pd.isna(x) or str(x).strip() == "":
        return default
    return str(x).strip().upper()


def safe_float(x) -> float:
    try:
        if x is None:
            return np.nan
        if isinstance(x, (list, tuple, set, np.ndarray)):
            return np.nan
        if pd.isna(x) or str(x).strip() == "":
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def parse_time_to_seconds(x) -> float:
    """Parse DICOM-ish HHMMSS / HH:MM:SS / datetime-like values."""
    if isinstance(x, (list, tuple, set, np.ndarray)):
        parsed = [parse_time_to_seconds(v) for v in x]
        parsed = [v for v in parsed if pd.notna(v)]
        return min(parsed) if parsed else np.nan

    if _is_missing(x):
        return np.nan

    s = str(x).strip()
    if not s:
        return np.nan

    dt = pd.to_datetime(s, errors="coerce")
    if pd.notna(dt) and not re.fullmatch(r"\d{6}(?:\.\d+)?", s):
        return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

    s = s.replace(":", "")
    m = re.match(r"^(\d{2})(\d{2})(\d{2})(?:\.(\d+))?$", s)
    if not m:
        return np.nan

    hh, mm, ss, frac = m.groups()
    if int(hh) > 23 or int(mm) > 59 or int(ss) > 59:
        return np.nan
    seconds = int(hh) * 3600 + int(mm) * 60 + int(ss)
    if frac:
        seconds += float("0." + frac)
    return seconds


def parse_pixel_spacing(x) -> tuple[float, float, float, float]:
    if isinstance(x, (list, tuple, set, np.ndarray)):
        x = next((v for v in x if not _is_missing(v)), None)
    if _is_missing(x):
        return np.nan, np.nan, np.nan, np.nan

    s = str(x).strip()
    s = (
        s.replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "\\")
        .replace(";", "\\")
    )
    parts = [p for p in re.split(r"[\\\s]+", s) if p]

    try:
        vals = [float(p) for p in parts]
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

    if not vals:
        return np.nan, np.nan, np.nan, np.nan

    sx, sy = (vals[0], vals[0]) if len(vals) == 1 else (vals[0], vals[1])
    mean_spacing = float(np.mean([sx, sy]))
    pixel_area = sx * sy
    return sx, sy, mean_spacing, pixel_area


def _resolve_text_cols(cols: Sequence[str] | None = None) -> list[str]:
    return (
        common.resolve_text_columns(TEXT_COLS_DEFAULT) if cols is None else list(cols)
    )


def get_dixon_text_cols(cols: Sequence[str] | None = None) -> list[str]:
    return [col for col in _resolve_text_cols(cols) if col != "ImageType"]


def iter_series_text_columns(
    row: pd.Series,
    cols: Sequence[str] | None = None,
):
    yield from common.iter_series_text_columns(row, _resolve_text_cols(cols))


def _first_text_column_value(
    row: pd.Series,
    evaluator,
    cols: Sequence[str] | None = None,
):
    return common.first_text_column_value(row, evaluator, _resolve_text_cols(cols))


def _first_text_decision(row: pd.Series, evaluator):
    """Retain the field behind a label/reason/confidence text decision."""
    for column, text in iter_series_text_columns(row):
        result = evaluator(text)
        if result is not None:
            label, reason, confidence = result
            if column != "SeriesDescription":
                reason += f"; {column}={text!r}"
            return label, reason, confidence
    return None


def _phase_text_column(row: pd.Series):
    return common.first_phase_text_column(
        row,
        _resolve_text_cols(),
        phase_rules=rules.PHASE_RULES,
        post=rules.RX_PHASE_POST_CONTRAST,
        extra_patterns=(rules.RX_PHASE_GENERIC_DYNAMIC,),
    )


def _first_phase_text_value(row: pd.Series, evaluator):
    evidence = _phase_text_column(row)
    return evaluator(evidence[1]) if evidence is not None else None


def _phase_matches_pattern(row: pd.Series, pattern: str) -> bool:
    return bool(_first_phase_text_value(row, lambda text: re.search(pattern, text)))


def _phase_matches_any_patterns(row: pd.Series, patterns: Sequence[str]) -> bool:
    return bool(
        _first_phase_text_value(
            row, lambda text: any(re.search(pattern, text) for pattern in patterns)
        )
    )


def _phase_matches_all_patterns(
    row: pd.Series,
    required_patterns: Sequence[str],
    excluded_patterns: Sequence[str] = (),
) -> bool:
    return bool(
        _first_phase_text_value(
            row,
            lambda text: all(re.search(pattern, text) for pattern in required_patterns)
            and not any(re.search(pattern, text) for pattern in excluded_patterns),
        )
    )


def _row_matches_pattern(
    row: pd.Series,
    pattern: str,
    cols: Sequence[str] | None = None,
) -> bool:
    return bool(
        _first_text_column_value(
            row,
            lambda text: True if re.search(pattern, text) else None,
            cols=cols,
        )
    )


def _feature_matches_pattern(row: pd.Series, pattern: str, family: Sequence[str]):
    evidence = common.first_text_column(
        row,
        lambda text: any(re.search(item, text) for item in family),
        _resolve_text_cols(),
    )
    return bool(evidence and re.search(pattern, evidence[1]))


def _sequence_matches_pattern(row: pd.Series, pattern: str) -> bool:
    return _feature_matches_pattern(
        row,
        pattern,
        (
            rules.RX_LOCALIZER,
            rules.RX_KEY_IMAGES,
            rules.RX_SEQUENCE_DWI,
            rules.RX_SEQUENCE_T1,
            rules.RX_SEQUENCE_T1_CONTRAST,
            rules.RX_SEQUENCE_T2,
        ),
    )


def _row_matches_any_patterns(
    row: pd.Series,
    patterns: Sequence[str],
    cols: Sequence[str] | None = None,
) -> bool:
    return bool(
        _first_text_column_value(
            row,
            lambda text: (
                True if any(re.search(pattern, text) for pattern in patterns) else None
            ),
            cols=cols,
        )
    )


def _row_matches_all_patterns(
    row: pd.Series,
    required_patterns: Sequence[str],
    excluded_patterns: Sequence[str] | None = None,
    cols: Sequence[str] | None = None,
) -> bool:
    excluded_patterns = list(excluded_patterns or [])
    return bool(
        _first_text_column_value(
            row,
            lambda text: (
                True
                if all(re.search(pattern, text) for pattern in required_patterns)
                and not any(re.search(pattern, text) for pattern in excluded_patterns)
                else None
            ),
            cols=cols,
        )
    )


def build_series_text(row: pd.Series, cols: Sequence[str] | None = None) -> str:
    """Combine the configured metadata fields into normalized display text."""
    return common.build_series_text(row, _resolve_text_cols(cols))


def build_display_text(row: pd.Series, cols: Sequence[str] | None = None) -> str:
    """Return display text from configured raw text columns."""
    cols = _resolve_text_cols(cols)
    parts = [_display_str(row.get(c)) for c in cols if c in row.index]
    return " | ".join(part for part in parts if part)


# -----------------------------------------------------------------------------
# Volume order
# -----------------------------------------------------------------------------


def add_volume_order_features(
    df: pd.DataFrame,
    patient_col: str = "patient_key",
    study_col: str | None = "study_id",
    series_col: str = "series_id",
    volume_col: str = "volume_id",
    time_col: str = "time",
) -> pd.DataFrame:
    """Add volume_order_in_series and n_volumes_in_series if possible."""
    out = df.drop(columns=["volume_index_in_series"], errors="ignore").copy()

    if "volume_order_in_series" in out.columns and "n_volumes_in_series" in out.columns:
        if "volume_order_source" not in out:
            out["volume_order_source"] = "provided"
        return out

    if series_col not in out.columns or volume_col not in out.columns:
        out["volume_order_in_series"] = 1
        out["n_volumes_in_series"] = 1
        out["is_multivolume_series"] = False
        return out

    series_group_cols = [
        c
        for c in [patient_col, study_col, series_col]
        if c is not None and c in out.columns
    ]
    volume_group_cols = [*series_group_cols, volume_col]

    work = out.copy()
    work["_sort_time_seconds"] = (
        work[time_col].apply(parse_time_to_seconds)
        if time_col in work.columns
        else np.nan
    )
    work["_row_order_for_volume_order"] = np.arange(len(work))

    work["_sort_acquisition_order"] = (
        work["acquisition_order"].map(_first_numeric)
        if "acquisition_order" in work
        else np.nan
    )
    work["_series_group_key"] = work.apply(
        lambda row: "||".join(_stable_text(row.get(c)) for c in series_group_cols),
        axis=1,
    )
    work["_volume_group_key"] = work.apply(
        lambda row: "||".join(_stable_text(row.get(c)) for c in volume_group_cols),
        axis=1,
    )

    # One representative row per volume for ordering.
    rep = (
        work.sort_values(
            [
                "_series_group_key",
                "_sort_time_seconds",
                "_sort_acquisition_order",
                "_row_order_for_volume_order",
            ],
            na_position="last",
        )
        .drop_duplicates("_volume_group_key")
        .copy()
    )

    rep["volume_order_in_series"] = (
        rep.groupby("_series_group_key", dropna=False).cumcount() + 1
    )
    rep["n_volumes_in_series"] = rep.groupby("_series_group_key", dropna=False)[
        "_volume_group_key"
    ].transform("size")
    rep["is_multivolume_series"] = rep["n_volumes_in_series"] > 1

    order_cols = [
        "_volume_group_key",
        "volume_order_in_series",
        "n_volumes_in_series",
        "is_multivolume_series",
    ]

    out = out.drop(
        columns=[
            c
            for c in [
                "volume_order_in_series",
                "n_volumes_in_series",
                "is_multivolume_series",
            ]
            if c in out.columns
        ],
        errors="ignore",
    )
    ordered = work[["_row_order_for_volume_order", "_volume_group_key"]].merge(
        rep[order_cols],
        on="_volume_group_key",
        how="left",
    )
    ordered = ordered.set_index("_row_order_for_volume_order").reindex(range(len(out)))
    for col in [
        "volume_order_in_series",
        "n_volumes_in_series",
        "is_multivolume_series",
    ]:
        out[col] = ordered[col].to_numpy()
    supported = (
        work.groupby("_series_group_key")[
            ["_sort_time_seconds", "_sort_acquisition_order"]
        ]
        .transform("nunique")
        .gt(1)
        .any(axis=1)
    )
    out["volume_order_source"] = np.where(
        supported.to_numpy(), "computed", "input_order"
    )
    return out


# -----------------------------------------------------------------------------
# Sequence classification
# -----------------------------------------------------------------------------


def detect_mri_sequence(row: pd.Series) -> tuple[str, str, str]:
    modality = safe_str(row.get("Modality")).upper()

    text_match = _first_text_decision(
        row,
        lambda text: (
            ("LOCALIZER", "matched localizer/scout/survey keyword", "high")
            if re.search(rules.RX_LOCALIZER, text)
            else (
                ("KEY_IMAGES", "matched key-image/processed marker", "high")
                if modality == "KO" or re.search(rules.RX_KEY_IMAGES, text)
                else (
                    ("DWI", "matched DWI/diffusion/ADC/b-value keyword", "high")
                    if re.search(rules.RX_SEQUENCE_DWI, text)
                    else (
                        ("T1", "matched T1 / VIBE-LAVA-THRIVE-Dixon-GRE family", "high")
                        if (
                            re.search(rules.RX_SEQUENCE_T1, text)
                            or re.search(rules.RX_SEQUENCE_T1_CONTRAST, text)
                        )
                        else (
                            ("T2", "matched T2/TSE/FSE/HASTE/BLADE/MRCP family", "high")
                            if re.search(rules.RX_SEQUENCE_T2, text)
                            else None
                        )
                    )
                )
            )
        ),
    )
    if text_match is not None:
        return text_match

    if modality == "KO":
        return "KEY_IMAGES", "matched key-image/processed marker", "high"

    # Weak TR/TE fallback, if available.
    tr = safe_float(row.get("RepetitionTime"))
    te = safe_float(row.get("EchoTime"))
    if pd.notna(tr) and pd.notna(te):
        if tr < 800 and te < 35:
            return "T1", f"TR={tr:g}, TE={te:g} compatible with T1", "medium"
        if tr > 1500 and te > 60:
            return "T2", f"TR={tr:g}, TE={te:g} compatible with T2", "medium"

    return "OTHER", "no MRI sequence rule matched", "low"


def add_mri_sequence_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    result = out.apply(detect_mri_sequence, axis=1)
    out[["mri_sequence", "mri_sequence_reason", "mri_sequence_confidence"]] = (
        pd.DataFrame(result.tolist(), index=out.index)
    )
    return out


def detect_t2_acquisition_family(row: pd.Series) -> str | None:
    """Classify T2 acquisitions using the first field with family evidence."""
    sequence = row.get("mri_sequence")
    if _is_missing(sequence):
        sequence = detect_mri_sequence(row)[0]
    if norm_label(sequence) != "T2":
        return None

    def evaluate(text):
        if re.search(rules.RX_T2_SINGLE_SHOT, text):
            return "single_shot"
        if re.search(rules.RX_T2_MULTI_SHOT, text):
            return "multi_shot"
        return None

    return _first_text_column_value(row, evaluate)


# -----------------------------------------------------------------------------
# T1 perfusion phase classification
# -----------------------------------------------------------------------------


def text_matches_art_port(row: pd.Series) -> bool:
    return _phase_matches_pattern(row, rules.RX_PHASE_ART_PORT_DYNAMIC)


def text_matches_art_port_late(row: pd.Series) -> bool:
    return _phase_matches_all_patterns(
        row,
        required_patterns=[
            rules.RX_PHASE_ARTERIAL,
            rules.RX_PHASE_PORTAL,
            rules.RX_PHASE_DELAYED,
        ],
        excluded_patterns=[rules.RX_PHASE_HEPATOBILIARY],
    )


def text_matches_mask_multiart(row: pd.Series) -> bool:
    return _phase_matches_pattern(row, rules.RX_PHASE_MASK_MULTIART_DYNAMIC)


def has_post_contrast_text(row: pd.Series) -> bool:
    return bool(
        _first_phase_text_value(
            row,
            lambda text: shared_rules.has_post_contrast_text(
                text, rules.RX_PHASE_POST_CONTRAST
            ),
        )
    )


def has_pure_post_contrast_text(row: pd.Series) -> bool:
    """Recognize acquisition postcontrast wording, not an agent or exam mention."""
    if bool(row.get("_numbered_postcontrast", False)):
        return True
    evidence = _phase_text_column(row)
    if evidence is None or evidence[0] == "StudyDescription":
        return False
    text = evidence[1]
    return bool(
        shared_rules.has_post_contrast_text(text, shared_rules.RX_PHASE_POST_CONTRAST)
        and not re.search(rules.RX_PHASE_NATIVE, text)
        and not shared_rules.has_pre_post_contrast_text(
            text, rules.RX_PHASE_NATIVE, rules.RX_PHASE_POST_CONTRAST
        )
    )


def detect_ordinal_phase_index(row: pd.Series) -> int | None:
    return _first_phase_text_value(
        row,
        lambda text: (
            next(
                (
                    int(group)
                    for group in re.search(rules.RX_PHASE_ORDINAL, text).groups()
                    if group is not None
                ),
                None,
            )
            if re.search(rules.RX_PHASE_ORDINAL, text)
            else None
        ),
    )


def detect_explicit_phase_from_text(row: pd.Series) -> tuple[str | None, str, str, str]:
    match = _first_phase_text_value(
        row,
        lambda text: match_phase(
            text,
            rules.PHASE_RULES,
            post=rules.RX_PHASE_POST_CONTRAST,
            prefer_delayed_time=True,
        ),
    )
    if match is not None:
        label, description = match
        return (
            label,
            f"matched explicit {description} evidence in {_phase_text_column(row)[0]}={_phase_text_column(row)[1]!r}",
            "explicit",
            "explicit_text",
        )

    return None, "no explicit T1 perfusion phase keyword matched", "unknown", "none"


def infer_special_t1_phase_from_volume_order(
    row: pd.Series,
) -> tuple[str | None, str, str, str]:
    """Special same-description/multiple-volume protocols."""
    if norm_label(row.get("mri_sequence")) != "T1":
        return None, "not T1", "unknown", "none"

    order = safe_float(row.get("volume_order_in_series"))
    n_volumes = safe_float(row.get("n_volumes_in_series"))
    if pd.isna(order) or pd.isna(n_volumes) or n_volumes < 2:
        return None, "no independent second volume detected", "unknown", "none"

    if (
        not float(order).is_integer()
        or not float(n_volumes).is_integer()
        or not 1 <= order <= n_volumes
    ):
        return (
            "OTHER",
            "invalid dynamic volume order/count",
            "unknown",
            "dynamic_profile_guard",
        )
    order = int(order)
    n_volumes = int(n_volumes)
    if text_matches_art_port_late(row) or text_matches_art_port(row):
        late = text_matches_art_port_late(row)
        label = (
            "ARTERIAL"
            if order == 1
            else (
                ("DELAYED" if late else "PORTAL_VENOUS")
                if order == n_volumes
                else (
                    "PORTAL_VENOUS"
                    if late and n_volumes == 3 and order == 2
                    else "OTHER"
                )
            )
        )
        profile = "ART/PORT/LATE" if late else "ART-PORT"
        return (
            label,
            f"{profile} endpoint fallback volume {order}/{n_volumes}"
            + (
                "; intermediate phase unresolved from profile order"
                if label == "OTHER"
                else ""
            ),
            "unknown" if label == "OTHER" else "inferred",
            "volume_order_art_port_late" if late else "volume_order_art_port",
        )

    if text_matches_mask_multiart(row):
        if order == 1:
            return (
                "NATIVE",
                f"inferred NATIVE from first Mask+Multiart volume {order}/{n_volumes}",
                "inferred",
                "volume_order_mask_multiart",
            )
        return (
            "ARTERIAL",
            f"inferred ARTERIAL from Mask+Multiart volume {order}/{n_volumes}",
            "inferred",
            "volume_order_mask_multiart",
        )

    return None, "no special dynamic T1 profile", "unknown", "none"


def detect_mri_phase_applicability(row: pd.Series) -> str | None:
    """Return why perfusion phase is not applicable to this MR row."""
    sequence = norm_label(row.get("mri_sequence"))
    if sequence == "LOCALIZER":
        return "LOCALIZER"
    if sequence in {"T2", "DWI"}:
        return "NON_T1_SEQUENCE"

    return None


def detect_t1_perfusion_phase(row: pd.Series) -> tuple[str, str, str, str]:
    """
    Volume-aware T1 phase classifier.

    Priority:
      1. Direct DICOM phase labels and pure named phase evidence.
      2. Established ART/PORT/LATE, ART-PORT, and Mask+Multiart inference.
      3. Defer generic dynamics and PhN prefixes to compatible chronology.
      4. OTHER.
    """
    seq = norm_label(row.get("mri_sequence"))

    # Key/secondary images can carry a named acquisition phase. Their sequence
    # classification still keeps them out of diagnostic candidate selection.
    if seq == "KEY_IMAGES":
        explicit_label, reason, confidence, source = detect_explicit_phase_from_text(
            row
        )
        if explicit_label is not None:
            return explicit_label, reason, confidence, source
        return "OTHER", "no phase keyword matched for key images", "unknown", "none"

    if seq != "T1":
        return "OTHER", f"sequence={seq}; phase not assigned", "unknown", "none"

    phase_values = {
        norm_label(value) for value in scalar_values(row.get("ContrastBolusAgentPhase"))
    }
    phase_mapping = {
        "PRE_CONTRAST": "NATIVE",
        "ARTERIAL": "ARTERIAL",
        "PORTAL_VENOUS": "PORTAL_VENOUS",
        "DELAYED": "DELAYED",
    }
    supported = {
        phase_mapping[value] for value in phase_values if value in phase_mapping
    }
    if supported and len(phase_values) > 1:
        return (
            "OTHER",
            "conflicting ContrastBolusAgentPhase values within volume",
            "unknown",
            "dicom_phase_conflict",
        )
    if len(supported) == 1:
        label = next(iter(supported))
        return label, "explicit ContrastBolusAgentPhase", "explicit", "dicom_phase"
    special_profile = (
        text_matches_art_port(row)
        or text_matches_art_port_late(row)
        or text_matches_mask_multiart(row)
    )
    if not special_profile:
        explicit = detect_explicit_phase_from_text(row)
        if explicit[0] is not None:
            return explicit
    if special_profile:
        text_column, text = _phase_text_column(row) or ("", "")
        durations = shared_rules._durations(text)
        if durations:
            labels = []
            for _, seconds in durations:
                matches = {
                    rule.label
                    for rule in rules.PHASE_RULES
                    if rule.accepts_time(seconds)
                }
                labels.append(next(iter(matches)) if len(matches) == 1 else "OTHER")
            label = labels[0] if len(set(labels)) == 1 else "OTHER"
            return (
                label,
                f"explicit duration in {text_column}={text!r}"
                + (
                    "; unsupported duration; order fallback blocked"
                    if label == "OTHER"
                    else ""
                ),
                "explicit" if label != "OTHER" else "unknown",
                "explicit_duration",
            )

    # A single derived product is not another acquisition in an ART/PORT
    # container. Only a real multivolume order can disambiguate its phase here;
    # do not let extra reconstructions shift the source acquisitions' ranks.
    if bool(row.get("is_derived_low_value")) and (
        text_matches_art_port(row) or text_matches_mask_multiart(row)
    ):
        special_label, reason, confidence, source = (
            infer_special_t1_phase_from_volume_order(row)
        )
        if special_label is not None:
            return special_label, reason, confidence, source
        return (
            "OTHER",
            "derived dynamic product lacks acquisition phase evidence",
            "unknown",
            "none",
        )

    if (
        _phase_matches_pattern(row, rules.RX_PHASE_MULTIART_DYNAMIC)
        and not text_matches_mask_multiart(row)
        and not text_matches_art_port(row)
        and not text_matches_art_port_late(row)
    ):
        return (
            "ARTERIAL",
            "explicit Multiart profile; all acquisitions arterial",
            "explicit",
            "explicit_text",
        )

    if text_matches_art_port_late(row):
        special_label, special_reason, special_conf, special_source = (
            infer_special_t1_phase_from_volume_order(row)
        )
        if special_label is not None:
            return special_label, special_reason, special_conf, special_source
        return (
            "OTHER",
            "matched single-volume ART/PORT/LATE text; awaiting exam acquisition context",
            "unknown",
            ART_PORT_LATE_CONTEXT_PENDING,
        )

    if text_matches_art_port(row):
        special_label, special_reason, special_conf, special_source = (
            infer_special_t1_phase_from_volume_order(row)
        )
        if special_label is not None:
            return special_label, special_reason, special_conf, special_source
        return (
            "OTHER",
            "matched single-volume ART-PORT text; awaiting exam acquisition context",
            "unknown",
            ART_PORT_CONTEXT_PENDING,
        )

    if text_matches_mask_multiart(row):
        special_label, special_reason, special_conf, special_source = (
            infer_special_t1_phase_from_volume_order(row)
        )
        if special_label is not None:
            return special_label, special_reason, special_conf, special_source
        return (
            "OTHER",
            "matched single-volume Mask+Multiart text; awaiting exam acquisition context",
            "unknown",
            MASK_MULTIART_CONTEXT_PENDING,
        )

    ordinal_index = detect_ordinal_phase_index(row)
    if ordinal_index is not None:
        return (
            "OTHER",
            f"ordinal phase Ph{ordinal_index} detected but exam context has not resolved it",
            "unknown",
            "ordinal_context",
        )

    if _phase_matches_pattern(row, rules.RX_PHASE_GENERIC_DYNAMIC):
        return (
            "OTHER",
            "generic dynamic text; awaiting compatible acquisition chronology and anchors",
            "unknown",
            GENERIC_DYNAMIC_CONTEXT_PENDING,
        )

    return "OTHER", "no supported T1 perfusion phase rule matched", "unknown", "none"


def _infer_special_profile_phases_by_acquisition_order(
    exam_rows: pd.DataFrame,
    *,
    candidate_sources: set[str],
    profile_name: str,
    rank_to_label,
    profile_match=None,
    minimum_count=2,
) -> list[tuple[int, str, str]]:
    """Resolve special dynamic profiles by acquisition order within Dixon component."""
    candidates = exam_rows.loc[
        exam_rows["mri_perfusion_source"].isin(candidate_sources)
        | (
            exam_rows["mri_perfusion_source"].eq("explicit_duration")
            & exam_rows.apply(profile_match, axis=1)
        )
    ].copy()
    if candidates.empty:
        return []

    n_volumes = candidates.get(
        "n_volumes_in_series", pd.Series(1, index=candidates.index)
    ).apply(safe_float)
    candidates["_is_single_volume"] = n_volumes.eq(1)
    candidates["_is_multivolume"] = n_volumes.gt(1)

    component = (
        candidates["dixon_component"].fillna("UNKNOWN")
        if "dixon_component" in candidates.columns
        else pd.Series("UNKNOWN", index=candidates.index)
    )
    assignments = []
    # Keep studies, sequence families, planes and reconstruction components apart.
    grouping = candidates.apply(_generic_dynamic_key, axis=1)
    for _, component_rows in candidates.groupby(grouping, sort=False):
        component_name = component.loc[component_rows.index[0]]
        has_single_volume = bool(component_rows["_is_single_volume"].any())
        has_multivolume = bool(component_rows["_is_multivolume"].any())
        if not has_single_volume:
            continue
        rejection = None
        if len(component_rows) < minimum_count:
            rejection = f"fewer than {minimum_count} acquisitions"
        # Require one complete, distinct ordering field; never break ties with
        # SeriesNumber, cross-series AcquisitionNumber, or input row order.
        order_values = None
        for column in ("acquisition_order", "time"):
            if column not in component_rows:
                continue
            parser = (
                _first_numeric
                if column == "acquisition_order"
                else parse_time_to_seconds
            )
            values = component_rows[column].map(parser)
            if values.notna().all() and values.nunique() == len(values):
                order_values = values
                break
        if len(component_rows) == 1 and minimum_count == 1:
            order_values = pd.Series([0], index=component_rows.index)
        if rejection is None and order_values is None:
            rejection = "missing or tied acquisition order"
        if rejection:
            assignments.extend(
                (
                    idx,
                    "OTHER",
                    f"ambiguous {profile_name}: {rejection}; independent phase evidence required",
                )
                for idx in component_rows.index
                if component_rows.loc[idx, "mri_perfusion_source"]
                != "explicit_duration"
            )
            continue

        ranked = order_values.sort_values(kind="stable").index
        context = (
            "mixed multivolume and single-volume matches"
            if has_multivolume
            else "single-volume series"
        )
        for rank, row_idx in enumerate(ranked, start=1):
            if (
                component_rows.loc[row_idx, "mri_perfusion_source"]
                == "explicit_duration"
            ):
                continue
            label = rank_to_label(rank, len(ranked))
            assignments.append(
                (
                    row_idx,
                    label,
                    (
                        f"inferred {label} from {profile_name} acquisition {rank}/{len(ranked)} "
                        f"for {component_name} {context}"
                    ),
                )
            )
    return assignments


def infer_art_port_phases_by_acquisition_order(
    exam_rows: pd.DataFrame,
) -> list[tuple[int, str, str]]:
    """Resolve ART-PORT rows by acquisition order when series context requires it."""
    return _infer_special_profile_phases_by_acquisition_order(
        exam_rows,
        candidate_sources={
            ART_PORT_CONTEXT_PENDING,
            "volume_order_art_port",
        },
        profile_name="ART-PORT",
        profile_match=lambda row: text_matches_art_port(row)
        and not text_matches_art_port_late(row),
        rank_to_label=lambda rank, count: (
            "ARTERIAL" if rank == 1 else "PORTAL_VENOUS" if rank == count else "OTHER"
        ),
    )


def infer_art_port_late_phases_by_acquisition_order(
    exam_rows: pd.DataFrame,
) -> list[tuple[int, str, str]]:
    """Resolve ART/PORT/LATE rows by acquisition order when series context requires it."""
    return _infer_special_profile_phases_by_acquisition_order(
        exam_rows,
        candidate_sources={
            ART_PORT_LATE_CONTEXT_PENDING,
            "volume_order_art_port_late",
        },
        profile_name="ART/PORT/LATE",
        profile_match=text_matches_art_port_late,
        rank_to_label=lambda rank, count: (
            "ARTERIAL"
            if rank == 1
            else (
                "DELAYED"
                if rank == count
                else "PORTAL_VENOUS" if count == 3 else "OTHER"
            )
        ),
    )


def infer_mask_multiart_phases_by_acquisition_order(
    exam_rows: pd.DataFrame,
) -> list[tuple[int, str, str]]:
    """Resolve Mask+Multiart rows by acquisition order when series context requires it."""
    return _infer_special_profile_phases_by_acquisition_order(
        exam_rows,
        candidate_sources={
            MASK_MULTIART_CONTEXT_PENDING,
            "volume_order_mask_multiart",
        },
        profile_name="Mask+Multiart",
        profile_match=text_matches_mask_multiart,
        minimum_count=1,
        rank_to_label=lambda rank, count: "NATIVE" if rank == 1 else "ARTERIAL",
    )


def _dynamic_series_id(row: pd.Series) -> str:
    return next(
        (
            common.stable_text(row.get(col))
            for col in ("series_id", "SeriesInstanceUID")
            if not common.is_missing(row.get(col)) and common.stable_text(row.get(col))
        ),
        "",
    )


def _generic_dynamic_key(row: pd.Series) -> tuple:
    sequence_patterns = (
        rules.RX_LOCALIZER,
        rules.RX_KEY_IMAGES,
        rules.RX_SEQUENCE_DWI,
        rules.RX_SEQUENCE_T1,
        rules.RX_SEQUENCE_T1_CONTRAST,
        rules.RX_SEQUENCE_T2,
        *(pattern for _, pattern in rules.GENERIC_DYNAMIC_FAMILIES),
    )
    evidence = common.first_text_column(
        row,
        lambda text: any(re.search(pattern, text) for pattern in sequence_patterns),
        _resolve_text_cols(),
    )
    family = next(
        (
            name
            for name, pattern in rules.GENERIC_DYNAMIC_FAMILIES
            if evidence and re.search(pattern, evidence[1])
        ),
        "UNSPECIFIED_T1",
    )
    plane = norm_label(row.get("acquisition_plane"), "UNKNOWN")
    if plane not in {"AXIAL", "CORONAL", "SAGITTAL"}:
        plane = norm_label(row.get("plane"), "UNKNOWN")
    return (
        *(
            common.stable_text(row.get(col))
            for col in ("patient_key", "date", "study_id", "StudyInstanceUID")
        ),
        tuple(parse_image_type_tokens(row.get("ImageType"))),
        family,
        plane,
        norm_label(row.get("dixon_component"), "NOT_DIXON"),
    )


def _matches_dynamic_group(row: pd.Series, key: tuple, series_ids: set[str]) -> bool:
    """Keep native/context matching consistent with the target dynamic protocol."""
    row_key = _generic_dynamic_key(row)
    if row_key == key:
        return True
    if row_key[:-1] != key[:-1]:
        return False
    # An explicitly identified water reconstruction can support an unidentified
    # Dixon reconstruction of the same protocol, never a known fat component.
    if row_key[-1] == "WATER" and key[-1] in {"DIXON_UNKNOWN", "NOT_DIXON"}:
        return True
    return (
        row_key[-1] in {"NOT_DIXON", "DIXON_UNKNOWN"}
        and _dynamic_series_id(row) in series_ids
    )


def _series_temporal_order(rows: pd.DataFrame):
    """Use volume ranks only within a known series."""
    if "volume_order_in_series" not in rows:
        return None, ""
    if (
        "volume_order_source" in rows
        and rows.volume_order_source.eq("input_order").any()
    ):
        return None, ""
    values = rows.volume_order_in_series.map(_first_numeric)
    counts = rows.get(
        "n_volumes_in_series", pd.Series(float("nan"), index=rows.index)
    ).map(_first_numeric)
    valid = values.ge(1) & values.le(counts) & values.mod(1).eq(0) & counts.mod(1).eq(0)
    if valid.all() and values.nunique() > 1:
        return values, "volume_order_in_series"
    return None, ""


def _generic_dynamic_chronology(rows: pd.DataFrame):
    """Order by canonical time, then acquisition order or series-local rank."""
    series = rows.apply(_dynamic_series_id, axis=1)
    local_orders = {}
    for sid in set(series) - {""}:
        local_orders[sid] = _series_temporal_order(rows.loc[series.eq(sid)])[0]
    choices = {}
    for column, parser in (
        ("time", parse_time_to_seconds),
        ("acquisition_order", _first_numeric),
    ):
        if column in rows:
            choices[column] = rows[column].map(parser)
    if series.ne("").all() and series.nunique() == 1:
        local = local_orders[series.iloc[0]]
        if local is not None:
            choices["volume_order_in_series within series"] = local
    complete = [
        column
        for column, values in choices.items()
        if values.notna().all() and values.nunique() > 1
    ]
    partial = [
        column
        for column, values in choices.items()
        if values.notna().any() and values.nunique() > 1
    ]
    if not (complete or partial):
        return None, "no comparable acquisition chronology"
    source = (complete or partial)[0]
    values = choices[source]
    evidence = pd.DataFrame(index=rows.index)
    evidence["order"] = np.nan
    evidence["order_source"] = source
    rank = 0
    for value in sorted(values.dropna().unique()):
        indices = values.index[values.eq(value)]
        frames = [indices]
        if source == "time" and len(indices) > 1:
            acquisition = choices.get("acquisition_order")
            if (
                acquisition is not None
                and acquisition.loc[indices].notna().all()
                and acquisition.loc[indices].nunique() > 1
            ):
                frames = [
                    indices[acquisition.loc[indices].eq(position)]
                    for position in sorted(acquisition.loc[indices].unique())
                ]
                evidence.loc[indices, "order_source"] += "; acquisition_order"
            else:
                local_series = [
                    sid
                    for sid in set(series.loc[indices]) - {""}
                    if local_orders[sid] is not None
                    and local_orders[sid]
                    .loc[indices.intersection(local_orders[sid].index)]
                    .nunique()
                    > 1
                ]
                if local_series:
                    if len(local_series) != 1 or series.loc[indices].nunique() != 1:
                        evidence.loc[
                            indices, "order_source"
                        ] += "; ambiguous volume order across series"
                        continue
                    positions = local_orders[local_series[0]].loc[indices]
                    frames = [
                        indices[positions.eq(position)]
                        for position in sorted(positions.unique())
                    ]
                    evidence.loc[
                        indices, "order_source"
                    ] += "; volume_order_in_series within series"
        for frame in frames:
            evidence.loc[frame, "order"] = rank
            rank += 1
    return evidence, ""


def _dynamic_time_deltas(rows: pd.DataFrame) -> pd.Series:
    """Measure seconds from the first volume in each compatible dynamic group."""
    delta = pd.Series(np.nan, index=rows.index, dtype=float)
    if "time" not in rows:
        return delta
    eligible = rows.loc[rows.mri_sequence.eq("T1")].copy()
    for column in ("is_subtraction", "is_quant_or_report"):
        if column in eligible:
            eligible = eligible.loc[~eligible[column].fillna(False)]
    if eligible.empty:
        return delta
    seconds = eligible.time.map(parse_time_to_seconds)
    for _, group in eligible.groupby(
        eligible.apply(_generic_dynamic_key, axis=1), sort=False
    ):
        if not group.apply(
            lambda row: _phase_matches_pattern(row, rules.RX_PHASE_GENERIC_DYNAMIC)
            or detect_ordinal_phase_index(row) is not None
            or _phase_matches_pattern(row, rules.RX_PHASE_MULTIART_DYNAMIC),
            axis=1,
        ).any():
            continue
        chronology, _ = _generic_dynamic_chronology(group)
        if chronology is None:
            # A shared populated time still gives an unambiguous zero delta.
            reference = seconds.loc[group.index]
        else:
            first = chronology.index[chronology.order.eq(chronology.order.min())]
            reference = seconds.loc[first]
        if reference.empty or reference.isna().any() or reference.nunique() != 1:
            continue
        delta.loc[group.index] = seconds.loc[group.index] - reference.iloc[0]
    return delta


def infer_generic_dynamic_phases_from_exam_context(
    exam_rows: pd.DataFrame,
) -> list[tuple[object, str, str, str, str]]:
    """Prefer default delay windows before the ordered dynamic template.

    Succeeding explicit phases bound generic inference. Within those bounds,
    elapsed seconds from time_delta precede the established order fallback.
    """

    eligible = exam_rows.loc[exam_rows.mri_sequence.map(norm_label).eq("T1")].copy()
    for column in ("is_subtraction", "is_quant_or_report"):
        if column in eligible:
            eligible = eligible.loc[~eligible[column].fillna(False)]
    if eligible.empty:
        return []
    generic = eligible.loc[
        eligible.mri_perfusion_source.isin(
            {GENERIC_DYNAMIC_CONTEXT_PENDING, "none", "volume_order", "ordinal_context"}
        )
        & eligible.apply(
            lambda row: (
                _phase_matches_pattern(row, rules.RX_PHASE_GENERIC_DYNAMIC)
                or detect_ordinal_phase_index(row) is not None
            )
            and not text_matches_art_port(row)
            and not text_matches_mask_multiart(row)
            and not _phase_matches_pattern(row, rules.RX_PHASE_MULTIART_DYNAMIC)
            and detect_explicit_phase_from_text(row)[0] is None,
            axis=1,
        )
    ]
    if generic.empty:
        return []
    anchors = eligible.loc[
        (
            eligible.mri_perfusion_source.isin(
                {
                    "explicit_text",
                    "dicom_phase",
                    "strategy_context",
                    "acquisition_phase",
                }
            )
            & eligible.mri_perfusion_label.isin(
                {"NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED"}
            )
        )
        | (
            eligible.mri_perfusion_label.eq("NATIVE")
            & eligible.mri_perfusion_source.isin(
                {
                    "exam_context",
                    "group_pre_post_order",
                    "volume_order_mask_multiart",
                    "acquisition_order_mask_multiart",
                }
            )
        )
    ]
    keys = eligible.apply(_generic_dynamic_key, axis=1)
    groups = {}
    for idx in generic.index:
        groups.setdefault(keys.loc[idx], []).append(idx)
    assignments = []
    phases = ["NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED"]
    phase_ranks = {phase: rank for rank, phase in enumerate(phases)}
    for key, indices in groups.items():
        candidates = generic.loc[indices]
        series_ids = set(candidates.apply(_dynamic_series_id, axis=1)) - {""}
        matched_anchors = anchors.loc[
            anchors.apply(
                lambda row: _matches_dynamic_group(row, key, series_ids), axis=1
            )
        ]
        context = pd.concat([candidates, matched_anchors])
        evidence, failure = _generic_dynamic_chronology(context)
        explicit_anchors = matched_anchors.loc[
            matched_anchors.mri_perfusion_source.isin(
                {
                    "explicit_text",
                    "dicom_phase",
                    "strategy_context",
                    "acquisition_phase",
                }
            )
        ]
        native_context = matched_anchors.drop(index=explicit_anchors.index)
        source = (
            "dynamic_explicit_anchor"
            if not explicit_anchors.empty
            else (
                "volume_order"
                if len(series_ids) == 1 and "acquisition_order" not in context
                else "acquisition_order_dixon_component"
            )
        )
        anchor_ids = ", ".join(
            str(row.get("volume_id", idx)) for idx, row in explicit_anchors.iterrows()
        )
        native_context_ids = ", ".join(
            str(row.get("volume_id", idx)) for idx, row in native_context.iterrows()
        )
        upper_bounds = {}
        if evidence is not None:
            anchor_ranks = explicit_anchors.mri_perfusion_label.map(phase_ranks)
            anchor_ranks = anchor_ranks.loc[anchor_ranks.gt(0)]
            time_order = (
                context.time.map(parse_time_to_seconds)
                if "time" in context
                else pd.Series(np.nan, index=context.index)
            )
            for idx in indices:
                # Ambiguous local ranks can still precede an anchor by canonical time.
                order = (
                    evidence.order
                    if pd.notna(evidence.loc[idx, "order"])
                    else time_order
                )
                later = order.index[order.gt(order.loc[idx])]
                upper = anchor_ranks.reindex(later).min()
                if pd.notna(upper):
                    upper_bounds[idx] = int(upper)

        def assign(idx, label, reason, *, use_delta=True, minimum_phase=1):
            decision_source = source
            upper = upper_bounds.get(idx) if use_delta else None
            seconds = safe_float(candidates.loc[idx].get("time_delta"))
            matches = {
                rule.label
                for rule in rules.PHASE_RULES
                if use_delta
                and rule.accepts_time(seconds)
                and (upper is None or phase_ranks.get(rule.label, len(phases)) < upper)
            }
            if len(matches) == 1:
                label = matches.pop()
                reason = f"matched default {label} delay window from time_delta={seconds:g}s since first volume in group"
                decision_source = "time_delta"
            if upper is not None:
                if phase_ranks.get(label, -1) >= upper:
                    label = phases[upper - 1] if upper - 1 >= minimum_phase else "OTHER"
                reason += f"; reserved succeeding explicit {phases[upper]} phase"
                if label == "OTHER":
                    reason += "; no compatible postcontrast phase remains"
            order_source = (
                evidence.loc[idx, "order_source"] if evidence is not None else ""
            )
            assignments.append(
                (
                    idx,
                    label,
                    (
                        "generic dynamic unresolved: "
                        if label == "OTHER"
                        else f"inferred {label}; "
                    )
                    + reason
                    + f"; ordered by {order_source}, family={key[-3]}, plane={key[-2]}, component={key[-1]}"
                    + (f"; explicit anchors={anchor_ids}" if anchor_ids else "")
                    + (
                        f"; native context={native_context_ids}"
                        if native_context_ids
                        else ""
                    ),
                    (
                        GENERIC_DYNAMIC_CONTEXT_PENDING
                        if label == "OTHER"
                        else decision_source
                    ),
                    order_source,
                )
            )
            return label

        if evidence is None:
            for idx in indices:
                assign(idx, "OTHER", failure)
            continue
        for idx in candidates.index[evidence.loc[candidates.index, "order"].isna()]:
            assign(idx, "OTHER", "missing or ambiguous acquisition order")
        usable = evidence.loc[evidence.order.notna()]
        native_indices = matched_anchors.index[
            matched_anchors.mri_perfusion_label.eq("NATIVE")
        ]
        ordered_native = native_indices.intersection(usable.index)
        if len(ordered_native):
            latest_native = usable.loc[ordered_native, "order"].max()
            early = candidates.index.intersection(
                usable.index[usable.order.le(latest_native)]
            )
            for idx in early:
                assign(
                    idx,
                    "OTHER",
                    "acquisition does not occur strictly after the resolved native context",
                )
            usable = usable.loc[
                usable.order.gt(latest_native) | usable.index.isin(ordered_native)
            ]
            # Repeated explicit natives do not consume postcontrast positions.
            usable = usable.loc[
                ~usable.index.isin(ordered_native) | usable.order.eq(latest_native)
            ]
        frames = [list(frame.index) for _, frame in usable.groupby("order", sort=True)]
        remaining = candidates.index.intersection(usable.index)
        if not frames or remaining.empty:
            continue
        has_native = len(native_indices) > 0
        acquisition_count = len(frames) + int(has_native and not len(ordered_native))
        if acquisition_count < 3:
            for idx in remaining:
                assign(
                    idx,
                    "OTHER",
                    "fewer than three distinct acquisitions for a generic dynamic template",
                )
            continue
        pure_post = (
            candidates.loc[remaining].apply(has_pure_post_contrast_text, axis=1).any()
        )
        first_labels = matched_anchors.loc[
            matched_anchors.index.intersection(frames[0]), "mri_perfusion_label"
        ]
        infer_native = not has_native and not pure_post and first_labels.empty
        previous_phase = -1 if infer_native else 0
        for frame in frames:
            named = matched_anchors.loc[
                matched_anchors.index.intersection(frame), "mri_perfusion_label"
            ]
            if named.nunique() > 1:
                for idx in remaining.intersection(frame):
                    assign(
                        idx,
                        "OTHER",
                        "conflicting explicit phases at one acquisition position",
                        use_delta=False,
                    )
                continue
            reason = "phase evidence=order only; ordered dynamic template"
            if not named.empty:
                label = named.iloc[0]
            else:
                label = phases[min(previous_phase + 1, len(phases) - 1)]
            if not named.empty:
                previous_phase = max(previous_phase, phase_ranks[label])
            for idx in remaining.intersection(frame):
                assigned = assign(
                    idx,
                    label,
                    reason,
                    use_delta=named.empty,
                    minimum_phase=max(previous_phase, 1),
                )
                if assigned != "OTHER":
                    previous_phase = max(
                        previous_phase,
                        phase_ranks.get(assigned, len(phases) - 1),
                    )
    return assignments


def is_native_fallback_candidate(row: pd.Series) -> bool:
    if norm_label(row.get("mri_sequence")) != "T1":
        return False
    if norm_label(row.get("mri_perfusion_label")) != "OTHER":
        return False
    if (
        bool(row.get("is_derived_low_value"))
        or row.get("mri_perfusion_source") == "dicom_phase_conflict"
    ):
        return False

    if (
        _row_matches_pattern(row, rules.RX_SUBTRACTION)
        or _row_matches_pattern(row, rules.RX_MIP_MPR)
        or _row_matches_pattern(row, rules.RX_QUANT_OR_REPORT)
        or has_pure_post_contrast_text(row)
        or detect_ordinal_phase_index(row) is not None
    ):
        return False

    explicit_label, *_ = detect_explicit_phase_from_text(row)
    if explicit_label is not None:
        return False

    dixon_component = row.get("dixon_component")
    if dixon_component in {"FAT", "FAT_FRACTION", "R2STAR", "DIXON_ALL"}:
        return False

    return bool(
        row.get("is_3d_gre")
        or _sequence_matches_pattern(row, rules.RX_T1_3D_GRE)
        or dixon_component in {"WATER", "IN_PHASE", "DIXON_UNKNOWN"}
    )


def _sort_key_for_native_fallback(row: pd.Series) -> tuple:
    component = row.get("dixon_component")
    component_score = {
        "WATER": 4,
        "IN_PHASE": 3,
        "DIXON_UNKNOWN": 2,
        "NOT_DIXON": 1,
    }.get(component, 0)
    quality_score = 0
    quality_score += 4 if row.get("plane") == "AXIAL" else 0
    quality_score += (
        3
        if bool(row.get("is_3d_gre"))
        or _sequence_matches_pattern(row, rules.RX_T1_3D_GRE)
        else 0
    )
    quality_score += 1 if bool(row.get("is_breath_hold")) else 0

    time_seconds = parse_time_to_seconds(row.get("time"))
    acquisition_order = safe_float(row.get("acquisition_order"))
    return (
        -quality_score,
        -component_score,
        time_seconds if pd.notna(time_seconds) else np.inf,
        acquisition_order if pd.notna(acquisition_order) else np.inf,
    )


def infer_missing_native_fallback(exam_rows: pd.DataFrame) -> tuple[int | None, str]:
    """Find a separate native before compatible dynamics or resolved perfusion phases."""
    if exam_rows.mri_perfusion_label.eq("NATIVE").any():
        return None, "native/precontrast already resolved in compatible protocol"
    candidates = exam_rows.loc[
        exam_rows.apply(
            lambda row: is_native_fallback_candidate(row)
            and not _phase_matches_pattern(row, rules.RX_PHASE_GENERIC_DYNAMIC),
            axis=1,
        )
    ]
    if candidates.empty:
        return None, "no suitable separate native fallback candidate"
    dynamics = exam_rows.loc[
        exam_rows.apply(
            lambda row: _phase_matches_pattern(row, rules.RX_PHASE_GENERIC_DYNAMIC)
            or detect_ordinal_phase_index(row) is not None,
            axis=1,
        )
    ]
    known_phases = exam_rows.loc[
        exam_rows.mri_perfusion_label.isin({"ARTERIAL", "PORTAL_VENOUS", "DELAYED"})
    ]
    context = pd.concat([dynamics, known_phases]).loc[
        lambda frame: ~frame.index.duplicated()
    ]
    if context.empty:
        return None, "no dynamic or resolved perfusion phase context"
    minimum_acquisitions = 2 if not known_phases.empty else 3
    ranked = sorted(
        candidates.index,
        key=lambda idx: _sort_key_for_native_fallback(candidates.loc[idx]),
    )
    for idx in ranked:
        chronology, _ = _generic_dynamic_chronology(
            pd.concat([candidates.loc[[idx]], context])
        )
        if (
            chronology is None
            or chronology.order.isna().any()
            or chronology.order.nunique() < minimum_acquisitions
        ):
            continue
        if chronology.loc[idx, "order"] >= chronology.loc[context.index, "order"].min():
            continue
        desc = candidates.loc[idx].get("SeriesDescription")
        return idx, (
            "selected fallback native/precontrast from compatible protocol context: "
            f"a separate candidate precedes compatible perfusion acquisitions: {desc}"
        )
    return None, "no supported native candidate before compatible dynamics"


def _infer_unnumbered_protocol_natives(rows: pd.DataFrame):
    """Match protocol text exactly after removing its numbered postcontrast token."""

    def protocol_name(text):
        return re.sub(
            rf"{SEP_CHARS}+", " ", re.sub(rules.RX_PHASE_ORDINAL, " ", text)
        ).strip()

    numbered = {}
    for idx, row in rows.iterrows():
        if detect_ordinal_phase_index(row) is None:
            continue
        if row.mri_perfusion_source in {"dicom_phase_conflict", "ordinal_phase_conflict"}:
            continue
        if row.mri_perfusion_label == "NATIVE" or bool(row.get("is_derived_low_value")):
            continue
        column, text = _phase_text_column(row)
        if column in {"StudyDescription", "ImageType"}:
            continue
        name = protocol_name(text)
        if name:
            key = (_generic_dynamic_key(row), column, name)
            numbered.setdefault(key, []).append(str(row.get("volume_id", idx)))

    if not numbered:
        return []
    assignments = []
    for idx, row in rows.iterrows():
        if (
            row.mri_perfusion_label != "OTHER"
            or row.mri_perfusion_source in {"dicom_phase_conflict", "ordinal_phase_conflict"}
            or bool(row.get("is_derived_low_value"))
            or detect_ordinal_phase_index(row) is not None
            or has_pure_post_contrast_text(row)
            or detect_explicit_phase_from_text(row)[0] is not None
        ):
            continue
        evidence = _phase_text_column(row)
        for column, text in iter_series_text_columns(row):
            if evidence is not None and evidence[0] != column:
                continue
            key = (_generic_dynamic_key(row), column, protocol_name(text))
            if key in numbered:
                assignments.append(
                    (
                        idx,
                        "inferred NATIVE from matching protocol without PhN: "
                        f"{column}={text!r}; numbered postcontrast volumes="
                        + ", ".join(numbered[key]),
                    )
                )
                break
    return assignments


def _apply_numbered_postcontrast_context(rows: pd.DataFrame) -> pd.DataFrame:
    """Require a distinct, supported unnumbered native before restricting PhN."""
    rows["_numbered_postcontrast"] = False
    native_sources = {
        "explicit_text",
        "dicom_phase",
        "strategy_context",
        "acquisition_phase",
        "exam_context",
        "group_pre_post_order",
        "volume_order_mask_multiart",
        "acquisition_order_mask_multiart",
    }
    natives = rows.loc[
        rows.mri_sequence.eq("T1")
        & rows.mri_perfusion_label.eq("NATIVE")
        & rows.mri_perfusion_source.isin(native_sources)
        & ~rows.get("is_derived_low_value", pd.Series(False, index=rows.index)).fillna(False)
        & rows.apply(lambda row: detect_ordinal_phase_index(row) is None, axis=1)
    ]
    if natives.empty:
        return rows
    for idx, row in rows.iterrows():
        ordinal = detect_ordinal_phase_index(row)
        if ordinal is None or row.mri_sequence != "T1":
            continue
        key = _generic_dynamic_key(row)
        series_ids = {_dynamic_series_id(row)} - {""}
        if not natives.apply(
            lambda native: _matches_dynamic_group(native, key, series_ids), axis=1
        ).any():
            continue
        rows.loc[idx, "_numbered_postcontrast"] = True
        if row.mri_perfusion_label == "NATIVE":
            rows.loc[idx, "mri_perfusion_label"] = "OTHER"
            rows.loc[idx, "mri_perfusion_confidence"] = "unknown"
            rows.loc[idx, "mri_perfusion_source"] = "ordinal_phase_conflict"
            rows.loc[idx, "mri_perfusion_reason"] = (
                f"Ph{ordinal} is postcontrast after resolving a compatible native; "
                "conflicting native phase evidence"
            )
    return rows


def _infer_missing_native_fallbacks(
    exam_rows: pd.DataFrame,
) -> list[tuple[object, str]]:
    """Resolve each compatible protocol independently; unrelated natives do not veto it."""
    eligible = exam_rows.loc[
        exam_rows.mri_sequence.eq("T1")
        & exam_rows.mri_perfusion_label.ne("HEPATOBILIARY")
    ].copy()
    for column in ("is_subtraction", "is_quant_or_report"):
        if column in eligible:
            eligible = eligible.loc[~eligible[column].fillna(False)]
    if eligible.empty:
        return []
    assignments = _infer_unnumbered_protocol_natives(eligible)
    for idx, _ in assignments:
        eligible.loc[idx, "mri_perfusion_label"] = "NATIVE"
        eligible.loc[idx, "mri_perfusion_source"] = "exam_context"
    phase_context = eligible.loc[
        eligible.mri_perfusion_label.isin({"ARTERIAL", "PORTAL_VENOUS", "DELAYED"})
        | eligible.apply(
            lambda row: _phase_matches_pattern(row, rules.RX_PHASE_GENERIC_DYNAMIC)
            or detect_ordinal_phase_index(row) is not None,
            axis=1,
        )
    ]
    if phase_context.empty:
        return assignments
    for key, rows in phase_context.groupby(
        phase_context.apply(_generic_dynamic_key, axis=1), sort=False
    ):
        series_ids = set(rows.apply(_dynamic_series_id, axis=1)) - {""}
        context = eligible.loc[
            eligible.apply(
                lambda row: _matches_dynamic_group(row, key, series_ids), axis=1
            )
        ]
        idx, reason = infer_missing_native_fallback(context)
        if idx is not None:
            assignments.append((idx, reason))
            eligible.loc[idx, "mri_perfusion_label"] = "NATIVE"
            eligible.loc[idx, "mri_perfusion_source"] = "exam_context"
    return assignments


def _share_acquisition_phase_evidence(rows: pd.DataFrame) -> pd.DataFrame:
    """Share authoritative phase labels across compatible reconstructions."""
    if "acquisition_id" not in rows:
        return rows
    out = rows.copy()
    keys = out.apply(
        lambda row: tuple(
            common.stable_text(row.get(col))
            for col in ("patient_key", "study_id", "StudyInstanceUID", "acquisition_id")
        ),
        axis=1,
    )
    for key, group in out.groupby(keys, sort=False):
        if not key[-1] or not (key[1] or key[2]) or len(group) < 2:
            continue
        if (
            not group.mri_sequence.eq("T1").all()
            or group.mri_perfusion_source.eq("dicom_phase_conflict").any()
        ):
            continue
        incompatible = False
        for column in (
            "FrameOfReferenceUID",
            "Rows",
            "Columns",
            "PixelSpacing",
            "ImageOrientationPatient",
            "ImagePositionPatient",
        ):
            if (
                column in group
                and len(set(group[column].map(common.stable_text)) - {""}) > 1
            ):
                incompatible = True
        for column in ("time", "acquisition_order", "volume_order_in_series"):
            if (
                column in group
                and group[column].dropna().map(common.stable_text).nunique() > 1
            ):
                incompatible = True
        if incompatible:
            continue
        anchors = group.loc[
            group.mri_perfusion_source.isin(
                {
                    "explicit_text",
                    "dicom_phase",
                    "strategy_context",
                }
            )
            & group.mri_perfusion_label.ne("OTHER")
        ]
        if anchors.empty or anchors.mri_perfusion_label.nunique() != 1:
            continue
        anchor = anchors.iloc[0]
        unresolved = group.index[group.mri_perfusion_label.eq("OTHER")]
        for idx in unresolved:
            out.loc[idx, "mri_perfusion_label"] = anchor.mri_perfusion_label
            out.loc[idx, "mri_perfusion_reason"] = (
                f"same validated acquisition_id={key[-1]} as authoritative phase: {anchor.mri_perfusion_reason}"
            )
            out.loc[idx, "mri_perfusion_confidence"] = "inferred"
            out.loc[idx, "mri_perfusion_source"] = "acquisition_phase"
    return out


def add_mri_perfusion_columns(
    df: pd.DataFrame,
    exam_group_cols: Sequence[str] | None = None,
    phase_curation: dict | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out["phase_text_column"] = out.apply(
        lambda row: evidence[0] if (evidence := _phase_text_column(row)) else "",
        axis=1,
    )
    out["phase_text_value"] = out.apply(
        lambda row: evidence[1] if (evidence := _phase_text_column(row)) else "",
        axis=1,
    )
    result = out.apply(detect_t1_perfusion_phase, axis=1)
    out[
        [
            "mri_perfusion_label",
            "mri_perfusion_reason",
            "mri_perfusion_confidence",
            "mri_perfusion_source",
        ]
    ] = pd.DataFrame(result.tolist(), index=out.index)
    out["mri_dynamic_order_source"] = ""
    out["time_delta"] = np.nan

    group_cols = [c for c in (exam_group_cols or []) if c in out.columns]
    if group_cols:
        grouped = out.groupby(
            [out[col].map(common.stable_text) for col in group_cols], dropna=False
        ).groups.values()
    else:
        grouped = [out.index]

    # Resolve only strategies ahead of rules. Their labels inform chronology,
    # while raw rule outputs and final strategy precedence remain independent.
    config = validate_phase_curation(phase_curation)
    preceding = []
    for strategy in config["strategies"]:
        if strategy["type"] == "rules":
            break
        preceding.append(strategy)
    authority = pd.DataFrame(index=out.index)
    if preceding:
        authority = apply_phase_curation(out, {**config, "strategies": preceding})

    def context_rows(indices):
        context = out.loc[indices].copy()
        if "phase_status" in authority:
            resolved = authority.loc[indices, "phase_status"].eq("RESOLVED")
            selected = resolved.index[resolved]
            context.loc[selected, "mri_perfusion_label"] = authority.loc[
                selected, "phase"
            ]
            context.loc[selected, "mri_perfusion_source"] = "strategy_context"
            context.loc[selected, "mri_perfusion_reason"] = authority.loc[
                selected, "phase_reason"
            ]
            context.loc[selected, "mri_perfusion_confidence"] = authority.loc[
                selected, "phase_confidence"
            ]
        return _apply_numbered_postcontrast_context(context)

    def apply_native_fallbacks(indices):
        for fallback_idx, reason in _infer_missing_native_fallbacks(
            context_rows(indices)
        ):
            logger.info("Fallback native selected for MRI protocol: %s", reason)
            out.loc[fallback_idx, "mri_perfusion_label"] = "NATIVE"
            out.loc[fallback_idx, "mri_perfusion_reason"] = reason
            out.loc[fallback_idx, "mri_perfusion_confidence"] = "fallback"
            out.loc[fallback_idx, "mri_perfusion_source"] = "exam_context"

    for idx in grouped:
        out.loc[idx, "time_delta"] = _dynamic_time_deltas(out.loc[idx])
        shared = _share_acquisition_phase_evidence(context_rows(idx))
        propagated = shared.index[shared.mri_perfusion_source.eq("acquisition_phase")]
        for column in (
            "mri_perfusion_label",
            "mri_perfusion_reason",
            "mri_perfusion_confidence",
            "mri_perfusion_source",
        ):
            out.loc[propagated, column] = shared.loc[propagated, column]
        exam_rows = context_rows(idx)

        for row_idx, label, reason in infer_art_port_late_phases_by_acquisition_order(
            exam_rows
        ):
            out.loc[row_idx, "mri_perfusion_label"] = label
            out.loc[row_idx, "mri_perfusion_reason"] = reason
            out.loc[row_idx, "mri_perfusion_confidence"] = (
                "unknown" if label == "OTHER" else "inferred"
            )
            out.loc[row_idx, "mri_perfusion_source"] = "acquisition_order_art_port_late"

        exam_rows = context_rows(idx)

        for row_idx, label, reason in infer_art_port_phases_by_acquisition_order(
            exam_rows
        ):
            out.loc[row_idx, "mri_perfusion_label"] = label
            out.loc[row_idx, "mri_perfusion_reason"] = reason
            out.loc[row_idx, "mri_perfusion_confidence"] = (
                "unknown" if label == "OTHER" else "inferred"
            )
            out.loc[row_idx, "mri_perfusion_source"] = "acquisition_order_art_port"

        exam_rows = context_rows(idx)

        for row_idx, label, reason in infer_mask_multiart_phases_by_acquisition_order(
            exam_rows
        ):
            out.loc[row_idx, "mri_perfusion_label"] = label
            out.loc[row_idx, "mri_perfusion_reason"] = reason
            out.loc[row_idx, "mri_perfusion_confidence"] = (
                "unknown" if label == "OTHER" else "inferred"
            )
            out.loc[row_idx, "mri_perfusion_source"] = "acquisition_order_mask_multiart"

        apply_native_fallbacks(idx)

        exam_rows = context_rows(idx)

        for (
            row_idx,
            label,
            reason,
            source,
            order_source,
        ) in infer_generic_dynamic_phases_from_exam_context(exam_rows):
            out.loc[row_idx, "mri_perfusion_label"] = label
            out.loc[row_idx, "mri_perfusion_reason"] = reason
            out.loc[row_idx, "mri_perfusion_confidence"] = (
                "inferred" if label != "OTHER" else "unknown"
            )
            out.loc[row_idx, "mri_perfusion_source"] = source
            out.loc[row_idx, "mri_dynamic_order_source"] = order_source

        # Timing can resolve perfusion evidence that was unavailable in the first pass.
        apply_native_fallbacks(idx)
        exam_rows = context_rows(idx)

        eligible = exam_rows.loc[
            exam_rows["mri_sequence"].eq("T1")
            & ~exam_rows["is_derived_low_value"].fillna(False)
        ]
        compatibility = [
            col
            for col in ("study_id", "StudyInstanceUID", "dixon_component", "plane")
            if col in eligible
        ]
        for row_idx, reason in infer_pre_post_native_acquisitions(
            eligible,
            text_columns=_resolve_text_cols(),
            group_columns=compatibility,
            phase_column="mri_perfusion_label",
            native=rules.RX_PHASE_NATIVE,
            post=rules.RX_PHASE_POST_CONTRAST,
            phase_rules=rules.PHASE_RULES,
            extra_patterns=(rules.RX_PHASE_GENERIC_DYNAMIC,),
        ):
            if bool(exam_rows.loc[row_idx, "_numbered_postcontrast"]):
                continue
            label = norm_label(out.loc[row_idx, "mri_perfusion_label"])
            source = out.loc[row_idx, "mri_perfusion_source"]
            if source == "dicom_phase_conflict":
                continue
            if label != "OTHER" and not (
                label == "NATIVE"
                and source in {"volume_order", "acquisition_order_dixon_component"}
            ):
                continue
            out.loc[row_idx, "mri_perfusion_label"] = "NATIVE"
            out.loc[row_idx, "mri_perfusion_reason"] = reason
            out.loc[row_idx, "mri_perfusion_confidence"] = "inferred"
            out.loc[row_idx, "mri_perfusion_source"] = "group_pre_post_order"

        # Mixed pre/post inference can establish the native context at this stage.
        exam_rows = context_rows(idx)
        conflicts = exam_rows.index[
            exam_rows.mri_perfusion_source.eq("ordinal_phase_conflict")
        ]
        for column in (
            "mri_perfusion_label",
            "mri_perfusion_confidence",
            "mri_perfusion_source",
            "mri_perfusion_reason",
        ):
            out.loc[conflicts, column] = exam_rows.loc[conflicts, column]

    for row_idx, row in out.iterrows():
        if row.phase_text_column and row.phase_text_value:
            detail = f"{row.phase_text_column}={row.phase_text_value!r}"
            if detail not in row.mri_perfusion_reason:
                out.loc[row_idx, "mri_perfusion_reason"] += f"; {detail}"
    return out


# -----------------------------------------------------------------------------
# Feature extraction and scoring
# -----------------------------------------------------------------------------


def detect_plane(value: pd.Series | str, cols: Sequence[str] | None = None) -> str:
    if isinstance(value, pd.Series):
        return _first_text_column_value(value, match_plane, cols=cols) or "UNKNOWN"
    return match_plane(safe_str(value)) or "UNKNOWN"


def parse_image_type_tokens(value: object) -> list[str]:
    """Return normalized, exact DICOM ImageType tokens."""
    if _is_missing(value):
        return []

    if isinstance(value, (list, tuple, set, np.ndarray)):
        tokens = []
        values = sorted(value, key=str) if isinstance(value, set) else value
        for item in values:
            tokens.extend(parse_image_type_tokens(item))
        return tokens

    text = str(value).strip()
    if not text:
        return []

    if text[:1] in "[(" and text[-1:] in ")]":
        try:
            parsed = literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if parsed is not None and not isinstance(parsed, str):
            return parse_image_type_tokens(parsed)

    raw_tokens = re.split(r"[\\,;\s]+", text)
    return [
        re.sub(r"[\s-]+", "_", token.strip().upper())
        for token in raw_tokens
        if token.strip()
    ]


IMAGE_TYPE_DIXON_COMPONENTS = {
    "W": "WATER",
    "WATER": "WATER",
    "F": "FAT",
    "FAT": "FAT",
    "IP": "IN_PHASE",
    "IN_PHASE": "IN_PHASE",
    "INPHASE": "IN_PHASE",
    "OP": "OPPOSED_PHASE",
    "OOP": "OPPOSED_PHASE",
    "OUT_PHASE": "OPPOSED_PHASE",
    "OUTOFPHASE": "OPPOSED_PHASE",
    "FF": "FAT_FRACTION",
    "FAT_FRACTION": "FAT_FRACTION",
    "FATFRACTION": "FAT_FRACTION",
    "R2STAR": "R2STAR",
    "R2*": "R2STAR",
    "R2S": "R2STAR",
}

IMAGE_TYPE_QUANTITATIVE_COMPONENTS = ("FAT_FRACTION", "R2STAR")


def _image_type_dixon_details(value: object) -> tuple[str | None, list[str]]:
    """Prefer quantitative tokens; reject contradictory reconstruction tokens."""
    tokens = parse_image_type_tokens(value)
    matched = {
        IMAGE_TYPE_DIXON_COMPONENTS[token]
        for token in tokens
        if token in IMAGE_TYPE_DIXON_COMPONENTS
    }
    if not matched:
        return None, tokens

    for component in IMAGE_TYPE_QUANTITATIVE_COMPONENTS:
        if component in matched:
            return component, tokens

    if len(matched) > 1:
        return "DIXON_UNKNOWN", tokens
    return next(iter(matched)), tokens


def detect_dixon_component_from_image_type(value: object) -> str | None:
    """Classify exact ImageType component tokens; contradictions are unknown."""
    component, _ = _image_type_dixon_details(value)
    return component


def _free_text_component_tokens(text: str) -> set[str]:
    """Parse compact reconstruction suffixes only after Dixon context is known."""
    return {token.upper() for token in re.split(rf"{SEP_CHARS}+", text) if token}


def _detect_dixon_component_from_text(text: str) -> tuple[str, str, str] | None:
    explicit_components = []
    for component, pattern in [
        ("WATER", rules.RX_DIXON_WATER),
        ("FAT", rules.RX_DIXON_FAT),
        ("IN_PHASE", rules.RX_DIXON_IN),
        ("OPPOSED_PHASE", rules.RX_DIXON_OPPOSED),
    ]:
        if re.search(pattern, text):
            explicit_components.append(component)

    has_dixon_context = bool(re.search(rules.RX_DIXON_CONTEXT, text))
    if has_dixon_context:
        # W in w/wo, wo-W, or w contrast describes injection, not a component.
        water_text = re.sub(rules.RX_DIXON_CONTRAST_SHORTHAND, "contrast", text)
        if re.search(rules.RX_DIXON_WATER_SUFFIX, water_text):
            explicit_components.append("WATER")
        suffix_tokens = _free_text_component_tokens(text)
        for token, component in {
            "F": "FAT",
            "IN": "IN_PHASE",
            "IP": "IN_PHASE",
            "OPP": "OPPOSED_PHASE",
            "OP": "OPPOSED_PHASE",
            "OOP": "OPPOSED_PHASE",
        }.items():
            if token in suffix_tokens:
                explicit_components.append(component)

    explicit_components = list(dict.fromkeys(explicit_components))
    if len(explicit_components) == 1:
        component = explicit_components[0]
        return component, f"matched explicit {component.lower()} text", "explicit_text"
    if len(explicit_components) > 1:
        return (
            "DIXON_UNKNOWN",
            f"contradictory explicit Dixon components: {', '.join(explicit_components)}",
            "explicit_text",
        )
    if has_dixon_context:
        return (
            "DIXON_UNKNOWN",
            "Dixon context detected without component",
            "dixon_context",
        )
    return None


def detect_dixon_component(row: pd.Series) -> tuple[str, str, str]:
    """Return normalized Dixon component, reason, and evidence source."""
    text_match = _first_text_column_value(
        row,
        lambda text: (
            ("FAT_FRACTION", "matched explicit fat-fraction/PDFF text", "explicit_text")
            if re.search(rules.RX_DIXON_FAT_FRACTION, text)
            else (
                ("R2STAR", "matched explicit R2*/T2* map text", "explicit_text")
                if re.search(rules.RX_DIXON_R2STAR, text)
                else (
                    (
                        "DIXON_ALL",
                        "matched explicit all-reconstructions text",
                        "explicit_text",
                    )
                    if re.search(rules.RX_DIXON_ALL, text)
                    else _detect_dixon_component_from_text(text)
                )
            )
        ),
        cols=get_dixon_text_cols(),
    )
    if text_match is not None and text_match[0] in {"FAT_FRACTION", "R2STAR"}:
        return text_match

    image_component, image_tokens = _image_type_dixon_details(row.get("ImageType"))
    if image_component is not None:
        matched_tokens = [
            token for token in image_tokens if token in IMAGE_TYPE_DIXON_COMPONENTS
        ]
        if image_component == "DIXON_UNKNOWN":
            return (
                image_component,
                f"contradictory ImageType Dixon tokens: {', '.join(matched_tokens)}",
                "image_type",
            )
        return (
            image_component,
            f"matched ImageType token {matched_tokens[0]}",
            "image_type",
        )

    if text_match is not None:
        return text_match

    return "NOT_DIXON", "no Dixon context or component token", "none"


def _product_text_flags(row: pd.Series) -> tuple[bool, ...]:
    patterns = (
        rules.RX_SUBTRACTION,
        rules.RX_MIP_MPR,
        rules.RX_QUANT_OR_REPORT,
        rules.RX_KEY_IMAGES,
    )

    component, *_ = detect_dixon_component(row)

    def evaluate(text):
        flags = [bool(re.search(pattern, text)) for pattern in patterns]
        if component not in {"FAT_FRACTION", "R2STAR"} and (
            re.search(rules.RX_DIXON_FAT_FRACTION, text)
            or re.search(rules.RX_DIXON_R2STAR, text)
        ):
            # Discard later quantitative wording contradicted by the selected
            # component. Subtraction/MPR remain compatible with water/fat data.
            flags[2] = False
        return tuple(flags) if any(flags) else None

    text_flags = _first_text_column_value(
        row, evaluate, cols=[col for col in _resolve_text_cols() if col != "ImageType"]
    ) or (False,) * len(patterns)
    # Structured product markers remain independent of text overrides.
    image_text = safe_str(row.get("ImageType"))
    image_flags = tuple(bool(re.search(pattern, image_text)) for pattern in patterns)
    return tuple(text or image for text, image in zip(text_flags, image_flags))


def add_basic_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["series_text"] = out.apply(build_series_text, axis=1)
    out["plane"] = out.apply(detect_plane, axis=1)

    products = out.apply(_product_text_flags, axis=1)
    out[["is_subtraction", "is_mip_mpr", "is_quant_or_report", "_is_key_product"]] = (
        pd.DataFrame(products.tolist(), index=out.index)
    )

    # T1 features.
    dixon = out.apply(detect_dixon_component, axis=1)
    out[["dixon_component", "dixon_component_reason", "dixon_component_source"]] = (
        pd.DataFrame(dixon.tolist(), index=out.index)
    )
    # Product eligibility is independent of whether its phase can be resolved.
    out["is_derived_low_value"] = (
        out[["is_subtraction", "is_mip_mpr", "is_quant_or_report"]].any(axis=1)
        | out["dixon_component"].isin(["FAT_FRACTION", "R2STAR"])
        | out.pop("_is_key_product")
    )
    out["is_3d_gre"] = out.apply(
        lambda row: _sequence_matches_pattern(row, rules.RX_T1_3D_GRE), axis=1
    )
    out["is_dynamic_t1_text"] = out.apply(
        lambda row: _phase_matches_pattern(row, rules.RX_T1_DYNAMIC),
        axis=1,
    )
    respiratory_patterns = (rules.RX_BREATH_HOLD, rules.RX_RESP_TRIGGERED)
    out["is_breath_hold"] = out.apply(
        lambda row: _feature_matches_pattern(
            row, rules.RX_BREATH_HOLD, respiratory_patterns
        ),
        axis=1,
    )
    out["is_resp_triggered"] = out.apply(
        lambda row: _feature_matches_pattern(
            row, rules.RX_RESP_TRIGGERED, respiratory_patterns
        ),
        axis=1,
    )

    # T2 features.
    out["mri_t2_acquisition_family"] = out.apply(detect_t2_acquisition_family, axis=1)
    t2_patterns = (
        rules.RX_T2_FATSAT,
        rules.RX_T2_MOTION_ROBUST,
        rules.RX_T2_HASTE_SSFSE,
        rules.RX_T2_TSE_FSE,
        rules.RX_T2_MRCP_BILIARY,
    )
    out["is_t2_fatsat"] = out.apply(
        lambda row: _feature_matches_pattern(row, rules.RX_T2_FATSAT, t2_patterns),
        axis=1,
    )
    out["is_t2_motion_robust"] = out.apply(
        lambda row: _feature_matches_pattern(
            row, rules.RX_T2_MOTION_ROBUST, t2_patterns
        ),
        axis=1,
    )
    out["is_t2_haste_ssfse"] = out.apply(
        lambda row: _feature_matches_pattern(row, rules.RX_T2_HASTE_SSFSE, t2_patterns),
        axis=1,
    )
    out["is_t2_tse_fse"] = out.apply(
        lambda row: _feature_matches_pattern(row, rules.RX_T2_TSE_FSE, t2_patterns),
        axis=1,
    )
    out["is_t2_mrcp_biliary"] = out.apply(
        lambda row: _feature_matches_pattern(
            row, rules.RX_T2_MRCP_BILIARY, t2_patterns
        ),
        axis=1,
    )

    return out


def score_t1(row: pd.Series) -> float:
    phase = norm_label(row.get("phase", row.get("mri_perfusion_label")))
    source = safe_str(row.get("mri_perfusion_source")) or "none"

    score = float(rules.T1_PHASE_PRIORITY.get(phase, 0))
    score += phase_selection_bonus(row)
    score += rules.T1_PHASE_SOURCE_PRIORITY.get(source, 0)

    score += {"AXIAL": 50, "CORONAL": 20, "SAGITTAL": 5}.get(row.get("plane"), 30)
    score += 25 if bool(row.get("is_3d_gre")) else 0

    # Dynamic containers are useful fallback, but explicit pure phase labels are preferred.
    if bool(row.get("is_dynamic_t1_text")) and source == "explicit_text":
        score += 5

    score += rules.DIXON_COMPONENT_PRIORITY.get(row.get("dixon_component"), 0)
    score += 8 if bool(row.get("is_resp_triggered")) else 0
    score += 5 if bool(row.get("is_breath_hold")) else 0

    score -= 150 if bool(row.get("is_subtraction")) else 0
    score -= 100 if bool(row.get("is_mip_mpr")) else 0
    score -= 200 if bool(row.get("is_quant_or_report")) else 0

    # score -= 20 if bool(row.get("SliceThickness")>5) else 0
    return float(score)


def score_t2(row: pd.Series) -> float:
    score = {"AXIAL": 50, "CORONAL": 20, "SAGITTAL": 5}.get(row.get("plane"), 30)
    score += 35 if bool(row.get("is_t2_fatsat")) else 0
    score += {"multi_shot": 20, "single_shot": 10}.get(
        row.get("mri_t2_acquisition_family"), 0
    )
    score += 8 if bool(row.get("is_resp_triggered")) else 0
    score += 5 if bool(row.get("is_breath_hold")) else 0
    score -= 60 if bool(row.get("is_t2_mrcp_biliary")) else 0
    score -= 100 if bool(row.get("is_mip_mpr")) else 0
    score -= 150 if bool(row.get("is_quant_or_report")) else 0
    return float(score)


def score_dwi(row: pd.Series) -> float:
    score = 70.0
    score += 10 if row.get("plane") == "AXIAL" else 0
    score += 10 if _row_matches_pattern(row, rules.RX_SEQUENCE_DWI) else 0
    score -= 50 if bool(row.get("is_mip_mpr")) else 0
    return score


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["t1_score"] = np.where(
        out["mri_sequence"].eq("T1"), out.apply(score_t1, axis=1), np.nan
    )
    out["t2_score"] = np.where(
        out["mri_sequence"].eq("T2"), out.apply(score_t2, axis=1), np.nan
    )
    out["dwi_score"] = np.where(
        out["mri_sequence"].eq("DWI"), out.apply(score_dwi, axis=1), np.nan
    )

    out["selection_slot"] = "OTHER"
    out.loc[out["mri_sequence"].eq("T2"), "selection_slot"] = "T2"
    out.loc[out["mri_sequence"].eq("DWI"), "selection_slot"] = "DWI"

    is_t1 = out["mri_sequence"].eq("T1")
    phase_column = "phase" if "phase" in out.columns else "mri_perfusion_label"
    t1_phase = out[phase_column].map(norm_label)
    out.loc[is_t1 & t1_phase.ne("OTHER"), "selection_slot"] = "T1_" + t1_phase
    out.loc[is_t1 & t1_phase.eq("OTHER"), "selection_slot"] = "T1_OTHER"

    out["selection_score"] = np.nan
    out.loc[out["mri_sequence"].eq("T1"), "selection_score"] = out.loc[
        out["mri_sequence"].eq("T1"), "t1_score"
    ]
    out.loc[out["mri_sequence"].eq("T2"), "selection_score"] = out.loc[
        out["mri_sequence"].eq("T2"), "t2_score"
    ]
    out.loc[out["mri_sequence"].eq("DWI"), "selection_score"] = out.loc[
        out["mri_sequence"].eq("DWI"), "dwi_score"
    ]
    return out


# -----------------------------------------------------------------------------
# Selection
# -----------------------------------------------------------------------------


def add_tiebreaker_columns(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    out = df.copy()
    out["_row_order"] = np.arange(len(out))

    if time_col in out.columns:
        out["_time_seconds"] = out[time_col].apply(parse_time_to_seconds)
    else:
        out["_time_seconds"] = np.nan

    for candidate_col in [
        "n_rows_in_volume",
        "n_sop_instances_in_volume",
        "n_rows_in_series",
        "NumberOfInstances",
    ]:
        if candidate_col in out.columns:
            out["_n_rows_proxy"] = out[candidate_col].apply(safe_float)
            break
    else:
        out["_n_rows_proxy"] = np.nan

    out["_slice_thickness"] = (
        out["SliceThickness"].apply(safe_float)
        if "SliceThickness" in out.columns
        else np.nan
    )

    if "PixelSpacing" in out.columns and not out.empty:
        spacing = out["PixelSpacing"].apply(parse_pixel_spacing).apply(pd.Series)
        spacing = spacing.reindex(columns=range(4))
        spacing.columns = ["_spacing_x", "_spacing_y", "_mean_spacing", "_pixel_area"]
        out = pd.concat([out, spacing], axis=1)
    else:
        out["_pixel_area"] = np.nan

    out["_z_coverage_proxy"] = out["_n_rows_proxy"] * out["_slice_thickness"]
    out["_volume_proxy"] = out["_z_coverage_proxy"] * out["_pixel_area"]
    return out


def select_best_candidates(
    curated: pd.DataFrame,
    patient_col: str = "patient_key",
    study_col: str | None = "study_id",
    date_col: str = "date",
    display_text_col_count: int | None = None,
    exam_group_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return selected_long and selected_wide.

    ``display_text_col_count`` limits candidate display text to the first N
    configured ``TEXT_COLS_DEFAULT`` fields. ``None`` retains all fields.
    """
    if display_text_col_count is not None:
        if isinstance(display_text_col_count, bool) or not isinstance(
            display_text_col_count, int
        ):
            raise TypeError("display_text_col_count must be a positive integer or None")
        if display_text_col_count < 1:
            raise ValueError("display_text_col_count must be at least 1 or None")

    data = curated.copy()
    if date_col in data.columns:
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce").dt.date

    data = add_tiebreaker_columns(data)
    exam_cols = get_exam_group_cols(
        data,
        patient_col=patient_col,
        study_col=study_col,
        date_col=date_col,
        exam_group_columns=exam_group_columns,
    )

    selectable = data[
        data["selection_slot"].isin(
            [
                "T2",
                "DWI",
                "T1_NATIVE",
                "T1_ARTERIAL",
                "T1_PORTAL_VENOUS",
                "T1_DELAYED",
                "T1_HEPATOBILIARY",
            ]
        )
    ].copy()

    # MIP/MPR reconstructions remain diagnostic candidates: their score penalty
    # prefers source images without leaving a phase slot empty when only a
    # reconstruction carries its phase. DERIVED alone is not an exclusion.
    # Only subtraction and quantitative/report flags exclude eligible products.
    excluded_products = selectable["is_subtraction"] | selectable["is_quant_or_report"]
    selectable = selectable.loc[~excluded_products].copy()
    selectable = selectable[selectable["selection_score"].fillna(-9999) > -500].copy()

    if selectable.empty:
        return selectable, pd.DataFrame(columns=exam_cols)

    # Stable keys retain exams with missing or aggregated identifier values.
    exam_key_cols = [f"_exam_key_{col}" for col in exam_cols]
    for col, key in zip(exam_cols, exam_key_cols):
        selectable[key] = selectable[col].apply(common.stable_text)
    exam_lookup = selectable[[*exam_key_cols, *exam_cols]].drop_duplicates(
        exam_key_cols
    )
    original_exam_cols = exam_cols
    exam_cols = exam_key_cols

    sort_cols = [
        *exam_cols,
        "selection_slot",
        "selection_score",
        "_volume_proxy",
        "_z_coverage_proxy",
        "_n_rows_proxy",
        "_time_seconds",
        "_row_order",
    ]
    ascending = [True] * len(exam_cols) + [True, False, False, False, False, True, True]

    ranked = selectable.sort_values(sort_cols, ascending=ascending, na_position="last")

    def _display(row: pd.Series) -> str:
        display_cols = _resolve_text_cols()[:display_text_col_count]
        desc = build_display_text(row, cols=display_cols)
        if (
            not desc
            and display_text_col_count is None
            and not common.has_text_columns_override()
        ):
            desc = _display_str(row.get("ProtocolName")) or row.get("series_text", "")

        details = []
        volume_order = safe_float(row.get("volume_order_in_series"))
        n_volumes = safe_float(row.get("n_volumes_in_series"))
        if pd.notna(volume_order) and pd.notna(n_volumes):
            details.append(f"vol={volume_order:g}/{n_volumes:g}")
        details.append(f"score={row.get('selection_score'):.1f}")
        return f"{desc} [{', '.join(details)}] {common.phase_evidence_summary(row)}"

    ranked["selected_candidate"] = ranked.apply(_display, axis=1)
    ranked["_candidate_rank"] = ranked.groupby(
        [*exam_cols, "selection_slot"]
    ).cumcount()

    selected_long = (
        ranked.groupby([*exam_cols, "selection_slot"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    selected_columns = (
        selected_long[[*exam_cols, "selection_slot", "selected_candidate"]]
        .pivot_table(
            index=exam_cols,
            columns="selection_slot",
            values="selected_candidate",
            aggfunc="first",
        )
        .reset_index()
    )
    other_candidates = (
        ranked.loc[ranked["_candidate_rank"] > 0]
        .groupby([*exam_cols, "selection_slot"], as_index=False)
        .agg(other_candidates=("selected_candidate", "; ".join))
        .pivot_table(
            index=exam_cols,
            columns="selection_slot",
            values="other_candidates",
            aggfunc="first",
        )
        .rename(columns=lambda slot: f"{slot}_other_candidates")
        .reset_index()
    )

    selected_wide = selected_columns.merge(other_candidates, on=exam_cols, how="left")
    selected_wide.columns.name = None

    slots = [col for col in selected_columns.columns if col not in exam_cols]
    candidate_cols = [f"{slot}_other_candidates" for slot in slots]
    for col in candidate_cols:
        if col not in selected_wide.columns:
            selected_wide[col] = pd.NA
    selected_wide = selected_wide[
        [
            *exam_cols,
            *(col for slot in slots for col in (slot, f"{slot}_other_candidates")),
        ]
    ]

    selected_wide = exam_lookup.merge(selected_wide, on=exam_cols, how="right")
    selected_wide = selected_wide.drop(columns=exam_key_cols)
    selected_wide = selected_wide[
        [
            *original_exam_cols,
            *[c for c in selected_wide if c not in original_exam_cols],
        ]
    ]
    selected_long = selected_long.drop(columns=exam_key_cols)
    return selected_long, selected_wide


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


@common.with_manifest_text_columns("MR")
def annotate_mri(
    df: pd.DataFrame,
    patient_col: str = "patient_key",
    study_col: str | None = "study_id",
    series_col: str = "series_id",
    volume_col: str = "volume_id",
    date_col: str = "date",
    phase_curation: dict | None = None,
    resolve_phase: bool = True,
) -> pd.DataFrame:
    """Add labels, features, and scores to a volume/series-level dataframe.

    Set ``resolve_phase=False`` to leave canonical phase resolution to a later
    cleaning step while still producing the modality's rule evidence.
    """
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.date

    out = add_volume_order_features(
        out,
        patient_col=patient_col,
        study_col=study_col,
        series_col=series_col,
        volume_col=volume_col,
    )
    out = add_mri_sequence_columns(out)
    out = add_basic_feature_columns(out)
    out["phase_applicability_reason"] = out.apply(
        detect_mri_phase_applicability, axis=1
    )
    exam_cols = get_exam_group_cols(
        out,
        patient_col=patient_col,
        study_col=study_col,
        date_col=date_col,
        exam_group_columns=validate_phase_curation(phase_curation)[
            "best_candidate_group_columns"
        ],
    )
    out = add_mri_perfusion_columns(
        out, exam_group_cols=exam_cols, phase_curation=phase_curation
    )
    out["rule_phase"] = out["mri_perfusion_label"]
    out["rule_phase_reason"] = out["mri_perfusion_reason"]
    out["rule_phase_confidence"] = out["mri_perfusion_confidence"]
    if resolve_phase:
        out = apply_phase_curation(out, phase_curation)
    out = add_scores(out)
    return out


@common.with_manifest_text_columns("MR")
def curate_mri(
    df: pd.DataFrame,
    patient_col: str = "patient_key",
    study_col: str | None = "study_id",
    series_col: str = "series_id",
    volume_col: str = "volume_id",
    date_col: str = "date",
    display_text_col_count: int | None = None,
    phase_curation: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """Full MRI curation pipeline.

    Set ``display_text_col_count`` to limit selected-candidate displays to the
    first N fields in ``TEXT_COLS_DEFAULT``; ``None`` displays all fields.
    """
    curated = annotate_mri(
        df,
        patient_col=patient_col,
        study_col=study_col,
        series_col=series_col,
        volume_col=volume_col,
        date_col=date_col,
        phase_curation=phase_curation,
    )
    selected_long, selected_wide = select_best_candidates(
        curated,
        patient_col=patient_col,
        study_col=study_col,
        date_col=date_col,
        display_text_col_count=display_text_col_count,
        exam_group_columns=validate_phase_curation(phase_curation)[
            "best_candidate_group_columns"
        ],
    )

    return {
        "curated": curated,
        "t1_candidates": curated[curated["mri_sequence"].eq("T1")].copy(),
        "t2_candidates": curated[curated["mri_sequence"].eq("T2")].copy(),
        "dwi_candidates": curated[curated["mri_sequence"].eq("DWI")].copy(),
        "selected_long": selected_long,
        "selected_wide": selected_wide,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MRI curation")
    parser.add_argument("input_csv")
    parser.add_argument("--output-prefix", default="mri_curated")
    args = parser.parse_args()

    df = read_csv(args.input_csv)
    results = curate_mri(df)
    results["curated"].to_csv(f"{args.output_prefix}_all.csv", index=False)
    results["selected_long"].to_csv(
        f"{args.output_prefix}_selected_long.csv", index=False
    )
    results["selected_wide"].to_csv(
        f"{args.output_prefix}_selected_wide.csv", index=False
    )
