"""
MRI curation utilities.

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
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import rules

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


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(Path(path).expanduser(), low_memory=False, **kwargs)


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


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


def clean_text(x) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())


def safe_str(x) -> str:
    if isinstance(x, (list, tuple, set, np.ndarray)):
        return clean_text(" ".join(safe_str(v) for v in x if safe_str(v)))
    return clean_text(x) if pd.notna(x) else ""


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
    return list(TEXT_COLS_DEFAULT if cols is None else cols)


def get_dixon_text_cols(cols: Sequence[str] | None = None) -> list[str]:
    return [col for col in _resolve_text_cols(cols) if col != "ImageType"]


def iter_series_text_columns(
    row: pd.Series,
    cols: Sequence[str] | None = None,
):
    for col in _resolve_text_cols(cols):
        if col not in row.index:
            continue
        text = safe_str(row.get(col))
        if text:
            yield col, text


def _first_text_column_value(
    row: pd.Series,
    evaluator,
    cols: Sequence[str] | None = None,
):
    for _, text in iter_series_text_columns(row, cols=cols):
        value = evaluator(text)
        if value is not None:
            return value
    return None


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


def _row_matches_any_patterns(
    row: pd.Series,
    patterns: Sequence[str],
    cols: Sequence[str] | None = None,
) -> bool:
    return bool(
        _first_text_column_value(
            row,
            lambda text: True if any(re.search(pattern, text) for pattern in patterns) else None,
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
    """Use all useful text fields, not only SeriesDescription."""
    cols = _resolve_text_cols(cols)
    parts = [safe_str(row.get(c)) for c in cols if c in row.index]
    return " | ".join(part for part in parts if part)


def build_display_text(row: pd.Series, cols: Sequence[str] | None = None) -> str:
    """Return display text from configured raw text columns."""
    cols = _resolve_text_cols(cols)
    parts = [_display_str(row.get(c)) for c in cols if c in row.index]
    return " | ".join(part for part in parts if part)


def get_exam_group_cols(
    df: pd.DataFrame,
    patient_col: str = "patient_key",
    study_col: str | None = "study_id",
    date_col: str = "date",
) -> list[str]:
    cols = [patient_col]
    if study_col is not None and study_col in df.columns:
        cols.append(study_col)
    if date_col in df.columns:
        cols.append(date_col)
    return [c for c in cols if c in df.columns]


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
        return out

    if series_col not in out.columns or volume_col not in out.columns:
        out["volume_order_in_series"] = 1
        out["n_volumes_in_series"] = 1
        out["is_multivolume_series"] = False
        return out

    series_group_cols = [
        c for c in [patient_col, study_col, series_col] if c is not None and c in out.columns
    ]
    volume_group_cols = [*series_group_cols, volume_col]

    work = out.copy()
    work["_sort_time_seconds"] = (
        work[time_col].apply(parse_time_to_seconds) if time_col in work.columns else np.nan
    )
    work["_row_order_for_volume_order"] = np.arange(len(work))

    for col in ["AcquisitionNumber", "InstanceNumber", "volume_id"]:
        if col not in work.columns:
            work[col] = np.nan

    work["_sort_acquisition_number"] = work["AcquisitionNumber"].apply(_first_numeric)
    work["_sort_instance_number"] = work["InstanceNumber"].apply(_first_numeric)
    work["_sort_volume_id"] = work[volume_col].apply(_stable_text)
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
                "_sort_acquisition_number",
                "_sort_instance_number",
                "_sort_volume_id",
            ],
            na_position="last",
        )
        .drop_duplicates("_volume_group_key")
        .copy()
    )

    rep["volume_order_in_series"] = (
        rep.groupby("_series_group_key", dropna=False).cumcount() + 1
    )
    rep["n_volumes_in_series"] = rep.groupby("_series_group_key", dropna=False)["_volume_group_key"].transform("size")
    rep["is_multivolume_series"] = rep["n_volumes_in_series"] > 1

    order_cols = [
        "_volume_group_key",
        "volume_order_in_series",
        "n_volumes_in_series",
        "is_multivolume_series",
    ]

    out = out.drop(
        columns=[
            c for c in [
                "volume_order_in_series",
                "n_volumes_in_series",
                "is_multivolume_series",
            ] if c in out.columns
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
    return out


# -----------------------------------------------------------------------------
# Sequence classification
# -----------------------------------------------------------------------------


def detect_mri_sequence(row: pd.Series) -> tuple[str, str, str]:
    modality = safe_str(row.get("Modality")).upper()

    text_match = _first_text_column_value(
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
    out[["mri_sequence", "mri_sequence_reason", "mri_sequence_confidence"]] = pd.DataFrame(
        result.tolist(), index=out.index
    )
    return out


# -----------------------------------------------------------------------------
# T1 perfusion phase classification
# -----------------------------------------------------------------------------


def text_matches_art_port(row: pd.Series) -> bool:
    return _row_matches_pattern(row, rules.RX_PHASE_ART_PORT_DYNAMIC)


def text_matches_art_port_late(row: pd.Series) -> bool:
    return _row_matches_all_patterns(
        row,
        required_patterns=[
            rules.RX_PHASE_ARTERIAL,
            rules.RX_PHASE_PORTAL,
            rules.RX_PHASE_DELAYED,
        ],
        excluded_patterns=[rules.RX_PHASE_HEPATOBILIARY],
    )


def text_matches_mask_multiart(row: pd.Series) -> bool:
    return _row_matches_pattern(row, rules.RX_PHASE_MASK_MULTIART_DYNAMIC)


def has_post_contrast_text(row: pd.Series) -> bool:
    return _row_matches_pattern(row, rules.RX_PHASE_POST_CONTRAST)


def detect_ordinal_phase_index(row: pd.Series) -> int | None:
    return _first_text_column_value(
        row,
        lambda text: next(
            (
                int(group)
                for group in re.search(rules.RX_PHASE_ORDINAL, text).groups()
                if group is not None
            ),
            None,
        )
        if re.search(rules.RX_PHASE_ORDINAL, text)
        else None,
    )


def detect_explicit_phase_from_text(row: pd.Series) -> tuple[str | None, str, str, str]:
    match = _first_text_column_value(
        row,
        lambda text: (
            ("NATIVE", "matched explicit native/non-injected keyword", "explicit", "explicit_text")
            if re.search(rules.RX_PHASE_NATIVE, text)
            else (
                (
                    "HEPATOBILIARY",
                    "matched explicit hepatobiliary/2h keyword",
                    "explicit",
                    "explicit_text",
                )
                if re.search(rules.RX_PHASE_HEPATOBILIARY, text)
                else (
                    (
                        "PORTAL_VENOUS",
                        "matched explicit portal/venous keyword",
                        "explicit",
                        "explicit_text",
                    )
                    if re.search(rules.RX_PHASE_PORTAL, text)
                    else (
                        ("ARTERIAL", "matched explicit arterial keyword", "explicit", "explicit_text")
                        if re.search(rules.RX_PHASE_ARTERIAL, text)
                        else (
                            ("DELAYED", "matched explicit delayed/tardif keyword", "explicit", "explicit_text")
                            if re.search(rules.RX_PHASE_DELAYED, text)
                            else None
                        )
                    )
                )
            )
        ),
    )
    if match is not None:
        return match

    return None, "no explicit T1 perfusion phase keyword matched", "unknown", "none"


def infer_special_t1_phase_from_volume_order(row: pd.Series) -> tuple[str | None, str, str, str]:
    """Special same-description/multiple-volume protocols."""
    if norm_label(row.get("mri_sequence")) != "T1":
        return None, "not T1", "unknown", "none"

    order = safe_float(row.get("volume_order_in_series"))
    n_volumes = safe_float(row.get("n_volumes_in_series"))
    if pd.isna(order) or pd.isna(n_volumes) or n_volumes < 2:
        return None, "no independent second volume detected", "unknown", "none"

    order = int(order)
    n_volumes = int(n_volumes)

    if text_matches_art_port_late(row):
        if order == 1:
            return (
                "ARTERIAL",
                f"inferred ARTERIAL from first ART/PORT/LATE volume {order}/{n_volumes}",
                "inferred",
                "volume_order_art_port_late",
            )
        if order == 2:
            return (
                "PORTAL_VENOUS",
                f"inferred PORTAL_VENOUS from second ART/PORT/LATE volume {order}/{n_volumes}",
                "inferred",
                "volume_order_art_port_late",
            )
        return (
            "DELAYED",
            f"inferred DELAYED from later ART/PORT/LATE volume {order}/{n_volumes}",
            "inferred",
            "volume_order_art_port_late",
        )

    if text_matches_art_port(row):
        if order == 1:
            return (
                "ARTERIAL",
                f"inferred ARTERIAL from first ART-PORT volume {order}/{n_volumes}",
                "inferred",
                "volume_order_art_port",
            )
        if order == 2:
            return (
                "PORTAL_VENOUS",
                f"inferred PORTAL_VENOUS from second ART-PORT volume {order}/{n_volumes}",
                "inferred",
                "volume_order_art_port",
            )
        return (
            "DELAYED",
            f"inferred DELAYED from later ART-PORT volume {order}/{n_volumes}",
            "inferred",
            "volume_order_art_port",
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


def has_generic_dynamic_t1_evidence(row: pd.Series) -> bool:
    if norm_label(row.get("mri_sequence")) != "T1":
        return False

    n_volumes = safe_float(row.get("n_volumes_in_series"))
    if pd.isna(n_volumes) or n_volumes < 3:
        return False

    return _row_matches_pattern(row, rules.RX_PHASE_GENERIC_DYNAMIC)


def infer_generic_t1_phase_from_volume_order(row: pd.Series) -> tuple[str | None, str, str, str]:
    if not has_generic_dynamic_t1_evidence(row):
        return None, "no generic dynamic multivolume T1 evidence", "unknown", "none"

    order = safe_float(row.get("volume_order_in_series"))
    n_volumes = safe_float(row.get("n_volumes_in_series"))
    if pd.isna(order) or pd.isna(n_volumes):
        return None, "missing volume order", "unknown", "none"

    order = int(order)
    n_volumes = int(n_volumes)

    mapping = {
        1: "NATIVE",
        2: "ARTERIAL",
        3: "PORTAL_VENOUS",
    }
    label = mapping.get(order, "DELAYED")
    return (
        label,
        f"inferred {label} from generic dynamic T1 volume order {order}/{n_volumes}",
        "inferred",
        "volume_order",
    )


def detect_t1_perfusion_phase(row: pd.Series) -> tuple[str, str, str, str]:
    """
    Volume-aware T1 phase classifier.

    Priority:
      1. Special multivolume ART/PORT/LATE, ART-PORT, or Mask+Multiart order inference.
      2. Defer single-volume special dynamic candidates to exam context.
      3. Explicit pure phase text, e.g. SANS IV, ART, PORT, TARDIF.
      4. Generic dynamic volume order inference.
      5. OTHER.
    """
    seq = norm_label(row.get("mri_sequence"))

    if seq != "T1":
        return "OTHER", f"sequence={seq}; phase not assigned", "unknown", "none"

    # Derived/subtraction rows are kept as T1 candidates but not valid phase labels.
    if (
        _row_matches_pattern(row, rules.RX_SUBTRACTION)
        or _row_matches_pattern(row, rules.RX_MIP_MPR)
        or _row_matches_pattern(row, rules.RX_QUANT_OR_REPORT)
    ):
        return "OTHER", "matched subtraction/derived/non-diagnostic marker", "unknown", "none"

    if text_matches_art_port_late(row):
        special_label, special_reason, special_conf, special_source = infer_special_t1_phase_from_volume_order(row)
        if special_label is not None:
            return special_label, special_reason, special_conf, special_source
        return (
            "OTHER",
            "matched single-volume ART/PORT/LATE text; awaiting exam acquisition context",
            "unknown",
            ART_PORT_LATE_CONTEXT_PENDING,
        )

    if text_matches_art_port(row):
        special_label, special_reason, special_conf, special_source = infer_special_t1_phase_from_volume_order(row)
        if special_label is not None:
            return special_label, special_reason, special_conf, special_source
        return (
            "OTHER",
            "matched single-volume ART-PORT text; awaiting exam acquisition context",
            "unknown",
            ART_PORT_CONTEXT_PENDING,
        )

    if text_matches_mask_multiart(row):
        special_label, special_reason, special_conf, special_source = infer_special_t1_phase_from_volume_order(row)
        if special_label is not None:
            return special_label, special_reason, special_conf, special_source
        return (
            "OTHER",
            "matched single-volume Mask+Multiart text; awaiting exam acquisition context",
            "unknown",
            MASK_MULTIART_CONTEXT_PENDING,
        )

    explicit_label, explicit_reason, explicit_conf, explicit_source = detect_explicit_phase_from_text(row)
    if explicit_label is not None:
        return explicit_label, explicit_reason, explicit_conf, explicit_source

    generic_label, generic_reason, generic_conf, generic_source = infer_generic_t1_phase_from_volume_order(row)
    if generic_label is not None:
        return generic_label, generic_reason, generic_conf, generic_source

    ordinal_index = detect_ordinal_phase_index(row)
    if ordinal_index is not None:
        return (
            "OTHER",
            f"ordinal phase Ph{ordinal_index} detected but exam context has not resolved it",
            "unknown",
            "ordinal_context",
        )

    return "OTHER", "no supported T1 perfusion phase rule matched", "unknown", "none"


def infer_phase_from_ordinal_context(
    row: pd.Series,
    exam_rows: pd.DataFrame,
) -> tuple[str | None, str, str, str]:
    ordinal_index = detect_ordinal_phase_index(row)
    if ordinal_index is None:
        return None, "no ordinal phase index detected", "unknown", "none"

    if norm_label(row.get("mri_sequence")) != "T1":
        return None, "ordinal phase ignored because sequence is not T1", "unknown", "ordinal_context"

    has_dynamic_text = _row_matches_any_patterns(
        row,
        [rules.RX_T1_DYNAMIC, rules.RX_T1_3D_GRE],
    )
    has_post_text = has_post_contrast_text(row)
    exam_has_post_ordinal = bool(
        exam_rows.apply(
            lambda r: detect_ordinal_phase_index(r) is not None and has_post_contrast_text(r),
            axis=1,
        ).any()
    )
    exam_has_explicit_native = bool(
        exam_rows["mri_perfusion_label"].map(norm_label).eq("NATIVE").any()
        and exam_rows["mri_perfusion_source"].eq("explicit_text").any()
    )
    exam_has_native_fallback = bool(
        exam_rows.apply(is_native_fallback_candidate, axis=1).any()
    )

    if has_post_text or (has_dynamic_text and exam_has_post_ordinal):
        mapping = {1: "ARTERIAL", 2: "PORTAL_VENOUS", 3: "DELAYED"}
        label = mapping.get(ordinal_index, "DELAYED")
        context = (
            "with explicit/fallback native context"
            if exam_has_explicit_native or exam_has_native_fallback
            else "from post-contrast dynamic ordinal context"
        )
        return (
            label,
            f"inferred {label} from Ph{ordinal_index} {context}",
            "inferred",
            "ordinal_context",
        )

    return (
        None,
        f"ordinal phase Ph{ordinal_index} detected but context is insufficient",
        "unknown",
        "ordinal_context",
    )


def _acquisition_sort_key(row: pd.Series, row_order: int) -> tuple:
    acquisition_order = _first_numeric(row.get("acquisition_order"))
    acquisition_number = _first_numeric(row.get("AcquisitionNumber"))
    acquisition_time = parse_time_to_seconds(row.get("time"))
    series_number = _first_numeric(row.get("SeriesNumber"))
    return (
        acquisition_order if pd.notna(acquisition_order) else np.inf,
        acquisition_number if pd.notna(acquisition_number) else np.inf,
        acquisition_time if pd.notna(acquisition_time) else np.inf,
        series_number if pd.notna(series_number) else np.inf,
        row_order,
    )


def _infer_special_profile_phases_by_acquisition_order(
    exam_rows: pd.DataFrame,
    *,
    candidate_sources: set[str],
    profile_name: str,
    rank_to_label,
) -> list[tuple[int, str, str]]:
    """Resolve special dynamic profiles by acquisition order within Dixon component."""
    candidates = exam_rows.loc[
        exam_rows["mri_perfusion_source"].isin(candidate_sources)
    ].copy()
    if candidates.empty:
        return []

    n_volumes = candidates.get(
        "n_volumes_in_series", pd.Series(1, index=candidates.index)
    ).apply(safe_float)
    candidates["_is_single_volume"] = n_volumes.eq(1)
    candidates["_is_multivolume"] = n_volumes.gt(1)

    row_order = {idx: order for order, idx in enumerate(candidates.index)}
    component = (
        candidates["dixon_component"].fillna("UNKNOWN")
        if "dixon_component" in candidates.columns
        else pd.Series("UNKNOWN", index=candidates.index)
    )
    assignments = []
    for component_name, component_rows in candidates.groupby(component, sort=False):
        has_single_volume = bool(component_rows["_is_single_volume"].any())
        has_multivolume = bool(component_rows["_is_multivolume"].any())
        if not has_single_volume:
            continue
        if not has_multivolume and len(component_rows) < 2:
            continue

        ranked = sorted(
            component_rows.index,
            key=lambda row_idx: _acquisition_sort_key(
                component_rows.loc[row_idx], row_order[row_idx]
            ),
        )
        context = (
            "mixed multivolume and single-volume matches"
            if has_multivolume
            else "single-volume series"
        )
        for rank, row_idx in enumerate(ranked, start=1):
            label = rank_to_label(rank)
            assignments.append((
                row_idx,
                label,
                (
                    f"inferred {label} from {profile_name} acquisition {rank}/{len(ranked)} "
                    f"for {component_name} {context}"
                ),
            ))
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
        rank_to_label=lambda rank: {
            1: "ARTERIAL",
            2: "PORTAL_VENOUS",
        }.get(rank, "DELAYED"),
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
        rank_to_label=lambda rank: {
            1: "ARTERIAL",
            2: "PORTAL_VENOUS",
        }.get(rank, "DELAYED"),
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
        rank_to_label=lambda rank: "NATIVE" if rank == 1 else "ARTERIAL",
    )


def infer_generic_dynamic_phases_from_exam_context(
    exam_rows: pd.DataFrame,
) -> list[tuple[int, str, str]]:
    """Infer generic dynamic phases within each available Dixon component."""

    def _is_candidate(row: pd.Series) -> bool:
        if norm_label(row.get("mri_sequence")) != "T1":
            return False
        if safe_str(row.get("mri_perfusion_source")) not in {"none", "volume_order"}:
            return False
        if not _row_matches_pattern(row, rules.RX_PHASE_GENERIC_DYNAMIC):
            return False
        if text_matches_art_port(row) or text_matches_mask_multiart(row):
            return False
        if detect_ordinal_phase_index(row) is not None:
            return False
        if (
            _row_matches_pattern(row, rules.RX_SUBTRACTION)
            or _row_matches_pattern(row, rules.RX_MIP_MPR)
            or _row_matches_pattern(row, rules.RX_QUANT_OR_REPORT)
        ):
            return False
        explicit_label, *_ = detect_explicit_phase_from_text(row)
        return explicit_label is None

    candidates = exam_rows.loc[exam_rows.apply(_is_candidate, axis=1)].copy()
    if candidates.empty:
        return []

    supported_components = {"WATER", "FAT", "IN_PHASE", "OPPOSED_PHASE"}
    components = candidates.get(
        "dixon_component", pd.Series("DIXON_UNKNOWN", index=candidates.index)
    ).fillna("DIXON_UNKNOWN")
    if components.isin(supported_components).any():
        group_keys = components.where(
            components.isin(supported_components), "DIXON_UNSPECIFIED"
        )
    else:
        group_keys = pd.Series("DYNAMIC", index=candidates.index)

    row_order = {idx: order for order, idx in enumerate(candidates.index)}
    assignments = []
    for component, component_rows in candidates.groupby(group_keys, sort=False):
        if len(component_rows) < 3:
            continue
        acquisition_order = component_rows.get(
            "acquisition_order", pd.Series(np.nan, index=component_rows.index)
        ).apply(_first_numeric)
        if acquisition_order.isna().any():
            continue
        ranked = sorted(
            component_rows.index,
            key=lambda row_idx: _acquisition_sort_key(
                component_rows.loc[row_idx], row_order[row_idx]
            ),
        )
        for rank, row_idx in enumerate(ranked, start=1):
            label = {1: "NATIVE", 2: "ARTERIAL", 3: "PORTAL_VENOUS"}.get(
                rank, "DELAYED"
            )
            assignments.append((
                row_idx,
                label,
                (
                    f"inferred {label} from generic dynamic acquisition {rank}/{len(ranked)} "
                    f"within Dixon component {component}"
                ),
            ))
    return assignments


def is_native_fallback_candidate(row: pd.Series) -> bool:
    if norm_label(row.get("mri_sequence")) != "T1":
        return False
    if norm_label(row.get("mri_perfusion_label")) != "OTHER":
        return False

    if (
        _row_matches_pattern(row, rules.RX_SUBTRACTION)
        or _row_matches_pattern(row, rules.RX_MIP_MPR)
        or _row_matches_pattern(row, rules.RX_QUANT_OR_REPORT)
        or has_post_contrast_text(row)
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
        or _row_matches_pattern(row, rules.RX_T1_3D_GRE)
        or dixon_component in {"WATER", "IN_PHASE", "DIXON_UNKNOWN"}
    )


def _exam_has_post_contrast_dynamic_phase(exam_rows: pd.DataFrame) -> bool:
    phase = exam_rows["mri_perfusion_label"].map(norm_label)
    source = exam_rows["mri_perfusion_source"].fillna("none")
    resolved_dynamic = phase.isin(["ARTERIAL", "PORTAL_VENOUS", "DELAYED"]) & source.isin(
        [
            "ordinal_context",
            "acquisition_order_art_port_late",
            "acquisition_order_art_port",
            "acquisition_order_mask_multiart",
            "acquisition_order_dixon_component",
            "volume_order_art_port_late",
            "volume_order",
            "volume_order_art_port",
            "volume_order_mask_multiart",
        ]
    )
    post_text_dynamic = exam_rows.apply(
        lambda r: has_post_contrast_text(r)
        and detect_ordinal_phase_index(r) is not None
        and norm_label(r.get("mri_sequence")) == "T1",
        axis=1,
    )
    return bool(resolved_dynamic.any() or post_text_dynamic.any())


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
    quality_score += 3 if bool(row.get("is_3d_gre")) or _row_matches_pattern(row, rules.RX_T1_3D_GRE) else 0
    quality_score += 1 if bool(row.get("is_breath_hold")) else 0

    time_seconds = parse_time_to_seconds(row.get("time"))
    series_number = safe_float(row.get("SeriesNumber"))
    acquisition_number = safe_float(row.get("AcquisitionNumber"))
    order = min(
        [x for x in [time_seconds, series_number, acquisition_number] if pd.notna(x)]
        or [np.inf]
    )
    return (-quality_score, -component_score, order)


def infer_missing_native_fallback(exam_rows: pd.DataFrame) -> tuple[int | None, str]:
    if exam_rows["mri_perfusion_label"].map(norm_label).eq("NATIVE").any():
        return None, "native/precontrast already resolved in exam"
    if not _exam_has_post_contrast_dynamic_phase(exam_rows):
        return None, "no post-contrast dynamic context for native fallback"

    candidates = exam_rows.loc[exam_rows.apply(is_native_fallback_candidate, axis=1)]
    if candidates.empty:
        return None, "no suitable native fallback candidate"

    ranked = sorted(candidates.index, key=lambda idx: _sort_key_for_native_fallback(candidates.loc[idx]))
    idx = ranked[0]
    desc = candidates.loc[idx].get("SeriesDescription")
    return (
        idx,
        (
            "selected fallback native/precontrast from exam context because no explicit "
            f"native series was found and post-contrast dynamic phases exist: {desc}"
        ),
    )


def add_mri_perfusion_columns(
    df: pd.DataFrame,
    exam_group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    result = out.apply(detect_t1_perfusion_phase, axis=1)
    out[["mri_perfusion_label", "mri_perfusion_reason", "mri_perfusion_confidence", "mri_perfusion_source"]] = pd.DataFrame(
        result.tolist(), index=out.index
    )

    group_cols = [c for c in (exam_group_cols or []) if c in out.columns]
    if group_cols:
        grouped = out.groupby(group_cols, dropna=False).groups.values()
    else:
        grouped = [out.index]

    for idx in grouped:
        exam_rows = out.loc[idx].copy()

        for row_idx, label, reason in infer_art_port_late_phases_by_acquisition_order(
            exam_rows
        ):
            out.loc[row_idx, "mri_perfusion_label"] = label
            out.loc[row_idx, "mri_perfusion_reason"] = reason
            out.loc[row_idx, "mri_perfusion_confidence"] = "inferred"
            out.loc[row_idx, "mri_perfusion_source"] = (
                "acquisition_order_art_port_late"
            )

        unresolved_art_port_late = out.loc[idx, "mri_perfusion_source"].eq(
            ART_PORT_LATE_CONTEXT_PENDING
        )
        for row_idx in out.loc[idx].index[unresolved_art_port_late]:
            out.loc[row_idx, "mri_perfusion_label"] = "DELAYED"
            out.loc[row_idx, "mri_perfusion_reason"] = (
                "ART/PORT/LATE exam context did not resolve multiple single-volume "
                "acquisitions; treated as delayed"
            )
            out.loc[row_idx, "mri_perfusion_confidence"] = "inferred"
            out.loc[row_idx, "mri_perfusion_source"] = (
                "explicit_text_art_port_late_single"
            )

        exam_rows = out.loc[idx].copy()

        for row_idx, label, reason in infer_art_port_phases_by_acquisition_order(
            exam_rows
        ):
            out.loc[row_idx, "mri_perfusion_label"] = label
            out.loc[row_idx, "mri_perfusion_reason"] = reason
            out.loc[row_idx, "mri_perfusion_confidence"] = "inferred"
            out.loc[row_idx, "mri_perfusion_source"] = "acquisition_order_art_port"

        unresolved_art_port = out.loc[idx, "mri_perfusion_source"].eq(ART_PORT_CONTEXT_PENDING)
        for row_idx in out.loc[idx].index[unresolved_art_port]:
            out.loc[row_idx, "mri_perfusion_label"] = "PORTAL_VENOUS"
            out.loc[row_idx, "mri_perfusion_reason"] = (
                "ART-PORT exam context did not resolve two single-volume acquisitions; "
                "treated as portal/transition"
            )
            out.loc[row_idx, "mri_perfusion_confidence"] = "inferred"
            out.loc[row_idx, "mri_perfusion_source"] = "explicit_text_art_port_single"

        exam_rows = out.loc[idx].copy()

        for row_idx, label, reason in infer_mask_multiart_phases_by_acquisition_order(
            exam_rows
        ):
            out.loc[row_idx, "mri_perfusion_label"] = label
            out.loc[row_idx, "mri_perfusion_reason"] = reason
            out.loc[row_idx, "mri_perfusion_confidence"] = "inferred"
            out.loc[row_idx, "mri_perfusion_source"] = (
                "acquisition_order_mask_multiart"
            )

        unresolved_mask_multiart = out.loc[idx, "mri_perfusion_source"].eq(
            MASK_MULTIART_CONTEXT_PENDING
        )
        for row_idx in out.loc[idx].index[unresolved_mask_multiart]:
            out.loc[row_idx, "mri_perfusion_label"] = "ARTERIAL"
            out.loc[row_idx, "mri_perfusion_reason"] = (
                "Mask+Multiart exam context did not contain multiple single-volume "
                "acquisitions; treated as arterial"
            )
            out.loc[row_idx, "mri_perfusion_confidence"] = "inferred"
            out.loc[row_idx, "mri_perfusion_source"] = (
                "explicit_text_mask_multiart_single"
            )

        exam_rows = out.loc[idx].copy()

        for row_idx, label, reason in infer_generic_dynamic_phases_from_exam_context(
            exam_rows
        ):
            out.loc[row_idx, "mri_perfusion_label"] = label
            out.loc[row_idx, "mri_perfusion_reason"] = reason
            out.loc[row_idx, "mri_perfusion_confidence"] = "inferred"
            out.loc[row_idx, "mri_perfusion_source"] = (
                "acquisition_order_dixon_component"
            )

        exam_rows = out.loc[idx].copy()

        for row_idx, row in exam_rows.iterrows():
            if norm_label(row.get("mri_perfusion_label")) != "OTHER":
                continue
            label, reason, confidence, source = infer_phase_from_ordinal_context(row, exam_rows)
            if label is None:
                if source == "ordinal_context":
                    out.loc[row_idx, "mri_perfusion_reason"] = reason
                    out.loc[row_idx, "mri_perfusion_confidence"] = confidence
                    out.loc[row_idx, "mri_perfusion_source"] = source
                continue
            out.loc[row_idx, "mri_perfusion_label"] = label
            out.loc[row_idx, "mri_perfusion_reason"] = reason
            out.loc[row_idx, "mri_perfusion_confidence"] = confidence
            out.loc[row_idx, "mri_perfusion_source"] = source

        exam_rows = out.loc[idx].copy()
        fallback_idx, reason = infer_missing_native_fallback(exam_rows)
        if fallback_idx is not None:
            logger.info("Fallback native selected for MRI exam: %s", reason)
            out.loc[fallback_idx, "mri_perfusion_label"] = "NATIVE"
            out.loc[fallback_idx, "mri_perfusion_reason"] = reason
            out.loc[fallback_idx, "mri_perfusion_confidence"] = "fallback"
            out.loc[fallback_idx, "mri_perfusion_source"] = "exam_context"

    return out


# -----------------------------------------------------------------------------
# Feature extraction and scoring
# -----------------------------------------------------------------------------


def detect_plane(value: pd.Series | str, cols: Sequence[str] | None = None) -> str:
    if isinstance(value, pd.Series):
        plane = _first_text_column_value(
            value,
            lambda text: (
                "AXIAL"
                if re.search(rules.RX_PLANE_AXIAL, text)
                else (
                    "CORONAL"
                    if re.search(rules.RX_PLANE_CORONAL, text)
                    else (
                        "SAGITTAL" if re.search(rules.RX_PLANE_SAGITTAL, text) else None
                    )
                )
            ),
            cols=cols,
        )
        return plane or "UNKNOWN"

    text = safe_str(value)
    if re.search(rules.RX_PLANE_AXIAL, text):
        return "AXIAL"
    if re.search(rules.RX_PLANE_CORONAL, text):
        return "CORONAL"
    if re.search(rules.RX_PLANE_SAGITTAL, text):
        return "SAGITTAL"
    return "UNKNOWN"


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
    return {
        token.upper()
        for token in re.split(r"[\s_.+\-/]+", text)
        if token
    }


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
        suffix_tokens = _free_text_component_tokens(text)
        for token, component in {
            "W": "WATER",
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
        return "DIXON_UNKNOWN", "Dixon context detected without component", "dixon_context"
    return None


def detect_dixon_component(row: pd.Series) -> tuple[str, str, str]:
    """Return normalized Dixon component, reason, and evidence source."""
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

    text_match = _first_text_column_value(
        row,
        lambda text: (
            ("FAT_FRACTION", "matched explicit fat-fraction/PDFF text", "explicit_text")
            if re.search(rules.RX_DIXON_FAT_FRACTION, text)
            else (
                ("R2STAR", "matched explicit R2*/T2* map text", "explicit_text")
                if re.search(rules.RX_DIXON_R2STAR, text)
                else (
                    ("DIXON_ALL", "matched explicit all-reconstructions text", "explicit_text")
                    if re.search(rules.RX_DIXON_ALL, text)
                    else _detect_dixon_component_from_text(text)
                )
            )
        ),
        cols=get_dixon_text_cols(),
    )
    if text_match is not None:
        return text_match

    return "NOT_DIXON", "no Dixon context or component token", "none"


def add_basic_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["series_text"] = out.apply(build_series_text, axis=1)
    out["plane"] = out.apply(detect_plane, axis=1)

    out["is_subtraction"] = out.apply(lambda row: _row_matches_pattern(row, rules.RX_SUBTRACTION), axis=1)
    out["is_mip_mpr"] = out.apply(lambda row: _row_matches_pattern(row, rules.RX_MIP_MPR), axis=1)
    out["is_quant_or_report"] = out.apply(
        lambda row: _row_matches_pattern(row, rules.RX_QUANT_OR_REPORT),
        axis=1,
    )

    # T1 features.
    dixon = out.apply(detect_dixon_component, axis=1)
    out[["dixon_component", "dixon_component_reason", "dixon_component_source"]] = (
        pd.DataFrame(dixon.tolist(), index=out.index)
    )
    out["is_3d_gre"] = out.apply(lambda row: _row_matches_pattern(row, rules.RX_T1_3D_GRE), axis=1)
    out["is_dynamic_t1_text"] = out.apply(
        lambda row: _row_matches_pattern(row, rules.RX_T1_DYNAMIC),
        axis=1,
    )
    out["is_breath_hold"] = out.apply(lambda row: _row_matches_pattern(row, rules.RX_BREATH_HOLD), axis=1)
    out["is_resp_triggered"] = out.apply(
        lambda row: _row_matches_pattern(row, rules.RX_RESP_TRIGGERED),
        axis=1,
    )

    # T2 features.
    out["is_t2_fatsat"] = out.apply(lambda row: _row_matches_pattern(row, rules.RX_T2_FATSAT), axis=1)
    out["is_t2_motion_robust"] = out.apply(
        lambda row: _row_matches_pattern(row, rules.RX_T2_MOTION_ROBUST),
        axis=1,
    )
    out["is_t2_haste_ssfse"] = out.apply(
        lambda row: _row_matches_pattern(row, rules.RX_T2_HASTE_SSFSE),
        axis=1,
    )
    out["is_t2_tse_fse"] = out.apply(lambda row: _row_matches_pattern(row, rules.RX_T2_TSE_FSE), axis=1)
    out["is_t2_mrcp_biliary"] = out.apply(
        lambda row: _row_matches_pattern(row, rules.RX_T2_MRCP_BILIARY),
        axis=1,
    )

    return out


def score_t1(row: pd.Series) -> float:
    phase = norm_label(row.get("mri_perfusion_label"))
    source = safe_str(row.get("mri_perfusion_source")) or "none"

    score = float(rules.T1_PHASE_PRIORITY.get(phase, 0))
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

    #score -= 20 if bool(row.get("SliceThickness")>5) else 0
    return float(score)


def score_t2(row: pd.Series) -> float:
    score = {"AXIAL": 50, "CORONAL": 20, "SAGITTAL": 5}.get(row.get("plane"), 30)
    score += 35 if bool(row.get("is_t2_fatsat")) else 0
    score += 35 if bool(row.get("is_t2_motion_robust")) else 0
    score += 15 if bool(row.get("is_t2_haste_ssfse")) else 0
    score += 10 if bool(row.get("is_t2_tse_fse")) else 0
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
    out["t1_score"] = np.where(out["mri_sequence"].eq("T1"), out.apply(score_t1, axis=1), np.nan)
    out["t2_score"] = np.where(out["mri_sequence"].eq("T2"), out.apply(score_t2, axis=1), np.nan)
    out["dwi_score"] = np.where(out["mri_sequence"].eq("DWI"), out.apply(score_dwi, axis=1), np.nan)

    out["selection_slot"] = "OTHER"
    out.loc[out["mri_sequence"].eq("T2"), "selection_slot"] = "T2"
    out.loc[out["mri_sequence"].eq("DWI"), "selection_slot"] = "DWI"

    is_t1 = out["mri_sequence"].eq("T1")
    t1_phase = out["mri_perfusion_label"].map(norm_label)
    out.loc[is_t1 & t1_phase.ne("OTHER"), "selection_slot"] = "T1_" + t1_phase
    out.loc[is_t1 & t1_phase.eq("OTHER"), "selection_slot"] = "T1_OTHER"

    out["selection_score"] = np.nan
    out.loc[out["mri_sequence"].eq("T1"), "selection_score"] = out.loc[out["mri_sequence"].eq("T1"), "t1_score"]
    out.loc[out["mri_sequence"].eq("T2"), "selection_score"] = out.loc[out["mri_sequence"].eq("T2"), "t2_score"]
    out.loc[out["mri_sequence"].eq("DWI"), "selection_score"] = out.loc[out["mri_sequence"].eq("DWI"), "dwi_score"]
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

    for candidate_col in ["n_rows_in_volume", "n_sop_instances_in_volume", "n_rows_in_series", "NumberOfInstances"]:
        if candidate_col in out.columns:
            out["_n_rows_proxy"] = out[candidate_col].apply(safe_float)
            break
    else:
        out["_n_rows_proxy"] = np.nan

    out["_slice_thickness"] = out["SliceThickness"].apply(safe_float) if "SliceThickness" in out.columns else np.nan

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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return selected_long and selected_wide."""
    data = curated.copy()
    if date_col in data.columns:
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce").dt.date

    data = add_tiebreaker_columns(data)
    exam_cols = get_exam_group_cols(data, patient_col=patient_col, study_col=study_col, date_col=date_col)

    selectable = data[
        data["selection_slot"].isin([
            "T2",
            "DWI",
            "T1_NATIVE",
            "T1_ARTERIAL",
            "T1_PORTAL_VENOUS",
            "T1_DELAYED",
            "T1_HEPATOBILIARY",
        ])
    ].copy()

    # Discard clearly invalid derived/subtraction candidates for final selection.
    selectable = selectable[selectable["selection_score"].fillna(-9999) > -500].copy()

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
        desc = build_display_text(row, cols=TEXT_COLS_DEFAULT)
        if not desc:
            desc = _display_str(row.get("ProtocolName")) or row.get("series_text", "")

        details = []
        volume_order = safe_float(row.get("volume_order_in_series"))
        n_volumes = safe_float(row.get("n_volumes_in_series"))
        if pd.notna(volume_order) and pd.notna(n_volumes):
            details.append(f"vol={volume_order:g}/{n_volumes:g}")
        details.append(f"score={row.get('selection_score'):.1f}")
        return f"{desc} [{', '.join(details)}]"

    ranked["selected_candidate"] = ranked.apply(_display, axis=1)
    ranked["_candidate_rank"] = ranked.groupby([*exam_cols, "selection_slot"]).cumcount()

    selected_long = (
        ranked
        .groupby([*exam_cols, "selection_slot"], as_index=False)
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
        [*exam_cols, *(col for slot in slots for col in (slot, f"{slot}_other_candidates"))]
    ]

    return selected_long, selected_wide


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def annotate_mri(
    df: pd.DataFrame,
    patient_col: str = "patient_key",
    study_col: str | None = "study_id",
    series_col: str = "series_id",
    volume_col: str = "volume_id",
    date_col: str = "date",
) -> pd.DataFrame:
    """Add labels, features, and scores to a volume/series-level dataframe."""
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
    exam_cols = get_exam_group_cols(
        out,
        patient_col=patient_col,
        study_col=study_col,
        date_col=date_col,
    )
    out = add_mri_perfusion_columns(out, exam_group_cols=exam_cols)
    out = add_scores(out)
    return out


def curate_mri(
    df: pd.DataFrame,
    patient_col: str = "patient_key",
    study_col: str | None = "study_id",
    series_col: str = "series_id",
    volume_col: str = "volume_id",
    date_col: str = "date",
) -> dict[str, pd.DataFrame]:
    """Full MRI curation pipeline."""
    curated = annotate_mri(
        df,
        patient_col=patient_col,
        study_col=study_col,
        series_col=series_col,
        volume_col=volume_col,
        date_col=date_col,
    )
    selected_long, selected_wide = select_best_candidates(
        curated,
        patient_col=patient_col,
        study_col=study_col,
        date_col=date_col,
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
    results["selected_long"].to_csv(f"{args.output_prefix}_selected_long.csv", index=False)
    results["selected_wide"].to_csv(f"{args.output_prefix}_selected_wide.csv", index=False)
