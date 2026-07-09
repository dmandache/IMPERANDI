"""
MRI curation utilities.

Core functionality:
1. Classify MRI sequence family: T1, T2, DWI, LOCALIZER, KEY_IMAGES, OTHER.
2. Classify T1 perfusion/contrast phase: PRECONTRAST, ARTERIAL,
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

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import rules


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

TEXT_COLS_DEFAULT = [
    "SeriesDescription",
    "ProtocolName",
    "StudyDescription",
    "ImageType",
    "ScanningSequence",
    "SequenceVariant",
    "ScanOptions",
    "SequenceName",
]


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(Path(path).expanduser(), low_memory=False, **kwargs)


def safe_str(x) -> str:
    return str(x).strip().lower() if pd.notna(x) else ""


def norm_label(x, default: str = "OTHER") -> str:
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
    if pd.isna(x):
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
    if pd.isna(x):
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


def build_series_text(row: pd.Series, cols: Sequence[str] | None = None) -> str:
    """Use all useful text fields, not only SeriesDescription."""
    cols = list(cols or TEXT_COLS_DEFAULT)
    return " | ".join(
        safe_str(row.get(c)) for c in cols if c in row.index and safe_str(row.get(c))
    )


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
    out = df.copy()

    if "volume_order_in_series" in out.columns and "n_volumes_in_series" in out.columns:
        return out

    if series_col not in out.columns or volume_col not in out.columns:
        out["volume_order_in_series"] = 1
        out["volume_index_in_series"] = 0
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

    for col in ["AcquisitionNumber", "InstanceNumber", "volume_id"]:
        if col not in work.columns:
            work[col] = np.nan

    # One representative row per volume for ordering.
    rep = (
        work.sort_values(
            [*series_group_cols, "_sort_time_seconds", "AcquisitionNumber", "InstanceNumber", volume_col],
            na_position="last",
        )
        .drop_duplicates(volume_group_cols)
        .copy()
    )

    rep["volume_index_in_series"] = rep.groupby(series_group_cols, dropna=False).cumcount()
    rep["volume_order_in_series"] = rep["volume_index_in_series"] + 1
    rep["n_volumes_in_series"] = rep.groupby(series_group_cols, dropna=False)[volume_col].transform("size")
    rep["is_multivolume_series"] = rep["n_volumes_in_series"] > 1

    order_cols = [
        *volume_group_cols,
        "volume_index_in_series",
        "volume_order_in_series",
        "n_volumes_in_series",
        "is_multivolume_series",
    ]

    out = out.drop(
        columns=[
            c for c in [
                "volume_index_in_series",
                "volume_order_in_series",
                "n_volumes_in_series",
                "is_multivolume_series",
            ] if c in out.columns
        ],
        errors="ignore",
    )
    return out.merge(rep[order_cols], on=volume_group_cols, how="left")


# -----------------------------------------------------------------------------
# Sequence classification
# -----------------------------------------------------------------------------


def detect_mri_sequence(row: pd.Series) -> tuple[str, str, str]:
    text = build_series_text(row)
    modality = safe_str(row.get("Modality")).upper()

    if re.search(rules.RX_LOCALIZER, text):
        return "LOCALIZER", "matched localizer/scout/survey keyword", "high"

    if modality == "KO" or re.search(rules.RX_KEY_IMAGES, text):
        return "KEY_IMAGES", "matched key-image/processed marker", "high"

    if re.search(rules.RX_SEQUENCE_DWI, text):
        return "DWI", "matched DWI/diffusion/ADC/b-value keyword", "high"

    if re.search(rules.RX_SEQUENCE_T1, text) or re.search(rules.RX_SEQUENCE_T1_CONTRAST, text):
        return "T1", "matched T1 / VIBE-LAVA-THRIVE-Dixon-GRE family", "high"

    if re.search(rules.RX_SEQUENCE_T2, text):
        return "T2", "matched T2/TSE/FSE/HASTE/BLADE/MRCP family", "high"

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
    return bool(re.search(rules.RX_PHASE_ART_PORT_DYNAMIC, build_series_text(row)))


def text_matches_mask_multiart(row: pd.Series) -> bool:
    return bool(re.search(rules.RX_PHASE_MASK_MULTIART_DYNAMIC, build_series_text(row)))


def infer_special_t1_phase_from_volume_order(row: pd.Series) -> tuple[str | None, str, str, str]:
    """Special same-description/multiple-volume protocols."""
    if norm_label(row.get("mri_sequence")) != "T1":
        return None, "not T1", "low", "none"

    order = safe_float(row.get("volume_order_in_series"))
    n_volumes = safe_float(row.get("n_volumes_in_series"))
    if pd.isna(order) or pd.isna(n_volumes) or n_volumes < 2:
        return None, "no independent second volume detected", "low", "none"

    order = int(order)
    n_volumes = int(n_volumes)

    if text_matches_art_port(row):
        if order == 1:
            return (
                "ARTERIAL",
                f"inferred ARTERIAL from first ART-PORT volume {order}/{n_volumes}",
                "medium",
                "volume_order_art_port",
            )
        if order == 2:
            return (
                "PORTAL_VENOUS",
                f"inferred PORTAL_VENOUS from second ART-PORT volume {order}/{n_volumes}",
                "medium",
                "volume_order_art_port",
            )
        return (
            "DELAYED",
            f"inferred DELAYED from later ART-PORT volume {order}/{n_volumes}",
            "low",
            "volume_order_art_port",
        )

    if text_matches_mask_multiart(row):
        if order == 1:
            return (
                "PRECONTRAST",
                f"inferred PRECONTRAST from first Mask+Multiart volume {order}/{n_volumes}",
                "medium",
                "volume_order_mask_multiart",
            )
        return (
            "ARTERIAL",
            f"inferred ARTERIAL from Mask+Multiart volume {order}/{n_volumes}",
            "medium",
            "volume_order_mask_multiart",
        )

    return None, "no special dynamic T1 profile", "low", "none"


def has_generic_dynamic_t1_evidence(row: pd.Series) -> bool:
    if norm_label(row.get("mri_sequence")) != "T1":
        return False

    n_volumes = safe_float(row.get("n_volumes_in_series"))
    if pd.isna(n_volumes) or n_volumes < 3:
        return False

    return bool(re.search(rules.RX_PHASE_GENERIC_DYNAMIC, build_series_text(row)))


def infer_generic_t1_phase_from_volume_order(row: pd.Series) -> tuple[str | None, str, str, str]:
    if not has_generic_dynamic_t1_evidence(row):
        return None, "no generic dynamic multivolume T1 evidence", "low", "none"

    order = safe_float(row.get("volume_order_in_series"))
    n_volumes = safe_float(row.get("n_volumes_in_series"))
    if pd.isna(order) or pd.isna(n_volumes):
        return None, "missing volume order", "low", "none"

    order = int(order)
    n_volumes = int(n_volumes)

    mapping = {
        1: "PRECONTRAST",
        2: "ARTERIAL",
        3: "PORTAL_VENOUS",
    }
    label = mapping.get(order, "DELAYED")
    return (
        label,
        f"inferred {label} from generic dynamic T1 volume order {order}/{n_volumes}",
        "medium",
        "volume_order",
    )


def detect_t1_perfusion_phase(row: pd.Series) -> tuple[str, str, str, str]:
    """
    Volume-aware T1 phase classifier.

    Priority:
      1. Special ART-PORT / Mask+Multiart order inference.
      2. Explicit pure phase text, e.g. SANS IV, ART, PORT, TARDIF.
      3. Generic dynamic volume order inference.
      4. OTHER.
    """
    text = build_series_text(row)
    seq = norm_label(row.get("mri_sequence"))

    if seq != "T1":
        return "OTHER", f"sequence={seq}; phase not assigned", "medium", "none"

    # Derived/subtraction rows are kept as T1 candidates but not valid phase labels.
    if (
        re.search(rules.RX_SUBTRACTION, text)
        or re.search(rules.RX_MIP_MPR, text)
        or re.search(rules.RX_QUANT_OR_REPORT, text)
    ):
        return "OTHER", "matched subtraction/derived/non-diagnostic marker", "high", "none"

    special_label, special_reason, special_conf, special_source = infer_special_t1_phase_from_volume_order(row)
    if special_label is not None:
        return special_label, special_reason, special_conf, special_source

    if re.search(rules.RX_PHASE_PRECONTRAST, text):
        return "PRECONTRAST", "matched explicit precontrast/non-injected keyword", "high", "explicit_text"

    if re.search(rules.RX_PHASE_HEPATOBILIARY, text):
        return "HEPATOBILIARY", "matched explicit hepatobiliary/2h keyword", "high", "explicit_text"

    if re.search(rules.RX_PHASE_DELAYED, text):
        return "DELAYED", "matched explicit delayed/tardif keyword", "high", "explicit_text"

    # Text-only ART-PORT fallback when volume ordering was not available.
    # This must be evaluated before the generic PORT regex, otherwise the source
    # is swallowed as explicit_text.
    if text_matches_art_port(row):
        return (
            "PORTAL_VENOUS",
            "matched ART-PORT text without usable volume order; treated as portal/transition",
            "medium",
            "explicit_text_art_port_single",
        )

    if re.search(rules.RX_PHASE_PORTAL, text):
        return "PORTAL_VENOUS", "matched explicit portal/venous keyword", "high", "explicit_text"

    if re.search(rules.RX_PHASE_ARTERIAL, text):
        return "ARTERIAL", "matched explicit arterial keyword", "high", "explicit_text"

    generic_label, generic_reason, generic_conf, generic_source = infer_generic_t1_phase_from_volume_order(row)
    if generic_label is not None:
        return generic_label, generic_reason, generic_conf, generic_source

    return "OTHER", "no supported T1 perfusion phase rule matched", "low", "none"


def add_mri_perfusion_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    result = out.apply(detect_t1_perfusion_phase, axis=1)
    out[["mri_perfusion_label", "mri_perfusion_reason", "mri_perfusion_confidence", "mri_perfusion_source"]] = pd.DataFrame(
        result.tolist(), index=out.index
    )
    return out


# -----------------------------------------------------------------------------
# Feature extraction and scoring
# -----------------------------------------------------------------------------


def detect_plane(text: str) -> str:
    if re.search(rules.RX_PLANE_AXIAL, text):
        return "AXIAL"
    if re.search(rules.RX_PLANE_CORONAL, text):
        return "CORONAL"
    if re.search(rules.RX_PLANE_SAGITTAL, text):
        return "SAGITTAL"
    return "UNKNOWN"


def detect_dixon_component(text: str) -> str:
    text = str(text or "").lower()
    has_dixon = bool(re.search(rules.RX_DIXON_CONTEXT, text))

    if re.search(rules.RX_DIXON_FAT_FRACTION, text):
        return "FAT_FRACTION"
    if re.search(rules.RX_DIXON_R2STAR, text):
        return "R2STAR"
    if re.search(rules.RX_DIXON_ALL, text):
        return "DIXON_ALL"

    # Compact suffixes are common: ART_W, ART_in, ART_opp, ART_F.
    if re.search(r"(?:^|[\s_.+\-/])w(?:$|[\s_.+\-/])", text) or re.search(rules.RX_DIXON_WATER, text):
        return "WATER"
    if re.search(r"(?:^|[\s_.+\-/])in(?:$|[\s_.+\-/])", text) or re.search(rules.RX_DIXON_IN, text):
        return "IN_PHASE"
    if re.search(r"(?:^|[\s_.+\-/])opp(?:$|[\s_.+\-/])", text) or re.search(rules.RX_DIXON_OPPOSED, text):
        return "OPPOSED_PHASE"
    if has_dixon and (re.search(r"(?:^|[\s_.+\-/])f(?:$|[\s_.+\-/])", text) or re.search(rules.RX_DIXON_FAT, text)):
        return "FAT"

    return "DIXON_UNKNOWN" if has_dixon else "NOT_DIXON"


def add_basic_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["series_text"] = out.apply(build_series_text, axis=1)
    out["plane"] = out["series_text"].apply(detect_plane)

    out["is_subtraction"] = out["series_text"].str.contains(rules.RX_SUBTRACTION, regex=True, na=False)
    out["is_mip_mpr"] = out["series_text"].str.contains(rules.RX_MIP_MPR, regex=True, na=False)
    out["is_quant_or_report"] = out["series_text"].str.contains(rules.RX_QUANT_OR_REPORT, regex=True, na=False)

    # T1 features.
    out["dixon_component"] = out["series_text"].apply(detect_dixon_component)
    out["is_3d_gre"] = out["series_text"].str.contains(rules.RX_T1_3D_GRE, regex=True, na=False)
    out["is_dynamic_t1_text"] = out["series_text"].str.contains(rules.RX_T1_DYNAMIC, regex=True, na=False)
    out["is_breath_hold"] = out["series_text"].str.contains(rules.RX_BREATH_HOLD, regex=True, na=False)
    out["is_resp_triggered"] = out["series_text"].str.contains(rules.RX_RESP_TRIGGERED, regex=True, na=False)

    # T2 features.
    out["is_t2_fatsat"] = out["series_text"].str.contains(rules.RX_T2_FATSAT, regex=True, na=False)
    out["is_t2_motion_robust"] = out["series_text"].str.contains(rules.RX_T2_MOTION_ROBUST, regex=True, na=False)
    out["is_t2_haste_ssfse"] = out["series_text"].str.contains(rules.RX_T2_HASTE_SSFSE, regex=True, na=False)
    out["is_t2_tse_fse"] = out["series_text"].str.contains(rules.RX_T2_TSE_FSE, regex=True, na=False)
    out["is_t2_mrcp_biliary"] = out["series_text"].str.contains(rules.RX_T2_MRCP_BILIARY, regex=True, na=False)

    return out


def score_t1(row: pd.Series) -> float:
    phase = norm_label(row.get("mri_perfusion_label"))
    source = safe_str(row.get("mri_perfusion_source")) or "none"

    score = float(rules.T1_PHASE_PRIORITY.get(phase, 0))
    score += rules.T1_PHASE_SOURCE_PRIORITY.get(source, 0)

    score += {"AXIAL": 50, "CORONAL": 20, "SAGITTAL": 5}.get(row.get("plane"), 50)
    score += 25 if bool(row.get("is_3d_gre")) else 0

    # Dynamic containers are useful fallback, but explicit pure phase labels are preferred.
    if bool(row.get("is_dynamic_t1_text")) and source == "explicit_text":
        score += 5

    score += rules.DIXON_COMPONENT_PRIORITY.get(row.get("dixon_component"), 0)
    score += 8 if bool(row.get("is_resp_triggered")) else 0
    score += 5 if bool(row.get("is_breath_hold")) else 0

    score -= 1000 if bool(row.get("is_subtraction")) else 0
    score -= 100 if bool(row.get("is_mip_mpr")) else 0
    score -= 150 if bool(row.get("is_quant_or_report")) else 0
    return float(score)


def score_t2(row: pd.Series) -> float:
    score = {"AXIAL": 50, "CORONAL": 20, "SAGITTAL": 5}.get(row.get("plane"), 50)
    score += 35 if bool(row.get("is_t2_fatsat")) else 0
    score += 35 if bool(row.get("is_t2_motion_robust")) else 0
    score += 15 if bool(row.get("is_t2_haste_ssfse")) else 0
    score += 10 if bool(row.get("is_t2_tse_fse")) else 0
    score += 8 if bool(row.get("is_resp_triggered")) else 0
    score += 5 if bool(row.get("is_breath_hold")) else 0
    score -= 40 if bool(row.get("is_t2_mrcp_biliary")) else 0
    score -= 100 if bool(row.get("is_mip_mpr")) else 0
    score -= 150 if bool(row.get("is_quant_or_report")) else 0
    return float(score)


def score_dwi(row: pd.Series) -> float:
    text = row.get("series_text", "")
    score = 70.0
    score += 10 if re.search(rules.RX_PLANE_AXIAL, text) else 0
    score += 10 if re.search(rules.RX_SEQUENCE_DWI, text) else 0
    score -= 50 if re.search(rules.RX_MIP_MPR, text) else 0
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
            "T1_PRECONTRAST",
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

    selected_long = (
        selectable.sort_values(sort_cols, ascending=ascending, na_position="last")
        .groupby([*exam_cols, "selection_slot"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    def _display(row: pd.Series) -> str:
        desc = row.get("SeriesDescription")
        if pd.isna(desc) or str(desc).strip() == "":
            desc = row.get("ProtocolName", row.get("series_text", ""))
        return f"{desc} [score={row.get('selection_score'):.1f}]"

    selected_long["selected_candidate"] = selected_long.apply(_display, axis=1)

    selected_wide = (
        selected_long[[*exam_cols, "selection_slot", "selected_candidate"]]
        .pivot_table(
            index=exam_cols,
            columns="selection_slot",
            values="selected_candidate",
            aggfunc="first",
        )
        .reset_index()
    )
    selected_wide.columns.name = None

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
    out = add_mri_perfusion_columns(out)
    out = add_basic_feature_columns(out)
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
