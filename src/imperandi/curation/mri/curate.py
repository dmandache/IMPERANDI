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


def build_series_text(row: pd.Series, cols: Sequence[str] | None = None) -> str:
    """Use all useful text fields, not only SeriesDescription."""
    cols = list(cols or TEXT_COLS_DEFAULT)
    parts = [safe_str(row.get(c)) for c in cols if c in row.index]
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

    rep["volume_index_in_series"] = rep.groupby("_series_group_key", dropna=False).cumcount()
    rep["volume_order_in_series"] = rep["volume_index_in_series"] + 1
    rep["n_volumes_in_series"] = rep.groupby("_series_group_key", dropna=False)["_volume_group_key"].transform("size")
    rep["is_multivolume_series"] = rep["n_volumes_in_series"] > 1

    order_cols = [
        "_volume_group_key",
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
    ordered = work[["_row_order_for_volume_order", "_volume_group_key"]].merge(
        rep[order_cols],
        on="_volume_group_key",
        how="left",
    )
    ordered = ordered.set_index("_row_order_for_volume_order").reindex(range(len(out)))
    for col in [
        "volume_index_in_series",
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


def has_post_contrast_text(row: pd.Series) -> bool:
    return bool(re.search(rules.RX_PHASE_POST_CONTRAST, build_series_text(row)))


def detect_ordinal_phase_index(row: pd.Series) -> int | None:
    match = re.search(rules.RX_PHASE_ORDINAL, build_series_text(row))
    if not match:
        return None
    for group in match.groups():
        if group is not None:
            return int(group)
    return None


def detect_explicit_phase_from_text(row: pd.Series) -> tuple[str | None, str, str, str]:
    text = build_series_text(row)

    if re.search(rules.RX_PHASE_NATIVE, text):
        return "NATIVE", "matched explicit native/non-injected keyword", "explicit", "explicit_text"

    if re.search(rules.RX_PHASE_HEPATOBILIARY, text):
        return (
            "HEPATOBILIARY",
            "matched explicit hepatobiliary/2h keyword",
            "explicit",
            "explicit_text",
        )

    if re.search(rules.RX_PHASE_DELAYED, text):
        return "DELAYED", "matched explicit delayed/tardif keyword", "explicit", "explicit_text"

    if re.search(rules.RX_PHASE_PORTAL, text):
        return "PORTAL_VENOUS", "matched explicit portal/venous keyword", "explicit", "explicit_text"

    if re.search(rules.RX_PHASE_ARTERIAL, text):
        return "ARTERIAL", "matched explicit arterial keyword", "explicit", "explicit_text"

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

    return bool(re.search(rules.RX_PHASE_GENERIC_DYNAMIC, build_series_text(row)))


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
      1. Explicit pure phase text, e.g. SANS IV, ART, PORT, TARDIF.
      2. Special ART-PORT / Mask+Multiart order inference.
      3. Generic dynamic volume order inference.
      4. OTHER.
    """
    text = build_series_text(row)
    seq = norm_label(row.get("mri_sequence"))

    if seq != "T1":
        return "OTHER", f"sequence={seq}; phase not assigned", "unknown", "none"

    # Derived/subtraction rows are kept as T1 candidates but not valid phase labels.
    if (
        re.search(rules.RX_SUBTRACTION, text)
        or re.search(rules.RX_MIP_MPR, text)
        or re.search(rules.RX_QUANT_OR_REPORT, text)
    ):
        return "OTHER", "matched subtraction/derived/non-diagnostic marker", "unknown", "none"

    if text_matches_art_port(row):
        special_label, special_reason, special_conf, special_source = infer_special_t1_phase_from_volume_order(row)
        if special_label is not None:
            return special_label, special_reason, special_conf, special_source
        return (
            "PORTAL_VENOUS",
            "matched ART-PORT text without usable volume order; treated as portal/transition",
            "inferred",
            "explicit_text_art_port_single",
        )

    if text_matches_mask_multiart(row):
        special_label, special_reason, special_conf, special_source = infer_special_t1_phase_from_volume_order(row)
        if special_label is not None:
            return special_label, special_reason, special_conf, special_source

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

    text = build_series_text(row)
    has_dynamic_text = bool(
        re.search(rules.RX_T1_DYNAMIC, text) or re.search(rules.RX_T1_3D_GRE, text)
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


def is_native_fallback_candidate(row: pd.Series) -> bool:
    if norm_label(row.get("mri_sequence")) != "T1":
        return False
    if norm_label(row.get("mri_perfusion_label")) != "OTHER":
        return False

    text = build_series_text(row)
    if (
        re.search(rules.RX_SUBTRACTION, text)
        or re.search(rules.RX_MIP_MPR, text)
        or re.search(rules.RX_QUANT_OR_REPORT, text)
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
        or re.search(rules.RX_T1_3D_GRE, text)
        or dixon_component in {"WATER", "IN_PHASE", "DIXON_UNKNOWN"}
    )


def _exam_has_post_contrast_dynamic_phase(exam_rows: pd.DataFrame) -> bool:
    phase = exam_rows["mri_perfusion_label"].map(norm_label)
    source = exam_rows["mri_perfusion_source"].fillna("none")
    resolved_dynamic = phase.isin(["ARTERIAL", "PORTAL_VENOUS", "DELAYED"]) & source.isin(
        ["ordinal_context", "volume_order", "volume_order_art_port", "volume_order_mask_multiart"]
    )
    post_text_dynamic = exam_rows.apply(
        lambda r: has_post_contrast_text(r)
        and detect_ordinal_phase_index(r) is not None
        and norm_label(r.get("mri_sequence")) == "T1",
        axis=1,
    )
    return bool(resolved_dynamic.any() or post_text_dynamic.any())


def _sort_key_for_native_fallback(row: pd.Series) -> tuple:
    text = build_series_text(row)
    component = row.get("dixon_component")
    component_score = {
        "WATER": 4,
        "IN_PHASE": 3,
        "DIXON_UNKNOWN": 2,
        "NOT_DIXON": 1,
    }.get(component, 0)
    quality_score = 0
    quality_score += 4 if row.get("plane") == "AXIAL" else 0
    quality_score += 3 if bool(row.get("is_3d_gre")) or re.search(rules.RX_T1_3D_GRE, text) else 0
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
        desc = row.get("SeriesDescription")
        if pd.isna(desc) or str(desc).strip() == "":
            desc = row.get("ProtocolName", row.get("series_text", ""))
        return f"{desc} [score={row.get('selection_score'):.1f}]"

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
        .groupby([*exam_cols, "selection_slot"], as_index=False)["selected_candidate"]
        .agg("; ".join)
        .pivot_table(
            index=exam_cols,
            columns="selection_slot",
            values="selected_candidate",
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
