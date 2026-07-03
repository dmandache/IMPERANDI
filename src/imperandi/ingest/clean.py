import argparse
import logging
import hashlib
import re
from ast import literal_eval
from datetime import time as dt_time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from pydicom.uid import UID
from unidecode import unidecode

from imperandi.utils.manifest import load_manifest, resolve_function_path
from imperandi.utils.geometry import (
    classify_plane_from_iop,
    standardize_iop,
)
from imperandi.ingest.hooks import (
    apply_patient_key_standardization,
    get_clean_hook_outputs,
)
from imperandi.utils.logging import log_task_summary, setup_logging
from imperandi.utils.misc import print_args, report_volumes, report_change
from imperandi.utils.datetime import to_dates, to_times
from imperandi.datasets_config.defaults import (
    DEFAULT_DICOM_TAGS,
    DEFAULT_MAX_PIXEL_SPACING_MM,
    DEFAULT_MAX_SLICE_THICKNESS_MM,
    DATE_CANDIDATES,
    TIME_CANDIDATES,
)

COLUMNS_TO_USE = [
    "patient_key",
    "_patient_key_raw",
    "study_id",
    "series_id",
    "dicom_path",
] + DEFAULT_DICOM_TAGS

BASE_INPUT_COLUMNS = [
    "patient_key",
    "_patient_key_raw",
    "study_id",
    "series_id",
    "dicom_path",
]


pd.options.mode.chained_assignment = None
logger = logging.getLogger(__name__)

DATETIME_TIME_RE = re.compile(
    r"datetime\.time\(\s*(\d{1,2})\s*,\s*(\d{1,2})(?:\s*,\s*(\d{1,2}))?(?:\s*,\s*(\d{1,6}))?\s*\)"
)
FLOAT_TOKEN_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
SUPPORTED_FILTER_OPERATORS = {
    "eq",
    "ne",
    "in",
    "not_in",
    "contains",
    "icontains",
    "regex",
    "lt",
    "lte",
    "gt",
    "gte",
    "is_null",
    "not_null",
}


def add_clean_arguments(
    parser: argparse.ArgumentParser,
    include_manifest: bool = True,
    include_csv_path: bool = True,
    include_csv_path_out: bool = True,
    include_dry_run: bool = True,
) -> None:
    """Add metadata-cleaning paths and manifest options."""
    if include_csv_path:
        parser.add_argument(
            "csv_path_pos",
            type=str,
            nargs="?",
            default=None,
            help=(
                "Path to the input CSV file (or a file pattern). "
                "Defaults to ./dicom_index.csv."
            ),
        )
    if include_csv_path and include_csv_path_out:
        parser.add_argument(
            "csv_path_out_pos",
            nargs="?",
            type=str,
            default=None,
            help="Optional output CSV path (positional alternative to --csv_path_out).",
        )
    if include_csv_path:
        parser.add_argument(
            "--csv_path",
            dest="csv_path_opt",
            nargs="+",
            type=str,
        )

    if include_csv_path_out:
        parser.add_argument(
            "--csv_path_out",
            type=str,
            required=False,
            default=None,
            help=(
                "Path to save the cleaned CSV file. "
                "Defaults to <csv_dir>/<csv_stem>_clean.csv."
            ),
        )
    if include_manifest:
        parser.add_argument(
            "--manifest",
            type=str,
            default=None,
            help="Dataset manifest name or path to manifest JSON.",
        )
    if include_dry_run:
        parser.add_argument(
            "--dry-run",
            dest="dry_run",
            action="store_true",
            default=False,
            help="Print planned actions without running.",
        )


def build_parser(
    add_help: bool = True,
    include_manifest: bool = True,
) -> argparse.ArgumentParser:
    """Build the standalone parser for metadata cleaning."""
    parser = argparse.ArgumentParser(
        description="Clean and process DICOM metadata CSV.",
        add_help=add_help,
    )
    add_clean_arguments(
        parser,
        include_manifest=include_manifest,
        include_csv_path=True,
        include_csv_path_out=True,
        include_dry_run=True,
    )
    return parser


def parse_arguments():
    """Parse and normalize arguments for the standalone clean command."""
    parser = build_parser()
    args = parser.parse_args()
    args = normalize_clean_args(args)

    logger.info("🚀 Running %s script with arguments: %s", Path(__file__).name, args)
    return args


def _default_clean_output_path(csv_path: Path) -> Path:
    stem = csv_path.stem
    if stem.endswith("_clean"):
        return csv_path.parent / f"{stem}_out.csv"
    return csv_path.parent / f"{stem}_clean.csv"


def normalize_clean_args(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve clean input/output paths in-place."""
    csv_in = (
        args.csv_path_opt
        if getattr(args, "csv_path_opt", None) is not None
        else getattr(args, "csv_path_pos", None)
    )
    if not csv_in:
        csv_paths = [Path.cwd() / "dicom_index.csv"]
    elif isinstance(csv_in, str):
        csv_paths = [Path(csv_in)]
    else:
        csv_paths = [Path(p) for p in csv_in]

    args.csv_path = [str(p) for p in csv_paths]
    first_csv = csv_paths[0]

    csv_path_out_pos = getattr(args, "csv_path_out_pos", None)
    csv_out = args.csv_path_out if args.csv_path_out else csv_path_out_pos
    if not csv_out:
        args.csv_path_out = str(_default_clean_output_path(first_csv))
    else:
        args.csv_path_out = str(csv_out)

    for attr in ("csv_path_pos", "csv_path_opt", "csv_path_out_pos"):
        if hasattr(args, attr):
            delattr(args, attr)

    return args


def read_csv_with_valid_columns(file, required_columns=None):
    """Read only recognized identifier and DICOM metadata columns from CSV."""
    available_columns = pd.read_csv(file, nrows=0).columns
    requested = set(COLUMNS_TO_USE)
    if required_columns:
        requested.update(required_columns)
    valid_columns = [col for col in available_columns if col in requested]
    return pd.read_csv(file, usecols=valid_columns)


def load_data(csv_path, required_columns=None):
    """Load and concatenate metadata CSVs, dropping empty helper columns."""
    if len(csv_path) == 1:
        df = read_csv_with_valid_columns(csv_path[0], required_columns=required_columns)
    else:
        dfs = [
            read_csv_with_valid_columns(file, required_columns=required_columns)
            for file in csv_path
        ]
        df = pd.concat(dfs, ignore_index=True)

    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    required = set(required_columns or [])
    empty_cols = [col for col in df.columns if df[col].isna().all() and col not in required]
    if empty_cols:
        df = df.drop(columns=empty_cols)
    logger.info("%s %s", df.shape, df.columns)

    return df


def filter_ct_modality(df):
    """Keep CT Image Storage rows when modality tags are available."""
    if "Modality" not in df.columns or "SOPClassUID" not in df.columns:
        return df
    df = df[df.Modality == "CT"]
    df["sop_class"] = df.SOPClassUID.apply(lambda x: UID(x).keyword)
    df = df[df.sop_class == "CTImageStorage"]
    return df


def remove_pet_ct(df):
    if "ModalitiesInStudy" not in df.columns:
        return df
    unwanted_modalities = ["PT", "NM"]
    df = df[
        ~df["ModalitiesInStudy"].apply(
            lambda mods: any(m in str(mods) for m in unwanted_modalities)
        )
    ]
    return df


def add_date(df, candidate_columns=None):
    candidates = candidate_columns or DATE_CANDIDATES
    candidate_cols = [col for col in candidates if col in df.columns]
    if not candidate_cols:
        return df

    parsed_by_candidate = {}
    for col in candidate_cols:
        parsed_by_candidate[col] = df[col].apply(_normalize_date_candidate)
    # Ordered fallback: first candidate has highest priority, next ones fill gaps.
    date = parsed_by_candidate[candidate_cols[0]].copy()
    fill_contrib = {}
    for col in candidate_cols[1:]:
        missing_before = int(date.isna().sum())
        date = date.fillna(parsed_by_candidate[col])
        missing_after = int(date.isna().sum())
        fill_contrib[col] = missing_before - missing_after

    df["date"] = date
    total_valid = int(df["date"].notna().sum())
    logger.info(
        "Date candidates (priority order) %s -> %d/%d valid%s",
        candidate_cols,
        total_valid,
        len(df),
        (
            f"; fallback filled {fill_contrib}"
            if any(v > 0 for v in fill_contrib.values())
            else ""
        ),
    )
    return df


def add_time(df, candidate_columns=None):
    candidates = candidate_columns or TIME_CANDIDATES
    candidate_cols = [col for col in candidates if col in df.columns]
    if not candidate_cols:
        return df

    parsed_by_candidate = {
        col: df[col].apply(_normalize_instance_creation_time) for col in candidate_cols
    }
    # Ordered fallback: first candidate has highest priority, next ones fill gaps.
    time = parsed_by_candidate[candidate_cols[0]].copy()
    fill_contrib = {}
    for col in candidate_cols[1:]:
        missing_before = int(time.isna().sum())
        time = time.fillna(parsed_by_candidate[col])
        missing_after = int(time.isna().sum())
        fill_contrib[col] = missing_before - missing_after

    df["time"] = time
    total_valid = int(df["time"].notna().sum())
    logger.info(
        "Time candidates (priority order) %s -> %d/%d valid%s",
        candidate_cols,
        total_valid,
        len(df),
        (
            f"; fallback filled {fill_contrib}"
            if any(v > 0 for v in fill_contrib.values())
            else ""
        ),
    )
    return df


def filter_image_type(df):
    if "ImageType" not in df.columns:
        return df
    df = df.dropna(subset=["ImageType"])
    parsed = []
    for value in df["ImageType"]:
        if isinstance(value, list):
            parsed.append(value)
            continue
        try:
            parsed.append(literal_eval(value))
        except (ValueError, SyntaxError):
            parsed.append([])
    df_imagetype = pd.DataFrame(parsed)
    df_imagetype = df_imagetype.add_prefix("ImageType_value_")
    df = pd.concat(
        [df.reset_index(drop=True), df_imagetype.reset_index(drop=True)], axis=1
    )
    return df


def remove_scouts_localizers(df):
    if "ImageType" not in df.columns or "SeriesDescription" not in df.columns:
        return df
    df = df.dropna(subset=["ImageType", "SeriesDescription"])
    df = df[~df["ImageType"].str.contains("LOCALIZER")]
    df = df[df["SeriesDescription"].apply(lambda x: "scout" not in str(x).lower())]
    return df


def remove_mpr(df):
    if "ImageType" not in df.columns or "SeriesDescription" not in df.columns:
        return df
    df = df[
        ~(
            (df.ImageType.str.contains("mpr", case=False, na=False))
            | (df.SeriesDescription.str.contains("mpr", case=False, na=False))
        )
    ]
    return df


def uniform_string(s):
    if s is None or _is_nan(s):
        return None
    s = str(s).rstrip(".0")
    s = " ".join(s.split())
    return unidecode(s.lower())


def remove_other_organs_description(df):
    if "SeriesDescription" not in df.columns:
        return df
    excluded_substrings = [
        "pelvis",
        "crane",
        "rachis",
        "prostate",
        "phlebo",
        "meckel",
        "femur",
        "orl",
    ]

    df["SeriesDescription"] = df["SeriesDescription"].apply(uniform_string)
    df = df[
        df["SeriesDescription"].apply(
            lambda x: not any(sub in str(x) for sub in excluded_substrings)
        )
    ]

    return df


def clean_scan_size(df):
    if "Rows" in df.columns and "Columns" in df.columns:
        df = df.dropna(subset=["Rows", "Columns"])
    if "SliceThickness" in df.columns:
        df = df[
            (df["SliceThickness"].astype(float) <= DEFAULT_MAX_SLICE_THICKNESS_MM)
            | (df["SliceThickness"].isna())
        ]
    return df


def clean_pixel_spacing(df):
    if "PixelSpacing" not in df.columns:
        return df
    df["PixelSpacingXY"] = df["PixelSpacing"].apply(
        lambda x: literal_eval(x)[0] if isinstance(x, str) else None
    )
    df["PixelSpacingXY"] = pd.to_numeric(df["PixelSpacingXY"], errors="coerce")
    df = df[
        (df["PixelSpacingXY"].isna())
        | (df["PixelSpacingXY"] <= DEFAULT_MAX_PIXEL_SPACING_MM)
    ]
    return df


def generate_volume_id(df, preferred_cols=None, fallback_cols=None):
    """Add a deterministic identifier for each candidate imaging volume."""

    preferred_cols = preferred_cols or [
        "patient_key",
        "study_id",
        "series_id",
        "ImageType",
        "AcquisitionNumber",
        "ImageOrientationPatient",
        "SliceThickness",
        "PixelSpacingXY",
    ]
    fallback_cols = fallback_cols or ["patient_key", "study_id", "series_id"]

    if "ImageOrientationPatient" in df.columns:
        df = df.copy()
        df["ImageOrientationPatient"] = df["ImageOrientationPatient"].apply(
            lambda x: (
                tuple(x) if isinstance(x, (list, tuple)) else tuple(literal_eval(x))
            )
        )

    # Choose the maximum available columns among preferred
    cols_to_use = [c for c in preferred_cols if c in df.columns]

    # If none of the preferred columns exist, enforce fallback
    if not cols_to_use:
        cols_to_use = [c for c in fallback_cols if c in df.columns]

    # If even fallback columns are missing, use any columns that exist
    if not cols_to_use:
        cols_to_use = list(df.columns)

    if cols_to_use:
        logger.info("For unique volume ID generation, using columns: %s", cols_to_use)

    # If df truly has no columns (or empty selection), assign a single id
    if not cols_to_use:
        logger.info(
            "For unique volume ID generation, using no columns (all rows get same ID)"
        )
        df = df.copy()
        df["volume_id"] = hashlib.sha1(b"volume").hexdigest()
        return df

    def _to_stable_str(x):
        # None/NaN -> empty
        if x is None:
            return ""
        try:
            if x != x:  # NaN
                return ""
        except Exception:
            pass
        # tuples/lists -> pipe-joined
        if isinstance(x, (list, tuple)):
            return "|".join(map(str, x))
        return str(x)

    # Build a per-row stable string and hash it
    joined = df[cols_to_use].applymap(_to_stable_str).agg("||".join, axis=1)

    df = df.copy()
    df["volume_id"] = joined.apply(
        lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()
    )
    return df


def filter_by_acquisition_plane(df, angle_thresh_deg=10.0):
    if "ImageOrientationPatient" not in df.columns:
        return df
    (
        df["acquisition_plane"],
        df["acquisition_angle"],
        df["acquisition_axis"],
    ) = zip(
        *df["ImageOrientationPatient"].map(
            lambda x: classify_plane_from_iop(x, angle_thresh_deg)
        )
    )
    df = df[df["acquisition_axis"] == "Z"]
    return df


def correct_volume_ids(
    df,
    z_tolerance=1e-3,
    group_columns=None,
    z_sources=None,
):
    """
    Merge "pseudo-volumes" (multiple volume_id values) that actually belong to the same volume,
    but do it robustly when DICOM tags/columns are missing.

    Strategy:
    - If volume_id missing -> return df unchanged.
    - Group by the *maximum available* columns from a preferred list.
      If none available -> fallback to grouping by patient_key, study_id, series_id (subset that exists).
    - Determine z positions using the best available source:
        1) ImagePositionPatient (z component)
        2) SliceLocation
      If neither usable -> skip that group.
    - If spacing between sorted z positions is consistent (within tolerance) -> merge volume_ids.
    """

    if "volume_id" not in df.columns:
        return df

    preferred_group_cols = group_columns or [
        "patient_key",
        "study_id",
        "series_id",
        "ImageType",
        "ImageOrientationPatient",
        "SliceThickness",
        "PixelSpacingXY",
    ]
    fallback_group_cols = ["patient_key", "study_id", "series_id"]
    z_sources = z_sources or ["ImagePositionPatient", "SliceLocation"]

    # Choose grouping columns: maximum available
    group_cols = [c for c in preferred_group_cols if c in df.columns]
    if not group_cols:
        group_cols = [c for c in fallback_group_cols if c in df.columns]
    if not group_cols:
        # last resort: keep everything in one group
        group_cols = None

    df = df.copy()

    # # Normalize position/orientation if present (but don't require them)
    # if "ImageOrientationPatient" in df.columns:
    #     df["ImageOrientationPatient"] = df["ImageOrientationPatient"].apply(
    #         lambda x: tuple(as_float_array(x)) if x is not None and x == x else None
    #     )

    # if "ImagePositionPatient" in df.columns:
    #     df["ImagePositionPatient"] = df["ImagePositionPatient"].apply(
    #         lambda x: tuple(as_float_array(x)) if x is not None and x == x else None
    #     )

    updated_ids = {}

    grouped = df.groupby(group_cols, dropna=False) if group_cols else [(None, df)]

    for _, group_df in grouped:
        volume_ids = group_df["volume_id"].dropna().unique()
        if len(volume_ids) <= 1:
            continue

        debug_cols = [
            c
            for c in ["patient_key", "date", "SeriesDescription"]
            if c in group_df.columns
        ]
        if debug_cols:
            summary = {
                c: group_df[c].dropna().unique().tolist()[:5] for c in debug_cols
            }
        else:
            summary = {}
        logger.debug(
            "Evaluating volume_id correction group: group_cols=%s, summary=%s, "
            "volume_ids=%s, rows=%s",
            group_cols,
            summary,
            list(map(str, volume_ids)),
            len(group_df),
        )

        # --- get z positions robustly ---
        z_positions = None

        if "ImagePositionPatient" in z_sources and "ImagePositionPatient" in group_df.columns:
            z_values = []
            ipp_parse_failures = 0
            for value in group_df["ImagePositionPatient"]:
                ipp = _parse_ipp(value)
                if ipp is not None:
                    z_values.append(ipp[2])
                else:
                    ipp_parse_failures += 1
            if z_values:
                z_positions = np.asarray(z_values, dtype=float)
            logger.debug(
                "ImagePositionPatient z extraction: parsed=%s, failed=%s, "
                "sample_z=%s",
                len(z_values),
                ipp_parse_failures,
                z_values[:10],
            )

        if (
            z_positions is None or len(z_positions) < 2
        ) and "SliceLocation" in z_sources and "SliceLocation" in group_df.columns:
            logger.debug(
                "Falling back to SliceLocation for z_positions: "
                "ipp_z_count=%s, rows=%s",
                0 if z_positions is None else len(z_positions),
                len(group_df),
            )
            s = group_df["SliceLocation"]
            mask = s.notna()
            if mask.any():
                try:
                    z_positions = s[mask].astype(float).to_numpy()
                    logger.debug(
                        "SliceLocation z extraction: parsed=%s, sample_z=%s",
                        len(z_positions),
                        z_positions[:10].tolist(),
                    )
                except Exception as exc:
                    # non-numeric slice locations -> skip
                    z_positions = None
                    logger.debug("SliceLocation z extraction failed: %s", exc)

        if z_positions is None or len(z_positions) < 2:
            logger.debug(
                "Skipping volume_id correction group: insufficient z_positions "
                "(count=%s)",
                0 if z_positions is None else len(z_positions),
            )
            continue

        # --- check consistent spacing ---
        z_sorted = np.sort(z_positions)
        z_diff = np.diff(z_sorted)

        consistent_spacing = np.all(np.isclose(z_diff, z_diff[0], atol=z_tolerance))
        logger.debug(
            "z_positions spacing check: z_sample=%s, diff_sample=%s, "
            "nonzero_diff_sample=%s, reference_spacing=%s, consistent=%s, "
            "z_tolerance=%s",
            z_sorted[:5].tolist(),
            z_diff[:5].tolist(),
            z_diff[:5].tolist(),
            float(z_diff[0]),
            bool(consistent_spacing),
            z_tolerance,
        )

        logger.info(
            "%s : %s pseudo-volumes, %s total files",
            summary,
            len(volume_ids),
            len(group_df),
        )

        if consistent_spacing:
            logger.info("👫 Merged")
            # canonical_id = sorted(map(str, volume_ids))[0]
            canonical_id = hashlib.sha1(
                "|".join(sorted(map(str, volume_ids))).encode()
            ).hexdigest()
            logger.debug(
                "Merging volume_ids into canonical_id=%s: %s",
                canonical_id,
                list(map(str, volume_ids)),
            )
            for vol_id in volume_ids:
                updated_ids[vol_id] = canonical_id
        else:
            logger.info("👍 They are different volumes")
            logger.debug(
                "Keeping volume_ids separate due to inconsistent z spacing: %s",
                list(map(str, volume_ids)),
            )

    df["volume_id"] = df["volume_id"].apply(lambda vid: updated_ids.get(vid, vid))
    return df


def _is_nan(value):
    try:
        return bool(value != value)
    except Exception:
        return False


def _as_python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _hashable_key(value):
    value = _as_python_scalar(value)
    if value is None:
        return ("none",)
    if _is_nan(value):
        return ("nan",)
    if isinstance(value, (int, float)):
        return ("num", float(value))
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return ("seq", tuple(_hashable_key(v) for v in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted((_hashable_key(k), _hashable_key(v)) for k, v in value.items())
            ),
        )
    try:
        hash(value)
    except Exception:
        return ("repr", repr(value))
    return ("val", value)


def _string_sort_key(value):
    value = _as_python_scalar(value)
    if isinstance(value, bytes):
        return value.decode(errors="ignore")
    return str(value)


def _parse_float(value):
    value = _as_python_scalar(value)
    if value is None or _is_nan(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except Exception:
            return None
    return None


def _parse_ipp(value):
    value = _as_python_scalar(value)
    if value is None or _is_nan(value):
        return None
    seq = None
    if isinstance(value, (list, tuple, np.ndarray)):
        seq = value
    elif isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in {"none", "nan", "nat"}:
            return None
        try:
            seq = literal_eval(s)
        except (ValueError, SyntaxError):
            seq = FLOAT_TOKEN_RE.findall(s)
            if len(seq) < 3:
                return None
    elif isinstance(value, bytes):
        return _parse_ipp(value.decode(errors="ignore"))
    elif hasattr(value, "__iter__") and not isinstance(value, dict):
        try:
            seq = list(value)
        except TypeError:
            return None
    else:
        return None

    try:
        if isinstance(seq, str):
            return _parse_ipp(seq)
        coords = np.asarray(seq, dtype=float).reshape(-1)
        if coords.size < 3:
            return None
        x = float(coords[0])
        y = float(coords[1])
        z = float(coords[2])
        return (x, y, z)
    except Exception:
        return None


def _sort_key_for_column(col_name):
    if col_name == "ImagePositionPatient":

        def key(v):
            ipp = _parse_ipp(v)
            if ipp is None:
                return (1, _string_sort_key(v))
            return (0, ipp[2], ipp[1], ipp[0])

        return key

    if col_name in {"SliceLocation", "InstanceNumber", "AcquisitionNumber"}:

        def key(v):
            num = _parse_float(v)
            if num is None:
                return (1, _string_sort_key(v))
            return (0, num)

        return key

    return _string_sort_key


def _sorted_unique(values, col_name):
    seen = {}
    for v in values:
        key = _hashable_key(v)
        if key not in seen:
            seen[key] = v
    unique_vals = list(seen.values())
    if len(unique_vals) <= 1:
        return unique_vals
    key_fn = _sort_key_for_column(col_name)
    return sorted(unique_vals, key=key_fn)


def group_volumes(df):
    """Aggregate instance-level metadata into one row per volume identifier."""

    def agg_fun(col):
        vals = list(col.dropna())
        if len(vals) == 0:
            return float("NaN")
        unique_vals = _sorted_unique(vals, col.name)
        if len(unique_vals) == 1:
            return unique_vals[0]
        return unique_vals

    df = df.groupby("volume_id").agg(agg_fun)
    df = df.reset_index()
    df = df.dropna(axis=1, how="all")
    return df


def calculate_volume_length(df):
    """Compute reconstructed volume length in millimetres from slice geometry."""
    def calculate_total_volume_length(row):
        try:
            n_files = row["n_files"]
            thickness = abs(row["SliceThickness"])
            spacing = row.get("SpacingBetweenSlices", thickness)
            spacing = abs(spacing) if pd.notna(spacing) else thickness
            total_length = thickness + (n_files - 1) * abs(spacing)
            return total_length
        except Exception:
            return None

    df["n_files"] = df["dicom_path"].apply(
        lambda x: len(x) if isinstance(x, list) else 1
    )
    df["volume_length"] = df.apply(calculate_total_volume_length, axis=1)
    return df


def filter_volumes_by_size(df, min_length_mm, max_length_mm):
    """Keep volumes within inclusive length bounds, retaining missing lengths."""
    df = df[
        (df["volume_length"].isna())
        | (
            (df["volume_length"] >= min_length_mm)
            & (df["volume_length"] <= max_length_mm)
        )
    ]
    return df


def map_series_description(df, csv_tag_dict):
    if not csv_tag_dict or "SeriesDescription" not in df.columns:
        return df

    df_dict = pd.read_csv(csv_tag_dict)
    df_dict["SeriesDescription"] = df_dict["SeriesDescription"].apply(uniform_string)

    data_dict = df_dict.set_index("SeriesDescription")["phase"].to_dict()

    df["SeriesDescription"] = df["SeriesDescription"].fillna("inconnu")
    df["phase"] = df["SeriesDescription"].apply(uniform_string).replace(data_dict)

    mixt_phase_mask = df["phase"].str.lower().eq("mixte")
    acq = pd.to_numeric(df["AcquisitionNumber"], errors="coerce")
    df.loc[mixt_phase_mask & (acq == 1), "phase"] = "arteriel"
    df.loc[mixt_phase_mask & (acq == 2), "phase"] = "portal"

    known_phases = ["sans_injection", "arteriel", "mixte", "portal", "tardif"]
    known_discards = ["inutile", "inconnu"]

    unknown_descriptions = df[~df.phase.isin(known_phases + known_discards)].phase
    unique_unknown_descriptions = unknown_descriptions.unique().tolist()

    if len(unknown_descriptions) == 0:
        logger.info("No unknown SeriesDescription in dataset.")
    else:
        logger.info(
            "%s unmapped SeriesDescription, %s unique : %s",
            len(unknown_descriptions),
            len(unique_unknown_descriptions),
            unique_unknown_descriptions,
        )

    df = df[df.phase != "inutile"]

    return df


def compute_visit_order(df):
    if "date" not in df.columns:
        return df
    df_study = (
        df.reset_index()
        .groupby(["patient_key", "study_id"], group_keys=False)
        .first()
        .groupby("patient_key", group_keys=False)
        .apply(lambda x: x.sort_values(by=["date"]))
    )

    df_study["delay_since_prev_exam"] = df_study.groupby("patient_key")["date"].diff()
    df_study["delay_since_first_exam"] = df_study.groupby("patient_key")[
        "delay_since_prev_exam"
    ].cumsum()
    df_study["visit_order"] = df_study.groupby("patient_key")["date"].cumcount()
    logger.info("%s %s", df.shape, df_study.shape)

    df = df.merge(
        df_study[["delay_since_prev_exam", "delay_since_first_exam", "visit_order"]],
        on=["patient_key", "study_id"],
        left_index=False,
        right_index=False,
        how="left",
    )
    return df


def _normalize_instance_creation_time(value):
    value = _as_python_scalar(value)

    if value is None or _is_nan(value):
        return None

    if isinstance(value, dt_time):
        return value

    if isinstance(value, pd.Timestamp):
        return value.time()

    if isinstance(value, (list, tuple, np.ndarray)):
        parsed = [_normalize_instance_creation_time(v) for v in value]
        parsed = [t for t in parsed if t is not None]
        return min(parsed) if parsed else None

    if isinstance(value, (int, float)):
        return _normalize_instance_creation_time(str(value))

    if not isinstance(value, str):
        return None

    s = value.strip()
    if not s or s.lower() in {"none", "nan", "nat"}:
        return None

    try:
        literal = literal_eval(s)
    except (ValueError, SyntaxError):
        literal = None
    if literal is not None:
        # Avoid recursive no-op loops for scalar literals like:
        # "120000.0" -> 120000.0 -> "120000.0" -> ...
        is_scalar_noop = isinstance(literal, (str, int, float, bool)) and (
            str(literal).strip() == s
        )
        if not is_scalar_noop:
            parsed = _normalize_instance_creation_time(literal)
            if parsed is not None:
                return parsed

    matches = DATETIME_TIME_RE.findall(s)
    if matches:
        parsed = []
        for hh, mm, ss, us in matches:
            try:
                parsed.append(
                    dt_time(
                        hour=int(hh),
                        minute=int(mm),
                        second=int(ss) if ss else 0,
                        microsecond=int(us) if us else 0,
                    )
                )
            except ValueError:
                continue
        if parsed:
            return min(parsed)

    for fmt in ("%H%M%S.%f", "%H%M%S", "%H:%M:%S.%f", "%H:%M:%S"):
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        if pd.notna(parsed):
            return parsed.time()

    return None


def _normalize_acquisition_sort_number(value):
    value = _as_python_scalar(value)

    if value is None or _is_nan(value):
        return np.nan

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, (list, tuple, np.ndarray)):
        parsed = [_normalize_acquisition_sort_number(v) for v in value]
        parsed = [v for v in parsed if pd.notna(v)]
        return min(parsed) if parsed else np.nan

    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in {"none", "nan", "nat"}:
            return np.nan

        direct = _parse_float(s)
        if direct is not None:
            return direct

        try:
            literal = literal_eval(s)
        except (ValueError, SyntaxError):
            return np.nan
        return _normalize_acquisition_sort_number(literal)

    return np.nan


def compute_acquisition_order(df):
    df = df.copy()
    if "time" in df.columns:
        df["time"] = df["time"].apply(_normalize_instance_creation_time)
        time_delta = df["time"].apply(
            lambda t: (
                pd.Timedelta(
                    hours=t.hour,
                    minutes=t.minute,
                    seconds=t.second,
                    microseconds=t.microsecond,
                )
                if t is not None
                else pd.NaT
            )
        )
        time_delta = pd.to_timedelta(time_delta, errors="coerce")
    else:
        time_delta = pd.Series(pd.NaT, index=df.index, dtype="timedelta64[ns]")

    if "date" in df.columns:
        date_values = pd.to_datetime(df["date"], errors="coerce")
    else:
        date_values = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    df["_acq_timestamp"] = date_values + time_delta

    if "SeriesNumber" in df.columns:
        df["_series_number_sort"] = df["SeriesNumber"].apply(
            _normalize_acquisition_sort_number
        )
    if "AcquisitionNumber" in df.columns:
        df["_acquisition_number_sort"] = df["AcquisitionNumber"].apply(
            _normalize_acquisition_sort_number
        )

    # One row per (patient, study, volume) with representative acquisition keys.
    # Sort primarily by acquisition timestamp when available, then fallback to
    # numeric SeriesNumber and AcquisitionNumber.
    agg_map = {"_acq_timestamp": ("_acq_timestamp", "min")}
    if "_series_number_sort" in df.columns:
        agg_map["_series_number_sort"] = ("_series_number_sort", "min")
    if "_acquisition_number_sort" in df.columns:
        agg_map["_acquisition_number_sort"] = ("_acquisition_number_sort", "min")

    df_study = df.groupby(["patient_key", "study_id", "volume_id"], as_index=False).agg(
        **agg_map
    )

    sort_cols = ["patient_key", "study_id", "_acq_timestamp"]
    if "_series_number_sort" in df_study.columns:
        sort_cols.append("_series_number_sort")
    if "_acquisition_number_sort" in df_study.columns:
        sort_cols.append("_acquisition_number_sort")
    sort_cols.append("volume_id")

    df_study = df_study.sort_values(
        by=sort_cols,
        kind="mergesort",
        na_position="last",
    )
    group_cols = ["patient_key", "study_id"]

    df_study["delay_since_prev_acq_sec"] = (
        df_study.groupby(group_cols)["_acq_timestamp"].diff().dt.total_seconds()
    )
    first_mask = df_study.groupby(group_cols).cumcount() == 0
    df_study.loc[first_mask, "delay_since_prev_acq_sec"] = 0.0

    first_ts = df_study.groupby(group_cols)["_acq_timestamp"].transform("min")
    df_study["delay_since_first_acq_sec"] = (
        df_study["_acq_timestamp"] - first_ts
    ).dt.total_seconds()
    df_study["acquisition_order"] = df_study.groupby(group_cols).cumcount()

    df = df.merge(
        df_study[
            [
                "patient_key",
                "study_id",
                "volume_id",
                "delay_since_prev_acq_sec",
                "delay_since_first_acq_sec",
                "acquisition_order",
            ]
        ],
        on=["patient_key", "study_id", "volume_id"],
        left_index=False,
        right_index=False,
        how="left",
    )
    df = df.drop(
        columns=[
            # "_acq_timestamp",
            "_series_number_sort",
            "_acquisition_number_sort",
        ],
        errors="ignore",
    )
    return df


def drop_irrelevant_dicom_tags(df):
    important_dicom_tags = [
        "SeriesDescription",
        "PixelSpacingXY",
        "Rows",
        "Columns",
        "SliceThickness",
        "SpacingBetweenSlices",
        "InstanceNumber",
        "AcquisitionNumber",
        "SliceLocation",
        "ImagePositionPatient",
    ] + df.columns[df.isna().sum() == 0].to_list()
    logger.info("%s", important_dicom_tags)
    dicom_tags = [
        col
        for col in df.columns
        if any(c.isupper() for c in col)
        and "UID" not in col
        and col not in important_dicom_tags
    ]
    df = df.drop(columns=dicom_tags)
    return df


def reorder_columns(df):
    PRIORITY_COLS = [
        "patient_key",
        "volume_id",
        "study_id",
        "series_id",
        "date",
        "time",
    ]

    cols = df.columns.tolist()

    ordered_cols = [c for c in PRIORITY_COLS if c in cols] + [
        c for c in cols if c not in PRIORITY_COLS
    ]

    return df[ordered_cols]


def reorder_rows(df):
    sort_cols = []
    tmp = pd.DataFrame(index=df.index)

    if "patient_key" in df.columns:
        tmp["_sort_patient_key"] = (
            df["patient_key"].astype("string").fillna("").str.strip()
        )
        tmp.loc[tmp["_sort_patient_key"] == "", "_sort_patient_key"] = "~"
        sort_cols.append("_sort_patient_key")

    if "date" in df.columns:
        tmp["_sort_date"] = pd.to_datetime(df["date"], errors="coerce")
        sort_cols.append("_sort_date")

    if "time" in df.columns:
        time_values = df["time"].apply(
            lambda value: value.isoformat() if isinstance(value, dt_time) else value
        )
        tmp["_sort_time"] = pd.to_timedelta(time_values, errors="coerce")
        sort_cols.append("_sort_time")

    if not sort_cols:
        return df

    ordered_idx = tmp.sort_values(
        by=sort_cols,
        kind="mergesort",
        na_position="last",
    ).index
    return df.loc[ordered_idx].reset_index(drop=True)


def _resolve_cleaning_config(manifest: dict) -> dict:
    """Extract and validate the top-level ``cleaning`` manifest block."""
    cleaning = manifest.get("cleaning")
    if not isinstance(cleaning, dict):
        raise ValueError("Cleaning manifest must define a 'cleaning' object.")
    if cleaning.get("version") != 1:
        raise ValueError(
            "Cleaning manifest must define 'cleaning.version' equal to 1."
        )
    steps = cleaning.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Cleaning manifest must define a non-empty 'steps' list.")
    return cleaning


def _get_hook_outputs(step: dict) -> list[str]:
    """Resolve a clean hook and return its declared output columns."""
    hook = resolve_function_path(step["function"])
    outputs = get_clean_hook_outputs(hook)
    if not outputs:
        raise ValueError(
            f"Hook {step['function']!r} must declare clean outputs via @clean_hook."
        )
    return outputs


def _get_step_inputs(step: dict) -> set[str]:
    """Return the columns a step expects to read before it runs."""
    step_type = step["type"]
    if step_type == "hook":
        return set(step["source_columns"])
    if step_type == "filter":
        return {rule["column"] for rule in step["rules"]}
    if step_type == "coalesce_date":
        return set(step.get("candidates", DATE_CANDIDATES))
    if step_type == "coalesce_time":
        return set(step.get("candidates", TIME_CANDIDATES))
    if step_type == "sop_class":
        return {"SOPClassUID"}
    if step_type == "parse_image_type":
        return {"ImageType"}
    if step_type == "clean_scan_size":
        return {"Rows", "Columns", "SliceThickness"}
    if step_type == "normalize_string":
        return {step["column"]}
    if step_type == "pixel_spacing_xy":
        return {"PixelSpacing"}
    if step_type in {"standardize_iop", "classify_acquisition_plane"}:
        return {"ImageOrientationPatient"}
    if step_type == "build_volume_id":
        return set(step.get("preferred_columns") or []) | set(
            step.get("fallback_columns") or []
        )
    if step_type == "merge_volume_ids":
        return (
            {"volume_id"}
            | set(step.get("group_columns") or [])
            | set(step.get("z_sources") or [])
        )
    if step_type == "group_volumes":
        return {"volume_id"}
    if step_type == "compute_volume_length":
        return {"dicom_path", "SliceThickness", "SpacingBetweenSlices"}
    if step_type == "compute_visit_order":
        return {"patient_key", "study_id", "date"}
    if step_type == "compute_acquisition_order":
        return {
            "patient_key",
            "study_id",
            "volume_id",
            "date",
            "time",
            "SeriesNumber",
            "AcquisitionNumber",
        }
    return set()


def _get_step_outputs(step: dict) -> set[str]:
    """Return the columns a step is expected to create or rewrite."""
    step_type = step["type"]
    if step_type == "hook":
        outputs = set(_get_hook_outputs(step))
        if outputs == {"patient_key"} and step.get("source_columns") == ["patient_key"]:
            outputs |= {"_patient_key_raw", "patient_key_std_failed"}
        return outputs
    if step_type == "coalesce_date":
        return {"date"}
    if step_type == "coalesce_time":
        return {"time"}
    if step_type == "sop_class":
        return {"sop_class"}
    if step_type == "normalize_string":
        return {step["column"]}
    if step_type == "pixel_spacing_xy":
        return {"PixelSpacingXY"}
    if step_type == "standardize_iop":
        return {"ImageOrientationPatient"}
    if step_type == "classify_acquisition_plane":
        return {"acquisition_plane", "acquisition_angle", "acquisition_axis"}
    if step_type == "build_volume_id":
        return {"volume_id"}
    if step_type == "merge_volume_ids":
        return {"volume_id"}
    if step_type == "compute_volume_length":
        return {"n_files", "volume_length"}
    if step_type == "compute_visit_order":
        return {"delay_since_prev_exam", "delay_since_first_exam", "visit_order"}
    if step_type == "compute_acquisition_order":
        return {
            "_acq_timestamp",
            "delay_since_prev_acq_sec",
            "delay_since_first_acq_sec",
            "acquisition_order",
        }
    return set()


def _validate_string_list(
    value,
    field_name: str,
    *,
    allow_empty: bool,
) -> None:
    """Validate that a manifest field is a list of non-empty strings."""
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of strings.")
    if not allow_empty and not value:
        raise ValueError(f"{field_name} must be a non-empty list of strings.")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field_name} must contain only non-empty strings.")


def _validate_optional_number(step: dict, field_name: str) -> None:
    """Validate an optional numeric step parameter when it is present."""
    value = step.get(field_name)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric when provided.")


def _validate_step_config(step: dict) -> None:
    """Validate step-specific options beyond the generic type checks."""
    step_type = step["type"]

    if step_type in {"coalesce_date", "coalesce_time"} and "candidates" in step:
        _validate_string_list(
            step["candidates"],
            f"{step_type}.candidates",
            allow_empty=False,
        )

    if step_type == "normalize_string":
        column = step.get("column")
        if not isinstance(column, str) or not column:
            raise ValueError("normalize_string steps must define a non-empty 'column'.")

    if step_type == "classify_acquisition_plane":
        _validate_optional_number(step, "angle_thresh_deg")

    if step_type == "build_volume_id":
        if "preferred_columns" in step:
            _validate_string_list(
                step["preferred_columns"],
                "build_volume_id.preferred_columns",
                allow_empty=True,
            )
        if "fallback_columns" in step:
            _validate_string_list(
                step["fallback_columns"],
                "build_volume_id.fallback_columns",
                allow_empty=True,
            )

    if step_type == "merge_volume_ids":
        if "group_columns" in step:
            _validate_string_list(
                step["group_columns"],
                "merge_volume_ids.group_columns",
                allow_empty=True,
            )
        if "z_sources" in step:
            _validate_string_list(
                step["z_sources"],
                "merge_volume_ids.z_sources",
                allow_empty=True,
            )
        _validate_optional_number(step, "z_tolerance")


def validate_cleaning_manifest(manifest: dict) -> list[dict]:
    """Validate ``cleaning.steps`` and return the normalized step list."""
    cleaning = _resolve_cleaning_config(manifest)
    validated_steps = []

    for index, step in enumerate(cleaning["steps"], start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Cleaning step {index} must be an object.")
        step_type = step.get("type")
        if step_type not in STEP_REGISTRY:
            raise ValueError(f"Unknown cleaning step type: {step_type!r}.")

        if step_type == "hook":
            if not isinstance(step.get("function"), str) or not step["function"]:
                raise ValueError("Hook steps must define a non-empty 'function'.")
            source_columns = step.get("source_columns")
            if not isinstance(source_columns, list) or not source_columns:
                raise ValueError("Hook steps must define non-empty 'source_columns'.")
            _validate_string_list(
                source_columns,
                "hook.source_columns",
                allow_empty=False,
            )
            _get_hook_outputs(step)

        if step_type == "filter":
            if step.get("kind") not in {"keep", "discard"}:
                raise ValueError("Filter steps must define kind='keep' or 'discard'.")
            if step.get("scope") not in {"row", "volume"}:
                raise ValueError("Filter steps must define scope='row' or 'volume'.")
            if step.get("logic") not in {"and", "or"}:
                raise ValueError("Filter steps must define logic='and' or 'or'.")
            rules = step.get("rules")
            if not isinstance(rules, list) or not rules:
                raise ValueError("Filter steps must define a non-empty 'rules' list.")
            if "keep_null" in step and not isinstance(step["keep_null"], bool):
                raise ValueError("Filter steps must define keep_null as a boolean.")
            for rule in rules:
                if not isinstance(rule, dict):
                    raise ValueError("Each filter rule must be an object.")
                if "column" not in rule or "op" not in rule:
                    raise ValueError("Each filter rule must define 'column' and 'op'.")
                if not isinstance(rule["column"], str) or not rule["column"]:
                    raise ValueError("Each filter rule must define a non-empty 'column'.")
                if not isinstance(rule["op"], str) or not rule["op"]:
                    raise ValueError("Each filter rule must define a non-empty 'op'.")
                if rule["op"] not in SUPPORTED_FILTER_OPERATORS:
                    raise ValueError(
                        f"Unsupported filter operator: {rule['op']!r}."
                    )
                if rule["op"] not in {"is_null", "not_null"} and "value" not in rule:
                    raise ValueError(
                        f"Filter operator {rule['op']!r} requires a 'value'."
                    )
                if rule["op"] in {"in", "not_in"} and not isinstance(
                    rule["value"], list
                ):
                    raise ValueError(
                        f"Filter operator {rule['op']!r} requires a list 'value'."
                    )

        _validate_step_config(step)

        validated_steps.append(step)

    return validated_steps


def _collect_required_input_columns(
    steps: list[dict],
) -> list[str]:
    """Infer which source columns must be loaded from the parsed CSV."""
    required = set(BASE_INPUT_COLUMNS)
    produced = set()
    for step in steps:
        for col in _get_step_inputs(step):
            if col not in produced:
                required.add(col)
        produced.update(_get_step_outputs(step))
    return sorted(required)


def _step_label(step: dict) -> str:
    """Build a human-readable label for logging and validation errors."""
    if step.get("name"):
        return str(step["name"])
    if step["type"] == "hook":
        return f"hook {step['function']}"
    if step["type"] == "filter":
        return f"{step['kind']} {step['scope']} filter"
    return step["type"].replace("_", " ")


def _ensure_columns_present(df: pd.DataFrame, columns: set[str], step: dict) -> None:
    """Raise when a step references columns that are missing from ``df``."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Step '{_step_label(step)}' requires missing columns: {sorted(missing)}"
        )


def _hook_output_frame(result: pd.Series, outputs: list[str]) -> pd.DataFrame:
    """Coerce hook results into a DataFrame aligned with declared outputs."""
    out = result.apply(pd.Series)
    if set(outputs).issubset(out.columns):
        return out.reindex(columns=outputs)
    if out.shape[1] != len(outputs):
        raise ValueError(
            f"Hook returned {out.shape[1]} columns, expected {len(outputs)} outputs."
        )
    out.columns = outputs
    return out


def _run_hook_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Execute a manifest hook step against one or more source columns."""
    source_columns = step["source_columns"]
    _ensure_columns_present(df, set(source_columns), step)

    hook = resolve_function_path(step["function"])
    outputs = _get_hook_outputs(step)

    if outputs == ["patient_key"] and source_columns == ["patient_key"]:
        return apply_patient_key_standardization(df, hook)

    df = df.copy()

    if len(source_columns) == 1:
        source_col = source_columns[0]
        result = df[source_col].apply(hook)
    else:
        result = df[source_columns].apply(lambda row: hook(row.copy()), axis=1)

    if len(outputs) == 1:
        df[outputs[0]] = result
    else:
        out = _hook_output_frame(result, outputs)
        for col in outputs:
            df[col] = out[col]

    return df


def _normalize_date_candidate(value):
    """Convert a date candidate into a scalar timestamp or ``NaT``."""
    if isinstance(value, list):
        return pd.NaT
    if value is None or _is_nan(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value
    return pd.to_datetime(value, errors="coerce")


def _rule_mask(df: pd.DataFrame, rule: dict) -> pd.Series:
    """Evaluate a single declarative filter rule against ``df``."""
    col = rule["column"]
    op = rule["op"]
    value = rule.get("value")
    series = df[col]

    if op == "eq":
        return series == value
    if op == "ne":
        return series != value
    if op == "in":
        return series.isin(value)
    if op == "not_in":
        return ~series.isin(value)
    if op == "contains":
        return series.astype("string").str.contains(str(value), regex=False, na=False)
    if op == "icontains":
        return series.astype("string").str.contains(
            str(value), regex=False, case=False, na=False
        )
    if op == "regex":
        return series.astype("string").str.contains(str(value), regex=True, na=False)
    if op == "lt":
        return pd.to_numeric(series, errors="coerce") < value
    if op == "lte":
        return pd.to_numeric(series, errors="coerce") <= value
    if op == "gt":
        return pd.to_numeric(series, errors="coerce") > value
    if op == "gte":
        return pd.to_numeric(series, errors="coerce") >= value
    if op == "is_null":
        return series.isna()
    if op == "not_null":
        return series.notna()
    raise ValueError(f"Unsupported filter operator: {op!r}")


def _run_filter_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Apply a manifest filter step using explicit keep/discard semantics."""
    required_columns = {rule["column"] for rule in step["rules"]}
    _ensure_columns_present(df, required_columns, step)

    masks = [_rule_mask(df, rule).fillna(False) for rule in step["rules"]]
    mask = masks[0].copy()
    for current in masks[1:]:
        if step["logic"] == "and":
            mask = mask & current
        else:
            mask = mask | current

    if step.get("keep_null"):
        null_mask = df[list(required_columns)].isna().any(axis=1)
        mask = mask | null_mask

    if step["kind"] == "keep":
        return df[mask].copy()
    return df[~mask].copy()


def _safe_uid_keyword(value):
    """Resolve a DICOM UID to its keyword, returning ``None`` on failure."""
    try:
        return UID(value).keyword
    except Exception:
        return None


def _run_coalesce_date_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Normalize candidate date columns and populate the canonical ``date``."""
    df = to_dates(df.copy())
    return add_date(df, candidate_columns=step.get("candidates"))


def _run_coalesce_time_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Normalize candidate time columns and populate the canonical ``time``."""
    df = to_times(df.copy())
    return add_time(df, candidate_columns=step.get("candidates"))


def _run_sop_class_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Translate ``SOPClassUID`` values into DICOM keyword names."""
    if "SOPClassUID" not in df.columns:
        return df
    df = df.copy()
    df["sop_class"] = df["SOPClassUID"].apply(_safe_uid_keyword)
    return df


def _run_parse_image_type_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Parse serialized ``ImageType`` values into a structured representation."""
    return filter_image_type(df)


def _run_clean_scan_size_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Normalize scan-size fields used by later geometry filters."""
    return clean_scan_size(df)


def _run_normalize_string_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Lowercase and normalize a configured string column in-place."""
    column = step["column"]
    if column not in df.columns:
        return df
    df = df.copy()
    df[column] = df[column].apply(uniform_string)
    return df


def _pixel_spacing_xy_value(value):
    """Extract the first in-plane spacing value from ``PixelSpacing``."""
    if value is None or _is_nan(value):
        return None
    if isinstance(value, str):
        try:
            parsed = literal_eval(value)
        except (ValueError, SyntaxError):
            return None
    else:
        parsed = value
    if isinstance(parsed, (list, tuple, np.ndarray)) and len(parsed) > 0:
        return parsed[0]
    return None


def _run_pixel_spacing_xy_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Derive numeric ``PixelSpacingXY`` values from ``PixelSpacing``."""
    if "PixelSpacing" not in df.columns:
        return df
    df = df.copy()
    df["PixelSpacingXY"] = df["PixelSpacing"].apply(_pixel_spacing_xy_value)
    df["PixelSpacingXY"] = pd.to_numeric(df["PixelSpacingXY"], errors="coerce")
    return df


def _run_standardize_iop_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Canonicalize ``ImageOrientationPatient`` values for downstream geometry."""
    if "ImageOrientationPatient" not in df.columns:
        return df
    df = df.copy()
    df["ImageOrientationPatient"] = df["ImageOrientationPatient"].apply(standardize_iop)
    return df


def _run_classify_acquisition_plane_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Classify acquisition plane, angle, and axis from standardized IOP values."""
    if "ImageOrientationPatient" not in df.columns:
        return df
    df = df.copy()
    angle_thresh_deg = float(step.get("angle_thresh_deg", 10.0))
    (
        df["acquisition_plane"],
        df["acquisition_angle"],
        df["acquisition_axis"],
    ) = zip(
        *df["ImageOrientationPatient"].map(
            lambda x: classify_plane_from_iop(x, angle_thresh_deg)
        )
    )
    return df


def _run_build_volume_id_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Build volume identifiers from the configured preferred and fallback columns."""
    return generate_volume_id(
        df,
        preferred_cols=step.get("preferred_columns"),
        fallback_cols=step.get("fallback_columns"),
    )


def _run_merge_volume_ids_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Merge near-duplicate volume ids using configurable geometry tolerances."""
    return correct_volume_ids(
        df,
        z_tolerance=float(step.get("z_tolerance", 1e-3)),
        group_columns=step.get("group_columns"),
        z_sources=step.get("z_sources"),
    )


def _run_group_volumes_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Aggregate slice-level rows into volume-level records."""
    return group_volumes(df)


def _run_compute_volume_length_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Compute reconstructed volume length and file counts per volume."""
    return calculate_volume_length(df)


def _run_compute_visit_order_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Annotate each volume with patient-level exam ordering metadata."""
    return compute_visit_order(df)


def _run_compute_acquisition_order_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Annotate each volume with within-study acquisition ordering metadata."""
    return compute_acquisition_order(df)


def _run_finalize_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Drop empty helper columns and restore the canonical output ordering."""
    df = df.dropna(axis=1, how="all")
    df = reorder_columns(df)
    df = reorder_rows(df)
    return df


STEP_REGISTRY: dict[str, Callable[[pd.DataFrame, dict], pd.DataFrame]] = {
    "hook": _run_hook_step,
    "filter": _run_filter_step,
    "coalesce_date": _run_coalesce_date_step,
    "coalesce_time": _run_coalesce_time_step,
    "sop_class": _run_sop_class_step,
    "parse_image_type": _run_parse_image_type_step,
    "clean_scan_size": _run_clean_scan_size_step,
    "normalize_string": _run_normalize_string_step,
    "pixel_spacing_xy": _run_pixel_spacing_xy_step,
    "standardize_iop": _run_standardize_iop_step,
    "classify_acquisition_plane": _run_classify_acquisition_plane_step,
    "build_volume_id": _run_build_volume_id_step,
    "merge_volume_ids": _run_merge_volume_ids_step,
    "group_volumes": _run_group_volumes_step,
    "compute_volume_length": _run_compute_volume_length_step,
    "compute_visit_order": _run_compute_visit_order_step,
    "compute_acquisition_order": _run_compute_acquisition_order_step,
    "finalize": _run_finalize_step,
}


def run_clean_pipeline(df: pd.DataFrame, steps: list[dict]) -> pd.DataFrame:
    """Run validated cleaning steps sequentially with per-step reporting."""
    report_volumes(df, "initial load")

    for step in steps:
        df_prev = df.copy()
        df = STEP_REGISTRY[step["type"]](df, step)
        label = _step_label(step)
        report_volumes(df, label)
        try:
            report_change(df, df_prev)
        except Exception:
            logger.debug("Could not compute detailed change report for step %s", label)

    return df


def clean_and_save_data(
    csv_path,
    csv_path_out,
    manifest,
):
    """Run the manifest-defined metadata-curation pipeline and write its CSV output."""
    steps = validate_cleaning_manifest(manifest)
    required_columns = _collect_required_input_columns(steps)
    df = load_data(csv_path, required_columns=required_columns)
    input_rows = len(df)
    df = run_clean_pipeline(df, steps)

    if csv_path_out:
        df.to_csv(csv_path_out, index=False)
        logger.info("Cleaned data saved to %s", csv_path_out)
    logger.info("shape : %s", df.shape)
    logger.info("columns : %s", df.columns)

    output_rows = len(df)
    filtered_rows = max(0, input_rows - output_rows)
    log_task_summary(
        logger,
        "Cleaning",
        total_rows=input_rows,
        processed_rows=input_rows,
        succeeded_rows=output_rows,
        skipped_rows=filtered_rows,
        failed_rows=0,
        success_label="retained",
        skipped_label="filtered out",
    )
    logger.info("Cleaning done ✔")


if __name__ == "__main__":
    setup_logging()
    args = parse_arguments()
    if args.dry_run:
        logger.info("Dry run: clean")
        print_args(args)
        raise SystemExit(0)
    manifest = load_manifest(
        args.manifest, base_path=Path(__file__).resolve().parents[1]
    )
    clean_and_save_data(
        args.csv_path,
        args.csv_path_out,
        manifest,
    )
