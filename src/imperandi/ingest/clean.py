import argparse
import logging
import hashlib
import re
from ast import literal_eval
from datetime import time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd
from pydicom.uid import UID
from unidecode import unidecode

from imperandi.utils.manifest import load_manifest
from imperandi.utils.geometry import (
    classify_plane_from_iop,
    standardize_iop,
)
from imperandi.ingest.hook_manifests import (
    apply_id_standardization,
    apply_derived_columns,
)
from imperandi.utils.logging import setup_logging
from imperandi.utils.misc import print_args, report_volumes, report_change
from imperandi.utils.datetime import to_dates, to_times
from imperandi.datasets_config.defaults import (
    DEFAULT_DICOM_TAGS,
    DEFAULT_VOLUME_LOWERBOUND,
    DEFAULT_VOLUME_UPPERBOUND,
    DEFAULT_MAX_PIXEL_SPACING_MM,
    DEFAULT_MAX_SLICE_THICKNESS_MM,
    DATE_CANDIDATES,
    TIME_CANDIDATES,
)

COLUMNS_TO_USE = [
    "patient_key",
    "study_id",
    "series_id",
    "dicom_path",
] + DEFAULT_DICOM_TAGS


pd.options.mode.chained_assignment = None
logger = logging.getLogger(__name__)

DATETIME_TIME_RE = re.compile(
    r"datetime\.time\(\s*(\d{1,2})\s*,\s*(\d{1,2})(?:\s*,\s*(\d{1,2}))?(?:\s*,\s*(\d{1,6}))?\s*\)"
)


def add_clean_arguments(
    parser: argparse.ArgumentParser,
    include_manifest: bool = True,
    include_csv_path: bool = True,
    include_csv_path_out: bool = True,
    include_dry_run: bool = True,
) -> None:
    if include_csv_path:
        parser.add_argument(
            "csv_path_pos",
            type=str,
            nargs="*",
            default=None,
            help=(
                "Path to the input CSV file / File pattern for multi-file CSV. "
                "Defaults to ./dicom_index.csv."
            ),
        )
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
    parser.add_argument(
        "--csv_dict_path",
        type=str,
        default=None,
        help="Path to the CSV tag dictionary file",
    )
    if include_manifest:
        parser.add_argument(
            "--manifest",
            type=str,
            default=None,
            help="Dataset manifest name or path to manifest JSON.",
        )
    parser.add_argument(
        "--volume_min",
        type=float,
        default=DEFAULT_VOLUME_LOWERBOUND,
        help="Minimum allowable volume depth.",
    )
    parser.add_argument(
        "--volume_max",
        type=float,
        default=DEFAULT_VOLUME_UPPERBOUND,
        help="Maximum allowable volume depth.",
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
    csv_in = (
        args.csv_path_opt
        if getattr(args, "csv_path_opt", None) is not None
        else getattr(args, "csv_path_pos", None)
    )
    if not csv_in:
        csv_paths = [Path.cwd() / "dicom_index.csv"]
    else:
        csv_paths = [Path(p) for p in csv_in]

    args.csv_path = [str(p) for p in csv_paths]
    first_csv = csv_paths[0]

    if not args.csv_path_out:
        args.csv_path_out = str(_default_clean_output_path(first_csv))

    for attr in ("csv_path_pos", "csv_path_opt"):
        if hasattr(args, attr):
            delattr(args, attr)

    return args


def read_csv_with_valid_columns(file):
    available_columns = pd.read_csv(file, nrows=0).columns
    valid_columns = [col for col in COLUMNS_TO_USE if col in available_columns]
    return pd.read_csv(file, usecols=valid_columns)


def load_data(csv_path):
    if len(csv_path) == 1:
        df = read_csv_with_valid_columns(csv_path[0])
    else:
        dfs = [read_csv_with_valid_columns(file) for file in csv_path]
        df = pd.concat(dfs, ignore_index=True)

    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.dropna(axis=1, how="all")
    logger.info("%s %s", df.shape, df.columns)

    return df


def filter_ct_modality(df):
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


def add_date(df):
    candidate_cols = [col for col in DATE_CANDIDATES if col in df.columns]
    if not candidate_cols:
        return df

    parsed_by_candidate = {
        col: pd.to_datetime(
            df[col].apply(lambda x: pd.NaT if isinstance(x, list) else x),
            errors="coerce",
        )
        for col in candidate_cols
    }
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


def add_time(df):
    candidate_cols = [col for col in TIME_CANDIDATES if col in df.columns]
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
            lambda x: not any(sub in x for sub in excluded_substrings)
        )
    ]

    return df


def clean_scan_size(df):
    df = df.dropna(axis=1, how="all")
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


def generate_volume_id(df):

    preferred_cols = [
        "patient_key",
        "study_id",
        "series_id",
        "ImageType",
        "AcquisitionNumber",
        "ImageOrientationPatient",
        "SliceThickness",
        "PixelSpacingXY",
    ]
    fallback_cols = ["patient_key", "study_id", "series_id"]

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

    # If df truly has no columns (or empty selection), assign a single id
    if not cols_to_use:
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
    joined = df[cols_to_use].map(_to_stable_str).agg("||".join, axis=1)

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


def correct_volume_ids(df, z_tolerance=1e-3):
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

    preferred_group_cols = [
        "patient_key",
        "study_id",
        "series_id",
        "ImageType",
        "ImageOrientationPatient",
        "SliceThickness",
        "PixelSpacingXY",
    ]
    fallback_group_cols = ["patient_key", "study_id", "series_id"]

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

        # --- get z positions robustly ---
        z_positions = None

        if "ImagePositionPatient" in group_df.columns:
            s = group_df["ImagePositionPatient"]
            # keep only rows with a usable tuple/list of len>=3
            mask = s.notna() & s.apply(
                lambda v: isinstance(v, (list, tuple)) and len(v) >= 3
            )
            if mask.any():
                z_positions = s[mask].apply(lambda v: float(v[2])).to_numpy()

        if (
            z_positions is None or len(z_positions) < 2
        ) and "SliceLocation" in group_df.columns:
            s = group_df["SliceLocation"]
            mask = s.notna()
            if mask.any():
                try:
                    z_positions = s[mask].astype(float).to_numpy()
                except Exception:
                    # non-numeric slice locations -> skip
                    z_positions = None

        if z_positions is None or len(z_positions) < 2:
            continue

        # --- check consistent spacing ---
        z_sorted = np.sort(z_positions)
        z_diff = np.diff(z_sorted)

        # If duplicates exist, drop zeros before checking (common with repeated slices)
        z_diff_nz = z_diff[np.abs(z_diff) > z_tolerance]

        if len(z_diff_nz) < 1:
            # all slices same z (or too few distinct) -> can't decide
            continue

        consistent_spacing = np.all(
            np.isclose(z_diff_nz, z_diff_nz[0], atol=z_tolerance)
        )

        # --- optional debug print (robust to missing cols) ---
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

        logger.info(
            "%s : %s pseudo-volumes, %s total files",
            summary,
            len(volume_ids),
            len(group_df),
        )

        if consistent_spacing:
            logger.info("👫 Merged")
            canonical_id = sorted(map(str, volume_ids))[0]
            for vol_id in volume_ids:
                updated_ids[vol_id] = canonical_id
        else:
            logger.info("👍 They are different volumes")

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
        try:
            seq = literal_eval(value)
        except (ValueError, SyntaxError):
            return None
    else:
        return None

    try:
        if len(seq) < 3:
            return None
        x = float(seq[0])
        y = float(seq[1])
        z = float(seq[2])
        return (z, y, x)
    except Exception:
        return None


def _sort_key_for_column(col_name):
    if col_name == "ImagePositionPatient":

        def key(v):
            ipp = _parse_ipp(v)
            if ipp is None:
                return (1, _string_sort_key(v))
            return (0, ipp[0], ipp[1], ipp[2])

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


def filter_volumes_by_size(df, t_min, t_max):
    df = df[
        (df["volume_length"].isna())
        | ((df["volume_length"] >= t_min) & (df["volume_length"] <= t_max))
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
            lambda t: pd.Timedelta(
                hours=t.hour,
                minutes=t.minute,
                seconds=t.second,
                microseconds=t.microsecond,
            )
            if t is not None
            else pd.NaT
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

    df_study = df.groupby(
        ["patient_key", "study_id", "volume_id"], as_index=False
    ).agg(**agg_map)

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
        (df_study["_acq_timestamp"] - first_ts).dt.total_seconds()
    )
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


def clean_and_save_data(
    csv_path, csv_path_out, csv_dict_path, manifest, volume_min, volume_max
):
    df = load_data(csv_path)
    report_volumes(df, "initial load")

    df = apply_id_standardization(df, manifest, logger=logger)
    report_volumes(df, "standardize patient key")

    df = to_dates(df)
    df = to_times(df)
    df = add_date(df)  # generic date column
    df = add_time(df)  # generic time column

    df_prev = df.copy()
    df = apply_derived_columns(df, manifest)
    report_volumes(df, "add new columns based on patient key")
    report_change(df, df_prev)

    df_prev = df.copy()
    df = filter_ct_modality(df)
    report_volumes(df, "filtering CT modality")
    report_change(df, df_prev, col="Modality")

    df_prev = df.copy()
    df = remove_pet_ct(df)
    report_volumes(df, "removing PET-CT modality")
    report_change(df, df_prev, col="ModalitiesInStudy")

    df_prev = df.copy()
    df = filter_image_type(df)
    report_volumes(df, "filtering image type")
    report_change(df, df_prev, col="ImageType")

    df_prev = df.copy()
    df = clean_scan_size(df)
    report_volumes(df, "cleaning scan size")
    report_change(df, df_prev)

    df_prev = df.copy()
    df = remove_scouts_localizers(df)
    report_volumes(df, "removing localizers / scouts")
    report_change(df, df_prev)

    df_prev = df.copy()
    df = remove_other_organs_description(df)
    report_volumes(df, "removing other organs")
    report_change(df, df_prev, col="SeriesDescription")

    df_prev = df.copy()
    df = clean_pixel_spacing(df)
    report_volumes(df, "cleaning pixel spacing")

    if "ImageOrientationPatient" in df.columns:
        df_prev = df.copy()
        df["ImageOrientationPatient"] = df["ImageOrientationPatient"].apply(
            standardize_iop
        )
        df = filter_by_acquisition_plane(df)
        report_volumes(df, "keeping only AXIAL acquisitions")
        report_change(df, df_prev)

    df_prev = df.copy()
    df = generate_volume_id(df)
    report_volumes(df, "generating volume IDs")
    report_change(df, df_prev)

    df_prev = df.copy()
    df = correct_volume_ids(df)
    report_volumes(df, "merging multi-acquisition volumes IDs")
    report_change(df, df_prev)

    df = group_volumes(df)
    report_volumes(df, "grouping by volume IDs")

    df = calculate_volume_length(df)
    report_volumes(df, "computing volume length")

    df_prev = df.copy()
    df = filter_volumes_by_size(df, volume_min, volume_max)
    report_volumes(df, f"filtering volumes by size [{volume_min}, {volume_max}]")
    report_change(df, df_prev)

    df_prev = df.copy()
    df = map_series_description(df, csv_dict_path)
    report_volumes(df, "mapping series descriptions")
    report_change(df, df_prev, col="SeriesDescription")

    df = compute_visit_order(df)
    df = compute_acquisition_order(df)

    df = df.dropna(axis=1, how="all") # drop empty columns

    df = reorder_columns(df)
    df = reorder_rows(df)

    if csv_path_out:
        df.to_csv(csv_path_out, index=False)
        logger.info("Cleaned data saved to %s", csv_path_out)
    logger.info("shape : %s", df.shape)
    logger.info("columns : %s", df.columns)

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
        args.csv_dict_path,
        manifest,
        args.volume_min,
        args.volume_max,
    )
