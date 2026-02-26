import warnings
import glob
import io
import json
import logging
import os
from pathlib import Path
import argparse
from multiprocessing import cpu_count
from typing import Optional, Union

import pandas as pd
from tqdm import tqdm
from pandarallel import pandarallel
from pydicom import dcmread, config

from imperandi.utils.archive_io import (
    DEFAULT_ARCHIVE_MAX_DEPTH,
    decode_archive_uri,
    discover_dicom_sources,
    is_archive_filename,
    is_archive_uri,
    read_archive_member_bytes,
)
from imperandi.utils.logging import setup_logging
from imperandi.utils.misc import print_args
from imperandi.utils.manifest import load_manifest
from imperandi.datasets_config.defaults import DEFAULT_DICOM_TAGS
from imperandi.ingest.hook_manifests import apply_id_standardization

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Make reading tolerant of non-conformant values
config.settings.reading_validation_mode = config.IGNORE  # or config.WARN


# -------------------------
# CLI
# -------------------------
def add_parse_arguments(
    parser: argparse.ArgumentParser,
    include_manifest: bool = True,
    include_dry_run: bool = True,
) -> None:
    parser.add_argument(
        "root_path_pos",
        type=str,
        nargs="?",
        default=None,
        help=(
            "Root path where the DICOM files are located. "
            "Supports glob patterns (e.g. '/data/site_*'). "
            "Defaults to current working directory."
        ),
    )
    parser.add_argument(
        "output_dir_pos",
        type=str,
        nargs="?",
        default=None,
        help="Directory to save output CSV files. Defaults to parent of root_path.",
    )
    parser.add_argument(
        "--root_path",
        dest="root_path_opt",
        type=str,
        help=(
            "Root path where the DICOM files are located. "
            "Supports glob patterns (e.g. '/data/site_*')."
        ),
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir_opt",
        type=str,
    )
    if include_manifest:
        parser.add_argument(
            "--manifest",
            type=str,
            default=None,
            help="Dataset manifest name or path to manifest JSON.",
        )

    # Tag reading
    parser.add_argument(
        "--tags",
        type=str,
        default="",
        help="Comma-separated list of additional DICOM keyword tags to read (e.g. PatientID,StudyInstanceUID,SeriesInstanceUID,Modality). "
        "These are added to the default selected tag set.",
    )
    parser.add_argument(
        "--force_dicom_read",
        action="store_true",
        default=False,
        help="Pass force=True to pydicom dcmread (useful for non-compliant DICOMs).",
    )

    # ID extraction
    parser.add_argument(
        "--id_source",
        type=str,
        default="auto",
        choices=["path", "tags", "auto"],
        help="'path': use root/patient/study/series structure. "
        "'tags': use DICOM tags. "
        "'auto': use tags when present else fallback to path.",
    )
    parser.add_argument(
        "--patient_key_from",
        type=str,
        default="PatientName",
        help="DICOM keyword to use for patient_key when id_source is 'tags' or 'auto' (e.g. PatientID or PatientName).",
    )
    parser.add_argument(
        "--study_id_from",
        type=str,
        default="StudyInstanceUID",
        help="DICOM keyword to use for study_id when id_source is 'tags' or 'auto' (e.g. StudyInstanceUID or StudyID).",
    )
    parser.add_argument(
        "--series_id_from",
        type=str,
        default="SeriesInstanceUID",
        help="DICOM keyword to use for series_id when id_source is 'tags' or 'auto' (e.g. SeriesInstanceUID or SeriesNumber).",
    )

    # Performance
    parser.add_argument(
        "--checkpoint_frequency",
        default=None,
        type=int,
        help="If set, process DICOM files in chunks of N and write checkpoint CSVs.",
    )
    parser.add_argument(
        "--snapshot_tags",
        action="store_true",
        default=False,
        help="Write a full recursive DICOM-tag snapshot for a sampled subset.",
    )
    parser.add_argument(
        "--snapshot_sample_size",
        default=500,
        type=int,
        help="Number of sampled DICOM files for full-tag snapshot generation.",
    )
    parser.add_argument(
        "--snapshot_seed",
        default=42,
        type=int,
        help="Random seed for deterministic snapshot sampling.",
    )
    parser.add_argument(
        "--num_workers",
        default=cpu_count(),
        type=int,
        help="Number of parallel workers (default: all CPUs).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=False, help="Verbose mode."
    )
    parser.add_argument(
        "--archive_max_depth",
        type=int,
        default=DEFAULT_ARCHIVE_MAX_DEPTH,
        help="Maximum recursion depth for nested archives.",
    )
    parser.add_argument(
        "--archive_detect_sample_size",
        type=int,
        default=128,
        help="Per-root deterministic sample size used to detect archive presence.",
    )
    parser.add_argument(
        "--archive_cache_dir",
        type=str,
        default=None,
        help="Optional cache directory for materialized archive members.",
    )
    parser.add_argument(
        "--keep_archive_cache",
        action="store_true",
        default=False,
        help="Keep materialized archive cache after the command finishes.",
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
    include_dry_run: bool = True,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process DICOM files: read selected header tags once, then compute patient/study/series IDs from tags or path."
        ),
        add_help=add_help,
    )
    add_parse_arguments(
        parser,
        include_manifest=include_manifest,
        include_dry_run=include_dry_run,
    )
    return parser


def resolve_root_paths(root_path: Optional[Union[str, Path]]) -> list[Path]:
    if root_path is None:
        return [Path.cwd()]

    root_str = str(root_path)
    if glob.has_magic(root_str):
        matches = [Path(p) for p in glob.glob(root_str, recursive=True)]
        return sorted(
            {
                p
                for p in matches
                if p.is_dir() or (p.is_file() and is_archive_filename(p.name))
            }
        )

    return [Path(root_str)]


def default_output_dir(root_path: Optional[str]) -> Path:
    if root_path is None:
        return Path.cwd().parent

    if glob.has_magic(str(root_path)):
        matched_roots = resolve_root_paths(root_path)
        if matched_roots:
            return matched_roots[0].parent
        return Path.cwd()

    return Path(root_path).parent


def _iter_root_files_deterministic(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            yield Path(dirpath) / filename


def detect_archive_mode_by_subsample(
    resolved_roots: list[Path], sample_size: int = 128
) -> bool:
    sample_size = max(1, int(sample_size))
    for root in resolved_roots:
        if root.is_file():
            if is_archive_filename(root.name):
                return True
            continue

        if not root.is_dir():
            continue

        scanned = 0
        for path in _iter_root_files_deterministic(root):
            if is_archive_filename(path.name):
                return True
            scanned += 1
            if scanned >= sample_size:
                break
    return False


def normalize_parse_args(args: argparse.Namespace) -> argparse.Namespace:
    root_in = (
        args.root_path_opt
        if getattr(args, "root_path_opt", None) is not None
        else (
            getattr(args, "root_path_pos", None)
            if getattr(args, "root_path_pos", None) is not None
            else getattr(args, "root_path", None)
        )
    )
    output_in = (
        args.output_dir_opt
        if getattr(args, "output_dir_opt", None) is not None
        else (
            getattr(args, "output_dir_pos", None)
            if getattr(args, "output_dir_pos", None) is not None
            else getattr(args, "output_dir", None)
        )
    )

    root_path = Path(root_in) if root_in else Path.cwd()
    output_dir = Path(output_in) if output_in else default_output_dir(root_in)
    args.root_path = str(root_path)
    args.output_dir = str(output_dir)
    args.archive_max_depth = int(
        getattr(args, "archive_max_depth", DEFAULT_ARCHIVE_MAX_DEPTH)
    )
    args.archive_detect_sample_size = max(
        1, int(getattr(args, "archive_detect_sample_size", 128))
    )
    args.archive_cache_dir = getattr(args, "archive_cache_dir", None)
    args.keep_archive_cache = bool(getattr(args, "keep_archive_cache", False))
    args.snapshot_tags = bool(getattr(args, "snapshot_tags", False))
    args.snapshot_sample_size = int(getattr(args, "snapshot_sample_size", 500))
    args.snapshot_seed = int(getattr(args, "snapshot_seed", 42))

    for attr in ("root_path_pos", "root_path_opt", "output_dir_pos", "output_dir_opt"):
        if hasattr(args, attr):
            delattr(args, attr)

    return args


def parse_arguments():
    parser = build_parser()
    args = parser.parse_args()
    args = normalize_parse_args(args)
    logger.info("🚀 Running %s with args: %s", Path(__file__).name, args)
    return args


# -------------------------
# IO helpers
# -------------------------
def ensure_directory_exists(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)


def get_dicom_path_entries(
    root_path: Union[str, Path], archive_max_depth: int = DEFAULT_ARCHIVE_MAX_DEPTH
) -> list[dict]:
    """
    Strategy:
    1) Resolve root_path as directories and/or archive files.
    2) Recursively discover DICOM sources, including nested archives.
    3) Prefer *.dcm and fallback to header validation when needed.
    """
    resolved_roots = resolve_root_paths(root_path)
    return discover_dicom_sources(resolved_roots, max_depth=archive_max_depth)


def get_dicom_paths(root_path):
    entries = get_dicom_path_entries(root_path)
    paths = []
    for entry in entries:
        src = entry["source_uri_or_path"]
        if entry.get("is_archive_member"):
            paths.append(src)
        else:
            paths.append(Path(src))
    return paths


# -------------------------
# DICOM tag extraction
# -------------------------
def extract_dicom_tags_recursive(ds, parent_key=""):
    tags = {}
    for elem in ds:
        key = f"{parent_key}_{elem.keyword}" if parent_key else elem.keyword
        if elem.VR == "SQ":
            for i, item in enumerate(elem.value or []):
                tags.update(extract_dicom_tags_recursive(item, f"{key}[{i}]"))
        else:
            v = elem.value
            tags[key] = (
                None
                if v is None
                else ([str(x) for x in v] if isinstance(v, (list, tuple)) else str(v))
            )
    return tags


def _normalize_dicom_value(value):
    if value is None:
        return None
    if hasattr(value, "value") and not isinstance(value, (str, bytes, list, tuple)):
        value = value.value
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return str(value)


def build_effective_tags(
    *,
    default_tags: list[str],
    user_tags: list[str],
    patient_tag: str,
    study_tag: str,
    series_tag: str,
) -> list[str]:
    effective = []
    seen = set()
    for tag in default_tags + user_tags + [patient_tag, study_tag, series_tag]:
        t = str(tag).strip() if tag is not None else ""
        if not t or t in seen:
            continue
        seen.add(t)
        effective.append(t)
    return effective


def _load_dicom_dataset_standard(source, *, force=False, specific_tags=None):
    kwargs = {
        "stop_before_pixels": True,
        "force": force,
    }
    if specific_tags is not None:
        kwargs["specific_tags"] = specific_tags
    return dcmread(Path(str(source)), **kwargs)


def _load_dicom_dataset_archive_aware(
    source,
    *,
    force=False,
    specific_tags=None,
    archive_max_depth=DEFAULT_ARCHIVE_MAX_DEPTH,
):
    kwargs = {
        "stop_before_pixels": True,
        "force": force,
    }
    if specific_tags is not None:
        kwargs["specific_tags"] = specific_tags

    src_str = str(source)
    if is_archive_uri(src_str):
        outer, entry_chain = decode_archive_uri(src_str)
        payload = read_archive_member_bytes(
            outer_archive_path=outer,
            entry_chain=entry_chain,
            max_depth=archive_max_depth,
        )
        return dcmread(io.BytesIO(payload), **kwargs)
    return dcmread(Path(src_str), **kwargs)


def read_dicom_header_selected_standard(source, *, tags: list[str], force=False):
    """
    Read selected tags from a DICOM header.
    Returns pd.Series with one key per requested tag.
    """
    try:
        ds = _load_dicom_dataset_standard(
            source,
            force=force,
            specific_tags=tags,
        )
        values = {tag: _normalize_dicom_value(ds.get(tag)) for tag in tags}
        return pd.Series(values)
    except Exception:
        return pd.Series({})


def read_dicom_header_selected_archive_aware(
    source, *, tags: list[str], force=False, archive_max_depth=DEFAULT_ARCHIVE_MAX_DEPTH
):
    try:
        ds = _load_dicom_dataset_archive_aware(
            source,
            force=force,
            specific_tags=tags,
            archive_max_depth=archive_max_depth,
        )
        values = {tag: _normalize_dicom_value(ds.get(tag)) for tag in tags}
        return pd.Series(values)
    except Exception:
        return pd.Series({})


def read_dicom_header_selected(
    source,
    *,
    tags: list[str],
    force=False,
    archive_aware: bool = False,
    archive_max_depth: int = DEFAULT_ARCHIVE_MAX_DEPTH,
):
    if archive_aware:
        return read_dicom_header_selected_archive_aware(
            source,
            tags=tags,
            force=force,
            archive_max_depth=archive_max_depth,
        )
    return read_dicom_header_selected_standard(source, tags=tags, force=force)


def read_dicom_header_standard(source, *, force=False):
    """
    Read header once: recursively extract all DICOM tags into columns.
    Returns pd.Series with all tags flattened.
    """
    try:
        ds = _load_dicom_dataset_standard(
            source,
            force=force,
        )
        tags = extract_dicom_tags_recursive(ds)
        return pd.Series(tags)

    except Exception:
        return pd.Series({})


def read_dicom_header_archive_aware(
    source, *, force=False, archive_max_depth=DEFAULT_ARCHIVE_MAX_DEPTH
):
    try:
        ds = _load_dicom_dataset_archive_aware(
            source,
            force=force,
            archive_max_depth=archive_max_depth,
        )
        tags = extract_dicom_tags_recursive(ds)
        return pd.Series(tags)
    except Exception:
        return pd.Series({})


def read_dicom_header(
    source,
    *,
    force=False,
    archive_aware: bool = False,
    archive_max_depth: int = DEFAULT_ARCHIVE_MAX_DEPTH,
):
    if archive_aware:
        return read_dicom_header_archive_aware(
            source,
            force=force,
            archive_max_depth=archive_max_depth,
        )
    return read_dicom_header_standard(source, force=force)


def read_dicom_header_with_force(fp, force):
    return read_dicom_header(fp, force=force)


def _normalize_snapshot_missing_strings(value):
    if isinstance(value, str):
        return None if value.strip() == "" else value
    if isinstance(value, list):
        return [_normalize_snapshot_missing_strings(v) for v in value]
    if isinstance(value, tuple):
        return [_normalize_snapshot_missing_strings(v) for v in value]
    if isinstance(value, dict):
        return {
            key: _normalize_snapshot_missing_strings(val)
            for key, val in value.items()
        }
    return value


def build_global_readers(
    *,
    initial_archive_mode: bool,
    tags: list[str],
    force: bool,
    archive_max_depth: int,
):
    state = {
        "archive_mode": bool(initial_archive_mode),
        "auto_switched": False,
    }

    def selected(source):
        if state["archive_mode"]:
            return read_dicom_header_selected_archive_aware(
                source,
                tags=tags,
                force=force,
                archive_max_depth=archive_max_depth,
            )
        if is_archive_uri(str(source)):
            state["archive_mode"] = True
            state["auto_switched"] = True
            logger.info(
                "[archive][detect] archive URI encountered at runtime; switched to archive-aware mode."
            )
            return read_dicom_header_selected_archive_aware(
                source,
                tags=tags,
                force=force,
                archive_max_depth=archive_max_depth,
            )
        return read_dicom_header_selected_standard(source, tags=tags, force=force)

    def full(source):
        if state["archive_mode"]:
            return read_dicom_header_archive_aware(
                source,
                force=force,
                archive_max_depth=archive_max_depth,
            )
        if is_archive_uri(str(source)):
            state["archive_mode"] = True
            state["auto_switched"] = True
            logger.info(
                "[archive][detect] archive URI encountered at runtime; switched to archive-aware mode."
            )
            return read_dicom_header_archive_aware(
                source,
                force=force,
                archive_max_depth=archive_max_depth,
            )
        return read_dicom_header_standard(source, force=force)

    return selected, full, state


def write_dicom_tags_snapshot(
    *,
    df: pd.DataFrame,
    output_path: Path,
    sample_size: int,
    seed: int,
    series_col: str = "SeriesInstanceUID",
    read_path_col: str = "_read_path",
    read_full_func=None,
) -> int:
    if sample_size <= 0 or df.empty:
        return 0

    sampling_pool = df
    if series_col in df.columns:
        non_empty_series = df[df[series_col].notna()].copy()
        if not non_empty_series.empty:
            non_empty_series = non_empty_series[
                non_empty_series[series_col].astype(str).str.strip() != ""
            ]
            if not non_empty_series.empty:
                sampling_pool = non_empty_series.drop_duplicates(
                    subset=[series_col], keep="first"
                )

    n = min(sample_size, len(sampling_pool))
    sampled = sampling_pool.sample(n=n, random_state=seed).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, row in sampled.iterrows():
            read_fp = row.get(read_path_col)
            if read_fp is None or str(read_fp).strip() == "":
                continue

            if read_full_func is None:
                tags_series = read_dicom_header(read_fp)
            else:
                tags_series = read_full_func(read_fp)

            record = {
                "dicom_path": row.get("dicom_path"),
                "_scan_root": row.get("_scan_root"),
                "_relative_path": row.get("_relative_path"),
                "snapshot_seed": seed,
                "snapshot_index": int(idx),
                "tags": _normalize_snapshot_missing_strings(tags_series.to_dict()),
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            written += 1
    return written


# -------------------------
# ID selection logic
# -------------------------


def choose_ids(
    df: pd.DataFrame,
    root_path: Path,
    id_source: str,
    patient_tag: str,
    study_tag: str,
    series_tag: str,
    scan_root_col: str = "_scan_root",
    relative_path_col: str = "_relative_path",
) -> pd.DataFrame:
    """
    Compute patient_key / study_id / series_id.

    Rules:
      - patient_key_path = first directory under root
      - study_id, series_id come from tags only
      - if study/series tags missing -> "0"
      - multiple files per series are expected
    """

    # -------------------------
    # Path-derived patient_key
    # -------------------------
    if relative_path_col in df.columns:
        rel = df[relative_path_col].map(
            lambda p: (
                Path(str(p)) if (not pd.isna(p) and str(p).strip() != "") else Path("")
            )
        )
    else:
        if scan_root_col in df.columns:
            scan_roots = df[scan_root_col].astype(str)
        else:
            scan_roots = pd.Series([str(root_path)] * len(df), index=df.index)

        def _relative_to_root(dicom_path: str, scan_root: str) -> Path:
            p = Path(dicom_path)
            try:
                return p.relative_to(Path(scan_root))
            except ValueError:
                # Fallback keeps parsing resilient when paths are not nested as expected.
                return p

        rel = pd.Series(
            [_relative_to_root(p, r) for p, r in zip(df["dicom_path"], scan_roots)],
            index=df.index,
        )
    df["patient_key_path"] = rel.map(lambda p: p.parts[0] if len(p.parts) > 1 else None)
    df["study_path"] = rel.map(lambda p: p.parts[1] if len(p.parts) > 2 else None)
    df["series_path"] = rel.map(lambda p: p.parts[2] if len(p.parts) > 3 else None)
    df["dicom_filename"] = rel.map(lambda p: p.name)

    # -------------------------
    # Tag-derived IDs
    # -------------------------
    def _tagcol(tag: str) -> pd.Series:
        if tag in df.columns:
            return df[tag].map(
                lambda v: None if pd.isna(v) or str(v).strip() == "" else str(v).strip()
            )
        return pd.Series([None] * len(df), index=df.index)

    patient_key_tags = _tagcol(patient_tag)
    study_id_tags = _tagcol(study_tag)
    series_id_tags = _tagcol(series_tag)

    # -------------------------
    # Choose source
    # -------------------------
    if id_source == "path":
        df["patient_key"] = df["patient_key_path"]
        df["study_id"] = df["study_path"].fillna("0")
        df["series_id"] = df["series_path"].fillna("0")

    elif id_source == "tags":
        df["patient_key"] = patient_key_tags
        df["study_id"] = study_id_tags.fillna("0")
        df["series_id"] = series_id_tags.fillna("0")

    else:  # auto
        df["patient_key"] = patient_key_tags.fillna(df["patient_key_path"]).fillna(
            "UNKNOWN"
        )
        df["study_id"] = study_id_tags.fillna(df["study_path"]).fillna("0")
        df["series_id"] = series_id_tags.fillna(df["series_path"]).fillna("0")

    df = df.drop(
        columns=[
            "patient_key_path",
            "study_path",
            "series_path",
            scan_root_col,
            relative_path_col,
            "_read_path",
        ],
        errors="ignore",
    )

    return df


# -------------------------
# Checkpointed processing (for tag read stage)
# -------------------------
def process_with_checkpoint(
    df_paths: pd.DataFrame,
    read_func,
    checkpoint_frequency: Optional[int],
    output_dir: Path,
    final_name: str,
    read_path_col: str = "dicom_path",
):
    """
    Apply read_func(dicom_path)->Series to df_paths with optional chunked checkpoint.
    """
    cols_to_drop_for_persist = [read_path_col] if read_path_col != "dicom_path" else []

    if checkpoint_frequency is None:
        tags_df = df_paths[read_path_col].parallel_apply(read_func)
        tags_df = tags_df.replace("", float("NaN")).infer_objects(copy=False).dropna(
            how="all", axis=1
        )
        out = pd.concat([df_paths, tags_df], axis=1)
        out.drop(columns=cols_to_drop_for_persist, errors="ignore").to_csv(
            output_dir / final_name, index=False
        )
        return out
    if checkpoint_frequency <= 0:
        raise ValueError("checkpoint_frequency must be a positive integer.")

    # chunked
    total_rows = df_paths.shape[0]
    with tqdm(total=total_rows, desc="Parse files", unit="file") as pbar:
        for i in range(0, total_rows, checkpoint_frequency):
            chunk_idx = i // checkpoint_frequency
            ckpt = output_dir / f"{Path(final_name).stem}_{chunk_idx:03d}.csv"
            chunk = df_paths.iloc[i : i + checkpoint_frequency].copy()
            chunk_len = len(chunk)
            if ckpt.exists():
                pbar.update(chunk_len)
                continue
            tags_chunk = chunk[read_path_col].parallel_apply(read_func)
            # Keep all selected tag columns in checkpoint outputs, even if a
            # column is entirely empty for a chunk, so final schema is stable.
            tags_chunk = tags_chunk.replace("", float("NaN")).infer_objects(copy=False)
            out_chunk = pd.concat([chunk, tags_chunk], axis=1)
            out_chunk.drop(columns=cols_to_drop_for_persist, errors="ignore").to_csv(
                ckpt, index=False
            )
            pbar.update(chunk_len)

    # merge
    csv_files = sorted(glob.glob(str(output_dir / f"{Path(final_name).stem}_0*.csv")))
    if not csv_files:
        raise RuntimeError("No checkpoint files found to merge.")
    merged = pd.concat([pd.read_csv(p) for p in csv_files], ignore_index=True)
    merged.to_csv(output_dir / final_name, index=False)
    return merged


# -------------------------
# Main
# -------------------------
def main(args):
    args = normalize_parse_args(args)
    root_path = args.root_path
    output_dir = Path(args.output_dir)
    ensure_directory_exists(output_dir)
    logger.info("Output directory: %s", output_dir)
    manifest = load_manifest(
        args.manifest, base_path=Path(__file__).resolve().parents[1]
    )

    matched_roots = resolve_root_paths(root_path)
    if glob.has_magic(str(root_path)):
        logger.info("Root pattern %s matched %s entries", root_path, len(matched_roots))
    archive_mode = detect_archive_mode_by_subsample(
        matched_roots, sample_size=args.archive_detect_sample_size
    )
    logger.info(
        "Archive mode detection (sample_size=%s): %s",
        args.archive_detect_sample_size,
        archive_mode,
    )

    dicom_entries = get_dicom_path_entries(
        root_path, archive_max_depth=args.archive_max_depth
    )
    logger.info("Found %s DICOM sources under %s", len(dicom_entries), root_path)

    # 1) base df with dicom_path only (no IDs computed yet)
    df = pd.DataFrame(
        {
            "dicom_path": [entry["source_uri_or_path"] for entry in dicom_entries],
            "_scan_root": [entry["scan_root"] for entry in dicom_entries],
            "_relative_path": [entry["relative_path"] for entry in dicom_entries],
        }
    )
    # 2) read selected tags once for all files
    user_tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    effective_tags = build_effective_tags(
        default_tags=DEFAULT_DICOM_TAGS,
        user_tags=user_tags,
        patient_tag=args.patient_key_from,
        study_tag=args.study_id_from,
        series_tag=args.series_id_from,
    )
    logger.info(
        "Reading selected DICOM tags (count=%s, user tags=%s)",
        len(effective_tags),
        user_tags or "none",
    )
    read_selected_func, read_full_func, archive_state = build_global_readers(
        initial_archive_mode=archive_mode,
        tags=effective_tags,
        force=args.force_dicom_read,
        archive_max_depth=args.archive_max_depth,
    )

    df = process_with_checkpoint(
        df_paths=df,
        read_func=read_selected_func,
        checkpoint_frequency=args.checkpoint_frequency,
        output_dir=output_dir,
        final_name="dicom_paths_with_tags.csv",
        read_path_col="dicom_path",
    )

    if args.snapshot_tags:
        snapshot_path = output_dir / "dicom_tags_snapshot.ndjson"
        written = write_dicom_tags_snapshot(
            df=df,
            output_path=snapshot_path,
            sample_size=args.snapshot_sample_size,
            seed=args.snapshot_seed,
            series_col=args.series_id_from,
            read_path_col="dicom_path",
            read_full_func=read_full_func,
        )
        logger.info("Saved tag snapshot: %s (records=%s)", snapshot_path, written)

    if archive_state.get("auto_switched"):
        logger.info(
            "[archive][detect] runtime auto-switch applied; parse finished in archive-aware mode."
        )

    logger.info("After tag extraction: %s columns=%s", df.shape, len(df.columns))

    # 3) compute IDs from tags/path using already-read tag columns
    df = choose_ids(
        df=df,
        root_path=Path(root_path),
        id_source=args.id_source,
        patient_tag=args.patient_key_from,
        study_tag=args.study_id_from,
        series_tag=args.series_id_from,
    )
    df = apply_id_standardization(df, manifest, logger=logger)
    #df = apply_derived_columns(df, manifest)

    # 4) output final df
    out_final = output_dir / "dicom_index.csv"
    df.to_csv(out_final, index=False)
    logger.info("Saved final index: %s", out_final)
    logger.info("Parsing done ✔")


if __name__ == "__main__":
    setup_logging()
    args = parse_arguments()
    if args.dry_run:
        setup_logging(verbose=getattr(args, "verbose", False))
        logger.info("Dry run: parse")
        print_args(args)
        raise SystemExit(0)
    setup_logging(verbose=getattr(args, "verbose", False))
    pandarallel.initialize(progress_bar=args.verbose, nb_workers=args.num_workers)
    main(args)
