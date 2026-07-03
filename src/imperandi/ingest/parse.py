from __future__ import annotations

import warnings
import glob
import hashlib
import io
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
from typing import Optional, Union

import pandas as pd
from tqdm import tqdm
from pydicom import dcmread, config

from imperandi.utils.archive_io import (
    DEFAULT_ARCHIVE_MAX_DEPTH,
    decode_archive_uri,
    discover_dicom_sources,
    is_archive_filename,
    is_archive_uri,
    read_archive_member_bytes,
)
from imperandi.utils.logging import log_task_summary, setup_logging
from imperandi.utils.misc import print_args
from imperandi.utils.manifest import load_manifest
from imperandi.utils.checkpoint_cli import add_checkpoint_arguments
from imperandi.utils.run_state import (
    STATE_SCHEMA_VERSION,
    atomic_write_csv,
    atomic_write_json,
    build_checkpoint_paths,
    compute_args_hash,
    fingerprint_inputs,
    load_state,
    now_epoch,
)
from imperandi.datasets_config.defaults import DEFAULT_DICOM_TAGS
from imperandi.ingest.hooks import apply_id_standardization

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
DEFAULT_CHECKPOINT_EVERY_ROWS = 10_000
DEFAULT_CHECKPOINT_EVERY_SEC = 5 * 60  # 5 minutes

PARSE_WORKER_BATCH_SIZE = 16384
PARSE_CHECKPOINT_SCHEMA_VERSION = 1

# Make reading tolerant of non-conformant values
config.settings.reading_validation_mode = config.IGNORE  # or config.WARN

_PARSE_WORKER_TAGS: tuple[str, ...] = tuple()
_PARSE_WORKER_FORCE = False
_PARSE_WORKER_ARCHIVE_MODE = False
_PARSE_WORKER_ARCHIVE_MAX_DEPTH = DEFAULT_ARCHIVE_MAX_DEPTH


# -------------------------
# CLI
# -------------------------
def add_parse_arguments(
    parser: argparse.ArgumentParser,
    include_manifest: bool = True,
    include_dry_run: bool = True,
) -> None:
    """Add DICOM discovery, identity, archive, and resume options to a parser."""
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
    add_checkpoint_arguments(
        parser,
        default_rows=DEFAULT_CHECKPOINT_EVERY_ROWS,
        default_sec=DEFAULT_CHECKPOINT_EVERY_SEC,
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
        default=4,
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
    """Build the standalone parser for DICOM metadata ingestion."""
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
    """Resolve a directory, archive, or glob into deterministic input roots.

    A missing path resolves to the current working directory. Glob results are
    de-duplicated and restricted to directories and supported archives.
    """
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
    """Choose the default parse output directory for an input path or glob."""
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
    """Return whether a deterministic sample of the roots contains archives."""
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
    """Resolve parse defaults and validate checkpoint settings in-place.

    Named path options take precedence over positional values. Temporary parser
    attributes are removed from the returned namespace.
    """
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
    args.checkpoint_every_rows = int(
        getattr(args, "checkpoint_every_rows", DEFAULT_CHECKPOINT_EVERY_ROWS)
    )
    args.checkpoint_every_sec = int(
        getattr(args, "checkpoint_every_sec", DEFAULT_CHECKPOINT_EVERY_SEC)
    )
    if args.checkpoint_every_rows <= 0:
        raise ValueError("checkpoint_every_rows must be a positive integer.")
    if args.checkpoint_every_sec <= 0:
        raise ValueError("checkpoint_every_sec must be a positive integer.")

    for attr in ("root_path_pos", "root_path_opt", "output_dir_pos", "output_dir_opt"):
        if hasattr(args, attr):
            delattr(args, attr)

    return args


def parse_arguments():
    """Parse and normalize arguments for the standalone parse command."""
    parser = build_parser()
    args = parser.parse_args()
    args = normalize_parse_args(args)
    logger.info("🚀 Running %s with args: %s", Path(__file__).name, args)
    return args


# -------------------------
# IO helpers
# -------------------------
def ensure_directory_exists(output_dir: Path):
    """Create an output directory and any missing parents."""
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
    """Return discovered DICOM paths for compatibility with older callers."""
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
    """Flatten all readable elements of a DICOM dataset into a dictionary."""
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
    """Combine default, requested, and identifier tags without duplicates."""
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
            key: _normalize_snapshot_missing_strings(val) for key, val in value.items()
        }
    return value


def build_global_readers(
    *,
    initial_archive_mode: bool,
    tags: list[str],
    force: bool,
    archive_max_depth: int,
):
    """Build selected/full header readers that can switch to archive mode.

    Returns:
        A selected-tag reader, a full-header reader, and their shared mutable
        archive-detection state.
    """
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
    """Write recursive tags from a deterministic source sample as NDJSON.

    Returns:
        The number of snapshot records written.
    """
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
        rel_norm = (
            df[relative_path_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.replace("\\", "/", regex=False)
            .str.strip("/")
        )
        n_parts = rel_norm.str.count("/") + 1
        n_parts = n_parts.where(rel_norm != "", 0)
        split = rel_norm.str.split("/", n=3, expand=True)

        def _split_col(position: int) -> pd.Series:
            if position in split.columns:
                return split[position]
            return pd.Series([None] * len(df), index=df.index)

        df["patient_key_path"] = _split_col(0).where(n_parts > 1, None)
        df["study_path"] = _split_col(1).where(n_parts > 2, None)
        df["series_path"] = _split_col(2).where(n_parts > 3, None)
        df["dicom_filename"] = rel_norm.str.rsplit("/", n=1).str[-1].fillna("")
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
        df["patient_key_path"] = rel.map(
            lambda p: p.parts[0] if len(p.parts) > 1 else None
        )
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
def _read_selected_values_dict(
    source,
    *,
    tags: list[str],
    force: bool,
    archive_mode: bool,
    archive_max_depth: int,
) -> dict:
    try:
        if archive_mode or is_archive_uri(str(source)):
            ds = _load_dicom_dataset_archive_aware(
                source,
                force=force,
                specific_tags=tags,
                archive_max_depth=archive_max_depth,
            )
        else:
            ds = _load_dicom_dataset_standard(
                source,
                force=force,
                specific_tags=tags,
            )
        return {tag: _normalize_dicom_value(ds.get(tag)) for tag in tags}
    except Exception:
        return {}


def _init_parse_worker(
    tags: tuple[str, ...],
    force: bool,
    archive_mode: bool,
    archive_max_depth: int,
):
    global _PARSE_WORKER_TAGS
    global _PARSE_WORKER_FORCE
    global _PARSE_WORKER_ARCHIVE_MODE
    global _PARSE_WORKER_ARCHIVE_MAX_DEPTH

    _PARSE_WORKER_TAGS = tuple(tags)
    _PARSE_WORKER_FORCE = bool(force)
    _PARSE_WORKER_ARCHIVE_MODE = bool(archive_mode)
    _PARSE_WORKER_ARCHIVE_MAX_DEPTH = int(archive_max_depth)


def _read_selected_worker(task: tuple[int, object]) -> dict:
    source_idx, source = task
    values = _read_selected_values_dict(
        source,
        tags=list(_PARSE_WORKER_TAGS),
        force=_PARSE_WORKER_FORCE,
        archive_mode=_PARSE_WORKER_ARCHIVE_MODE,
        archive_max_depth=_PARSE_WORKER_ARCHIVE_MAX_DEPTH,
    )
    values["_source_idx"] = int(source_idx)
    return values


def _coerce_header_result_to_dict(value) -> dict:
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, dict):
        return dict(value)
    return {}


def _lightweight_parse_input_fingerprint(inputs: list[object]) -> list[dict]:
    canonical = [str(x) for x in inputs]
    digest = hashlib.sha256()
    for item in canonical:
        digest.update(item.encode("utf-8", errors="ignore"))
        digest.update(b"\0")
    if canonical:
        first = canonical[0]
        last = canonical[-1]
    else:
        first = None
        last = None
    return [
        {
            "mode": "lightweight",
            "count": len(canonical),
            "first": first,
            "last": last,
            "sha256": digest.hexdigest(),
        }
    ]


def _parse_state_matches(
    state: dict | None,
    *,
    command: str,
    args_hash: str,
    input_fingerprint: list[dict],
) -> bool:
    if not isinstance(state, dict):
        return False
    return (
        state.get("schema_version") == STATE_SCHEMA_VERSION
        and state.get("parse_checkpoint_schema") == PARSE_CHECKPOINT_SCHEMA_VERSION
        and state.get("command") == command
        and state.get("args_hash") == args_hash
        and state.get("input_fingerprint") == input_fingerprint
    )


def _write_parse_state(
    *,
    state_path: Path,
    args_hash: str,
    input_fingerprint: list[dict],
    next_source_idx: int,
    finished: bool,
) -> None:
    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "parse_checkpoint_schema": PARSE_CHECKPOINT_SCHEMA_VERSION,
        "command": "parse",
        "args_hash": args_hash,
        "input_fingerprint": input_fingerprint,
        "next_source_idx": int(max(0, next_source_idx)),
        "updated_at_epoch": now_epoch(),
        "finished": bool(finished),
    }
    atomic_write_json(state_path, payload)


def _append_checkpoint_rows(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        df.to_csv(handle, index=False, header=write_header)


def _load_checkpoint_columns(path: Path) -> list[str]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path, nrows=0).columns.tolist()


def _series_has_any_true(series: pd.Series) -> bool:
    if series.empty:
        return False

    def _to_bool(value) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}

    return bool(series.map(_to_bool).any())


def process_with_checkpoint(
    df_paths: pd.DataFrame,
    read_func,
    checkpoint_every_rows: int,
    checkpoint_every_sec: int,
    resume: bool,
    strict_resume: bool,
    output_dir: Path,
    final_name: str,
    read_path_col: str = "dicom_path",
    *,
    num_workers: int = 1,
    worker_config: Optional[dict] = None,
    transform_chunk=None,
    return_df: bool = True,
    expected_columns: Optional[list[str]] = None,
    resume_signature: Optional[dict] = None,
):
    """
    Apply DICOM header reads in chunked mode with append-only checkpointing.
    """
    if checkpoint_every_rows <= 0:
        raise ValueError("checkpoint_every_rows must be a positive integer.")
    if checkpoint_every_sec <= 0:
        raise ValueError("checkpoint_every_sec must be a positive integer.")
    if read_path_col not in df_paths.columns:
        raise KeyError(f"column '{read_path_col}' missing")

    cols_to_drop_for_persist = [read_path_col] if read_path_col != "dicom_path" else []
    output_path = output_dir / final_name
    error_path = output_dir / f"{Path(final_name).stem}_errors.csv"
    paths = build_checkpoint_paths(
        output_path=output_path, error_path=error_path, command="parse"
    )

    df = df_paths.copy()
    if "_source_idx" not in df.columns:
        df["_source_idx"] = pd.RangeIndex(start=0, stop=len(df), step=1, dtype="int64")
    df = df.sort_values("_source_idx").reset_index(drop=True)

    input_values = df[read_path_col].tolist()
    input_fingerprint = (
        fingerprint_inputs(input_values, strict=True)
        if strict_resume
        else _lightweight_parse_input_fingerprint(input_values)
    )
    runtime_args = argparse.Namespace(
        read_path_col=read_path_col,
        dataframe_columns=list(df.columns),
        num_workers=int(num_workers),
        worker_config=(worker_config or {}),
        resume_signature=(resume_signature or {}),
        checkpoint_every_rows=checkpoint_every_rows,
        checkpoint_every_sec=checkpoint_every_sec,
        resume=resume,
        strict_resume=strict_resume,
    )
    args_hash = compute_args_hash(
        runtime_args,
        exclude_keys=(
            "resume",
            "checkpoint_every_rows",
            "checkpoint_every_sec",
            "strict_resume",
        ),
    )

    state = load_state(paths.state_path)
    state_is_compatible = bool(resume) and _parse_state_matches(
        state,
        command="parse",
        args_hash=args_hash,
        input_fingerprint=input_fingerprint,
    )
    if bool(resume) and state and not state_is_compatible:
        logger.info(
            "Existing parse checkpoint is incompatible with current checkpoint schema/signature; starting fresh."
        )

    already_finished = (
        state_is_compatible
        and bool((state or {}).get("finished"))
        and output_path.exists()
    )
    can_resume = state_is_compatible and paths.main_checkpoint_path.exists()

    if already_finished:
        logger.info(
            "Resume enabled and matching parse run already finished; skipping execution."
        )
        log_task_summary(
            logger,
            "Parse",
            total_rows=len(df),
            processed_rows=0,
            succeeded_rows=0,
            skipped_rows=len(df),
            failed_rows=0,
            success_label="parsed",
            extra_counts={"skipped by finished checkpoint": len(df)},
        )
        if return_df:
            return pd.read_csv(output_path)
        return None

    if not can_resume:
        for stale in (
            paths.main_checkpoint_path,
            paths.error_checkpoint_path,
            paths.state_path,
        ):
            if stale.exists():
                stale.unlink()
        next_source_idx = 0
    else:
        next_source_idx = int((state or {}).get("next_source_idx", 0))
        next_source_idx = max(0, min(next_source_idx, len(df)))
        logger.info("Resuming parse from source index %s", next_source_idx)
    resume_skipped_count = next_source_idx if can_resume else 0

    requested_workers = max(1, int(num_workers))
    active_worker_config = worker_config or {}
    if active_worker_config and "tags" not in active_worker_config:
        raise ValueError("worker_config must include a 'tags' key when provided.")

    checkpoint_columns = list(expected_columns) if expected_columns else []
    if can_resume and not checkpoint_columns:
        checkpoint_columns = _load_checkpoint_columns(paths.main_checkpoint_path)
    if checkpoint_columns and "_source_idx" not in checkpoint_columns:
        checkpoint_columns = ["_source_idx", *checkpoint_columns]

    chunk_size = max(1, min(PARSE_WORKER_BATCH_SIZE, int(checkpoint_every_rows)))
    buffer_frames: list[pd.DataFrame] = []
    buffered_rows = 0
    pending_next_source_idx = next_source_idx
    last_flush_at = time.time()

    def _flush_buffer(force_state: bool = False):
        nonlocal buffer_frames, buffered_rows, checkpoint_columns, last_flush_at
        if not buffer_frames and not force_state:
            return

        if buffer_frames:
            flush_df = pd.concat(buffer_frames, ignore_index=True)
            if "_source_idx" not in flush_df.columns:
                raise KeyError(
                    "internal error: _source_idx missing from flush dataframe"
                )

            if not checkpoint_columns:
                checkpoint_columns = flush_df.columns.tolist()
            extra_cols = [c for c in flush_df.columns if c not in checkpoint_columns]
            if extra_cols:
                checkpoint_columns = [*checkpoint_columns, *extra_cols]
                if paths.main_checkpoint_path.exists():
                    existing = pd.read_csv(paths.main_checkpoint_path)
                    existing = existing.reindex(
                        columns=checkpoint_columns, fill_value=None
                    )
                    atomic_write_csv(existing, paths.main_checkpoint_path, index=False)

            flush_df = flush_df.reindex(columns=checkpoint_columns, fill_value=None)
            _append_checkpoint_rows(paths.main_checkpoint_path, flush_df)
            buffer_frames = []
            buffered_rows = 0

        _write_parse_state(
            state_path=paths.state_path,
            args_hash=args_hash,
            input_fingerprint=input_fingerprint,
            next_source_idx=pending_next_source_idx,
            finished=False,
        )
        last_flush_at = time.time()

    parse_executor = None
    if active_worker_config and requested_workers > 1:
        parse_executor = ProcessPoolExecutor(
            max_workers=requested_workers,
            initializer=_init_parse_worker,
            initargs=(
                tuple(active_worker_config.get("tags", [])),
                bool(active_worker_config.get("force", False)),
                bool(active_worker_config.get("archive_mode", False)),
                int(
                    active_worker_config.get(
                        "archive_max_depth", DEFAULT_ARCHIVE_MAX_DEPTH
                    )
                ),
            ),
        )

    total_rows = len(df)
    try:
        with tqdm(
            total=total_rows,
            desc="Parse files",
            unit="file",
            miniters=1,
        ) as pbar:
            if next_source_idx > 0:
                pbar.update(next_source_idx)

            for start in range(next_source_idx, total_rows, chunk_size):
                end = min(total_rows, start + chunk_size)
                chunk = df.iloc[start:end].copy()
                tasks = [
                    (int(src_idx), source)
                    for src_idx, source in zip(
                        chunk["_source_idx"].tolist(), chunk[read_path_col].tolist()
                    )
                ]
                if active_worker_config:
                    if parse_executor is not None:
                        map_chunk_size = max(1, len(tasks) // (requested_workers * 4))
                        records = list(
                            parse_executor.map(
                                _read_selected_worker,
                                tasks,
                                chunksize=map_chunk_size,
                            )
                        )
                    else:
                        records = []
                        for source_idx, source in tasks:
                            row = _read_selected_values_dict(
                                source,
                                tags=list(active_worker_config.get("tags", [])),
                                force=bool(active_worker_config.get("force", False)),
                                archive_mode=bool(
                                    active_worker_config.get("archive_mode", False)
                                ),
                                archive_max_depth=int(
                                    active_worker_config.get(
                                        "archive_max_depth",
                                        DEFAULT_ARCHIVE_MAX_DEPTH,
                                    )
                                ),
                            )
                            row["_source_idx"] = int(source_idx)
                            records.append(row)
                else:
                    if read_func is None:
                        raise ValueError(
                            "read_func is required when worker_config is not provided."
                        )
                    records = []
                    for source_idx, source in tasks:
                        row = _coerce_header_result_to_dict(read_func(source))
                        row["_source_idx"] = int(source_idx)
                        records.append(row)

                tags_chunk = pd.DataFrame.from_records(records)
                chunk_out = chunk.merge(tags_chunk, on="_source_idx", how="left")
                chunk_out = chunk_out.replace("", pd.NA).infer_objects()
                if transform_chunk is not None:
                    chunk_out = transform_chunk(chunk_out)
                    if "_source_idx" not in chunk_out.columns:
                        raise KeyError(
                            "transform_chunk must preserve '_source_idx' for checkpoint dedupe."
                        )

                buffer_frames.append(chunk_out)
                buffered_rows += len(chunk_out)
                pending_next_source_idx = end

                now = time.time()
                if (
                    buffered_rows >= checkpoint_every_rows
                    or (now - last_flush_at) >= checkpoint_every_sec
                ):
                    _flush_buffer(force_state=False)

                pbar.update(len(chunk))
    finally:
        if parse_executor is not None:
            parse_executor.shutdown(wait=True)

    _flush_buffer(force_state=True)

    if paths.main_checkpoint_path.exists():
        out = pd.read_csv(paths.main_checkpoint_path)
    else:
        out = df.copy()
    if "_source_idx" in out.columns:
        out = (
            out.drop_duplicates(subset=["_source_idx"], keep="last")
            .sort_values("_source_idx")
            .reset_index(drop=True)
        )

    out = out.drop(columns=["_source_idx"], errors="ignore")
    out = out.drop(columns=cols_to_drop_for_persist, errors="ignore")
    if "patient_key_std_failed" in out.columns and not _series_has_any_true(
        out["patient_key_std_failed"]
    ):
        out = out.drop(columns=["patient_key_std_failed"], errors="ignore")

    atomic_write_csv(out, output_path, index=False)
    _write_parse_state(
        state_path=paths.state_path,
        args_hash=args_hash,
        input_fingerprint=input_fingerprint,
        next_source_idx=len(df),
        finished=True,
    )
    processed_rows = max(0, len(df) - resume_skipped_count)
    log_task_summary(
        logger,
        "Parse",
        total_rows=len(df),
        processed_rows=processed_rows,
        succeeded_rows=processed_rows,
        skipped_rows=resume_skipped_count,
        failed_rows=0,
        success_label="parsed",
        extra_counts={"skipped by resume": resume_skipped_count},
    )
    if return_df:
        return out
    return None


# -------------------------
# Main
# -------------------------
def main(args):
    """Run DICOM discovery and metadata parsing for a normalized namespace."""
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
    _, read_full_func, _ = build_global_readers(
        initial_archive_mode=archive_mode,
        tags=effective_tags,
        force=args.force_dicom_read,
        archive_max_depth=args.archive_max_depth,
    )

    if (not archive_mode) and any(
        is_archive_uri(str(p)) for p in df["dicom_path"].tolist()
    ):
        logger.info(
            "[archive][detect] archive URI encountered in discovered sources; archive-aware reads will be used per file."
        )

    def _transform_parse_chunk(chunk_df: pd.DataFrame) -> pd.DataFrame:
        out = choose_ids(
            df=chunk_df,
            root_path=Path(root_path),
            id_source=args.id_source,
            patient_tag=args.patient_key_from,
            study_tag=args.study_id_from,
            series_tag=args.series_id_from,
        )
        out = apply_id_standardization(out, manifest, logger=logger)
        if "patient_key_std_failed" not in out.columns:
            out["patient_key_std_failed"] = False
        else:
            out["patient_key_std_failed"] = out["patient_key_std_failed"].fillna(False)
        return out

    schema_seed = pd.DataFrame(
        columns=[
            "_source_idx",
            "dicom_path",
            "_scan_root",
            "_relative_path",
            *effective_tags,
        ]
    )
    schema_seed = _transform_parse_chunk(schema_seed)
    expected_columns = list(dict.fromkeys(schema_seed.columns.tolist()))
    if "patient_key_std_failed" not in expected_columns:
        expected_columns.append("patient_key_std_failed")

    worker_config = {
        "tags": effective_tags,
        "force": bool(args.force_dicom_read),
        "archive_mode": bool(archive_mode),
        "archive_max_depth": int(args.archive_max_depth),
    }
    resume_signature = {
        "effective_tags": effective_tags,
        "force_dicom_read": bool(args.force_dicom_read),
        "archive_mode": bool(archive_mode),
        "archive_max_depth": int(args.archive_max_depth),
        "id_source": args.id_source,
        "patient_key_from": args.patient_key_from,
        "study_id_from": args.study_id_from,
        "series_id_from": args.series_id_from,
        "manifest": args.manifest,
        "manifest_config": manifest,
    }
    process_with_checkpoint(
        df_paths=df,
        read_func=None,
        checkpoint_every_rows=args.checkpoint_every_rows,
        checkpoint_every_sec=args.checkpoint_every_sec,
        resume=bool(args.resume),
        strict_resume=bool(args.strict_resume),
        output_dir=output_dir,
        final_name="dicom_index.csv",
        read_path_col="dicom_path",
        num_workers=int(args.num_workers),
        worker_config=worker_config,
        transform_chunk=_transform_parse_chunk,
        return_df=False,
        expected_columns=expected_columns,
        resume_signature=resume_signature,
    )

    out_final = output_dir / "dicom_index.csv"
    if args.snapshot_tags:
        snapshot_source = pd.read_csv(
            out_final,
            usecols=["dicom_path", args.series_id_from],
        )
        snapshot_source = snapshot_source.merge(
            df[["dicom_path", "_scan_root", "_relative_path"]],
            on="dicom_path",
            how="left",
        )
        snapshot_source["_read_path"] = snapshot_source["dicom_path"]
        snapshot_path = output_dir / "dicom_tags_snapshot.ndjson"
        written = write_dicom_tags_snapshot(
            df=snapshot_source,
            output_path=snapshot_path,
            sample_size=args.snapshot_sample_size,
            seed=args.snapshot_seed,
            series_col=args.series_id_from,
            read_path_col="_read_path",
            read_full_func=read_full_func,
        )
        logger.info("Saved tag snapshot: %s (records=%s)", snapshot_path, written)
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
    main(args)
