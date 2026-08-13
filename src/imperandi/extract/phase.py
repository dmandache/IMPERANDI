from __future__ import annotations

import argparse
import logging
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Tuple

import nibabel as nib
import pandas as pd
from tqdm import tqdm

from imperandi.curation.phase import (
    apply_phase_curation,
    phase_needs_strategy,
    validate_phase_curation,
)
from imperandi.utils.manifest import load_manifest
from imperandi.utils.logging import log_task_summary, setup_logging
from imperandi.utils.misc import print_args
from imperandi.utils.checkpoint_cli import add_checkpoint_arguments
from imperandi.utils.run_state import (
    atomic_write_csv,
    CheckpointManager,
    ensure_source_id_column,
    merge_with_existing_output,
    normalize_source_id,
    normalize_source_ids,
    prepare_resume_context,
    source_id_resume_signature,
)

logger = logging.getLogger(__name__)
DEFAULT_CHECKPOINT_EVERY_ROWS = 50
DEFAULT_CHECKPOINT_EVERY_SEC = 5 * 60


def _load_phase_extractor() -> Callable[[Any], Dict[str, Any]]:
    try:
        from totalsegmentator.bin.totalseg_get_phase import get_ct_contrast_phase
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The 'phase' command requires optional dependencies. "
            "Install with: pip install -e .[segment]"
        ) from exc
    return get_ct_contrast_phase


def add_phase_arguments(
    parser: argparse.ArgumentParser,
    include_dry_run: bool = True,
) -> None:
    """Add phase-extraction paths, force, and resume options to a parser."""
    parser.add_argument(
        "csv_path_pos",
        nargs="?",
        type=str,
        default=None,
        help="Path to input CSV with a `nifti_path` column. Defaults to ./nifti_index.csv.",
    )
    parser.add_argument(
        "csv_path_out_pos",
        nargs="?",
        type=str,
        default=None,
        help="Optional output CSV path (positional alternative to --csv_path_out).",
    )
    parser.add_argument(
        "--csv_path",
        dest="csv_path_opt",
        type=str,
    )
    parser.add_argument(
        "--csv_path_out",
        type=str,
        required=False,
        default=None,
        help="Output CSV path (default: overwrite input CSV).",
    )
    parser.add_argument(
        "--error_csv_path",
        type=str,
        default=None,
        help="CSV path for failed rows only (default: <csv_dir>/phase_errors.csv).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Recompute phase even when `totalseg_phase` is already populated.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="generic",
        help="YAML manifest defining phase-curation strategy precedence.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    add_checkpoint_arguments(
        parser,
        default_rows=DEFAULT_CHECKPOINT_EVERY_ROWS,
        default_sec=DEFAULT_CHECKPOINT_EVERY_SEC,
    )
    if include_dry_run:
        parser.add_argument(
            "--dry-run",
            dest="dry_run",
            action="store_true",
            default=False,
            help="Print planned actions without running.",
        )


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    """Build the standalone manifest-driven phase-curation parser."""
    parser = argparse.ArgumentParser(
        description="Curate contrast phase with manifest-defined fallbacks.",
        add_help=add_help,
    )
    add_phase_arguments(parser)
    return parser


def normalize_phase_args(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve phase input, output, and error paths in-place.

    Raises:
        FileNotFoundError: If the input table does not exist.
        ValueError: If the input table is not a CSV file.
    """
    csv_in = args.csv_path_opt if args.csv_path_opt is not None else args.csv_path_pos
    csv_path = Path(csv_in) if csv_in else (Path.cwd() / "nifti_index.csv")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Not a CSV file: {csv_path}")

    args.csv_path = str(csv_path.resolve())

    csv_path_out_pos = getattr(args, "csv_path_out_pos", None)
    csv_out = args.csv_path_out if args.csv_path_out else csv_path_out_pos
    if not csv_out:
        args.csv_path_out = args.csv_path
    else:
        args.csv_path_out = str(Path(csv_out))

    if args.error_csv_path:
        args.error_csv_path = str(Path(args.error_csv_path))
    else:
        args.error_csv_path = str(Path(args.csv_path).parent / "phase_errors.csv")

    del args.csv_path_pos
    del args.csv_path_opt
    if hasattr(args, "csv_path_out_pos"):
        del args.csv_path_out_pos
    return args


def parse_arguments() -> argparse.Namespace:
    """Parse and normalize arguments for standalone phase extraction."""
    parser = build_parser()
    args = parser.parse_args()
    args = normalize_phase_args(args)
    logger.info("🚀 Running %s script with arguments: %s", Path(__file__).name, args)
    return args


def _has_populated_value(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def process_single_volume(
    idx: int,
    row: Mapping[str, Any],
    *,
    phase_extractor: Callable[[Any], Dict[str, Any]],
    verbose: bool = False,
) -> Tuple[int, Dict[str, Any] | None, str | None]:
    """Extract phase metadata for one cohort row without raising row errors.

    Returns:
        ``(row_index, metadata, error)``. Successful metadata keys are prefixed
        with ``totalseg_``; failures return an error string instead.
    """
    nifti_path_value = row.get("nifti_path")
    if not isinstance(nifti_path_value, str) or not nifti_path_value.strip():
        return idx, None, "column 'nifti_path' missing or invalid"

    nifti_path = Path(nifti_path_value)
    if not nifti_path.exists():
        return idx, None, f"file not found: {nifti_path}"

    try:
        nifti_image = nib.load(str(nifti_path))
        phase_info = phase_extractor(nifti_image, quiet=not verbose)
    except Exception as exc:
        logger.debug("Traceback for %s:\n%s", nifti_path, traceback.format_exc())
        return idx, None, str(exc)

    if not isinstance(phase_info, dict):
        return idx, None, "phase extractor did not return a dictionary"
    if not phase_info:
        return idx, None, "phase extractor returned no values"

    normalized = {f"totalseg_{key}": value for key, value in phase_info.items()}
    return idx, normalized, None


def main(args: argparse.Namespace) -> None:
    """Curate phase metadata for a cohort with checkpoint-aware resuming."""
    manifest_arg = getattr(args, "manifest", "generic")
    manifest = load_manifest(
        manifest_arg,
        base_path=Path(__file__).resolve().parents[1],
    )
    if "phase_curation" not in manifest:
        raise ValueError("Manifest must define a phase_curation section.")
    phase_curation = validate_phase_curation(manifest["phase_curation"])
    logger.info(
        "Phase strategy order: %s",
        " -> ".join(
            f"{strategy['name']} ({strategy['type']})"
            for strategy in phase_curation["strategies"]
        ),
    )

    output_path = Path(args.csv_path_out)
    error_path = Path(args.error_csv_path)
    exclude_hash_args = {
        "csv_path_out",
        "dry_run",
        "verbose",
        "resume",
        "checkpoint_every_rows",
        "checkpoint_every_sec",
        "strict_resume",
    }
    source_id_signature = source_id_resume_signature(args.csv_path)
    resume_values = {
        **vars(args),
        "checkpoint_manifest_config": manifest,
        "checkpoint_phase_curation": phase_curation,
    }
    if source_id_signature:
        resume_values["checkpoint_source_id"] = source_id_signature
    resume_args = argparse.Namespace(**resume_values)
    resume_ctx = prepare_resume_context(
        args=resume_args,
        command="phase",
        inputs=args.csv_path,
        output_path=output_path,
        error_path=error_path,
        exclude_hash_args=exclude_hash_args,
    )
    paths = resume_ctx["paths"]
    state = resume_ctx["state"]
    can_resume = resume_ctx["can_resume"]
    already_finished = resume_ctx["already_finished"]
    ckpt = CheckpointManager(paths=paths, config=resume_ctx["config"])

    if already_finished:
        logger.info(
            "Resume enabled and matching phase run already finished; skipping execution."
        )
        return

    if can_resume and paths.main_checkpoint_path.exists():
        logger.info("Resuming phase from checkpoint: %s", paths.main_checkpoint_path)
        df = pd.read_csv(paths.main_checkpoint_path).copy()
    else:
        df = pd.read_csv(args.csv_path).copy()
    df = ensure_source_id_column(df)

    errors_by_idx: Dict[str, Dict[str, Any]] = {}
    completed_indices: set[str] = set()
    force = bool(getattr(args, "force", False))
    resume_skipped_count = 0
    prefilled_skipped_count = 0
    prediction_skipped_count = 0
    if can_resume:
        completed_indices = normalize_source_ids(
            (state or {}).get("completed_indices", [])
        )
        resume_skipped_count = len(completed_indices)
        if paths.error_checkpoint_path.exists():
            err_ckpt = pd.read_csv(paths.error_checkpoint_path)
            for _, row in err_ckpt.iterrows():
                if "_source_idx" in row:
                    try:
                        source_idx = normalize_source_id(row["_source_idx"])
                        if source_idx:
                            errors_by_idx[source_idx] = row.to_dict()
                    except Exception:
                        pass

    needs_prediction = phase_needs_strategy(
        df,
        phase_curation,
        "totalsegmentator",
    )
    prediction_strategy = next(
        (
            strategy
            for strategy in phase_curation["strategies"]
            if strategy["type"] == "totalsegmentator"
        ),
        None,
    )
    if prediction_strategy is not None:
        logger.info(
            "Phase extraction step: %s (totalsegmentator) selected %d/%d volume(s)",
            prediction_strategy["name"],
            int(needs_prediction.sum()),
            len(df),
        )
    prediction_skipped_indices = {
        normalize_source_id(df.at[idx, "_source_idx"])
        for idx in df.index
        if not bool(needs_prediction.at[idx])
    }
    newly_skipped = prediction_skipped_indices - completed_indices
    if newly_skipped:
        prediction_skipped_count = len(newly_skipped)
        completed_indices.update(newly_skipped)
        for source_idx in newly_skipped:
            errors_by_idx.pop(source_idx, None)

    if not force and "totalseg_phase" in df.columns:
        prefilled_indices = {
            normalize_source_id(df.at[idx, "_source_idx"])
            for idx in df.index
            if bool(needs_prediction.at[idx])
            and _has_populated_value(df.at[idx, "totalseg_phase"])
        }
        newly_completed = prefilled_indices - completed_indices
        if newly_completed:
            prefilled_skipped_count = len(newly_completed)
            completed_indices.update(prefilled_indices)
            for source_idx in newly_completed:
                errors_by_idx.pop(source_idx, None)
            logger.info(
                "Skipping %d row(s) with existing totalseg_phase (use --force to recompute)",
                len(newly_completed),
            )

    def _checkpoint_write(*, force: bool = False) -> None:
        err_df = (
            pd.DataFrame(list(errors_by_idx.values()))
            if errors_by_idx
            else pd.DataFrame()
        )
        ckpt.flush(
            main_df=df,
            error_df=err_df,
            completed_indices=completed_indices,
            force=force,
        )

    row_indices = [
        idx
        for idx in df.index.tolist()
        if bool(needs_prediction.at[idx])
        and normalize_source_id(df.at[idx, "_source_idx"]) not in completed_indices
    ]
    phase_extractor = _load_phase_extractor() if row_indices else None
    processed_source_ids = {
        normalize_source_id(df.at[idx, "_source_idx"]) for idx in row_indices
    }
    for idx in tqdm(row_indices, total=len(row_indices), desc="Phase"):
        src_idx = normalize_source_id(df.at[idx, "_source_idx"])
        _, phase_info, err_msg = process_single_volume(
            idx,
            df.loc[idx].to_dict(),
            phase_extractor=phase_extractor,
            verbose=args.verbose,
        )
        if phase_info:
            for key, value in phase_info.items():
                df.at[idx, key] = value
            if src_idx in errors_by_idx:
                del errors_by_idx[src_idx]
        else:
            error_row = df.loc[idx].to_dict()
            error_row["error_message"] = err_msg or "unknown"
            errors_by_idx[src_idx] = error_row
            if args.verbose and err_msg:
                logger.warning("Row %s failed: %s", idx, err_msg)
        completed_indices.add(src_idx)
        ckpt.mark_processed()
        _checkpoint_write(force=False)

    df = apply_phase_curation(
        df,
        phase_curation,
        progress_logger=logger,
    )
    _checkpoint_write(force=True)
    df_out = df.drop(columns=["_source_idx"], errors="ignore")
    df_out = merge_with_existing_output(
        df_out,
        args.csv_path_out,
        preferred_keys=["volume_id", "nifti_path", "_source_idx"],
        strict=True,
    )
    atomic_write_csv(df_out, args.csv_path_out, index=False)
    logger.info("Wrote main table -> %s", args.csv_path_out)

    if errors_by_idx:
        df_err = pd.DataFrame(list(errors_by_idx.values())).drop(
            columns=["_source_idx"], errors="ignore"
        )
        atomic_write_csv(df_err, args.error_csv_path, index=False)
        logger.warning("%d rows failed -> %s", len(df_err), args.error_csv_path)
    ckpt.finalize_state(completed_indices=completed_indices)

    run_failed_count = len(processed_source_ids & set(errors_by_idx))
    log_task_summary(
        logger,
        "Phase extraction",
        total_rows=len(df),
        processed_rows=len(row_indices),
        succeeded_rows=max(0, len(row_indices) - run_failed_count),
        skipped_rows=(
            resume_skipped_count + prefilled_skipped_count + prediction_skipped_count
        ),
        failed_rows=run_failed_count,
        success_label="phase extracted",
        extra_counts={
            "skipped by resume": resume_skipped_count,
            "skipped with existing phase": prefilled_skipped_count,
            "not sent to TotalSegmentator": prediction_skipped_count,
        },
    )
    logger.info("Phase extraction done ✔")


if __name__ == "__main__":
    setup_logging()
    args = parse_arguments()
    setup_logging(verbose=getattr(args, "verbose", False))
    if getattr(args, "dry_run", False):
        logger.info("Dry run: phase")
        print_args(args)
        raise SystemExit(0)
    main(args)
