"""segment.py
=================
Batch-process a list of 3-D volumes to obtain masks with a configurable
segmentation backend (default: TotalSegmentator v2).

This module intentionally keeps the historical public API stable while delegating
implementation details to smaller internal modules:
- ``segment_config.py`` for configuration/postprocess resolution,
- ``segment_io.py`` for mask I/O and output discovery,
- ``segment_workflow.py`` for orchestration workflows.
"""

from __future__ import annotations

import argparse
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from imperandi.utils.logging import setup_logging
from imperandi.utils.manifest import DEFAULT_MANIFEST_NAME
from imperandi.utils.misc import report_volumes  # type: ignore

from .segment_io import (
    clean_and_merge_masks,
)
from .segment_workflow import (
    TotalSegmentatorBackend,
    prefetch_totalsegmentator_models as _prefetch_totalsegmentator_models,
    run_segment_batch_workflow,
    run_segment_volume_workflow,
)

DEFAULT_TIMEOUT = 15 * 60  # in seconds

logger = logging.getLogger(__name__)


def prefetch_totalsegmentator_models(
    tasks_config: Dict[str, Any], *, fast: bool
) -> None:
    """Download required TotalSegmentator weights before multiprocessing."""
    _prefetch_totalsegmentator_models(tasks_config, fast=fast)


def segment_volume(
    nifti_path: Path,
    output_dir: Path,
    tasks_config: Dict[str, Any],
    *,
    fast: bool,
    verbose: bool = False,
    force: bool = False,
    backend: TotalSegmentatorBackend | None = None,
) -> List[str]:
    """Run segmentation tasks and optional post-processing."""
    return run_segment_volume_workflow(
        nifti_path=nifti_path,
        output_dir=output_dir,
        tasks_config=tasks_config,
        fast=fast,
        verbose=verbose,
        force=force,
        backend=backend,
        clean_and_merge_masks_fn=clean_and_merge_masks,
        logger_obj=logger,
    )


def process_single_volume(
    idx: int,
    row: Dict[str, Any],  # must be JSON-serialisable
    tasks_config: Dict[str, Any],
    *,
    fast: bool,
    verbose: bool,
    force: bool,
    backend: TotalSegmentatorBackend | None = None,
) -> Tuple[int, str | None, str | None, str | None]:
    """Return ``(idx, output_dir|None, error_msg|None, warning_msg|None)``."""
    setup_logging(verbose=verbose)

    try:
        nifti_path = Path(row["nifti_path"])
    except KeyError:
        return idx, None, "column 'nifti_path' missing", None

    if not nifti_path.exists():
        return idx, None, "file not found", None

    try:
        warnings = segment_volume(
            nifti_path,
            nifti_path.parent,
            tasks_config,
            fast=fast,
            verbose=verbose,
            force=force,
            backend=backend,
        )
        warning_msg = " | ".join(warnings) if warnings else None
        return idx, str(nifti_path.parent), None, warning_msg
    except Exception as exc:
        logger.debug("Traceback for %s:\n%s", nifti_path.name, traceback.format_exc())
        return idx, None, str(exc), None


def add_segment_arguments(
    parser: argparse.ArgumentParser,
    include_manifest: bool = True,
    include_dry_run: bool = True,
) -> None:
    """Add command-line arguments for segment."""
    parser.add_argument(
        "csv_path_pos",
        nargs="?",
        type=str,
        default=None,
        help="Path to the input CSV file. Defaults to ./nifti_index.csv.",
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
        help="Output CSV (default: overwrite input).",
    )
    parser.add_argument(
        "--error_csv_path",
        type=str,
        default=None,
        help="CSV for failures only (default: alongside input CSV).",
    )
    parser.add_argument(
        "--tasks_config",
        type=str,
        default=None,
        help=(
            "JSON config for segmentation tasks. "
            "If omitted, use the manifest's segmentation section."
        ),
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Pool size")
    parser.add_argument(
        "--fast", action="store_true", help="Use TotalSegmentator fast mode"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if output masks already exist",
    )
    parser.add_argument(
        "--start_method",
        choices=["spawn", "fork", "forkserver"],
        default="spawn",
        help="multiprocessing start method: spawn=robust, fork=faster (Linux)",
    )
    parser.add_argument(
        "--timeout_sec",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Per-volume timeout in seconds",
    )
    if include_manifest:
        parser.add_argument(
            "--manifest",
            type=str,
            default=DEFAULT_MANIFEST_NAME,
            help=(
                "Dataset manifest name or path to manifest JSON "
                f"(default: {DEFAULT_MANIFEST_NAME})."
            ),
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
    """Build and return the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Batch segmentation with TotalSegmentator v2",
        add_help=add_help,
    )
    add_segment_arguments(parser)
    return parser


def normalize_segment_args(args: argparse.Namespace) -> argparse.Namespace:
    """Normalize parsed command-line arguments and fill derived defaults."""
    csv_in = args.csv_path_opt if args.csv_path_opt is not None else args.csv_path_pos

    if csv_in is None:
        csv_path = Path.cwd() / "nifti_index.csv"
    else:
        csv_path = Path(csv_in)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    args.csv_path = str(csv_path.resolve())

    if not args.csv_path_out:
        args.csv_path_out = args.csv_path
    else:
        args.csv_path_out = str(Path(args.csv_path_out))

    if args.error_csv_path:
        args.error_csv_path = str(Path(args.error_csv_path))
    else:
        args.error_csv_path = str(Path(args.csv_path).parent / "seg_errors.csv")

    if args.tasks_config:
        args.tasks_config = str(Path(args.tasks_config))

    del args.csv_path_pos
    del args.csv_path_opt

    return args


def main(args: argparse.Namespace) -> None:
    """Run the module entry point."""
    setup_logging(verbose=getattr(args, "verbose", False))

    run_segment_batch_workflow(
        args=args,
        manifest_base_path=Path(__file__).resolve().parents[1],
        process_single_volume_fn=process_single_volume,
        prefetch_totalsegmentator_models_fn=prefetch_totalsegmentator_models,
        process_pool_executor_cls=ProcessPoolExecutor,
        as_completed_fn=as_completed,
        tqdm_fn=tqdm,
        report_volumes_fn=report_volumes,
        logger_obj=logger,
    )


if __name__ == "__main__":
    setup_logging()
    args = build_parser().parse_args()
    args = normalize_segment_args(args)
    if getattr(args, "dry_run", False):
        logger.info("Dry run: segment")
        logger.info("%s", args)
        raise SystemExit(0)
    main(args)
