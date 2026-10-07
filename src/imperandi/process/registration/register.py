"""Command-line entry point for the ``imperandi register`` stage."""

import argparse
import logging
import multiprocessing as mp
from pathlib import Path

import pandas as pd

from imperandi.utils.checkpoint_cli import add_checkpoint_arguments
from imperandi.utils.manifest import load_manifest
from imperandi.utils.run_state import build_checkpoint_paths
from .config import CONSENSUS_METHODS, RegistrationConfig
from .cohort import prepare_cohort
from .reporting import group_label

logger = logging.getLogger(__name__)


def add_registration_arguments(parser):
    parser.add_argument(
        "csv_path_pos", nargs="?", help="Input CSV (default: ./nifti_index.csv)."
    )
    parser.add_argument("csv_path_out_pos", nargs="?", help="Optional output CSV.")
    parser.add_argument("--csv_path", dest="csv_path_opt")
    parser.add_argument("--csv_path_out", help="Default: <input_stem>_registered.csv.")
    parser.add_argument("--error_csv_path", help="Default: register_errors.csv.")
    parser.add_argument(
        "--qc_csv_path",
        help="Stage Dice and provenance table (default: register_qc.csv).",
    )
    parser.add_argument(
        "--output_dir", help="Artifact root (default: registration beside input CSV)."
    )
    parser.add_argument(
        "--manifest", help="Built-in manifest name or YAML path (default: generic)."
    )
    parser.add_argument("--method", choices=CONSENSUS_METHODS)
    affine = parser.add_mutually_exclusive_group()
    affine.add_argument("--affine", action="store_true", default=None)
    affine.add_argument("--no_affine", action="store_false", dest="affine")
    elastic = parser.add_mutually_exclusive_group()
    elastic.add_argument("--elastic", action="store_true", default=None)
    elastic.add_argument("--no_elastic", action="store_false", dest="elastic")
    parser.add_argument(
        "--demons_smoothing_sigma_mm",
        type=float,
        help="Gaussian smoothing sigma for the Demons displacement field in mm.",
    )
    parser.add_argument("--visit_column")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Maximum concurrent visit/modality groups.",
    )
    parser.add_argument(
        "--threads_per_worker",
        type=int,
        default=1,
        help="SimpleITK threads per group worker.",
    )
    parser.add_argument(
        "--start_method", choices=mp.get_all_start_methods(), default="spawn"
    )
    parser.add_argument(
        "--timeout_sec",
        type=float,
        default=900,
        help="Hard group timeout, including startup; 0 disables.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute all groups, ignoring existing checkpoints.",
    )
    parser.add_argument(
        "--retry_failed",
        action="store_true",
        help="Retry groups with recorded errors on resume.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        dest="dry_run",
        action="store_true",
        help="Validate and log group plan without writing files or loading images.",
    )
    add_checkpoint_arguments(parser, default_rows=50, default_sec=300)


def build_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="Register organs and build tumor consensus.", add_help=add_help
    )
    add_registration_arguments(parser)
    return parser


def normalize_registration_args(args):
    """Resolve paths and validate runtime flags; safe to call more than once."""
    csv_in = (
        getattr(args, "csv_path_opt", None)
        or getattr(args, "csv_path", None)
        or getattr(args, "csv_path_pos", None)
        or "nifti_index.csv"
    )
    source = Path(csv_in).expanduser().resolve()
    output = (
        Path(
            getattr(args, "csv_path_out", None)
            or getattr(args, "csv_path_out_pos", None)
            or source.with_name(source.stem + "_registered.csv")
        )
        .expanduser()
        .resolve()
    )
    error = (
        Path(
            getattr(args, "error_csv_path", None)
            or output.with_name("register_errors.csv")
        )
        .expanduser()
        .resolve()
    )
    qc = (
        Path(getattr(args, "qc_csv_path", None) or output.with_name("register_qc.csv"))
        .expanduser()
        .resolve()
    )
    args.qc_csv_path = str(qc)
    checkpoint_paths = build_checkpoint_paths(output, error, "register")
    paths = [
        source,
        output,
        error,
        qc,
        checkpoint_paths.state_path,
        checkpoint_paths.main_checkpoint_path,
        checkpoint_paths.error_checkpoint_path,
    ]
    if len(set(paths)) != len(paths):
        raise ValueError(
            "Registration input, output, error, QC and checkpoint paths must differ"
        )
    if not source.is_file():
        raise FileNotFoundError(f"CSV file not found: {source}")
    if source.suffix.lower() != ".csv":
        raise ValueError(f"Not a CSV file: {source}")
    args.csv_path, args.csv_path_out, args.error_csv_path = (
        str(source),
        str(output),
        str(error),
    )
    args.output_dir = str(
        Path(getattr(args, "output_dir", None) or source.parent / "registration")
        .expanduser()
        .resolve()
    )
    defaults = dict(
        num_workers=1,
        threads_per_worker=1,
        timeout_sec=900,
        start_method="spawn",
        force=False,
        retry_failed=False,
        verbose=False,
        dry_run=False,
        resume=True,
        strict_resume=False,
        checkpoint_every_rows=50,
        checkpoint_every_sec=300,
        manifest=None,
        method=None,
        affine=None,
        elastic=None,
        demons_smoothing_sigma_mm=None,
        visit_column=None,
    )
    for name, default in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, default)
    for name in [
        "num_workers",
        "threads_per_worker",
        "checkpoint_every_rows",
        "checkpoint_every_sec",
    ]:
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not 0 <= args.timeout_sec < float("inf"):
        raise ValueError("timeout_sec must be finite and nonnegative")
    if args.start_method not in mp.get_all_start_methods():
        raise ValueError(f"Unavailable start method: {args.start_method}")
    for name in ["csv_path_opt", "csv_path_pos", "csv_path_out_pos"]:
        vars(args).pop(name, None)
    return args


def resolve_config(args):
    manifest = load_manifest(args.manifest or "generic", base_path=Path.cwd())
    raw = manifest.get("registration", {})
    RegistrationConfig.from_mapping(raw)
    settings = dict(raw)
    for name in [
        "method",
        "affine",
        "elastic",
        "demons_smoothing_sigma_mm",
        "visit_column",
    ]:
        value = getattr(args, name)
        if value is not None:
            settings[name] = value
    return RegistrationConfig.from_mapping(settings), manifest


def main(args):
    args = normalize_registration_args(args)
    config, manifest = resolve_config(args)
    table = pd.read_csv(
        args.csv_path, dtype={"patient_key": str, config.visit_column: str}
    )
    if args.dry_run:
        planned = prepare_cohort(table, config)
        logger.info(
            "Registration dry run: series=%d, groups=%d, method=%s, affine=%s, "
            "elastic=%s",
            len(planned),
            planned.registration_group_id.nunique(),
            config.method,
            config.affine,
            config.elastic,
        )
        for _, group in planned.groupby("registration_group_id", sort=True):
            logger.info(
                "Registration group planned: %s, series=%d",
                group_label(group.iloc[0], config.visit_column),
                len(group),
            )
        return None
    from .runtime import run_registration

    return run_registration(args, table, config, manifest)


if __name__ == "__main__":
    from imperandi.utils.logging import setup_logging

    arguments = build_parser().parse_args()
    setup_logging(verbose=arguments.verbose)
    main(arguments)
