"""Phase extraction routines and CLI plumbing for CT exam workflows.

The definitions in this module are part of the Imperandi codebase and are
intended to be reused by higher-level workflows and CLI entry points.
"""

from __future__ import annotations

import argparse
import logging
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Tuple

import nibabel as nib
import pandas as pd
from tqdm import tqdm

from imperandi.utils.logging import setup_logging
from imperandi.utils.manifest import DEFAULT_MANIFEST_NAME, load_manifest
from imperandi.utils.misc import print_args

logger = logging.getLogger(__name__)


def _load_phase_extractor() -> Callable[[Any], Dict[str, Any]]:
    """Load phase extractor.

    Returns:
        Callable[[Any], Dict[str, Any]]: Loaded object returned by this routine.

    Raises:
        RuntimeError: If runtime prerequisites or optional dependencies are unavailable.
    """
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
    include_manifest: bool = True,
    include_dry_run: bool = True,
) -> None:
    """Add command-line arguments for phase.

    Args:
        parser (argparse.ArgumentParser): Argument parser instance to configure.
        include_manifest (bool): Boolean flag controlling optional behavior. Defaults to `True`.
        include_dry_run (bool): Boolean flag controlling optional behavior. Defaults to `True`.
    """
    parser.add_argument(
        "csv_path_pos",
        nargs="?",
        type=str,
        default=None,
        help="Path to input CSV with a `nifti_path` column. Defaults to ./nifti_index.csv.",
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
        help="Re-extract phase even when `totalseg_phase` already has a value.",
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
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    if include_dry_run:
        parser.add_argument(
            "--dry-run",
            dest="dry_run",
            action="store_true",
            default=False,
            help="Print planned actions without running.",
        )


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    """Build and return the command-line parser.

    Args:
        add_help (bool): Boolean flag controlling optional behavior. Defaults to `True`.

    Returns:
        argparse.ArgumentParser: Configured argument parser instance.
    """
    parser = argparse.ArgumentParser(
        description="Extract CT contrast phase metadata from NIfTI volumes.",
        add_help=add_help,
    )
    add_phase_arguments(parser)
    return parser


def normalize_phase_args(args: argparse.Namespace) -> argparse.Namespace:
    """Normalize parsed command-line arguments and fill derived defaults.

    Args:
        args (argparse.Namespace): Parsed command-line arguments namespace.

    Returns:
        argparse.Namespace: Parsed and normalized argument namespace.

    Raises:
        ValueError: If provided inputs fail validation.
        FileNotFoundError: If an expected input file cannot be found.
    """
    csv_in = args.csv_path_opt if args.csv_path_opt is not None else args.csv_path_pos
    csv_path = Path(csv_in) if csv_in else (Path.cwd() / "nifti_index.csv")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Not a CSV file: {csv_path}")

    args.csv_path = str(csv_path.resolve())

    if not args.csv_path_out:
        args.csv_path_out = args.csv_path
    else:
        args.csv_path_out = str(Path(args.csv_path_out))

    if args.error_csv_path:
        args.error_csv_path = str(Path(args.error_csv_path))
    else:
        args.error_csv_path = str(Path(args.csv_path).parent / "phase_errors.csv")

    del args.csv_path_pos
    del args.csv_path_opt
    return args


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for this module.

    Returns:
        argparse.Namespace: Parsed and normalized argument namespace.
    """
    parser = build_parser()
    args = parser.parse_args()
    args = normalize_phase_args(args)
    logger.info("Running %s script with arguments: %s", Path(__file__).name, args)
    return args


def process_single_volume(
    idx: int,
    row: Mapping[str, Any],
    *,
    phase_extractor: Callable[[Any], Dict[str, Any]],
) -> Tuple[int, Dict[str, Any] | None, str | None]:
    """Perform single volume.

    Args:
        idx (int): Input value for idx.
        row (Mapping[str, Any]): Input value for row.
        phase_extractor (Callable[[Any], Dict[str, Any]]): Input value for phase extractor.

    Returns:
        Tuple[int, Dict[str, Any] | None, str | None]: Tuple containing outputs from this step.
    """
    nifti_path_value = row.get("nifti_path")
    if not isinstance(nifti_path_value, str) or not nifti_path_value.strip():
        return idx, None, "column 'nifti_path' missing or invalid"

    nifti_path = Path(nifti_path_value)
    if not nifti_path.exists():
        return idx, None, f"file not found: {nifti_path}"

    try:
        nifti_image = nib.load(str(nifti_path))
        phase_info = phase_extractor(nifti_image)
    except Exception as exc:
        logger.debug("Traceback for %s:\n%s", nifti_path, traceback.format_exc())
        return idx, None, str(exc)

    if not isinstance(phase_info, dict):
        return idx, None, "phase extractor did not return a dictionary"
    if not phase_info:
        return idx, None, "phase extractor returned no values"

    normalized = {f"totalseg_{key}": value for key, value in phase_info.items()}
    return idx, normalized, None


def _has_existing_totalseg_phase(value: Any) -> bool:
    """Return whether one `totalseg_phase` cell should be treated as already set."""
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return False
    return True


def extract_phase_volumes(
    df: pd.DataFrame,
    *,
    verbose: bool,
    force: bool = False,
    phase_extractor: Callable[[Any], Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract phase volumes.

    Args:
        df (pd.DataFrame): Input pandas DataFrame to process.
        verbose (bool): Boolean flag controlling optional behavior.
        phase_extractor (Callable[[Any], Dict[str, Any]]): Input value for phase extractor.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Processed pandas DataFrame.

    Raises:
        KeyError: If required keys are missing from a mapping-like input.
    """
    if "nifti_path" not in df.columns:
        unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
        if unnamed:
            df = df.drop(columns=unnamed)
    if "nifti_path" not in df.columns:
        raise KeyError("column 'nifti_path' missing")

    errors = []
    iterator = df.iterrows()
    if verbose:
        iterator = tqdm(iterator, total=len(df), desc="Phase")

    has_totalseg_phase = "totalseg_phase" in df.columns
    for idx, row in iterator:
        if (
            not force
            and has_totalseg_phase
            and _has_existing_totalseg_phase(row.get("totalseg_phase"))
        ):
            continue

        _, phase_info, err_msg = process_single_volume(
            idx,
            row.to_dict(),
            phase_extractor=phase_extractor,
        )
        if phase_info:
            for key, value in phase_info.items():
                df.at[idx, key] = value
            continue

        error_row = row.to_dict()
        error_row["error_message"] = err_msg or "unknown"
        errors.append(error_row)
        if verbose and err_msg:
            logger.warning("Row %s failed: %s", idx, err_msg)

    return df, pd.DataFrame(errors)


def main(args: argparse.Namespace) -> None:
    """Run this module entry point.

    Args:
        args (argparse.Namespace): Parsed command-line arguments namespace.
    """
    load_manifest(
        getattr(args, "manifest", None), base_path=Path(__file__).resolve().parents[1]
    )
    phase_extractor = _load_phase_extractor()

    df = pd.read_csv(args.csv_path).copy()
    df, df_err = extract_phase_volumes(
        df,
        verbose=args.verbose,
        force=bool(getattr(args, "force", False)),
        phase_extractor=phase_extractor,
    )

    df.to_csv(args.csv_path_out, index=False)
    logger.info("Wrote main table -> %s", args.csv_path_out)

    if not df_err.empty:
        df_err.to_csv(args.error_csv_path, index=False)
        logger.warning("%d rows failed -> %s", len(df_err), args.error_csv_path)


if __name__ == "__main__":
    setup_logging()
    args = parse_arguments()
    setup_logging(verbose=getattr(args, "verbose", False))
    if getattr(args, "dry_run", False):
        logger.info("Dry run: phase")
        print_args(args)
        raise SystemExit(0)
    main(args)
