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
from imperandi.utils.misc import print_args
from imperandi.utils.run_state import (
    atomic_write_csv,
    atomic_write_json,
    compute_args_hash,
    fingerprint_inputs,
    load_state,
    now_epoch,
    state_matches,
)

logger = logging.getLogger(__name__)
DEFAULT_CHECKPOINT_EVERY_ROWS = 50
DEFAULT_CHECKPOINT_EVERY_SEC = 350


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
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--checkpoint_every_rows",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY_ROWS,
        help="Flush checkpoint files every N processed rows.",
    )
    parser.add_argument(
        "--checkpoint_every_sec",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY_SEC,
        help="Flush checkpoint files every T seconds.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from matching checkpoint state if available.",
    )
    parser.add_argument(
        "--strict_resume",
        action="store_true",
        default=False,
        help="Use content hashing for input fingerprint when resuming.",
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
    parser = argparse.ArgumentParser(
        description="Extract CT contrast phase metadata from NIfTI volumes.",
        add_help=add_help,
    )
    add_phase_arguments(parser)
    return parser


def normalize_phase_args(args: argparse.Namespace) -> argparse.Namespace:
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
    parser = build_parser()
    args = parser.parse_args()
    args = normalize_phase_args(args)
    logger.info("🚀 Running %s script with arguments: %s", Path(__file__).name, args)
    return args


def process_single_volume(
    idx: int,
    row: Mapping[str, Any],
    *,
    phase_extractor: Callable[[Any], Dict[str, Any]],
) -> Tuple[int, Dict[str, Any] | None, str | None]:
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


def extract_phase_volumes(
    df: pd.DataFrame,
    *,
    verbose: bool,
    phase_extractor: Callable[[Any], Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    for idx, row in iterator:
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
    phase_extractor = _load_phase_extractor()

    output_path = Path(args.csv_path_out)
    error_path = Path(args.error_csv_path)
    state_path = output_path.parent / f".{output_path.stem}.phase.state.json"
    checkpoint_main_path = output_path.parent / f".{output_path.stem}.phase.checkpoint.csv"
    checkpoint_err_path = error_path.parent / f".{error_path.stem}.phase.checkpoint.csv"
    exclude_hash_args = {
        "csv_path_out",
        "error_csv_path",
        "dry_run",
        "verbose",
        "resume",
        "checkpoint_every_rows",
        "checkpoint_every_sec",
        "strict_resume",
    }
    args_hash = compute_args_hash(args, exclude_keys=exclude_hash_args)
    input_fp = fingerprint_inputs(
        args.csv_path, strict=bool(getattr(args, "strict_resume", False))
    )
    state = load_state(state_path)
    can_resume = bool(getattr(args, "resume", False)) and state_matches(
        state, command="phase", args_hash=args_hash, input_fingerprint=input_fp
    )

    if can_resume and checkpoint_main_path.exists():
        logger.info("Resuming phase from checkpoint: %s", checkpoint_main_path)
        df = pd.read_csv(checkpoint_main_path).copy()
    else:
        df = pd.read_csv(args.csv_path).copy()
        df["_source_idx"] = df.index.astype(int)
    if "_source_idx" not in df.columns:
        df["_source_idx"] = df.index.astype(int)

    errors_by_idx: Dict[int, Dict[str, Any]] = {}
    completed_indices: set[int] = set()
    if can_resume:
        completed_indices = {
            int(i)
            for i in (state or {}).get("completed_indices", [])
            if isinstance(i, int)
        }
        if checkpoint_err_path.exists():
            err_ckpt = pd.read_csv(checkpoint_err_path)
            for _, row in err_ckpt.iterrows():
                if "_source_idx" in row:
                    try:
                        errors_by_idx[int(row["_source_idx"])] = row.to_dict()
                    except Exception:
                        pass

    checkpoint_every_rows = max(
        1, int(getattr(args, "checkpoint_every_rows", DEFAULT_CHECKPOINT_EVERY_ROWS))
    )
    checkpoint_every_sec = max(
        1, int(getattr(args, "checkpoint_every_sec", DEFAULT_CHECKPOINT_EVERY_SEC))
    )
    processed_since_checkpoint = 0
    last_checkpoint_time = now_epoch()

    def _checkpoint_write(*, force: bool = False) -> None:
        nonlocal processed_since_checkpoint, last_checkpoint_time
        elapsed = now_epoch() - last_checkpoint_time
        if not force and processed_since_checkpoint < checkpoint_every_rows and elapsed < checkpoint_every_sec:
            return
        atomic_write_csv(df, checkpoint_main_path, index=False)
        if errors_by_idx:
            atomic_write_csv(pd.DataFrame(list(errors_by_idx.values())), checkpoint_err_path, index=False)
        elif checkpoint_err_path.exists():
            checkpoint_err_path.unlink()
        atomic_write_json(
            state_path,
            {
                "command": "phase",
                "args_hash": args_hash,
                "input_fingerprint": input_fp,
                "completed_indices": sorted(completed_indices),
                "updated_at_epoch": now_epoch(),
            },
        )
        processed_since_checkpoint = 0
        last_checkpoint_time = now_epoch()

    iterator = df.index.tolist()
    if args.verbose:
        iterator = tqdm(iterator, total=len(iterator), desc="Phase")
    for idx in iterator:
        src_idx = int(df.at[idx, "_source_idx"])
        if src_idx in completed_indices:
            continue
        _, phase_info, err_msg = process_single_volume(
            idx,
            df.loc[idx].to_dict(),
            phase_extractor=phase_extractor,
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
        processed_since_checkpoint += 1
        _checkpoint_write(force=False)

    _checkpoint_write(force=True)
    df_out = df.drop(columns=["_source_idx"], errors="ignore")
    atomic_write_csv(df_out, args.csv_path_out, index=False)
    logger.info("Wrote main table -> %s", args.csv_path_out)

    if errors_by_idx:
        df_err = pd.DataFrame(list(errors_by_idx.values())).drop(
            columns=["_source_idx"], errors="ignore"
        )
        atomic_write_csv(df_err, args.error_csv_path, index=False)
        logger.warning("%d rows failed -> %s", len(df_err), args.error_csv_path)
    atomic_write_json(
        state_path,
        {
            "command": "phase",
            "args_hash": args_hash,
            "input_fingerprint": input_fp,
            "completed_indices": sorted(completed_indices),
            "updated_at_epoch": now_epoch(),
            "finished": True,
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
