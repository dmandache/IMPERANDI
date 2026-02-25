from __future__ import annotations

import traceback
import logging
from pathlib import Path
import os
import argparse
import pandas as pd
from ast import literal_eval
import dicom2nifti
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile

from imperandi.utils.archive_io import (
    ArchiveSession,
    DEFAULT_ARCHIVE_MAX_DEPTH,
    is_archive_uri,
)
from imperandi.utils.files import copy_files_to_temp_dir, check_file, is_valid_nifti
from imperandi.utils.logging import setup_logging
from imperandi.utils.misc import report_volumes, report_change, print_args
from imperandi.utils.manifest import load_manifest
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


# Function to parse command-line arguments
def add_convert_arguments(
    parser: argparse.ArgumentParser,
    include_manifest: bool = True,
    include_dry_run: bool = True,
):
    parser.add_argument(
        "csv_path_pos",
        nargs="?",
        type=str,
        default=None,
        help="Path to the input CSV file(s). Defaults to ./dicom_index.csv.",
    )
    parser.add_argument(
        "--csv_path",
        dest="csv_path_opt",
        nargs="+",
        type=str,
    )

    parser.add_argument(
        "output_dir_pos",
        nargs="?",
        type=str,
        default=None,
        help="Root directory for NIFTI data.",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir_opt",
        type=str,
    )

    parser.add_argument(
        "--csv_path_out",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to save the final output CSV file. "
            "Defaults to <csv_dir>/nifti_index.csv."
        ),
    )
    parser.add_argument(
        "--error_csv_path",
        type=str,
        default=None,
        help=(
            "Path to save the error CSV file. " "Defaults to <csv_dir>/conv_errors.csv."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose mode")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of parallel jobs",
    )
    parser.add_argument(
        "--archive_max_depth",
        type=int,
        default=DEFAULT_ARCHIVE_MAX_DEPTH,
        help="Maximum recursion depth for nested archives.",
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


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert DICOM Series to NIFTI file",
        add_help=add_help,
    )
    add_convert_arguments(parser)
    return parser


def parse_arguments():
    parser = build_parser()
    args = parser.parse_args()
    args = normalize_convert_args(args)
    logger.info("🚀 Running %s script with arguments: %s", Path(__file__).name, args)
    return args


def normalize_convert_args(args: argparse.Namespace) -> argparse.Namespace:
    # pick optionals over positionals
    csv_in = args.csv_path_opt if args.csv_path_opt is not None else args.csv_path_pos
    out_in = (
        args.output_dir_opt if args.output_dir_opt is not None else args.output_dir_pos
    )

    # csv_path -> list[str]
    if csv_in is None:
        csv_paths = [Path.cwd() / "dicom_index.csv"]
    elif isinstance(csv_in, str):
        csv_paths = [Path(csv_in)]
    else:
        csv_paths = [Path(p) for p in csv_in]

    for p in csv_paths:
        if not p.exists():
            raise FileNotFoundError(f"CSV file not found: {p}")
        if p.suffix.lower() != ".csv":
            raise ValueError(f"Not a CSV file: {p}")

    args.csv_path = [str(p.resolve()) for p in csv_paths]

    # output_dir -> directory
    if out_in is None:
        raise ValueError("output_dir is required (positional or --output_dir).")

    args.output_dir = out_in

    del args.csv_path_pos
    del args.csv_path_opt
    del args.output_dir_pos
    del args.output_dir_opt

    first_csv = Path(args.csv_path[0])
    csv_dir = first_csv.parent

    if not args.csv_path_out:
        args.csv_path_out = str(csv_dir / "nifti_index.csv")

    if not args.error_csv_path:
        args.error_csv_path = str(csv_dir / "conv_errors.csv")

    args.archive_max_depth = int(
        getattr(args, "archive_max_depth", DEFAULT_ARCHIVE_MAX_DEPTH)
    )
    args.archive_cache_dir = getattr(args, "archive_cache_dir", None)
    args.keep_archive_cache = bool(getattr(args, "keep_archive_cache", False))

    return args


# Function to convert string representation of lists to actual lists
def convert_list_str_to_list(cell):
    """
    Convert a string representation of a list to an actual list using `literal_eval`.

    Args:
        cell (str): String that represents a list.

    Returns:
        list or original value: If the string can be converted to a list, return the list, else return the original value.
    """
    try:
        return literal_eval(cell)
    except (ValueError, SyntaxError):
        return cell


def _flatten_dicom_paths(cell) -> list[str]:
    if isinstance(cell, list):
        return [str(v) for v in cell]
    if isinstance(cell, str):
        return [cell]
    return []


def _apply_uri_mapping_to_cell(cell, uri_map: dict[str, str | None]):
    if isinstance(cell, list):
        mapped = []
        for value in cell:
            s = str(value)
            if is_archive_uri(s):
                local = uri_map.get(s)
                if local:
                    mapped.append(local)
            else:
                mapped.append(s)
        return mapped

    if isinstance(cell, str):
        if is_archive_uri(cell):
            return uri_map.get(cell)
        return cell

    return cell


def materialize_archive_dicom_paths(
    df: pd.DataFrame,
    archive_session: ArchiveSession,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replace archive:// DICOM paths with local materialized paths.
    Returns updated dataframe and an error dataframe for rows that could not be materialized.
    """
    if "dicom_path" not in df.columns:
        return df, pd.DataFrame()

    unique_uris = sorted(
        {
            p
            for cell in df["dicom_path"]
            for p in _flatten_dicom_paths(cell)
            if is_archive_uri(p)
        }
    )
    if not unique_uris:
        return df, pd.DataFrame()

    uri_map: dict[str, str | None] = {}
    for uri in unique_uris:
        try:
            uri_map[uri] = str(archive_session.materialize(uri))
        except Exception as exc:
            logger.warning("[archive][materialize] convert skip %s (%s)", uri, exc)
            uri_map[uri] = None

    out = df.copy()
    out["dicom_path"] = out["dicom_path"].apply(
        lambda cell: _apply_uri_mapping_to_cell(cell, uri_map)
    )

    error_rows = []
    keep_mask = []
    for idx, cell in out["dicom_path"].items():
        if isinstance(cell, list):
            clean_list = [p for p in cell if isinstance(p, str) and p.strip()]
            out.at[idx, "dicom_path"] = clean_list
            if clean_list:
                keep_mask.append(True)
                continue
            row = out.loc[idx].copy()
            row["error"] = "all archive members failed to materialize"
            error_rows.append(row)
            keep_mask.append(False)
            continue

        if isinstance(cell, str) and cell.strip():
            keep_mask.append(True)
            continue

        row = out.loc[idx].copy()
        row["error"] = "archive path failed to materialize"
        error_rows.append(row)
        keep_mask.append(False)

    out = out.loc[pd.Series(keep_mask, index=out.index)].reset_index(drop=True)
    df_err = pd.DataFrame(error_rows) if error_rows else pd.DataFrame()
    return out, df_err


# Function to convert a single DICOM volume to NIfTI (parallel task)
def process_single_volume(k, row, output_dir, verbose, return_status=False):
    """
    Convert a single DICOM series to a NIfTI file, saving the result to the
    specified output directory.

    Args:
        k (int): Index of the current volume being processed.
        row (pd.Series): Metadata for the current DICOM series.
        output_dir (str): Directory to save the NIfTI files.
        verbose (bool): If true, configure verbose logging for worker setup.

    Returns:
        tuple:
            - default (return_status=False): (index, export_path, error_row)
            - with status (return_status=True):
              (index, export_path, error_row, status), where status is
              "converted", "skipped", or "failed".
    """
    setup_logging(verbose=verbose)

    def _result(export_path, error_row, status):
        if return_status:
            return k, export_path, error_row, status
        return k, export_path, error_row

    try:
        dicom_dir_path = row["series_dir"]
        files_in_vol = row["dicom_path"]
        files_in_dir = list(Path(dicom_dir_path).iterdir())

        n_files_in_vol = len(files_in_vol) if isinstance(files_in_vol, list) else 1
        n_files_in_dir = len(files_in_dir)
        series_id = (
            row.series_id + "_" + str(row.volume_ordinal_in_series)
            if row.volume_ordinal_in_series > 1
            else row.series_id
        )

        export_dir = Path(output_dir) / str(row.patient_key) / str(row.study_id) / str(
            series_id
        )
        export_path = export_dir / "scan.nii.gz"

        # Reuse existing valid outputs silently to avoid per-file success logs.
        if (
            export_path.exists()
            and export_path.is_file()
            and is_valid_nifti(export_path)
            and export_path.stat().st_size > 0
        ):
            return _result(export_path, None, "skipped")

        if not export_dir.exists():
            os.makedirs(export_dir, exist_ok=True)

        def read_dicom_write_nifti(dicom_dir_one_volume):
            dicom_input = dicom2nifti.common.read_dicom_directory(dicom_dir_one_volume)
            dicom2nifti.convert_dicom.dicom_array_to_nifti(
                dicom_input, export_path, reorient_nifti=False
            )

        if n_files_in_dir != n_files_in_vol:
            temp_dir_root = ".tmp"
            os.makedirs(temp_dir_root, exist_ok=True)
            with tempfile.TemporaryDirectory(
                dir=temp_dir_root, prefix="temp_convert_"
            ) as temp_dir:
                copy_files_to_temp_dir(paths=files_in_vol, temp_dir=temp_dir)
                read_dicom_write_nifti(temp_dir)
        else:
            read_dicom_write_nifti(dicom_dir_path)

        if is_valid_nifti(export_path):
            return _result(export_path, None, "converted")

        logger.error(
            "Error processing volume %s: output is not a valid NIfTI file.",
            k,
        )
        row["error"] = "output not valid nifti"
        return _result(None, row, "failed")

    except Exception:
        error_msg = traceback.format_exc()
        logger.error("Error processing volume %s: %s", k, error_msg)
        row["error"] = error_msg
        return _result(None, row, "failed")


# Function to convert DICOM to NIfTI in parallel
def convert_dicom_to_nifti_parallel(
    df,
    output_dir,
    print_flag,
    num_workers,
    *,
    on_result=None,
):
    """
    Convert multiple DICOM volumes to NIfTI in parallel using multiprocessing.

    Args:
        df (pd.DataFrame): DataFrame containing DICOM metadata.
        output_dir (str): Directory to save the NIfTI files.
        print_flag (bool): Whether to display progress using `tqdm`.
        num_workers (int): Number of parallel processes to use.

    Returns:
        tuple: The updated DataFrame with NIfTI paths and a DataFrame with any
        errors encountered.
    """
    n_samples = len(df)
    df_err = pd.DataFrame()
    converted_count = 0
    skipped_count = 0
    failed_count = 0

    logger.info("%s volumes to convert", n_samples)

    df["volume_ordinal_in_series"] = df.groupby("series_id").cumcount() + 1

    if "series_dir" not in df.columns:
        df["series_dir"] = df["dicom_path"].apply(
            lambda x: Path(x[0]).parent if isinstance(x, list) else Path(x).parent
        )

    # Use ProcessPoolExecutor to parallelize the task.
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_single_volume,
                k,
                df.iloc[k],
                output_dir,
                print_flag,
                True,
            )
            for k in range(n_samples)
        ]

        # Prepare iterator with optional progress bar.
        iterator = as_completed(futures)
        if print_flag:
            iterator = tqdm(as_completed(futures), total=n_samples)

        # Collect results.
        for future in iterator:
            try:
                result = future.result(timeout=600)  # wait 10 mins max
            except Exception as e:
                logger.error("Future failed to execute under 10 minutes: %s", e)
                failed_count += 1
                continue

            if len(result) == 4:
                k, export_path, error_row, status = result
            else:
                k, export_path, error_row = result
                status = "failed" if error_row is not None else "converted"

            if status == "converted":
                converted_count += 1
            elif status == "skipped":
                skipped_count += 1
            elif status == "failed":
                failed_count += 1

            if export_path is not None:
                df.iloc[k, df.columns.get_loc("nifti_path")] = str(export_path)
            elif error_row is not None:
                # Append error row to df_err.
                try:
                    df_err = pd.concat(
                        [df_err, error_row.to_frame().T], ignore_index=True
                    )
                except Exception:
                    # Fallback: create a dataframe from dict.
                    df_err = pd.concat(
                        [df_err, pd.DataFrame([error_row])], ignore_index=True
                    )
            if on_result is not None:
                on_result(k, export_path, error_row, status)

    logger.info(
        "Conversion summary: %s converted, %s skipped (already valid), %s failed",
        converted_count,
        skipped_count,
        failed_count,
    )

    return df, df_err


# Main function
def main(args):
    """
    Main function to convert DICOM series to NIfTI files in parallel and save the results to CSV.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    output_path = Path(args.csv_path_out)
    error_path = Path(args.error_csv_path)
    state_path = output_path.parent / f".{output_path.stem}.convert.state.json"
    checkpoint_main_path = output_path.parent / f".{output_path.stem}.convert.checkpoint.csv"
    checkpoint_err_path = error_path.parent / f".{error_path.stem}.convert.checkpoint.csv"

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
        state,
        command="convert",
        args_hash=args_hash,
        input_fingerprint=input_fp,
    )

    if args.verbose:
        for p in args.csv_path:
            check_file(p)

    if hasattr(args, "manifest") and args.manifest:
        load_manifest(args.manifest, base_path=Path(__file__).resolve().parents[1])

    if can_resume and checkpoint_main_path.exists():
        logger.info("Resuming convert from checkpoint: %s", checkpoint_main_path)
        df_all = pd.read_csv(checkpoint_main_path).copy()
    else:
        df_list = [pd.read_csv(p) for p in args.csv_path]
        df_all = pd.concat(df_list, ignore_index=True)
        df_all["_source_idx"] = df_all.index.astype(int)

    if "nifti_path" not in df_all.columns:
        df_all["nifti_path"] = None
    if "_source_idx" not in df_all.columns:
        df_all["_source_idx"] = df_all.index.astype(int)

    df_all = df_all.map(lambda x: convert_list_str_to_list(x) if isinstance(x, str) else x)

    logger.info("Before conversion:")
    report_volumes(df_all)
    df_prev = df_all.copy()

    if args.dry_run:
        logger.info("Dry run: convert")
        print_args(args)
        return

    completed_indices: set[int] = set()
    errors_by_idx: dict[int, dict] = {}
    if can_resume:
        completed_indices = {
            int(i)
            for i in (state or {}).get("completed_indices", [])
            if isinstance(i, int)
        }
        logger.info(
            "Resume enabled: %d completed rows restored from state",
            len(completed_indices),
        )
        if checkpoint_err_path.exists():
            err_ckpt = pd.read_csv(checkpoint_err_path)
            for _, row in err_ckpt.iterrows():
                if "_source_idx" in row:
                    try:
                        errors_by_idx[int(row["_source_idx"])] = row.to_dict()
                    except Exception:
                        continue

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
        if (
            not force
            and processed_since_checkpoint < checkpoint_every_rows
            and elapsed < checkpoint_every_sec
        ):
            return
        atomic_write_csv(df_all, checkpoint_main_path, index=False)
        if errors_by_idx:
            atomic_write_csv(
                pd.DataFrame(list(errors_by_idx.values())),
                checkpoint_err_path,
                index=False,
            )
        elif checkpoint_err_path.exists():
            checkpoint_err_path.unlink()
        atomic_write_json(
            state_path,
            {
                "command": "convert",
                "args_hash": args_hash,
                "input_fingerprint": input_fp,
                "completed_indices": sorted(completed_indices),
                "updated_at_epoch": now_epoch(),
            },
        )
        processed_since_checkpoint = 0
        last_checkpoint_time = now_epoch()

    with ArchiveSession(
        cache_dir=args.archive_cache_dir,
        keep_cache=args.keep_archive_cache,
        max_depth=args.archive_max_depth,
    ) as archive_session:
        work_df = df_all[~df_all["_source_idx"].isin(completed_indices)].copy()
        work_df = work_df.reset_index(drop=True)
        work_df, df_archive_err = materialize_archive_dicom_paths(work_df, archive_session)
        if not df_archive_err.empty:
            for _, row in df_archive_err.iterrows():
                if "_source_idx" in row:
                    source_idx = int(row["_source_idx"])
                    errors_by_idx[source_idx] = row.to_dict()
                    completed_indices.add(source_idx)

        def _on_result(k: int, export_path, error_row, status: str) -> None:
            nonlocal processed_since_checkpoint
            processed_since_checkpoint += 1
            source_idx = int(work_df.iloc[k]["_source_idx"])
            completed_indices.add(source_idx)
            if export_path is not None:
                mask = df_all["_source_idx"] == source_idx
                df_all.loc[mask, "nifti_path"] = str(export_path)
                errors_by_idx.pop(source_idx, None)
            elif error_row is not None:
                err_dict = (
                    error_row.to_dict()
                    if hasattr(error_row, "to_dict")
                    else dict(error_row)
                )
                err_dict["_source_idx"] = source_idx
                errors_by_idx[source_idx] = err_dict
            _checkpoint_write(force=False)

        _, _ = convert_dicom_to_nifti_parallel(
            work_df,
            args.output_dir,
            args.verbose,
            args.num_workers,
            on_result=_on_result,
        )
        _checkpoint_write(force=True)

    logger.info("After conversion:")
    report_volumes(df_all)
    report_change(df_all, df_prev)

    df_success = df_all[df_all["nifti_path"].notna()].copy()
    if "_source_idx" in df_success.columns:
        df_success = df_success.drop(columns=["_source_idx"], errors="ignore")
    atomic_write_csv(df_success, args.csv_path_out, index=False)
    if errors_by_idx:
        df_err = pd.DataFrame(list(errors_by_idx.values())).drop(
            columns=["_source_idx"], errors="ignore"
        )
        logger.warning(
            "⚠️ DICOM to Nifti Conversion Errors on patients : %s",
            df_err.patient_key.unique(),
        )
        report_volumes(df_err)
        atomic_write_csv(df_err, args.error_csv_path, index=False)

    logger.info("Conversion done ✔")
    atomic_write_json(
        state_path,
        {
            "command": "convert",
            "args_hash": args_hash,
            "input_fingerprint": input_fp,
            "completed_indices": sorted(completed_indices),
            "updated_at_epoch": now_epoch(),
            "finished": True,
        },
    )


if __name__ == "__main__":
    setup_logging()
    args = parse_arguments()
    if args.dry_run:
        logger.info("Dry run: convert")
        print_args(args)
        raise SystemExit(0)
    setup_logging(verbose=getattr(args, "verbose", False))
    main(args)
