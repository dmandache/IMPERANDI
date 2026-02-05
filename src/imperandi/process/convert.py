import warnings
import traceback
from pathlib import Path
import os
import argparse
import pandas as pd
from ast import literal_eval
import dicom2nifti
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile

from imperandi.utils.files import copy_files_to_temp_dir, check_file, is_valid_nifti
from imperandi.utils.misc import report_volumes, report_change, print_args
from imperandi.utils.manifest import load_manifest


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
            "Path to save the error CSV file. "
            "Defaults to <csv_dir>/nifti_index_errors.csv."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose mode")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of parallel jobs",
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
    print(f"Running {Path(__file__).name} script with arguments: {args}")
    return args


def normalize_convert_args(args: argparse.Namespace) -> argparse.Namespace:
    # pick optionals over positionals
    csv_in = args.csv_path_opt if args.csv_path_opt is not None else args.csv_path_pos
    out_in = args.output_dir_opt if args.output_dir_opt is not None else args.output_dir_pos

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
        args.error_csv_path = str(csv_dir / "nifti_index_errors.csv")

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


# Function to convert a single DICOM volume to NIfTI (parallel task)
def process_single_volume(k, row, output_dir, verbose):
    """
    Convert a single DICOM series to a NIfTI file, saving the result to the specified output directory.

    Args:
        k (int): Index of the current volume being processed.
        row (pd.Series): Metadata for the current DICOM series.
        output_dir (str): Directory to save the NIfTI files.
        verbose (bool): If true, print additional logs for debugging.

    Returns:
        tuple: A tuple containing the index, export path (if successful), and error row (if unsuccessful).
    """
    try:
        dicom_dir_path = row["series_dir"]
        files_in_vol = row["dicom_path"]
        files_in_dir = list(Path(dicom_dir_path).iterdir())

        n_files_in_vol = len(files_in_vol) if isinstance(files_in_vol, list) else 1
        n_files_in_dir = len(files_in_dir)

        if row.volume_ordinal_in_series > 1:
            if verbose:
                print("Multi-volume series")
            series_id = row.series_id + "_" + str(row.volume_ordinal_in_series)
        else:
            series_id = row.series_id

        export_dir = (
            Path(output_dir)
            / str(row.patient_key)
            / str(row.study_id)
            / str(series_id)  # / row.Modality
        )
        # export_dir = Path(output_dir) / row.patient_key / row.date / row.Modality / row.volume_id
        export_path = export_dir / "scan.nii.gz"

        # Check if file exists and skip
        if (
            export_path.exists()
            and export_path.is_file()
            and is_valid_nifti(export_path)
        ):
            sz = export_path.stat().st_size
            if sz > 0:
                if verbose:
                    print(
                        f"✅ File {export_path} exists and has size {sz*1e-6:.2f} MB. Skipping..."
                    )
                return k, export_path, None

        # Create export directory if it doesn't exist
        if not export_dir.exists():
            os.makedirs(export_dir, exist_ok=True)

        # # Conversion process
        # if n_files_in_dir != n_files_in_vol:
        #     print(f"{n_files_in_vol} files in volume vs. {n_files_in_dir} files in series dir")
        #     with tempfile.TemporaryDirectory(dir='/data/scratch/bdr220003/temp/', prefix='temp_convert_') as temp_dir:
        #         print(f"Using intermediary temp dir {temp_dir}")
        #         copy_files_to_temp_dir(paths=files_in_vol, temp_dir=temp_dir)
        #         dicom2nifti.dicom_series_to_nifti(temp_dir, export_path, reorient_nifti=False)
        # else:
        #     dicom2nifti.dicom_series_to_nifti(dicom_dir_path, export_path, reorient_nifti=False)

        # Conversion process
        def read_dicom_write_nifti(dicom_dir_one_volume):
            dicom_input = dicom2nifti.common.read_dicom_directory(dicom_dir_one_volume)
            dicom2nifti.convert_dicom.dicom_array_to_nifti(
                dicom_input, export_path, reorient_nifti=False
            )

        if n_files_in_dir != n_files_in_vol:
            if verbose:
                print(
                    f"{n_files_in_vol} files in volume vs. {n_files_in_dir} files in series dir"
                )

            temp_dir_root = ".tmp" #"/data/scratch/bdr220003/temp/"
            os.makedirs(temp_dir_root, exist_ok=True)
            with tempfile.TemporaryDirectory(
                dir=temp_dir_root, prefix="temp_convert_"
            ) as temp_dir:
                if verbose:
                    print(f"Using intermediary temp dir {temp_dir}")
                copy_files_to_temp_dir(paths=files_in_vol, temp_dir=temp_dir)
                read_dicom_write_nifti(temp_dir)
        else:
            if verbose:
                print(f"🧠 Processing volume {k}")
            read_dicom_write_nifti(dicom_dir_path)

        if is_valid_nifti(export_path):
            return k, export_path, None
        else:
            print(
                f"⚠️ Error processing volume {k}: output is not valid nifti file.",
                flush=True,
            )
            row["error"] = "output not valid nifti"
            return k, None, row

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"⚠️ Error processing volume {k}: {error_msg}", flush=True)
        row["error"] = error_msg
        return k, None, row


# Function to convert DICOM to NIfTI in parallel
def convert_dicom_to_nifti_parallel(df, output_dir, print_flag, num_workers):
    """
    Convert multiple DICOM volumes to NIfTI in parallel using multiprocessing.

    Args:
        df (pd.DataFrame): DataFrame containing DICOM metadata.
        output_dir (str): Directory to save the NIfTI files.
        print_flag (bool): Whether to display progress using `tqdm`.
        num_workers (int): Number of parallel processes to use.

    Returns:
        tuple: The updated DataFrame with NIfTI paths and a DataFrame with any errors encountered.
    """
    n_samples = len(df)
    df_err = pd.DataFrame()

    print(f"{n_samples} volumes to convert", flush=True)

    df["volume_ordinal_in_series"] = df.groupby("series_id").cumcount() + 1

    if "series_dir" not in df.columns:
        df["series_dir"] = df["dicom_path"].apply(
            lambda x: Path(x[0]).parent if isinstance(x, list) else Path(x).parent
        )

    # Use ProcessPoolExecutor to parallelize the task
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_single_volume, k, df.iloc[k], output_dir, print_flag
            )
            for k in range(n_samples)
        ]

        # Prepare iterator with optional progress bar
        iterator = as_completed(futures)
        if print_flag:
            iterator = tqdm(as_completed(futures), total=n_samples)

        # Collect results
        for future in iterator:
            try:
                k, export_path, error_row = future.result(
                    timeout=600
                )  # wait 10 mins max
            except Exception as e:
                print(f"⚠️ Future failed to execute under 10mins: {e}")
                k, export_path, error_row = None, None, None

            if export_path is not None:
                df.iloc[k, df.columns.get_loc("nifti_path")] = str(export_path)
            elif error_row is not None:
                # append error row to df_err as a row
                try:
                    df_err = pd.concat(
                        [df_err, error_row.to_frame().T], ignore_index=True
                    )
                except Exception:
                    # fallback: create a dataframe from dict
                    df_err = pd.concat(
                        [df_err, pd.DataFrame([error_row])], ignore_index=True
                    )

    df = df[df["nifti_path"].notna()]

    return df, df_err


# Main function
def main(args):
    """
    Main function to convert DICOM series to NIfTI files in parallel and save the results to CSV.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # If verbose, check input csv files
    if args.verbose:
        for p in args.csv_path:
            check_file(p)

    # Load manifest if provided
    manifest = None
    if hasattr(args, "manifest") and args.manifest:
        manifest = load_manifest(
            args.manifest, base_path=Path(__file__).resolve().parents[1]
        )

    # Load data (support multiple CSV paths)
    df_list = [pd.read_csv(p) for p in args.csv_path]
    df = pd.concat(df_list, ignore_index=True)

    # Columns placeholder
    df["nifti_path"] = None

    # Convert any list-like strings to actual lists
    df = df.map(lambda x: convert_list_str_to_list(x) if isinstance(x, str) else x)

    print("Before conversion:")
    report_volumes(df)
    df_prev = df.copy()

    if args.dry_run:
        print("Dry run: convert")
        print_args(args)
        return

    # Convert DICOM to NIfTI in parallel
    df, df_err = convert_dicom_to_nifti_parallel(
        df, args.output_dir, args.verbose, args.num_workers
    )

    print("After conversion:")
    report_volumes(df)
    report_change(df, df_prev)

    # Save results
    df.to_csv(args.csv_path_out, index=False)  # Save the updated CSV with NIfTI paths
    if not df_err.empty:
        print(
            f"⚠️ DICOM to Nifti Conversion Errors on patients : {df_err.patient_key.unique()}"
        )
        report_volumes(df_err)
        df_err.to_csv(args.error_csv_path, index=False)


if __name__ == "__main__":
    args = parse_arguments()
    if args.dry_run:
        print("Dry run: convert")
        print_args(args)
        raise SystemExit(0)
    main(args)
