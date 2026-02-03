import warnings
import glob
from pathlib import Path
import argparse
from multiprocessing import cpu_count

import pandas as pd
from tqdm import tqdm
from pandarallel import pandarallel
from pydicom import dcmread
from pydicom.errors import InvalidDicomError

warnings.filterwarnings("ignore")


# -------------------------
# CLI
# -------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process DICOM files: read header tags once, then compute patient/study/series IDs from tags or path."
    )
    parser.add_argument(
        "--root_path",
        type=str,
        required=True,
        help="Root path where the DICOM files are located.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output CSV files.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Dataset manifest name or path to manifest JSON.",
    )

    # Tag reading
    parser.add_argument(
        "--flatten_all_tags",
        action="store_true",
        default=False,
        help="If set, flattens (recursive) all DICOM header tags into columns (can be huge).",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default="",
        help="Comma-separated list of DICOM keyword tags to read (e.g. PatientID,StudyInstanceUID,SeriesInstanceUID,Modality). "
             "If empty and --flatten_all_tags is not set, only the ID tags are read.",
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
        "--num_workers",
        default=cpu_count(),
        type=int,
        help="Number of parallel workers (default: all CPUs).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=False, help="Verbose mode."
    )

    args = parser.parse_args()
    print(f"Running {Path(__file__).name} with args: {args}")
    return args


from imperandi.utils.manifest import load_manifest, resolve_hook


def apply_id_standardization(df: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    hook = resolve_hook(manifest.get("id_standardization") or {})
    if "patient_key" not in df.columns:
        return df

    # save raw once
    if "patient_key_raw" not in df.columns:
        df["patient_key_raw"] = df["patient_key"]

    if not hook:
        return df

    df["patient_key"] = df["patient_key_raw"].apply(hook)

    raw_ok = df["patient_key_raw"].notna() & (df["patient_key_raw"].astype(str).str.strip() != "")
    std_bad = df["patient_key"].isna() | (df["patient_key"].astype(str).str.strip() == "")
    failed = raw_ok & std_bad

    if failed.any():
        df["patient_key_std_failed"] = failed
        n_keys = int(df.loc[failed, "patient_key_raw"].nunique())
        print(f"[id_standardization] failed on unique raw keys={n_keys}")

    return df


def apply_derived_columns(df: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    derived_columns = manifest.get("derived_columns", [])
    if not derived_columns:
        return df

    for derived in derived_columns:
        from_column = derived.get("from_column")
        if not from_column or from_column not in df.columns:
            continue
        hook = resolve_hook(derived)
        if not hook:
            continue
        derived_values = df[from_column].apply(hook)
        derived_df = derived_values.apply(pd.Series)
        if derived_df.empty:
            continue
        join_mode = derived.get("join_mode", "missing_only")
        if join_mode == "overwrite":
            df = df.drop(columns=[col for col in derived_df.columns if col in df.columns])
            df = df.join(derived_df)
        else:
            derived_df = derived_df.loc[:, ~derived_df.columns.isin(df.columns)]
            if not derived_df.empty:
                df = df.join(derived_df)
    return df


# -------------------------
# IO helpers
# -------------------------
def ensure_directory_exists(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)


def get_dicom_paths(root_path):
    """
    Strategy:
    1) Try *.dcm
    2) If none, scan all files and attempt dcmread header
    """
    root_path = Path(root_path)
    dicom_paths = [p for p in root_path.rglob("*.dcm") if p.is_file()]
    if dicom_paths:
        return dicom_paths

    dicom_paths = []
    for p in root_path.rglob("*"):
        if not p.is_file():
            continue
        try:
            dcmread(p, stop_before_pixels=True, force=True)
            dicom_paths.append(p)
        except (InvalidDicomError, PermissionError, OSError):
            continue
    return dicom_paths

# -------------------------
# DICOM tag extraction
# -------------------------
def extract_dicom_tags_recursive(ds, parent_key=""):
    tags = {}
    for elem in ds:
        key = f"{parent_key}_{elem.keyword}" if parent_key else elem.keyword
        if elem.VR == "SQ":
            for i, item in enumerate(elem.value):
                tags.update(extract_dicom_tags_recursive(item, f"{key}[{i}]"))
        else:
            tags[key] = elem.value if elem.value else None
    return tags


def read_dicom_header(fp, *, specific_tags=None, force=False, flatten_all=False):
    """
    Read header once:
      - flatten_all=True  => recursive flatten all tags into columns
      - flatten_all=False => only read specific_tags (fast) and return those columns
    Returns pd.Series
    """
    try:
        ds = dcmread(
            fp,
            stop_before_pixels=True,
            force=force,
            specific_tags=specific_tags if (specific_tags and not flatten_all) else None,
        )

        if flatten_all:
            tags = extract_dicom_tags_recursive(ds)
            return pd.Series(tags)

        out = {}
        for t in (specific_tags or []):
            out[t] = getattr(ds, t, None)
        return pd.Series(out)

    except Exception:
        if flatten_all:
            return pd.Series({})
        return pd.Series({t: None for t in (specific_tags or [])})


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
    rel = df["dicom_path"].map(lambda p: Path(p).relative_to(root_path))
    df["patient_key_path"] = rel.map(lambda p: p.parts[0] if len(p.parts) > 0 else None)
    df["dicom_filename"]   = rel.map(lambda p: p.name)

    # -------------------------
    # Tag-derived IDs
    # -------------------------
    def _tagcol(tag: str) -> pd.Series:
        if tag in df.columns:
            return df[tag].map(
                lambda v: None if v is None or str(v).strip() == "" else str(v).strip()
            )
        return pd.Series([None] * len(df), index=df.index)

    patient_key_tags = _tagcol(patient_tag)
    study_id_tags    = _tagcol(study_tag)
    series_id_tags   = _tagcol(series_tag)

    # -------------------------
    # Choose source
    # -------------------------
    if id_source == "path":
        df["patient_key"] = df["patient_key_path"]
        df["study_id"]    = "0"
        df["series_id"]   = "0"

    elif id_source == "tags":
        df["patient_key"] = patient_key_tags
        df["study_id"]    = study_id_tags.fillna("0")
        df["series_id"]   = series_id_tags.fillna("0")

    else:  # auto
        df["patient_key"] = patient_key_tags.fillna(df["patient_key_path"])
        df["study_id"]    = study_id_tags.fillna("0")
        df["series_id"]   = series_id_tags.fillna("0")

    return df



# -------------------------
# Checkpointed processing (for tag read stage)
# -------------------------
def process_with_checkpoint(
    df_paths: pd.DataFrame,
    read_func,
    checkpoint_frequency: int | None,
    output_dir: Path,
    final_name: str,
):
    """
    Apply read_func(dicom_path)->Series to df_paths with optional chunked checkpoint.
    """
    if checkpoint_frequency is None:
        tags_df = df_paths["dicom_path"].parallel_apply(read_func)
        tags_df = tags_df.replace("", float("NaN")).dropna(how="all", axis=1)
        out = pd.concat([df_paths, tags_df], axis=1)
        out.to_csv(output_dir / final_name, index=False)
        return out

    # chunked
    total_rows = df_paths.shape[0]
    for i in tqdm(range(0, total_rows, checkpoint_frequency)):
        chunk_idx = i // checkpoint_frequency
        ckpt = output_dir / f"{Path(final_name).stem}_{chunk_idx:03d}.csv"
        if ckpt.exists():
            continue
        chunk = df_paths.iloc[i:i + checkpoint_frequency].copy()
        tags_chunk = chunk["dicom_path"].parallel_apply(read_func)
        tags_chunk = tags_chunk.replace("", float("NaN")).dropna(how="all", axis=1)
        out_chunk = pd.concat([chunk, tags_chunk], axis=1)
        out_chunk.to_csv(ckpt, index=False)

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
    root_path = Path(args.root_path)
    output_dir = Path(args.output_dir)
    ensure_directory_exists(output_dir)
    print(f"Output directory: {output_dir}")
    manifest = load_manifest(args.manifest, base_path=Path(__file__).resolve().parents[1])

    dicom_paths = get_dicom_paths(root_path)
    print(f"Found {len(dicom_paths)} DICOM files under {root_path}")

    # 1) base df with dicom_path only (no IDs computed yet)
    df = pd.DataFrame({"dicom_path": [str(p) for p in dicom_paths]})

    # 2) read tags once for all files
    #    - always include the ID tags if we might use them (tags/auto)
    user_tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    id_tags = [args.patient_key_from, args.study_id_from, args.series_id_from]
    tags_to_read = sorted(set(user_tags + (id_tags if not args.flatten_all_tags else [])))

    print(f"Reading DICOM tags: flatten_all={args.flatten_all_tags}, tags_to_read={tags_to_read}")

    read_func = lambda fp: read_dicom_header(
        fp,
        specific_tags=tags_to_read,
        force=args.force_dicom_read,
        flatten_all=args.flatten_all_tags,
    )

    df = process_with_checkpoint(
        df_paths=df,
        read_func=read_func,
        checkpoint_frequency=args.checkpoint_frequency,
        output_dir=output_dir,
        final_name="dicom_paths_with_tags.csv",
    )

    print(f"After tag extraction: {df.shape} columns={len(df.columns)}")

    # 3) compute IDs from tags/path using already-read tag columns
    df = choose_ids(
        df=df,
        root_path=root_path,
        id_source=args.id_source,
        patient_tag=args.patient_key_from,
        study_tag=args.study_id_from,
        series_tag=args.series_id_from,
    )
    df = apply_id_standardization(df, manifest)
    df = apply_derived_columns(df, manifest)

    # 4) output final df
    out_final = output_dir / "dicom_index.csv"
    df.to_csv(out_final, index=False)
    print(f"Saved final index: {out_final}")
    print("Done.")


if __name__ == "__main__":
    args = parse_arguments()
    pandarallel.initialize(progress_bar=args.verbose, nb_workers=args.num_workers)
    main(args)
