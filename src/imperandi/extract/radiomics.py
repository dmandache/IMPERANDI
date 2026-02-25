from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

DEFAULT_SETTINGS = {
    "binWidth": 25,
    "resampledPixelSpacing": [1, 1, 1],
    "resegmentRange": [-150, 250],
}


def _load_radiomics_dependencies():
    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The 'radiomics' command requires optional dependencies. "
            "Install with: pip install pyradiomics SimpleITK"
        ) from exc

    try:
        from radiomics import featureextractor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The 'radiomics' command requires optional dependencies. "
            "Install with: pip install pyradiomics"
        ) from exc

    return sitk, featureextractor


def _create_radiomics_extractor(featureextractor_module, settings: Dict[str, Any]):
    return featureextractor_module.RadiomicsFeatureExtractor(**settings)


def add_radiomics_arguments(
    parser: argparse.ArgumentParser,
    include_dry_run: bool = True,
) -> None:
    parser.add_argument(
        "csv_path_pos",
        nargs="?",
        type=str,
        default=None,
        help="Path to input CSV with nifti/mask paths. Defaults to ./nifti_index.csv.",
    )
    parser.add_argument(
        "--csv_path",
        dest="csv_path_opt",
        type=str,
    )
    parser.add_argument(
        "--csv_path_out",
        type=str,
        default=None,
        help=(
            "Path to save radiomics-enriched CSV. "
            "Defaults to <csv_dir>/<csv_stem>_radiomics.csv."
        ),
    )
    parser.add_argument(
        "--error_csv_path",
        type=str,
        default=None,
        help="Path to save rows with extraction errors (default: <csv_dir>/radiomics_errors.csv).",
    )
    parser.add_argument(
        "--skip_filter",
        action="store_true",
        default=False,
        help="Skip legacy cohort filtering and process all rows.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
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
        description="Extract PyRadiomics features from CT volumes and masks.",
        add_help=add_help,
    )
    add_radiomics_arguments(parser)
    return parser


def normalize_radiomics_args(args: argparse.Namespace) -> argparse.Namespace:
    csv_in = args.csv_path_opt if args.csv_path_opt is not None else args.csv_path_pos
    csv_path = Path(csv_in) if csv_in else (Path.cwd() / "nifti_index.csv")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Not a CSV file: {csv_path}")

    csv_path = csv_path.resolve()
    args.csv_path = str(csv_path)

    if args.csv_path_out:
        args.csv_path_out = str(Path(args.csv_path_out))
    else:
        args.csv_path_out = str(csv_path.parent / f"{csv_path.stem}_radiomics.csv")

    if args.error_csv_path:
        args.error_csv_path = str(Path(args.error_csv_path))
    else:
        args.error_csv_path = str(csv_path.parent / "radiomics_errors.csv")

    del args.csv_path_pos
    del args.csv_path_opt
    return args


def parse_arguments() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    args = normalize_radiomics_args(args)
    logger.info("🚀 Running %s with args: %s", Path(__file__).name, args)
    return args


def get_patients_with_complete_exams(df: pd.DataFrame):
    unique_combos = (
        df[["patient_key", "followup_months", "phase"]].dropna().drop_duplicates()
    )

    all_months = unique_combos["followup_months"].unique()
    all_phases = unique_combos["phase"].unique()
    all_combos = pd.MultiIndex.from_product(
        [all_months, all_phases], names=["followup_months", "phase"]
    )

    combo_counts = (
        unique_combos.groupby("patient_key")
        .apply(lambda g: pd.MultiIndex.from_frame(g[["followup_months", "phase"]]))
        .apply(set)
        .reset_index(name="combos")
    )

    full_set = set(all_combos)
    combo_counts["has_all_combinations"] = combo_counts["combos"].apply(
        lambda combos: combos == full_set
    )

    return combo_counts[combo_counts["has_all_combinations"]].patient_key.unique()


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "followup_months" in df.columns:
        df = df[df["followup_months"].isin([0, 3])]

    if "phase" in df.columns:
        df = df[df["phase"].isin(["arteriel", "portal"])]

    if "accord_progression_6mois" in df.columns:
        df = df[df["accord_progression_6mois"].isin(["Non", "Oui"])]

    if "6m_global_progresssion" in df.columns:
        df = df[df["6m_global_progresssion"].isin(["NP", "P"])]
    elif "progression_group_bin" in df.columns:
        df = df[~df["progression_group_bin"].isna()]

    if all(
        col in df.columns
        for col in ["patient_key", "date", "phase", "liver_gaussian_noise"]
    ):
        df = df.loc[
            df.groupby(["patient_key", "date", "phase"])[
                "liver_gaussian_noise"
            ].idxmin()
        ].reset_index(drop=True)

    if (
        "patient_key" in df.columns
        and "followup_months" in df.columns
        and "phase" in df.columns
    ):
        patients_complete_exams = get_patients_with_complete_exams(df)
        df = df[df["patient_key"].isin(patients_complete_exams)]

    return df


def mask_has_voxels(mask, sitk_module) -> bool:
    return bool(sitk_module.GetArrayViewFromImage(mask).sum() > 0)


def extract_radiomics_safe(
    image_path: str,
    mask_path: Optional[str],
    prefix: str,
    *,
    extractor,
    sitk_module,
) -> Tuple[Dict[str, Any], Optional[str]]:
    if not mask_path or not Path(mask_path).exists():
        return {}, f"{prefix} mask path is missing: {mask_path}"

    try:
        mask_image = sitk_module.ReadImage(mask_path)
        if not mask_has_voxels(mask_image, sitk_module):
            return {}, f"{prefix} mask is empty: {mask_path}"

        image = sitk_module.ReadImage(image_path)
        result = extractor.execute(image, mask_image)
        features = {
            f"{prefix}_{k}": v
            for k, v in result.items()
            if str(k).startswith("original")
        }
        return features, None
    except Exception as exc:
        return {}, f"Error extracting {prefix} features: {exc}"


def extract_radiomics_liver_minus_tumor(
    image_path: str,
    liver_mask_path: Optional[str],
    tumor_mask_path: Optional[str],
    *,
    extractor,
    sitk_module,
    prefix: str = "liver",
) -> Tuple[Dict[str, Any], Optional[str]]:
    if not liver_mask_path or not Path(liver_mask_path).exists():
        return {}, "missing liver mask"

    try:
        img = sitk_module.ReadImage(image_path)
        liver = sitk_module.ReadImage(liver_mask_path)

        if sitk_module.GetArrayViewFromImage(liver).sum() == 0:
            return {}, "empty liver mask"

        liver_bin = sitk_module.Cast(
            sitk_module.NotEqual(liver, 0), sitk_module.sitkUInt8
        )

        if tumor_mask_path and Path(tumor_mask_path).exists():
            tumor = sitk_module.ReadImage(tumor_mask_path)
            if sitk_module.GetArrayViewFromImage(tumor).sum() > 0:
                if (
                    tumor.GetSize(),
                    tumor.GetSpacing(),
                    tumor.GetOrigin(),
                    tumor.GetDirection(),
                ) != (
                    liver.GetSize(),
                    liver.GetSpacing(),
                    liver.GetOrigin(),
                    liver.GetDirection(),
                ):
                    rs = sitk_module.ResampleImageFilter()
                    rs.SetReferenceImage(liver)
                    rs.SetInterpolator(sitk_module.sitkNearestNeighbor)
                    rs.SetDefaultPixelValue(0)
                    tumor = rs.Execute(tumor)

                tumor_bin = sitk_module.Cast(
                    sitk_module.NotEqual(tumor, 0), sitk_module.sitkUInt8
                )
                liver_bin = sitk_module.And(
                    liver_bin,
                    sitk_module.Cast(sitk_module.Not(tumor_bin), sitk_module.sitkUInt8),
                )

        if sitk_module.GetArrayViewFromImage(liver_bin).sum() == 0:
            return {}, "liver_minus_tumor mask is empty"

        result = extractor.execute(img, liver_bin)
        features = {
            f"{prefix}_{k}": v
            for k, v in result.items()
            if str(k).startswith("original")
        }
        return features, None
    except Exception as exc:
        return {}, f"Error extracting liver_minus_tumor: {exc}"


def extract_radiomics_from_dataframe(
    df: pd.DataFrame,
    *,
    extractor,
    sitk_module,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_features = []
    errors = []

    iterator = df.iterrows()
    if verbose:
        iterator = tqdm(iterator, total=len(df), desc="Radiomics")

    for idx, row in iterator:
        image_path = row.get("nifti_path")
        liver_mask_path = row.get("mask_liver")  # row.get("liver_path")
        tumor_mask_path = row.get("mask_liver_tumor")  # row.get("liver_tumor_path")

        if not isinstance(image_path, str) or not Path(image_path).exists():
            error_row = row.to_dict()
            error_row["error_message"] = (
                f"CT image path is missing or invalid: {image_path}"
            )
            errors.append(error_row)
            all_features.append({})
            continue

        features = {}
        messages = []

        liver_features, liver_msg = extract_radiomics_liver_minus_tumor(
            image_path,
            liver_mask_path,
            tumor_mask_path,
            extractor=extractor,
            sitk_module=sitk_module,
            prefix="liver",
        )
        features.update(liver_features)
        if liver_msg:
            messages.append(liver_msg)

        tumor_features, tumor_msg = extract_radiomics_safe(
            image_path,
            tumor_mask_path,
            "tumor",
            extractor=extractor,
            sitk_module=sitk_module,
        )
        features.update(tumor_features)
        if tumor_msg:
            messages.append(tumor_msg)

        all_features.append(features)
        if not features:
            error_row = row.to_dict()
            error_row["error_message"] = (
                " | ".join(messages) if messages else "no features extracted"
            )
            errors.append(error_row)

    features_df = pd.DataFrame(all_features)
    df_features = pd.concat(
        [df.reset_index(drop=True), features_df.reset_index(drop=True)],
        axis=1,
    )
    return df_features, pd.DataFrame(errors)


def main(args: argparse.Namespace) -> None:
    sitk_module, featureextractor_module = _load_radiomics_dependencies()
    extractor = _create_radiomics_extractor(featureextractor_module, DEFAULT_SETTINGS)

    output_path = Path(args.csv_path_out)
    error_path = Path(args.error_csv_path)
    state_path = output_path.parent / f".{output_path.stem}.radiomics.state.json"
    checkpoint_main_path = (
        output_path.parent / f".{output_path.stem}.radiomics.checkpoint.csv"
    )
    checkpoint_err_path = error_path.parent / f".{error_path.stem}.radiomics.checkpoint.csv"
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
        state, command="radiomics", args_hash=args_hash, input_fingerprint=input_fp
    )

    if can_resume and checkpoint_main_path.exists():
        logger.info("Resuming radiomics from checkpoint: %s", checkpoint_main_path)
        df = pd.read_csv(checkpoint_main_path).copy()
    else:
        df = pd.read_csv(args.csv_path).copy()
        df["_source_idx"] = df.index.astype(int)
    if "_source_idx" not in df.columns:
        df["_source_idx"] = df.index.astype(int)

    if "nifti_path" not in df.columns:
        raise KeyError("column 'nifti_path' missing")
    if not args.skip_filter:
        df = filter_df(df)

    logger.info("Extracting radiomics from %d rows", len(df))
    completed_indices: set[int] = set()
    if can_resume:
        completed_indices = {
            int(i)
            for i in (state or {}).get("completed_indices", [])
            if isinstance(i, int)
        }
    errors_by_idx: dict[int, dict[str, Any]] = {}
    if can_resume and checkpoint_err_path.exists():
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
                "command": "radiomics",
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
        iterator = tqdm(iterator, total=len(iterator), desc="Radiomics")

    for idx in iterator:
        src_idx = int(df.at[idx, "_source_idx"])
        if src_idx in completed_indices:
            continue
        row = df.loc[idx]
        image_path = row.get("nifti_path")
        liver_mask_path = row.get("mask_liver")
        tumor_mask_path = row.get("mask_liver_tumor")

        if not isinstance(image_path, str) or not Path(image_path).exists():
            error_row = row.to_dict()
            error_row["error_message"] = (
                f"CT image path is missing or invalid: {image_path}"
            )
            errors_by_idx[src_idx] = error_row
            completed_indices.add(src_idx)
            processed_since_checkpoint += 1
            _checkpoint_write(force=False)
            continue

        features = {}
        messages = []
        liver_features, liver_msg = extract_radiomics_liver_minus_tumor(
            image_path,
            liver_mask_path,
            tumor_mask_path,
            extractor=extractor,
            sitk_module=sitk_module,
            prefix="liver",
        )
        features.update(liver_features)
        if liver_msg:
            messages.append(liver_msg)
        tumor_features, tumor_msg = extract_radiomics_safe(
            image_path,
            tumor_mask_path,
            "tumor",
            extractor=extractor,
            sitk_module=sitk_module,
        )
        features.update(tumor_features)
        if tumor_msg:
            messages.append(tumor_msg)

        for key, value in features.items():
            df.at[idx, key] = value
        if features:
            if src_idx in errors_by_idx:
                del errors_by_idx[src_idx]
        else:
            error_row = row.to_dict()
            error_row["error_message"] = (
                " | ".join(messages) if messages else "no features extracted"
            )
            errors_by_idx[src_idx] = error_row

        completed_indices.add(src_idx)
        processed_since_checkpoint += 1
        _checkpoint_write(force=False)

    _checkpoint_write(force=True)
    df_features = df.drop(columns=["_source_idx"], errors="ignore")
    atomic_write_csv(df_features, args.csv_path_out, index=False)
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
            "command": "radiomics",
            "args_hash": args_hash,
            "input_fingerprint": input_fp,
            "completed_indices": sorted(completed_indices),
            "updated_at_epoch": now_epoch(),
            "finished": True,
        },
    )

    logger.info("Radiomics extraction done ✔")


if __name__ == "__main__":
    setup_logging()
    args = parse_arguments()
    setup_logging(verbose=getattr(args, "verbose", False))
    if getattr(args, "dry_run", False):
        logger.info("Dry run: radiomics")
        print_args(args)
        raise SystemExit(0)
    main(args)
