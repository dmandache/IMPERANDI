from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from imperandi.utils.logging import setup_logging
from imperandi.utils.misc import print_args
from imperandi.utils.checkpoint_cli import add_checkpoint_arguments
from imperandi.utils.run_state import (
    atomic_write_csv,
    CheckpointManager,
    prepare_resume_context,
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


def _extract_original_features(
    result: Dict[str, Any],
    *,
    prefix: str,
    include_shape: bool = True,
    include_non_shape: bool = True,
) -> Dict[str, Any]:
    features: Dict[str, Any] = {}
    for key, value in result.items():
        skey = str(key)
        if not skey.startswith("original"):
            continue
        is_shape = skey.startswith("original_shape_")
        if (is_shape and not include_shape) or ((not is_shape) and not include_non_shape):
            continue
        features[f"{prefix}_{skey}"] = value
    return features


def _resample_to_reference_if_needed(mask_image, reference_image, sitk_module):
    if (
        mask_image.GetSize(),
        mask_image.GetSpacing(),
        mask_image.GetOrigin(),
        mask_image.GetDirection(),
    ) == (
        reference_image.GetSize(),
        reference_image.GetSpacing(),
        reference_image.GetOrigin(),
        reference_image.GetDirection(),
    ):
        return mask_image

    rs = sitk_module.ResampleImageFilter()
    rs.SetReferenceImage(reference_image)
    rs.SetInterpolator(sitk_module.sitkNearestNeighbor)
    rs.SetDefaultPixelValue(0)
    return rs.Execute(mask_image)


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
        features = _extract_original_features(result, prefix=prefix)
        return features, None
    except Exception as exc:
        return {}, f"Error extracting {prefix} features: {exc}"


def extract_radiomics_organ_minus_tumor(
    image_path: str,
    liver_mask_path: Optional[str],
    tumor_mask_path: Optional[str],
    *,
    extractor,
    sitk_module,
    prefix: str = "liver",
) -> Tuple[Dict[str, Any], Optional[str]]:
    if not liver_mask_path or not Path(liver_mask_path).exists():
        return {}, f"missing {prefix} mask"

    try:
        img = sitk_module.ReadImage(image_path)
        organ = sitk_module.ReadImage(liver_mask_path)

        if sitk_module.GetArrayViewFromImage(organ).sum() == 0:
            return {}, f"empty {prefix} mask"

        organ_bin = sitk_module.Cast(
            sitk_module.NotEqual(organ, 0), sitk_module.sitkUInt8
        )
        has_tumor = bool(tumor_mask_path and Path(tumor_mask_path).exists())

        if not has_tumor:
            result = extractor.execute(img, organ_bin)
            return _extract_original_features(result, prefix=prefix), None

        tumor = sitk_module.ReadImage(tumor_mask_path)
        if sitk_module.GetArrayViewFromImage(tumor).sum() == 0:
            result = extractor.execute(img, organ_bin)
            return _extract_original_features(result, prefix=prefix), None

        tumor = _resample_to_reference_if_needed(tumor, organ, sitk_module)
        tumor_bin = sitk_module.Cast(sitk_module.NotEqual(tumor, 0), sitk_module.sitkUInt8)
        organ_minus_tumor = sitk_module.And(
            organ_bin,
            sitk_module.Cast(sitk_module.Not(tumor_bin), sitk_module.sitkUInt8),
        )

        shape_result = extractor.execute(img, organ_bin)
        features = _extract_original_features(
            shape_result,
            prefix=prefix,
            include_shape=True,
            include_non_shape=False,
        )

        if sitk_module.GetArrayViewFromImage(organ_minus_tumor).sum() == 0:
            return features, f"{prefix}_minus_tumor mask is empty"

        non_shape_result = extractor.execute(img, organ_minus_tumor)
        features.update(
            _extract_original_features(
                non_shape_result,
                prefix=prefix,
                include_shape=False,
                include_non_shape=True,
            )
        )
        return features, None
    except Exception as exc:
        return {}, f"Error extracting {prefix}_minus_tumor: {exc}"


def _get_mask_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("mask_")]


def _extract_row_features(
    row: pd.Series,
    mask_columns: list[str],
    *,
    extractor,
    sitk_module,
) -> Tuple[Dict[str, Any], list[str]]:
    image_path = row.get("nifti_path")
    features: Dict[str, Any] = {}
    messages: list[str] = []

    if not isinstance(image_path, str) or not Path(image_path).exists():
        return {}, [f"CT image path is missing or invalid: {image_path}"]

    mask_columns_set = set(mask_columns)
    for mask_col in mask_columns:
        prefix = mask_col.replace("mask_", "", 1)
        mask_path = row.get(mask_col)

        if prefix.endswith("_tumor"):
            roi_features, roi_msg = extract_radiomics_safe(
                image_path,
                mask_path,
                prefix,
                extractor=extractor,
                sitk_module=sitk_module,
            )
        else:
            tumor_col = f"{mask_col}_tumor"
            tumor_path = row.get(tumor_col) if tumor_col in mask_columns_set else None
            roi_features, roi_msg = extract_radiomics_organ_minus_tumor(
                image_path,
                mask_path,
                tumor_path,
                extractor=extractor,
                sitk_module=sitk_module,
                prefix=prefix,
            )

        features.update(roi_features)
        if roi_msg:
            messages.append(roi_msg)

    return features, messages


def extract_radiomics_from_dataframe(
    df: pd.DataFrame,
    *,
    extractor,
    sitk_module,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_features = []
    errors = []
    mask_columns = _get_mask_columns(df)

    iterator = df.iterrows()
    if verbose:
        iterator = tqdm(iterator, total=len(df), desc="Radiomics")

    for idx, row in iterator:
        features, messages = _extract_row_features(
            row,
            mask_columns,
            extractor=extractor,
            sitk_module=sitk_module,
        )

        if messages and "CT image path is missing or invalid" in messages[0]:
            error_row = row.to_dict()
            error_row["error_message"] = messages[0]
            errors.append(error_row)
            all_features.append({})
            continue

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
    output_path = Path(args.csv_path_out)
    error_path = Path(args.error_csv_path)
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
    resume_ctx = prepare_resume_context(
        args=args,
        command="radiomics",
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
            "Resume enabled and matching radiomics run already finished; skipping execution."
        )
        return

    sitk_module, featureextractor_module = _load_radiomics_dependencies()
    extractor = _create_radiomics_extractor(featureextractor_module, DEFAULT_SETTINGS)

    if can_resume and paths.main_checkpoint_path.exists():
        logger.info("Resuming radiomics from checkpoint: %s", paths.main_checkpoint_path)
        df = pd.read_csv(paths.main_checkpoint_path).copy()
    else:
        df = pd.read_csv(args.csv_path).copy()
        df["_source_idx"] = df.index.astype(int)
    if "_source_idx" not in df.columns:
        df["_source_idx"] = df.index.astype(int)

    if "nifti_path" not in df.columns:
        raise KeyError("column 'nifti_path' missing")
    if not args.skip_filter:
        df = filter_df(df)
    mask_columns = _get_mask_columns(df)

    logger.info("Extracting radiomics from %d rows", len(df))
    completed_indices: set[int] = set()
    if can_resume:
        completed_indices = {
            int(i)
            for i in (state or {}).get("completed_indices", [])
            if isinstance(i, int)
        }
    errors_by_idx: dict[int, dict[str, Any]] = {}
    if can_resume and paths.error_checkpoint_path.exists():
        err_ckpt = pd.read_csv(paths.error_checkpoint_path)
        for _, row in err_ckpt.iterrows():
            if "_source_idx" in row:
                try:
                    errors_by_idx[int(row["_source_idx"])] = row.to_dict()
                except Exception:
                    pass

    def _checkpoint_write(*, force: bool = False) -> None:
        err_df = (
            pd.DataFrame(list(errors_by_idx.values())) if errors_by_idx else pd.DataFrame()
        )
        ckpt.flush(
            main_df=df,
            error_df=err_df,
            completed_indices=completed_indices,
            force=force,
        )

    iterator = df.index.tolist()
    if args.verbose:
        iterator = tqdm(iterator, total=len(iterator), desc="Radiomics")

    for idx in iterator:
        src_idx = int(df.at[idx, "_source_idx"])
        if src_idx in completed_indices:
            continue
        row = df.loc[idx]

        features, messages = _extract_row_features(
            row,
            mask_columns,
            extractor=extractor,
            sitk_module=sitk_module,
        )

        if messages and "CT image path is missing or invalid" in messages[0]:
            error_row = row.to_dict()
            error_row["error_message"] = messages[0]
            errors_by_idx[src_idx] = error_row
            completed_indices.add(src_idx)
            ckpt.mark_processed()
            _checkpoint_write(force=False)
            continue

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
        ckpt.mark_processed()
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
    ckpt.finalize_state(completed_indices=completed_indices)

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
