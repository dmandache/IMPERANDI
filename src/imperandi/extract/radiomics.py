from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from imperandi.utils.misc import print_args

logger = logging.getLogger(__name__)

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
    print(f"Running {Path(__file__).name} with args: {args}")
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
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    sitk_module, featureextractor_module = _load_radiomics_dependencies()
    extractor = _create_radiomics_extractor(featureextractor_module, DEFAULT_SETTINGS)

    df = pd.read_csv(args.csv_path).copy()
    if "nifti_path" not in df.columns:
        raise KeyError("column 'nifti_path' missing")
    if not args.skip_filter:
        df = filter_df(df)

    logger.info("Extracting radiomics from %d rows", len(df))
    df_features, df_err = extract_radiomics_from_dataframe(
        df,
        extractor=extractor,
        sitk_module=sitk_module,
        verbose=args.verbose,
    )

    df_features.to_csv(args.csv_path_out, index=False)
    logger.info("Wrote main table -> %s", args.csv_path_out)

    if not df_err.empty:
        df_err.to_csv(args.error_csv_path, index=False)
        logger.warning("%d rows failed -> %s", len(df_err), args.error_csv_path)


if __name__ == "__main__":
    args = parse_arguments()
    if getattr(args, "dry_run", False):
        print("Dry run: radiomics")
        print_args(args)
        raise SystemExit(0)
    main(args)
