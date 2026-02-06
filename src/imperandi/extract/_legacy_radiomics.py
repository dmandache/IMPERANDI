import logging
from pathlib import Path
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm
import numpy as np
import os
import argparse

# Radiomics settings
settings = {
    "binWidth": 25,
    "resampledPixelSpacing": [1, 1, 1],
    "resegmentRange": [-150, 250],
}

print(settings)

extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

logging.basicConfig(
    level=logging.ERROR, format="%(levelname)s: %(message)s", force=True
)


def parse_arguments():
    """
    Parse command-line arguments for segmenting liver from 3D NIfTI volumes using TotalSegmentatorV2.

    Returns:
        argparse.Namespace: Parsed arguments including input/output paths and verbose flag.
    """
    parser = argparse.ArgumentParser(
        description="Extract Radiomic Features using PyRadiomics"
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        help="Path to the input CSV file",
        default="/data/hdd/bdr220003/workspace_WP1/data_preparation/data_tables/CHC_RETRO_CT_Scans_Training_Data_with_Labels_diag.csv",
    )

    parser.add_argument(
        "--csv_path_out",
        type=str,
        help="Path to save the final output CSV file",
        default="/data/hdd/bdr220003/workspace_WP1/data_preparation/data_tables/CHC_RETRO_CT_Scans_Training_Data_with_Labels_radiomics.csv",
    )

    args = parser.parse_args()

    logging.info(f"Running {Path(__file__).name} script with arguments: {args}")
    return args


def get_patients_with_complete_exams(df):
    # Drop duplicates to work with unique combinations
    unique_combos = (
        df[["patient_key", "followup_months", "phase"]].dropna().drop_duplicates()
    )

    # Get all possible combinations of followup_months and phase
    all_months = unique_combos["followup_months"].unique()
    all_phases = unique_combos["phase"].unique()
    all_combos = pd.MultiIndex.from_product(
        [all_months, all_phases], names=["followup_months", "phase"]
    )

    # Count combinations per patient
    combo_counts = (
        unique_combos.groupby("patient_key")
        .apply(lambda g: pd.MultiIndex.from_frame(g[["followup_months", "phase"]]))
        .apply(set)
        .reset_index(name="combos")
    )

    # Expected set of all combos
    full_set = set(all_combos)

    # Check if each patient has full set
    combo_counts["has_all_combinations"] = combo_counts["combos"].apply(
        lambda s: s == full_set
    )

    patients_complete_exams = combo_counts[
        combo_counts.has_all_combinations
    ].patient_key.unique()

    return patients_complete_exams


def filter_df(df):
    # Make a working copy
    df = df.copy()

    # 1. Filter followup months
    if "followup_months" in df.columns:
        df = df[df["followup_months"].isin([0, 3])]

    # 2. Filter acquisition phase
    if "phase" in df.columns:
        df = df[df["phase"].isin(["arteriel", "portal"])]

    # 3. Filter known progression labels
    if "accord_progression_6mois" in df.columns:
        df = df[df["accord_progression_6mois"].isin(["Non", "Oui"])]

    if "6m_global_progresssion" in df.columns:
        df = df[df["6m_global_progresssion"].isin(["NP", "P"])]

    elif "progression_group_bin" in df.columns:
        df = df[~df["progression_group_bin"].isna()]

    # 4. Select lowest noise scan per (patient, timepoint, phase)
    if all(
        col in df.columns
        for col in ["patient_key", "date", "phase", "liver_gaussian_noise"]
    ):
        df = df.loc[
            df.groupby(["patient_key", "date", "phase"])[
                "liver_gaussian_noise"
            ].idxmin()
        ].reset_index(drop=True)

    # 5. Keep only patients with complete exams (via helper function)
    if (
        "patient_key" in df.columns
        and "followup_months" in df.columns
        and "phase" in df.columns
    ):
        patients_complete_exams = get_patients_with_complete_exams(df)
        df = df[df["patient_key"].isin(patients_complete_exams)]

    return df


def mask_has_voxels(mask):
    """Check if the mask has any non-zero voxels."""
    return sitk.GetArrayViewFromImage(mask).sum() > 0


def extract_radiomics_safe(image_path, mask_path, prefix):
    """Safely extract radiomics features with a prefix."""
    features = {}
    if not mask_path or not os.path.exists(mask_path):
        logging.warning(f"{prefix} mask path is missing: {mask_path}")
        return features
    try:
        mask_image = sitk.ReadImage(mask_path)
        if not mask_has_voxels(mask_image):
            logging.info(f"{prefix} mask is empty (no voxels): {mask_path}")
            return features
        image = sitk.ReadImage(image_path)
        result = extractor.execute(image, mask_image)
        features = {
            f"{prefix}_{k}": v for k, v in result.items() if k.startswith("original")
        }
    except Exception as e:
        logging.error(f"Error extracting {prefix} features: {e}")
    return features


def extract_radiomics_liver_minus_tumor(
    image_path, liver_mask_path, tumor_mask_path, prefix="liver"
):
    """Extract radiomics from (liver - tumor) without saving intermediate masks."""
    features = {}
    if not liver_mask_path or not os.path.exists(liver_mask_path):
        logging.warning("missing liver mask")
        return features
    try:
        img = sitk.ReadImage(image_path)
        liver = sitk.ReadImage(liver_mask_path)

        if sitk.GetArrayViewFromImage(liver).sum() == 0:
            logging.info("empty liver mask")
            return features

        # binarize liver
        liver_bin = sitk.Cast(sitk.NotEqual(liver, 0), sitk.sitkUInt8)

        # subtract tumor if present & non-empty
        if tumor_mask_path and os.path.exists(tumor_mask_path):
            tumor = sitk.ReadImage(tumor_mask_path)
            if sitk.GetArrayViewFromImage(tumor).sum() > 0:
                # quick geometry guard: resample tumor to liver if needed
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
                    rs = sitk.ResampleImageFilter()
                    rs.SetReferenceImage(liver)
                    rs.SetInterpolator(sitk.sitkNearestNeighbor)
                    rs.SetDefaultPixelValue(0)
                    tumor = rs.Execute(tumor)
                tumor_bin = sitk.Cast(sitk.NotEqual(tumor, 0), sitk.sitkUInt8)
                liver_bin = sitk.And(
                    liver_bin, sitk.Cast(sitk.Not(tumor_bin), sitk.sitkUInt8)
                )

        if sitk.GetArrayViewFromImage(liver_bin).sum() == 0:
            logging.info("liver_minus_tumor became empty; skipping")
            return features

        # run pyradiomics (label=1 ROI)
        result = extractor.execute(img, liver_bin)
        features = {
            f"{prefix}_{k}": v for k, v in result.items() if k.startswith("original")
        }
    except Exception as e:
        logging.error(f"Error extracting liver_minus_tumor: {e}")
    return features


def extract_radiomics_from_dataframe(df):
    """Extract radiomics from liver and tumor masks with robust error handling."""
    all_features = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_path = row.get("nifti_path")
        liver_mask_path = row.get("liver_path")
        tumor_mask_path = row.get("liver_tumor_path")

        if not image_path or not os.path.exists(image_path):
            logging.error(f"CT image path is missing or invalid: {image_path}")
            all_features.append({})
            continue

        features = {}
        # features.update(extract_radiomics_safe(image_path, liver_mask_path, "liver"))
        features.update(
            extract_radiomics_liver_minus_tumor(
                image_path, liver_mask_path, tumor_mask_path, "liver"
            )
        )
        features.update(extract_radiomics_safe(image_path, tumor_mask_path, "tumor"))
        all_features.append(features)

    # Merge with input DataFrame
    features_df = pd.DataFrame(all_features)
    return pd.concat(
        [df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1
    )


def main(args):
    df = pd.read_csv(args.csv_path)
    df = filter_df(df)

    logging.info(f"Extracting radiomics from {df.patient_key.nunique()} patients")
    df_features = extract_radiomics_from_dataframe(df)

    df_features.to_csv(args.csv_path_out)


# Entry point
if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    main(args)
