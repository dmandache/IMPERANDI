from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from imperandi.utils.logging import setup_logging
from imperandi.utils.misc import print_args
from imperandi.utils.manifest import load_manifest
from imperandi.utils.checkpoint_cli import add_checkpoint_arguments
from imperandi.utils.run_state import (
    atomic_write_csv,
    CheckpointManager,
    merge_with_existing_output,
    prepare_resume_context,
)

logger = logging.getLogger(__name__)
DEFAULT_CHECKPOINT_EVERY_ROWS = 50
DEFAULT_CHECKPOINT_EVERY_SEC = 350

DEFAULT_SETTINGS = {
    "binWidth": 25,
    "resampledPixelSpacing": [1, 1, 1], # 1mm isotropic resampling
    "resegmentRange": [-150, 250],  # typical HU range for soft tissue
    "correctMask": True, # enable mask correction to ensure valid feature extraction, especially for shape features
}
YAML_SUFFIXES = (".yaml", ".yml")


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


def _create_radiomics_extractors(
    featureextractor_module,
    settings: Optional[Dict[str, Any]] = None,
    *,
    settings_path: Optional[str] = None,
    settings_dict: Optional[Dict[str, Any]] = None,
):
    mode_count = int(settings is not None) + int(settings_path is not None) + int(
        settings_dict is not None
    )
    if mode_count != 1:
        raise ValueError(
            "Exactly one settings source must be provided "
            "(settings kwargs, settings_path, or settings_dict)."
        )

    def _new_extractor():
        if settings_dict is not None:
            return featureextractor_module.RadiomicsFeatureExtractor(settings_dict)
        if settings_path is not None:
            return featureextractor_module.RadiomicsFeatureExtractor(settings_path)
        return featureextractor_module.RadiomicsFeatureExtractor(**settings)

    extractor_all = _new_extractor()

    extractor_shape = _new_extractor()
    extractor_shape.disableAllFeatures()
    extractor_shape.enableFeatureClassByName("shape")

    extractor_non_shape = _new_extractor()
    extractor_non_shape.disableAllFeatures()
    for feature_class_name in extractor_non_shape.featureClassNames:
        if str(feature_class_name).lower() == "shape":
            continue
        extractor_non_shape.enableFeatureClassByName(feature_class_name)

    return {
        "all": extractor_all,
        "shape": extractor_shape,
        "non_shape": extractor_non_shape,
    }


def _configure_pyradiomics_output(*, enabled: bool, verbose: bool = False) -> None:
    pyradiomics_logger = logging.getLogger("radiomics")
    if enabled:
        pyradiomics_logger.disabled = False
        pyradiomics_logger.propagate = True
        pyradiomics_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        logger.info(
            "PyRadiomics output enabled (level=%s)",
            "DEBUG" if verbose else "INFO",
        )
    else:
        pyradiomics_logger.setLevel(logging.CRITICAL + 1)
        pyradiomics_logger.propagate = False
        pyradiomics_logger.disabled = True
        logger.info("PyRadiomics output disabled")

    try:
        import radiomics as pyradiomics
    except ModuleNotFoundError:
        return

    if hasattr(pyradiomics, "setVerbosity"):
        if enabled and verbose:
            verbosity_level = logging.DEBUG
        elif enabled:
            verbosity_level = logging.INFO
        else:
            verbosity_level = logging.CRITICAL
        pyradiomics.setVerbosity(verbosity_level)


def _execute_extractor(
    extractor: Any,
    image: Any,
    mask: Any,
    *,
    organ: str,
    extractor_name: str,
    row_idx: Optional[int] = None,
) -> Dict[str, Any]:
    row_label = row_idx if row_idx is not None else "-"
    logger.debug(
        "Executing PyRadiomics extractor | row=%s | organ=%s | extractor=%s",
        row_label,
        organ,
        extractor_name,
    )
    return extractor.execute(image, mask)


def _normalize_pyradiomics_settings_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    path = Path(raw)
    if path.suffix.lower() not in YAML_SUFFIXES:
        raise ValueError(
            "Expected a YAML file for --pyradiomics_settings "
            f"(accepted: {', '.join(YAML_SUFFIXES)}): {path}"
        )
    if not path.exists():
        raise FileNotFoundError(f"PyRadiomics settings YAML file not found: {path}")
    if not path.is_file():
        raise ValueError(f"PyRadiomics settings path is not a file: {path}")
    return str(path.resolve())


def _load_manifest_radiomics_settings(
    manifest_arg: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not manifest_arg:
        return None
    manifest = load_manifest(
        manifest_arg,
        base_path=Path(__file__).resolve().parents[1],
    )
    radiomics_settings = manifest.get("radiomics")
    if radiomics_settings is None:
        return None
    if not isinstance(radiomics_settings, dict):
        raise ValueError(
            "Manifest radiomics settings must be an object under key 'radiomics'."
        )
    return radiomics_settings


def _resolve_pyradiomics_settings_source(
    args: argparse.Namespace,
) -> tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    manifest_arg = getattr(args, "manifest", None)
    cli_settings_path = _normalize_pyradiomics_settings_path(
        getattr(args, "pyradiomics_settings", None)
    )
    manifest_settings: Optional[Dict[str, Any]] = None

    if manifest_arg and cli_settings_path:
        logger.warning(
            "Both --manifest and --pyradiomics_settings were provided; "
            "checking manifest radiomics settings first."
        )
    if manifest_arg:
        manifest_settings = _load_manifest_radiomics_settings(manifest_arg)

    if manifest_settings is not None:
        if cli_settings_path:
            logger.warning(
                "Manifest contains a 'radiomics' settings section; "
                "preferring manifest settings over --pyradiomics_settings."
            )
        logger.info("PyRadiomics settings source: manifest")
        return "manifest", None, manifest_settings

    if cli_settings_path:
        logger.info("PyRadiomics settings source: cli_file (%s)", cli_settings_path)
        return "cli_file", cli_settings_path, None

    logger.info("PyRadiomics settings source: defaults")
    return "defaults", None, None


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
        "csv_path_out_pos",
        nargs="?",
        type=str,
        default=None,
        help="Optional output CSV path (positional alternative to --csv_path_out).",
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
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Dataset manifest name or path to manifest JSON.",
    )
    parser.add_argument(
        "--pyradiomics_settings",
        type=str,
        default=None,
        help="Path to a PyRadiomics YAML settings file.",
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

    csv_path_out_pos = getattr(args, "csv_path_out_pos", None)
    csv_out = args.csv_path_out if args.csv_path_out else csv_path_out_pos
    if csv_out:
        args.csv_path_out = str(Path(csv_out))
    else:
        args.csv_path_out = str(csv_path.parent / f"{csv_path.stem}_radiomics.csv")

    if args.error_csv_path:
        args.error_csv_path = str(Path(args.error_csv_path))
    else:
        args.error_csv_path = str(csv_path.parent / "radiomics_errors.csv")

    args.pyradiomics_settings = _normalize_pyradiomics_settings_path(
        getattr(args, "pyradiomics_settings", None)
    )
    if getattr(args, "manifest", None) is not None:
        args.manifest = str(args.manifest)

    del args.csv_path_pos
    del args.csv_path_opt
    if hasattr(args, "csv_path_out_pos"):
        del args.csv_path_out_pos
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
    return bool(_array_sum(sitk_module.GetArrayViewFromImage(mask)) > 0)


def _array_sum(values: Any) -> float:
    if hasattr(values, "sum"):
        return float(values.sum())
    try:
        return float(sum(values))
    except TypeError:
        return float(values)


def _is_existing_path(value: Any) -> bool:
    """Return True only for non-empty string paths that exist on disk."""
    if not isinstance(value, str):
        return False
    path = value.strip()
    return bool(path) and Path(path).exists()


def _project_radiomics_features(result: Dict[str, Any], *, prefix: str) -> Dict[str, Any]:
    features: Dict[str, Any] = {}
    for key, value in result.items():
        skey = str(key)
        if skey.startswith("diagnostics_"):
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
    extractors,
    sitk_module,
    row_idx: Optional[int] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    if not _is_existing_path(mask_path):
        return {}, f"{prefix} mask path is missing: {mask_path}"

    try:
        mask_image = sitk_module.ReadImage(mask_path)
        if not mask_has_voxels(mask_image, sitk_module):
            return {}, f"{prefix} mask is empty: {mask_path}"

        image = sitk_module.ReadImage(image_path)
        result = _execute_extractor(
            extractors["all"],
            image,
            mask_image,
            organ=prefix,
            extractor_name="all",
            row_idx=row_idx,
        )
        features = _project_radiomics_features(result, prefix=prefix)
        return features, None
    except Exception as exc:
        return {}, f"Error extracting {prefix} features (extractor=all): {exc}"


def extract_radiomics_organ_minus_tumor(
    image_path: str,
    organ_mask_path: Optional[str],
    tumor_mask_path: Optional[str],
    *,
    extractors,
    sitk_module,
    prefix: str = "liver",
    row_idx: Optional[int] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    if not _is_existing_path(organ_mask_path):
        return {}, f"missing {prefix} mask"

    try:
        img = sitk_module.ReadImage(image_path)
        organ = sitk_module.ReadImage(organ_mask_path)

        if _array_sum(sitk_module.GetArrayViewFromImage(organ)) == 0:
            return {}, f"empty {prefix} mask"

        organ_bin = sitk_module.Cast(
            sitk_module.NotEqual(organ, 0), sitk_module.sitkUInt8
        )
        has_tumor = _is_existing_path(tumor_mask_path)

        if not has_tumor:
            result = _execute_extractor(
                extractors["all"],
                img,
                organ_bin,
                organ=prefix,
                extractor_name="all",
                row_idx=row_idx,
            )
            return _project_radiomics_features(result, prefix=prefix), None

        tumor = sitk_module.ReadImage(tumor_mask_path)
        if _array_sum(sitk_module.GetArrayViewFromImage(tumor)) == 0:
            result = _execute_extractor(
                extractors["all"],
                img,
                organ_bin,
                organ=prefix,
                extractor_name="all",
                row_idx=row_idx,
            )
            return _project_radiomics_features(result, prefix=prefix), None

        tumor = _resample_to_reference_if_needed(tumor, organ, sitk_module)
        tumor_bin = sitk_module.Cast(sitk_module.NotEqual(tumor, 0), sitk_module.sitkUInt8)
        organ_minus_tumor = sitk_module.And(
            organ_bin,
            sitk_module.Cast(sitk_module.Not(tumor_bin), sitk_module.sitkUInt8),
        )

        shape_result = _execute_extractor(
            extractors["shape"],
            img,
            organ_bin,
            organ=prefix,
            extractor_name="shape",
            row_idx=row_idx,
        )
        features = _project_radiomics_features(shape_result, prefix=prefix)

        if _array_sum(sitk_module.GetArrayViewFromImage(organ_minus_tumor)) == 0:
            return features, f"{prefix}_minus_tumor mask is empty"

        non_shape_result = _execute_extractor(
            extractors["non_shape"],
            img,
            organ_minus_tumor,
            organ=prefix,
            extractor_name="non_shape",
            row_idx=row_idx,
        )
        features.update(_project_radiomics_features(non_shape_result, prefix=prefix))
        return features, None
    except Exception as exc:
        return {}, f"Error extracting {prefix}_minus_tumor: {exc}"


def _get_mask_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("mask_")]


def _build_dataset_strategy(mask_columns: list[str]) -> list[str]:
    strategy: list[str] = []
    mask_columns_sorted = sorted(set(mask_columns))
    mask_columns_set = set(mask_columns_sorted)

    for mask_col in mask_columns_sorted:
        prefix = mask_col.replace("mask_", "", 1)
        if prefix.endswith("_tumor"):
            strategy.append(f"{prefix}: all on {mask_col}")
            continue

        tumor_col = f"{mask_col}_tumor"
        if tumor_col in mask_columns_set:
            strategy.append(
                f"{prefix}: shape on {mask_col}; non_shape on {prefix}_minus_tumor; "
                f"fallback all on {mask_col} if {tumor_col} missing/empty"
            )
        else:
            strategy.append(f"{prefix}: all on {mask_col} (no paired tumor mask column)")

    return strategy


def _log_dataset_strategy(mask_columns: list[str]) -> None:
    strategy = _build_dataset_strategy(mask_columns)
    if not strategy:
        logger.info("Radiomics strategy (dataset): no mask_* columns found")
        return

    logger.info(
        "Radiomics strategy (dataset): %d ROI plans detected",
        len(strategy),
    )
    for line in strategy:
        logger.info("Radiomics strategy (dataset) | %s", line)


def _extract_row_features(
    row: pd.Series,
    mask_columns: list[str],
    *,
    extractors,
    sitk_module,
    row_idx: Optional[int] = None,
) -> Tuple[Dict[str, Any], list[str]]:
    image_path = row.get("nifti_path")
    features: Dict[str, Any] = {}
    messages: list[str] = []

    if not isinstance(image_path, str) or not Path(image_path).exists():
        row_label = row_idx if row_idx is not None else "-"
        logger.warning(
            "Radiomics issue | row=%s | organ=all | CT image path is missing or invalid: %s",
            row_label,
            image_path,
        )
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
                extractors=extractors,
                sitk_module=sitk_module,
                row_idx=row_idx,
            )
        else:
            tumor_col = f"{mask_col}_tumor"
            tumor_path = row.get(tumor_col) if tumor_col in mask_columns_set else None
            roi_features, roi_msg = extract_radiomics_organ_minus_tumor(
                image_path,
                mask_path,
                tumor_path,
                extractors=extractors,
                sitk_module=sitk_module,
                prefix=prefix,
                row_idx=row_idx,
            )

        features.update(roi_features)
        if roi_msg:
            row_label = row_idx if row_idx is not None else "-"
            logger.warning(
                "Radiomics issue | row=%s | organ=%s | %s",
                row_label,
                prefix,
                roi_msg,
            )
            messages.append(roi_msg)
        else:
            row_label = row_idx if row_idx is not None else "-"
            logger.debug(
                "Radiomics features extracted | row=%s | organ=%s | feature_count=%d",
                row_label,
                prefix,
                len(roi_features),
            )

    return features, messages


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
    _configure_pyradiomics_output(
        enabled=bool(getattr(args, "verbose", False)),
        verbose=bool(getattr(args, "verbose", False)),
    )
    source_kind, settings_path, settings_dict = _resolve_pyradiomics_settings_source(
        args
    )
    if source_kind == "manifest":
        extractors = _create_radiomics_extractors(
            featureextractor_module,
            settings_dict=settings_dict,
        )
    elif source_kind == "cli_file":
        extractors = _create_radiomics_extractors(
            featureextractor_module,
            settings_path=settings_path,
        )
    else:
        extractors = _create_radiomics_extractors(
            featureextractor_module,
            DEFAULT_SETTINGS,
        )

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

    logger.info("Extracting radiomics from %d rows and ROIs: %s", len(df), mask_columns)
    _log_dataset_strategy(mask_columns)
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

    row_indices = [
        idx
        for idx in df.index.tolist()
        if int(df.at[idx, "_source_idx"]) not in completed_indices
    ]
    for idx in tqdm(row_indices, total=len(row_indices), desc="Radiomics", unit="row"):
        src_idx = int(df.at[idx, "_source_idx"])
        logger.debug("Processing radiomics row=%s", src_idx)
        row = df.loc[idx]

        features, messages = _extract_row_features(
            row,
            mask_columns,
            extractors=extractors,
            sitk_module=sitk_module,
            row_idx=src_idx,
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
    df_features = merge_with_existing_output(
        df_features,
        args.csv_path_out,
        preferred_keys=["nifti_path", "_source_idx"],
        strict=True,
    )
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
