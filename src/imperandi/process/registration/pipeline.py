"""Cohort orchestration; backend and consensus functions are replaceable interfaces."""

from datetime import datetime, timezone
import logging
from pathlib import Path
import time
import uuid

import numpy as np
import pandas as pd

from .artifacts import write_artifact
from .config import REGISTRATION_STAGES, RegistrationConfig
from .consensus import fuse_tumors
from .grouping import (
    normalize_modalities,
    prepare_cohort,
    reference_rank,
)
from .labels import group_label
from .organ import backend, dice, read_image, read_mask, register_pair, resample
from .qc import ERROR_COLUMNS, build_error_record, publish_group_log, record_stages

logger = logging.getLogger(__name__)


def _registration_report(result):
    """Return serializable stage diagnostics from a registration result."""
    return {
        key: value
        for key, value in vars(result).items()
        if key not in {"reference_to_scan", "scan_to_reference", "stage_transforms"}
    }


def register_cohort(
    table,
    output_dir,
    config=None,
    *,
    pair_registration=register_pair,
    fusion=fuse_tumors,
):
    """Return (cohort, errors), grouping strictly by patient, visit and modality.

    This image-processing API performs fresh work. The CLI runner adds group
    scheduling, checkpointing, and resume around this function.
    pair_registration and fusion may be injected by alternative implementations.
    """
    config = config or RegistrationConfig()
    sitk = backend()
    df = prepare_cohort(table, config)
    ids = df.registration_scan_id.tolist()
    modalities = normalize_modalities(df)
    root = Path(output_dir).expanduser().resolve() / uuid.uuid4().hex
    root.mkdir(parents=True)
    errors = []

    def error(index, stage, exc):
        logger.error(
            "Registration series failed: %s, stage=%s, error_type=%s; "
            "see the error table for details",
            df.at[index, "registration_scan_label"],
            stage,
            type(exc).__name__,
        )
        errors.append(build_error_record(df.loc[index], config, stage=stage, error=exc))

    groups = df.groupby(
        [df.patient_key, df[config.visit_column], modalities], sort=True
    )
    for _, group in groups:
        group_id = group.iloc[0].registration_group_id
        label = group_label(group.iloc[0], config.visit_column)
        logger.info("Registration group started: %s, series=%d", label, len(group))
        directory = root / group_id
        group_started = time.perf_counter()
        df.loc[group.index, "registration_started_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        df.loc[group.index, "registration_group_id"] = group_id
        modality = modalities.loc[group.index[0]]
        priorities = config.reference_priority.get(modality, [])
        order = sorted(
            group.index, key=lambda index: reference_rank(df.loc[index], priorities)
        )
        loaded = {}
        for i in order:
            try:
                image = read_image(df.at[i, "nifti_path"])
                organ = read_mask(df.at[i, config.organ_column], image)
                if np.count_nonzero(sitk.GetArrayViewFromImage(organ)) < 4:
                    raise ValueError("Empty or insufficient organ mask")
                loaded[i] = image, organ
            except (RuntimeError, ValueError, TypeError) as exc:
                df.at[i, "registration_status"] = "failed"
                error(i, "input", exc)
        if not loaded:
            logger.warning("Registration reference unavailable: no valid organ masks")
            df.loc[group.index, "consensus_status"] = "no_reference"
            df.loc[group.index, "registration_elapsed_seconds"] = (
                time.perf_counter() - group_started
            )
            publish_group_log(df, group.index, errors, config, directory)
            continue
        ref = next(iter(loaded))
        reference, reference_organ = loaded[ref]
        df.loc[group.index, "registration_reference_id"] = ids[ref]
        df.loc[group.index, "registration_reference_label"] = df.at[
            ref, "registration_scan_label"
        ]
        logger.info(
            "Registration reference selected: %s",
            df.at[ref, "registration_scan_label"],
        )
        transforms, inverse_transforms, pair_results, native_organs = (
            {},
            {},
            {},
            {},
        )
        for i, (image, organ) in loaded.items():
            started = time.perf_counter()
            logger.info(
                "Registration organ alignment started: %s",
                df.at[i, "registration_scan_label"],
            )
            try:
                if i == ref:
                    tx = sitk.Euler3DTransform()
                    report = {
                        "stage": "reference",
                        "dice_before": 1.0,
                        "dice_after": 1.0,
                        "warnings": [],
                        "stages": {
                            stage: {
                                "dice": 1.0 if stage == "baseline" else None,
                                "status": (
                                    "evaluated" if stage == "baseline" else "not_run"
                                ),
                            }
                            for stage in REGISTRATION_STAGES
                        },
                    }
                else:
                    result = pair_registration(reference_organ, organ, config)
                    tx = result.reference_to_scan
                    pair_results[i] = result
                    report = _registration_report(result)
                record_stages(df, i, report)
                inverse = (
                    getattr(result, "scan_to_reference", None) if i != ref else None
                )
                inverse = inverse or tx.GetInverse()
                pair_dir = directory / ids[i]
                native_organ = resample(reference_organ, image, inverse)
                df.at[i, "reg_organ_native_path"] = write_artifact(
                    native_organ,
                    pair_dir / "organ_native.nii.gz",
                )
                df.at[i, config.organ_column] = df.at[i, "reg_organ_native_path"]
                transforms[i] = tx
                inverse_transforms[i] = inverse
                native_organs[i] = native_organ
                df.at[i, "registration_status"] = "reference" if i == ref else "ok"
            except (RuntimeError, ValueError) as exc:
                if hasattr(exc, "result"):
                    pair_results[i] = exc.result
                    report = _registration_report(exc.result)
                    record_stages(df, i, report)
                df.at[i, "registration_status"] = "failed"
                error(i, "organ", exc)
            finally:
                df.at[i, "registration_elapsed_seconds"] = time.perf_counter() - started
        if ref not in transforms:
            df.loc[group.index, "consensus_status"] = "no_reference"
            df.loc[group.index, "registration_elapsed_seconds"] = (
                time.perf_counter() - group_started
            )
            publish_group_log(df, group.index, errors, config, directory)
            continue
        tumors = {}
        # Read all available inputs so tumor Dice can be reported independently
        # from the selected consensus policy.
        for i in [ref] + [i for i in order if i != ref]:
            if i not in transforms:
                continue
            path = df.at[i, config.tumor_column] if config.tumor_column in df else None
            if pd.isna(path) or not str(path).strip():
                continue
            try:
                tumors[i] = read_mask(path, loaded[i][0])
            except (RuntimeError, ValueError) as exc:
                error(i, "tumor_input", exc)

        # Tumor overlap is diagnostic only and never affects transform selection.
        if ref in tumors:
            df.at[ref, "registration_tumor_dice_baseline"] = 1.0
            for i, tumor in tumors.items():
                if i == ref:
                    continue
                for stage, tx in getattr(
                    pair_results.get(i), "stage_transforms", {}
                ).items():
                    try:
                        df.at[i, f"registration_tumor_dice_{stage}"] = dice(
                            tumors[ref], tumor, tx
                        )
                    except (RuntimeError, ValueError) as exc:
                        logger.warning(
                            "Registration tumor QC unavailable: %s, stage=%s, "
                            "error_type=%s",
                            df.at[i, "registration_scan_label"],
                            stage,
                            type(exc).__name__,
                        )

        contributors = list(tumors)
        if config.method == "anchor":
            contributors = contributors[:1]
        masks, coverages = [], []
        for i in contributors:
            tumor = tumors[i]
            coverage = sitk.Image(tumor.GetSize(), sitk.sitkUInt8) + 1
            coverage.CopyInformation(tumor)
            masks.append(resample(tumor, reference, transforms[i]))
            coverages.append(resample(coverage, reference, transforms[i]))
        df.loc[group.index, "consensus_status"] = "registration_failed"
        df.loc[list(transforms), "consensus_status"] = "no_tumor_input"
        if masks:
            contributor_labels = [
                df.at[i, "registration_scan_label"] for i in contributors
            ]
            logger.info(
                "Registration tumor consensus started: method=%s, contributors=%d [%s]",
                config.method,
                len(contributors),
                " | ".join(contributor_labels),
            )
            try:
                fused = fusion(
                    masks, coverages, method=config.method, threshold=config.threshold
                )
                for i, tx in transforms.items():
                    try:
                        inverse = inverse_transforms[i]
                        image = loaded[i][0]
                        pair_dir = directory / ids[i]
                        if config.method == "anchor":
                            composed = sitk.CompositeTransform(3)
                            composed.AddTransform(transforms[contributors[0]])
                            composed.AddTransform(inverse)
                            native = resample(
                                tumors[contributors[0]], image, composed
                            )
                        else:
                            native = resample(fused.mask, image, inverse)
                        native_coverage = resample(fused.coverage, image, inverse)
                        native = sitk.And(native, native_coverage)
                        if config.constrain_tumor_to_organ:
                            native = sitk.And(native, native_organs[i])
                        df.at[i, "reg_tumor_native_path"] = write_artifact(
                            native, pair_dir / "tumor_native.nii.gz"
                        )
                        df.at[i, config.tumor_column] = df.at[
                            i, "reg_tumor_native_path"
                        ]
                        df.at[i, "consensus_status"] = (
                            "single_contributor" if len(masks) == 1 else "ok"
                        )
                    except (RuntimeError, ValueError) as exc:
                        df.at[i, "consensus_status"] = "failed"
                        error(i, "transfer", exc)
            except (RuntimeError, ValueError) as exc:
                df.loc[group.index, "consensus_status"] = "failed"
                error(ref, "consensus", exc)
        else:
            logger.warning(
                "Registration tumor consensus skipped: no tumor masks are available"
            )
        publish_group_log(df, group.index, errors, config, directory)
        registration_counts = (
            df.loc[group.index, "registration_status"].value_counts().to_dict()
        )
        consensus_counts = (
            df.loc[group.index, "consensus_status"].value_counts().to_dict()
        )
        logger.info(
            "Registration group completed: registration_status_counts=%s, "
            "consensus_status_counts=%s",
            registration_counts,
            consensus_counts,
        )
    return df, pd.DataFrame(
        errors,
        columns=ERROR_COLUMNS,
    )
