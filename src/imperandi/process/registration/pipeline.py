"""Cohort orchestration; backend and consensus functions are replaceable interfaces."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
import uuid

import numpy as np
import pandas as pd

from .artifacts import write_artifact
from .config import REGISTRATION_STAGES, RegistrationConfig
from .consensus import fuse_tumors
from .completeness import (
    assess_organ_mask,
    coverage_image,
    overlap_qc,
    registration_confidence,
)
from .grouping import (
    normalize_modalities,
    prepare_cohort,
    reference_rank,
)
from .labels import group_label, scan_label
from .organ import backend, dice, read_image, read_mask, register_pair, resample
from .qc import (
    ERROR_COLUMNS,
    build_error_record,
    publish_group_log,
    record_stages,
    record_organ_qc,
)

logger = logging.getLogger(__name__)


def _missing_file(path):
    """Return whether a CSV path value is blank or does not name a file."""
    try:
        missing = pd.isna(path) or not str(path).strip()
    except (TypeError, ValueError):
        return True
    return missing or not Path(str(path)).expanduser().is_file()


def _registration_report(result):
    """Return serializable stage diagnostics from a registration result."""
    return {
        key: value
        for key, value in vars(result).items()
        if key not in {"reference_to_scan", "scan_to_reference", "stage_transforms"}
    }


def _record_consensus_inputs(df, indices, contributors, config):
    """Record group counts and per-series reasons, including no-reference groups."""
    for i in indices:
        if pd.isna(df.at[i, "tumor_consensus_input_status"]):
            df.at[i, "tumor_consensus_input_status"] = "excluded_" + str(
                df.at[i, "registration_status"]
            )
    statuses = df.loc[indices, "tumor_consensus_input_status"]
    reasons = statuses[statuses != "contributed"].value_counts().to_dict()
    df.loc[indices, "registration_consensus_contributors"] = len(contributors)
    df.loc[indices, "registration_consensus_excluded"] = len(indices) - len(
        contributors
    )
    df.loc[indices, "registration_consensus_exclusion_reasons"] = json.dumps(
        reasons, sort_keys=True
    )
    df.loc[indices, "registration_consensus_support_policy"] = (
        "common_fov_only" if config.method == "staple" else "per_voxel"
    )


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
        loaded, mask_qc = {}, {}
        for i in order:
            organ_path = df.at[i, config.organ_column]
            if _missing_file(organ_path):
                df.at[i, "registration_status"] = "skipped"
                df.at[i, "registration_skip_reason"] = "missing_organ_mask"
                df.at[i, "consensus_status"] = "skipped"
                logger.warning(
                    "Registration series skipped: %s, reason=missing organ mask",
                    df.at[i, "registration_scan_label"],
                )
                continue
            try:
                image = read_image(df.at[i, "nifti_path"])
                organ = read_mask(organ_path, image)
                quality = assess_organ_mask(organ, config)
                mask_qc[i] = quality
                record_organ_qc(df, i, quality)
                foreground = np.count_nonzero(sitk.GetArrayViewFromImage(organ))
                if foreground == 0:
                    df.at[i, "registration_status"] = "skipped"
                    df.at[i, "registration_skip_reason"] = "empty_organ_mask"
                    df.at[i, "consensus_status"] = "skipped"
                    logger.warning(
                        "Registration series skipped: %s, reason=empty organ mask",
                        df.at[i, "registration_scan_label"],
                    )
                    continue
                if quality.status == "invalid" or (
                    quality.partial and not config.allow_partial_organs
                ):
                    df.at[i, "registration_status"] = "invalid_organ_mask"
                    df.at[i, "registration_confidence"] = "invalid_organ_mask"
                    df.at[i, "registration_skip_reason"] = (
                        ",".join(quality.reasons) or "partial_organs_disabled"
                    )
                    error(
                        i, "organ_qc", ValueError(df.at[i, "registration_skip_reason"])
                    )
                    continue
                loaded[i] = image, organ
            except (RuntimeError, ValueError, TypeError) as exc:
                df.at[i, "registration_status"] = "failed"
                df.at[i, "registration_confidence"] = "failed"
                error(i, "input", exc)
        active_order = sorted(
            loaded,
            key=lambda i: (mask_qc[i].partial, reference_rank(df.loc[i], priorities)),
        )
        loaded = {i: loaded[i] for i in active_order}
        df.loc[group.index, "registration_series_number"] = None
        df.loc[group.index, "registration_group_size"] = None
        for position, i in enumerate(active_order, start=1):
            df.at[i, "registration_series_number"] = position
            df.at[i, "registration_group_size"] = len(active_order)
        for i in group.index:
            df.at[i, "registration_scan_label"] = scan_label(df.loc[i])
        if not loaded:
            logger.warning("Registration reference unavailable: no valid organ masks")
            unset = df.loc[group.index, "consensus_status"].isna()
            df.loc[unset.index[unset], "consensus_status"] = "no_reference"
            df.loc[group.index, "registration_elapsed_seconds"] = (
                time.perf_counter() - group_started
            )
            _record_consensus_inputs(df, group.index, [], config)
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
                metrics = overlap_qc(reference_organ, organ, tx)
                confidence = registration_confidence(
                    metrics, mask_qc[ref].partial or mask_qc[i].partial, config
                )
                if i != ref and getattr(result, "confidence", "ok") in {
                    "failed",
                    "low_confidence",
                }:
                    confidence = result.confidence
                df.at[i, "registration_confidence"] = confidence
                df.at[i, "registration_organ_volume_ratio"] = (
                    mask_qc[ref].volume_mm3 / mask_qc[i].volume_mm3
                )
                for key in ("dice_full", "dice_common_fov", "common_fov_fraction"):
                    df.at[i, f"registration_{key}"] = metrics[key]
                if confidence in {"failed", "low_confidence"}:
                    df.at[i, "registration_status"] = confidence
                    continue
                inverse = (
                    getattr(result, "scan_to_reference", None) if i != ref else None
                )
                inverse = inverse or tx.GetInverse()
                pair_dir = directory / ids[i]
                native_organ = resample(reference_organ, image, inverse)
                if mask_qc[ref].partial:
                    # Reference has no annotation outside its FOV: retain the
                    # scan's own organ there instead of erasing unobserved tissue.
                    known = resample(coverage_image(reference), image, inverse)
                    native_organ = sitk.Or(
                        native_organ, sitk.And(organ, sitk.Equal(known, 0))
                    )
                df.at[i, "reg_organ_native_path"] = write_artifact(
                    native_organ,
                    pair_dir / "organ_native.nii.gz",
                )
                df.at[i, config.organ_column] = df.at[i, "reg_organ_native_path"]
                transforms[i] = tx
                inverse_transforms[i] = inverse
                native_organs[i] = native_organ
                df.at[i, "registration_status"] = (
                    "reference" if i == ref else confidence
                )
            except (RuntimeError, ValueError) as exc:
                if hasattr(exc, "result"):
                    pair_results[i] = exc.result
                    report = _registration_report(exc.result)
                    record_stages(df, i, report)
                df.at[i, "registration_status"] = "failed"
                df.at[i, "registration_confidence"] = "failed"
                error(i, "organ", exc)
            finally:
                df.at[i, "registration_elapsed_seconds"] = time.perf_counter() - started
        if ref not in transforms:
            df.loc[group.index, "consensus_status"] = "no_reference"
            df.loc[group.index, "registration_elapsed_seconds"] = (
                time.perf_counter() - group_started
            )
            _record_consensus_inputs(df, group.index, [], config)
            publish_group_log(df, group.index, errors, config, directory)
            continue
        tumors = {}
        # Read all available inputs so tumor Dice can be reported independently
        # from the selected consensus policy.
        for i in [ref] + [i for i in order if i != ref]:
            if i not in transforms:
                df.at[i, "tumor_consensus_input_status"] = "excluded_" + str(
                    df.at[i, "registration_status"]
                )
                continue
            path = df.at[i, config.tumor_column] if config.tumor_column in df else None
            if _missing_file(path):
                df.at[i, "tumor_consensus_input_status"] = "skipped_missing"
                continue
            try:
                tumor = read_mask(path, loaded[i][0])
                if not np.count_nonzero(sitk.GetArrayViewFromImage(tumor)):
                    df.at[i, "tumor_consensus_input_status"] = "skipped_empty"
                    continue
                tumors[i] = tumor
                df.at[i, "tumor_consensus_input_status"] = "contributed"
            except (RuntimeError, ValueError) as exc:
                df.at[i, "tumor_consensus_input_status"] = "failed"
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
        for i in tumors:
            if i not in contributors:
                df.at[i, "tumor_consensus_input_status"] = "excluded_anchor_policy"
        _record_consensus_inputs(df, group.index, contributors, config)
        masks, coverages = [], []
        for i in contributors:
            tumor = tumors[i]
            coverage = coverage_image(tumor)
            masks.append(resample(tumor, reference, transforms[i]))
            coverages.append(resample(coverage, reference, transforms[i]))
        failed = df.loc[group.index, "registration_status"].isin(
            ["failed", "low_confidence", "invalid_organ_mask"]
        )
        df.loc[failed.index[failed], "consensus_status"] = "registration_failed"
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
                            native = resample(tumors[contributors[0]], image, composed)
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
                df.loc[list(transforms), "consensus_status"] = "failed"
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
