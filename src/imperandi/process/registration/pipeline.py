"""Cohort orchestration; backend and consensus functions are replaceable interfaces."""

from dataclasses import asdict
from datetime import datetime, timezone
import logging
from pathlib import Path
import time
import uuid

import numpy as np
import pandas as pd

from imperandi.utils.run_state import atomic_write_json
from .artifacts import write_artifact
from .config import RegistrationConfig
from .consensus import fuse_tumors
from .grouping import (
    normalize_modalities,
    prepare_cohort,
    reference_rank,
    stable_id,
)
from .labels import group_context, group_label
from .organ import backend, read_image, read_mask, register_pair, resample
from .qc import STAGES, publish_group_qc, record_stages

logger = logging.getLogger(__name__)


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
        context = group_context(df.loc[index], config.visit_column)
        logger.error(
            "Registration failed: scan={%s}, stage=%s, error_type=%s; "
            "see the error table for details",
            df.at[index, "registration_scan_label"],
            stage,
            type(exc).__name__,
        )
        errors.append(
            {
                **context,
                "registration_group_label": df.at[index, "registration_group_label"],
                "registration_scan_label": df.at[index, "registration_scan_label"],
                "registration_scan_id": ids[index],
                "stage": stage,
                "error": str(exc),
            }
        )

    groups = df.groupby(
        [df.patient_key, df[config.visit_column], modalities], sort=True
    )
    for group_key, group in groups:
        group_id = stable_id(tuple(str(value) for value in group_key))
        label = group_label(group.iloc[0], config.visit_column)
        logger.info("Starting registration group: %s, series=%d", label, len(group))
        directory = root / group_id
        group_started = time.perf_counter()
        df.loc[group.index, "registration_started_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        df.loc[group.index, "registration_group_id"] = group_id
        priorities = config.reference_priority.get(group_key[2], [])
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
            logger.warning("No valid registration reference")
            df.loc[group.index, "consensus_status"] = "no_reference"
            df.loc[group.index, "registration_elapsed_seconds"] = (
                time.perf_counter() - group_started
            )
            publish_group_qc(df, group.index, errors, config, directory)
            continue
        ref = next(iter(loaded))
        reference, reference_organ = loaded[ref]
        df.loc[group.index, "registration_reference_id"] = ids[ref]
        df.loc[group.index, "registration_reference_label"] = df.at[
            ref, "registration_scan_label"
        ]
        logger.info(
            "Selected registration reference: %s",
            df.at[ref, "registration_scan_label"],
        )
        transforms = {}
        pair_reports = []
        for i, (image, organ) in loaded.items():
            started = time.perf_counter()
            logger.info(
                "Starting organ alignment: %s",
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
                            for stage in STAGES
                        },
                    }
                else:
                    result = pair_registration(reference_organ, organ, config)
                    tx = result.reference_to_scan
                    report = {
                        k: v
                        for k, v in vars(result).items()
                        if k != "reference_to_scan"
                    }
                record_stages(df, i, report)
                inverse = tx.GetInverse()
                pair_dir = directory / ids[i]
                df.at[i, "reg_reference_to_scan_path"] = write_artifact(
                    tx, pair_dir / "reference_to_scan.tfm", transform=True
                )
                df.at[i, "reg_scan_to_reference_path"] = write_artifact(
                    inverse, pair_dir / "scan_to_reference.tfm", transform=True
                )
                df.at[i, "reg_nifti_path"] = write_artifact(
                    resample(image, reference, tx, label=False),
                    pair_dir / "image_registered.nii.gz",
                )
                df.at[i, "reg_organ_path"] = write_artifact(
                    resample(organ, reference, tx), pair_dir / "organ_registered.nii.gz"
                )
                df.at[i, "reg_organ_native_path"] = write_artifact(
                    resample(reference_organ, image, inverse),
                    pair_dir / "organ_native.nii.gz",
                )
                df.at[i, config.organ_column] = df.at[i, "reg_organ_native_path"]
                transforms[i] = tx
                df.at[i, "registration_status"] = "reference" if i == ref else "ok"
                pair_reports.append(
                    {
                        "scan_id": ids[i],
                        "scan": df.at[i, "registration_scan_label"],
                        **report,
                    }
                )
            except (RuntimeError, ValueError) as exc:
                if hasattr(exc, "result"):
                    report = {
                        k: v
                        for k, v in vars(exc.result).items()
                        if k != "reference_to_scan"
                    }
                    record_stages(df, i, report)
                    pair_reports.append(
                        {
                            "scan_id": ids[i],
                            "scan": df.at[i, "registration_scan_label"],
                            "rejected": True,
                            **report,
                        }
                    )
                df.at[i, "registration_status"] = "failed"
                error(i, "organ", exc)
            finally:
                df.at[i, "registration_elapsed_seconds"] = time.perf_counter() - started
        if ref not in transforms:
            df.loc[group.index, "consensus_status"] = "no_reference"
            df.loc[group.index, "registration_elapsed_seconds"] = (
                time.perf_counter() - group_started
            )
            publish_group_qc(df, group.index, errors, config, directory)
            continue
        masks, coverages, contributors, native_masks = [], [], [], {}
        # Reference first, then availability in configured priority order.
        for i in [ref] + [i for i in order if i != ref]:
            if i not in transforms:
                continue
            path = df.at[i, config.tumor_column] if config.tumor_column in df else None
            if pd.isna(path) or not str(path).strip():
                continue
            try:
                tumor = read_mask(path, loaded[i][0])
                coverage = sitk.Image(tumor.GetSize(), sitk.sitkUInt8) + 1
                coverage.CopyInformation(tumor)
                masks.append(resample(tumor, reference, transforms[i]))
                coverages.append(resample(coverage, reference, transforms[i]))
                contributors.append(i)
                native_masks[i] = tumor
                if config.method == "anchor":
                    break
            except (RuntimeError, ValueError) as exc:
                error(i, "tumor_input", exc)
        df.loc[group.index, "consensus_status"] = "registration_failed"
        df.loc[list(transforms), "consensus_status"] = "no_tumor_input"
        if masks:
            contributor_labels = [
                df.at[i, "registration_scan_label"] for i in contributors
            ]
            logger.info(
                "Starting tumor consensus: method=%s, contributors=%d [%s]",
                config.method,
                len(contributors),
                " | ".join(contributor_labels),
            )
            try:
                fused = fusion(
                    masks, coverages, method=config.method, threshold=config.threshold
                )
                common = write_artifact(fused.mask, directory / "tumor_common.nii.gz")
                support = write_artifact(
                    fused.coverage, directory / "coverage_common.nii.gz"
                )
                probability = (
                    write_artifact(
                        fused.probability, directory / "probability_common.nii.gz"
                    )
                    if fused.probability is not None
                    else None
                )
                for i, tx in transforms.items():
                    try:
                        inverse = tx.GetInverse()
                        image = loaded[i][0]
                        pair_dir = directory / ids[i]
                        if config.method == "anchor":
                            composed = sitk.CompositeTransform(3)
                            composed.AddTransform(transforms[contributors[0]])
                            composed.AddTransform(inverse)
                            native = resample(
                                native_masks[contributors[0]], image, composed
                            )
                        else:
                            native = resample(fused.mask, image, inverse)
                        native_coverage = resample(fused.coverage, image, inverse)
                        native = sitk.And(native, native_coverage)
                        df.at[i, "reg_tumor_native_path"] = write_artifact(
                            native, pair_dir / "tumor_native.nii.gz"
                        )
                        df.at[i, "reg_tumor_coverage_native_path"] = write_artifact(
                            native_coverage,
                            pair_dir / "coverage_native.nii.gz",
                        )
                        if probability:
                            df.at[
                                i,
                                "reg_tumor_probability_native_path",
                            ] = write_artifact(
                                resample(
                                    fused.probability, image, inverse, label=False
                                ),
                                pair_dir / "probability_native.nii.gz",
                            )
                        df.at[i, "reg_tumor_common_path"] = common
                        df.at[i, "reg_tumor_coverage_common_path"] = support
                        df.at[i, "reg_tumor_probability_common_path"] = probability
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
            logger.warning("Tumor consensus skipped: no tumor masks are available")
        report_path = directory / "group.json"
        atomic_write_json(
            report_path,
            {
                "group": [str(value) for value in group_key],
                "group_context": group_context(group.iloc[0], config.visit_column),
                "group_label": label,
                "reference": ids[ref],
                "reference_label": df.at[ref, "registration_scan_label"],
                "config": asdict(config),
                "contributors": [ids[i] for i in contributors],
                "pairs": pair_reports,
                "simpleitk_version": sitk.Version_VersionString(),
                "errors": [
                    e
                    for e in errors
                    if e["registration_scan_id"] in set(group.registration_scan_id)
                ],
            },
        )
        df.loc[group.index, "registration_report_path"] = str(report_path.resolve())
        publish_group_qc(df, group.index, errors, config, directory)
        logger.info(
            "Finished registration group: registration_status=%s, consensus_status=%s",
            df.loc[group.index, "registration_status"].tolist(),
            df.loc[group.index, "consensus_status"].tolist(),
        )
    return df, pd.DataFrame(
        errors,
        columns=[
            "patient_id",
            "date",
            "visit_order",
            "visit",
            "modality",
            "registration_group_label",
            "registration_scan_label",
            "registration_scan_id",
            "stage",
            "error",
        ],
    )
