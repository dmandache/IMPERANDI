"""Cohort preparation, organ alignment, tumor fusion, and artifact publication."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import time
import uuid

import numpy as np
import pandas as pd

from .config import (
    CONSENSUS_METHODS,
    REGISTRATION_STAGES,
    SUPPORTED_MODALITIES,
    RegistrationConfig,
)
from .alignment import (
    assess_organ_mask,
    backend,
    coverage_image,
    dice,
    overlap_qc,
    read_image,
    read_mask,
    register_pair,
    registration_confidence,
    resample,
)
from .reporting import (
    ERROR_COLUMNS,
    QC_FIELDS,
    build_error_record,
    group_label,
    scan_label,
    publish_group_log,
    record_stages,
    record_organ_qc,
)

logger = logging.getLogger(__name__)


OBSOLETE_ARTIFACT_COLUMNS = (
    "registration_report_path",
    "reg_reference_to_scan_path",
    "reg_scan_to_reference_path",
    "reg_tumor_common_path",
    "reg_tumor_coverage_common_path",
    "reg_tumor_coverage_native_path",
    "reg_tumor_probability_common_path",
    "reg_tumor_probability_native_path",
    "reg_nifti_path",
    "reg_organ_path",
)


def stable_id(value) -> str:
    """Return a filesystem-safe, deterministic identifier for structured values."""
    payload = json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def normalize_label(value) -> str:
    return "" if pd.isna(value) else str(value).strip().upper()


def normalize_modalities(table: pd.DataFrame) -> pd.Series:
    """Normalize supported modality labels without changing source columns."""
    return table.Modality.map(normalize_label).replace({"MRI": "MR"})


def reference_rank(row, priorities) -> tuple[int, str]:
    """Rank a reference candidate using configured selectors and a stable tie-break."""
    rank = next(
        (
            index
            for index, selector in enumerate(priorities)
            if all(
                normalize_label(row.get(key)) == normalize_label(value)
                for key, value in selector.items()
            )
        ),
        len(priorities),
    )
    return rank, str(row["registration_scan_id"])


def prepare_cohort(table: pd.DataFrame, config) -> pd.DataFrame:
    """Validate identities and initialize registration output columns."""
    df = table.drop(columns=OBSOLETE_ARTIFACT_COLUMNS, errors="ignore").reset_index(
        drop=True
    )

    # Source columns remain authoritative across repeated registration runs.
    for column in (config.organ_column, config.tumor_column):
        source = f"source_{column}"
        if source not in df:
            if column == config.organ_column and column not in df:
                continue
            df[source] = df[column] if column in df else None
        df[column] = df[source].astype(object)

    required = [
        "patient_key",
        config.visit_column,
        "Modality",
        "nifti_path",
        config.organ_column,
    ]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing registration columns: {sorted(missing)}")
    for column in ["patient_key", config.visit_column, "Modality", "nifti_path"]:
        if df[column].isna().any() or df[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing registration identity: {column}")

    modalities = normalize_modalities(df)
    if not modalities.isin(SUPPORTED_MODALITIES).all():
        raise ValueError("Registration supports CT and MR only")

    identities = [
        [
            str(row.patient_key),
            str(row[config.visit_column]),
            modalities[index],
            str(Path(row.nifti_path).expanduser().resolve()),
        ]
        for index, row in df.iterrows()
    ]
    scan_ids = [stable_id(identity) for identity in identities]
    if len(scan_ids) != len(set(scan_ids)):
        raise ValueError("Duplicate scan identity in registration input")

    df["registration_scan_id"] = scan_ids
    df["registration_group_id"] = [
        stable_id(
            (str(row.patient_key), str(row[config.visit_column]), modalities[index])
        )
        for index, row in df.iterrows()
    ]
    df["registration_group_label"] = [
        group_label(row, config.visit_column) for _, row in df.iterrows()
    ]
    df["registration_series_number"] = None
    df["registration_group_size"] = None
    for _, group in df.groupby("registration_group_id", sort=False):
        modality = modalities.loc[group.index[0]]
        priorities = config.reference_priority.get(modality, [])
        ordered = sorted(
            group.index,
            key=lambda index: reference_rank(df.loc[index], priorities),
        )
        for position, index in enumerate(ordered, start=1):
            df.at[index, "registration_series_number"] = position
            df.at[index, "registration_group_size"] = len(group)
    df["registration_scan_label"] = [scan_label(row) for _, row in df.iterrows()]

    derived_columns = [
        "registration_qc_path",
        "registration_log_path",
        *QC_FIELDS,
        "registration_reference_id",
        "registration_reference_label",
        "registration_status",
        "consensus_status",
        "reg_tumor_native_path",
        "reg_organ_native_path",
    ]
    for column in derived_columns:
        df[column] = pd.Series([None] * len(df), dtype=object)

    return df


@dataclass
class ConsensusResult:
    mask: object
    coverage: object
    probability: object | None
    observation_count: object | None = None
    support_policy: str = "per_voxel"


def fuse_tumors(masks, coverages, *, method, threshold=0.5):
    """Fuse aligned masks with their spatial observations; anchor is first input.

    Voting uses strict > threshold; STAPLE uses >= threshold. Unknown spatial
    coverage is excluded from every estimator, and returned separately. STAPLE
    is restricted to the intersection of observed FOVs, as its backend cannot
    model missing observations. Native empty masks are filtered by the caller.
    """
    sitk = backend()
    if method not in CONSENSUS_METHODS:
        raise ValueError(f"Unknown consensus method: {method}")
    if not masks or len(masks) != len(coverages):
        raise ValueError("Consensus requires masks and matching coverage")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between zero and one")
    reference = masks[0]
    for img in [*masks, *coverages]:
        if (img.GetSize(), img.GetOrigin(), img.GetSpacing(), img.GetDirection()) != (
            reference.GetSize(),
            reference.GetOrigin(),
            reference.GetSpacing(),
            reference.GetDirection(),
        ):
            raise ValueError("Consensus inputs must share one grid")
    # Native empty-mask exclusion belongs to the pipeline. A nonempty native
    # mask can become empty on the reference grid and still supplies negative
    # observations inside its FOV; do not silently remove that contributor.
    if method == "anchor":
        masks, coverages = masks[:1], coverages[:1]
    values = np.stack([sitk.GetArrayFromImage(m) > 0 for m in masks])
    observations = np.stack([sitk.GetArrayFromImage(c) > 0 for c in coverages])
    counts = observations.sum(axis=0)
    votes = (values & observations).sum(axis=0)
    support = counts == len(masks) if method == "staple" else counts > 0
    if not support.any():
        raise ValueError("Tumor inputs have no observed support for this method")
    probability = np.divide(
        votes, counts, out=np.zeros(counts.shape, np.float32), where=counts > 0
    )
    if method == "staple" and len(masks) > 1 and values[:, support].any():
        # STAPLE's estimator is voxelwise: pack only observed samples so padding
        # cannot affect its estimated prior or rater performance.
        packed = [
            sitk.GetImageFromArray(v[support].astype(np.uint8)[None, :]) for v in values
        ]
        estimator = sitk.STAPLEImageFilter()
        estimator.SetForegroundValue(1)
        estimator.SetMaximumIterations(100)
        estimated = sitk.GetArrayFromImage(estimator.Execute(packed)).ravel()
        if not np.isfinite(estimated).all():
            raise ValueError("STAPLE produced nonfinite probabilities")
        probability = np.zeros(support.shape, np.float32)
        probability[support] = estimated
        binary = probability >= threshold
    elif method in {"majority", "staple"}:
        binary = (
            probability > threshold
            if method == "majority"
            else probability >= threshold
        )
    elif method == "intersection":
        binary = (votes == counts) & (counts > 0)
    else:
        binary = votes > 0

    def image(array, dtype):
        out = sitk.GetImageFromArray(array.astype(dtype))
        out.CopyInformation(reference)
        return out

    return ConsensusResult(
        image(binary & support, np.uint8),
        image(support, np.uint8),
        image(probability * support, np.float32),
        image(counts, np.uint32),
        "common_fov_only" if method == "staple" else "per_voxel",
    )


def write_artifact(image, path) -> str:
    """Write an image atomically and return its absolute path."""
    sitk = backend()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{uuid.uuid4().hex}.{destination.name}")
    try:
        sitk.WriteImage(image, str(temporary))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return str(destination.resolve())


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
                        ",".join(quality.reasons)
                        if quality.status == "invalid"
                        else "partial_organs_disabled"
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
