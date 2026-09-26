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
    ORGAN_CONSENSUS_METHODS,
    REGISTRATION_STAGES,
    TUMOR_CONSENSUS_METHODS,
    RegistrationConfig,
)
from .alignment import (
    assess_organ_mask,
    backend,
    coverage_image,
    dice,
    expand_organ_distance,
    organ_distance_field,
    overlap_qc,
    read_image,
    read_mask,
    register_pair,
    registration_confidence,
    resample,
    resample_organ_distance,
    threshold_organ_distance,
)
from .reporting import (
    ERROR_COLUMNS,
    QC_FIELDS,
    build_error_record,
    group_label,
    scan_label,
    publish_group_log,
    record_final_organ_qc,
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


def identity_value(column, value) -> str:
    """Normalize values used in stable scan and group identities."""
    label = "" if pd.isna(value) else str(value).strip()
    if column == "Modality":
        label = normalize_label(value)
        return "MR" if label == "MRI" else label
    return label


def reference_rank(row, priorities) -> tuple[tuple, str]:
    """Apply ordered categorical/numeric criteria, then a deterministic scan ID.

    Missing/unlisted categories follow listed values. Numeric strings from CSVs
    are supported; missing, malformed, boolean, and nonfinite numbers rank last
    for both min and max. Each criterion breaks ties in the preceding criteria.
    """
    ranks = []
    for criterion in priorities:
        column, preference = next(iter(criterion.items()))
        value = row.get(column)
        if isinstance(preference, list):
            labels = [normalize_label(label) for label in preference]
            if column == "Modality":
                labels = ["MR" if label == "MRI" else label for label in labels]
            label = normalize_label(value)
            if column == "Modality" and label == "MRI":
                label = "MR"
            ranks.append(labels.index(label) if label in labels else len(labels))
        else:
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                number = float("nan")
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(number):
                ranks.append(float("inf"))
            else:
                ranks.append(number if preference == "min" else -number)
    return tuple(ranks), str(row["registration_scan_id"])


def prepare_cohort(table: pd.DataFrame, config) -> pd.DataFrame:
    """Validate identities and initialize registration output columns."""
    df = table.drop(columns=OBSOLETE_ARTIFACT_COLUMNS, errors="ignore").reset_index(
        drop=True
    )

    # Source columns remain authoritative across repeated registration runs.
    for column in (config.organ_mask_column, config.tumor_mask_column):
        source = f"source_{column}"
        if source not in df:
            if column == config.organ_mask_column and column not in df:
                continue
            df[source] = df[column] if column in df else None
        df[column] = df[source].astype(object)

    required = [*config.grouping_columns, "nifti_path", config.organ_mask_column]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing registration columns: {sorted(missing)}")
    for column in [*config.grouping_columns, "nifti_path"]:
        if df[column].isna().any() or df[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing registration identity: {column}")

    identities = [
        [
            *(
                identity_value(column, row[column])
                for column in config.grouping_columns
            ),
            str(Path(row.nifti_path).expanduser().resolve()),
        ]
        for _, row in df.iterrows()
    ]
    scan_ids = [stable_id(identity) for identity in identities]
    if len(scan_ids) != len(set(scan_ids)):
        raise ValueError("Duplicate scan identity in registration input")

    df["registration_scan_id"] = scan_ids
    df["registration_group_id"] = [
        stable_id(
            tuple(
                identity_value(column, row[column])
                for column in config.grouping_columns
            )
        )
        for _, row in df.iterrows()
    ]
    df["registration_group_label"] = [
        group_label(row, config.grouping_columns) for _, row in df.iterrows()
    ]
    # Clear previous-run values before planning, including organ volume used
    # as a priority criterion. Image-derived criteria become available after QC.
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

    df["registration_series_number"] = None
    df["registration_group_size"] = None
    for _, group in df.groupby("registration_group_id", sort=False):
        ordered = sorted(
            group.index,
            key=lambda index: reference_rank(
                df.loc[index], config.reference_selection_priority
            ),
        )
        for position, index in enumerate(ordered, start=1):
            df.at[index, "registration_series_number"] = position
            df.at[index, "registration_group_size"] = len(group)
    df["registration_scan_label"] = [scan_label(row) for _, row in df.iterrows()]

    return df


@dataclass
class ConsensusResult:
    mask: object
    coverage: object
    probability: object | None
    observation_count: object | None = None
    support_policy: str = "per_voxel"
    distance_field: object | None = None


def _organ_consensus_domain(
    reference,
    contributors,
    loaded,
    inverse_transforms,
    mask_qc,
    *,
    padding_mm,
):
    """Return a reference-lattice ROI covering every transformed organ bound."""
    sitk = backend()
    points = []
    for index in contributors:
        organ = loaded[index][1]
        bounds = mask_qc[index].bbox_index
        start = np.asarray(bounds[:3], dtype=float)
        stop = start + np.asarray(bounds[3:], dtype=float) - 1.0
        inverse = inverse_transforms[index]
        for x in (start[0], stop[0]):
            for y in (start[1], stop[1]):
                for z in (start[2], stop[2]):
                    native_point = organ.TransformContinuousIndexToPhysicalPoint(
                        (float(x), float(y), float(z))
                    )
                    reference_point = inverse.TransformPoint(native_point)
                    points.append(
                        reference.TransformPhysicalPointToContinuousIndex(
                            reference_point
                        )
                    )
    points = np.asarray(points)
    padding = np.ceil(float(padding_mm) / np.asarray(reference.GetSpacing())).astype(
        int
    )
    start = np.floor(points.min(axis=0)).astype(int) - padding
    stop = np.ceil(points.max(axis=0)).astype(int) + 1 + padding
    start = np.maximum(0, start)
    stop = np.minimum(np.asarray(reference.GetSize()), stop)
    if np.any(stop <= start):
        raise ValueError("Transformed organ bounds have no reference-grid support")
    domain = sitk.Image([int(value) for value in stop - start], sitk.sitkFloat32)
    domain.SetOrigin(
        reference.TransformIndexToPhysicalPoint([int(value) for value in start])
    )
    domain.SetSpacing(reference.GetSpacing())
    domain.SetDirection(reference.GetDirection())
    return domain


def fuse_organs(distance_fields, coverages, *, organ_consensus):
    """Fuse organ surfaces with a coverage-aware upper distance median.

    The upper median reproduces strict binary majority at ties and has a 50%
    breakdown point: with at least three observations, one arbitrarily distant
    outlier cannot move the fused boundary outside the majority surfaces.
    Processing in axial slabs bounds temporary memory instead of stacking every
    full-volume contributor.
    """
    sitk = backend()
    if organ_consensus not in ORGAN_CONSENSUS_METHODS:
        raise ValueError(f"Unknown organ consensus: {organ_consensus}")
    if not distance_fields or len(distance_fields) != len(coverages):
        raise ValueError("Organ consensus requires fields and matching coverage")
    reference = distance_fields[0]
    for image in [*distance_fields, *coverages]:
        if (
            image.GetSize(),
            image.GetOrigin(),
            image.GetSpacing(),
            image.GetDirection(),
        ) != (
            reference.GetSize(),
            reference.GetOrigin(),
            reference.GetSpacing(),
            reference.GetDirection(),
        ):
            raise ValueError("Organ consensus inputs must share one grid")
    if organ_consensus == "anchor":
        distance_fields, coverages = distance_fields[:1], coverages[:1]
    field_views = [sitk.GetArrayViewFromImage(field) for field in distance_fields]
    coverage_views = [sitk.GetArrayViewFromImage(image) for image in coverages]
    shape = field_views[0].shape
    outside = float(max(reference.GetSpacing()))
    fused = np.full(shape, outside, dtype=np.float32)
    probability = np.zeros(shape, dtype=np.float32)
    counts = np.zeros(shape, dtype=np.uint32)
    contributor_count = len(distance_fields)
    slice_bytes = max(1, contributor_count * shape[1] * shape[2] * 5)
    slab_depth = max(1, min(shape[0], (32 * 2**20) // slice_bytes))
    for start in range(0, shape[0], slab_depth):
        slab = slice(start, min(shape[0], start + slab_depth))
        values = np.empty(
            (contributor_count, slab.stop - slab.start, shape[1], shape[2]),
            dtype=np.float32,
        )
        slab_counts = np.zeros(values.shape[1:], dtype=np.uint32)
        votes = np.zeros(values.shape[1:], dtype=np.uint32)
        for index, (field_values, coverage_values) in enumerate(
            zip(field_views, coverage_views)
        ):
            observed = coverage_values[slab] > 0
            values[index] = field_values[slab]
            values[index][~observed] = np.inf
            slab_counts += observed
            votes += observed & (field_values[slab] <= 0.0)
        values.sort(axis=0)
        positions = slab_counts // 2
        selected = np.take_along_axis(values, positions[None, ...], axis=0)[0]
        slab_support = slab_counts > 0
        fused_slab = fused[slab]
        fused_slab[slab_support] = selected[slab_support]
        probability[slab] = np.divide(
            votes,
            slab_counts,
            out=np.zeros(votes.shape, dtype=np.float32),
            where=slab_support,
        )
        counts[slab] = slab_counts
    support = counts > 0
    if not support.any():
        raise ValueError("Organ inputs have no observed support for this consensus")
    output = (fused <= 0.0) & support

    def consensus_image(array, dtype):
        # Avoid an extra full-volume copy when the accumulator already has the
        # requested dtype. GetImageFromArray necessarily owns its output copy.
        image = sitk.GetImageFromArray(np.asarray(array, dtype=dtype))
        image.CopyInformation(reference)
        return image

    fused_field = consensus_image(fused, np.float32)
    fused_field.SetMetaData("imperandi_outside_distance_mm", repr(outside))
    return ConsensusResult(
        consensus_image(output, np.uint8),
        consensus_image(support, np.uint8),
        consensus_image(probability, np.float32),
        consensus_image(counts, np.uint32),
        distance_field=fused_field,
    )


def fuse_tumors(masks, coverages, *, tumor_consensus, threshold=0.5):
    """Fuse aligned masks with their spatial observations; anchor is first input.

    Voting uses strict > threshold; STAPLE uses >= threshold. Unknown spatial
    coverage is excluded from every estimator, and returned separately. STAPLE
    is restricted to the intersection of observed FOVs, as its backend cannot
    model missing observations. Native empty masks are filtered by the caller.
    """
    sitk = backend()
    if tumor_consensus not in TUMOR_CONSENSUS_METHODS:
        raise ValueError(f"Unknown tumor consensus: {tumor_consensus}")
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
    if tumor_consensus == "anchor":
        masks, coverages = masks[:1], coverages[:1]
    values = np.stack([sitk.GetArrayFromImage(m) > 0 for m in masks])
    observations = np.stack([sitk.GetArrayFromImage(c) > 0 for c in coverages])
    counts = observations.sum(axis=0)
    votes = (values & observations).sum(axis=0)
    support = counts == len(masks) if tumor_consensus == "staple" else counts > 0
    if not support.any():
        raise ValueError("Tumor inputs have no observed support for this consensus")
    probability = np.divide(
        votes, counts, out=np.zeros(counts.shape, np.float32), where=counts > 0
    )
    if tumor_consensus == "staple" and len(masks) > 1 and values[:, support].any():
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
    elif tumor_consensus in {"majority", "staple"}:
        binary = (
            probability > threshold
            if tumor_consensus == "majority"
            else probability >= threshold
        )
    elif tumor_consensus == "intersection":
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
        "common_fov_only" if tumor_consensus == "staple" else "per_voxel",
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


@dataclass(frozen=True)
class OrganTransferResult:
    mask: object
    distance_field: object
    residual_seam_voxels: int


@dataclass(frozen=True)
class FinalOrganQC:
    component_count: int
    component_sizes_mm3: tuple[float, ...]
    fragment_count: int
    fragment_sizes_mm3: tuple[float, ...]
    volume_mm3: float
    source_volume_mm3: float
    volume_change_fraction: float
    residual_seam_voxels: int
    has_residual_seam: bool
    has_fragments: bool


def _coverage_seam_voxels(output, coverage, source):
    """Count new binary jumps exactly across the observed/unobserved boundary."""
    out = backend().GetArrayViewFromImage(output) > 0
    observed = backend().GetArrayViewFromImage(coverage) > 0
    original = backend().GetArrayViewFromImage(source) > 0
    seams = 0
    for axis in range(out.ndim):
        low = [slice(None)] * out.ndim
        high = [slice(None)] * out.ndim
        low[axis] = slice(None, -1)
        high[axis] = slice(1, None)
        low, high = tuple(low), tuple(high)
        coverage_edge = observed[low] != observed[high]
        introduced_edge = (out[low] != out[high]) & (original[low] == original[high])
        seams += int(np.count_nonzero(coverage_edge & introduced_edge))
    return seams


def _fill_unobserved(consensus_distance, coverage, source_distance, *, blend_width_mm):
    """Blend fields inside coverage, retain source outside, and threshold once."""
    sitk = backend()
    observed = sitk.Cast(coverage > 0, sitk.sitkUInt8)
    observed_values = sitk.GetArrayViewFromImage(observed) > 0
    source_mask = threshold_organ_distance(source_distance)
    if not observed_values.any():
        blended = sitk.Image(source_distance)
    elif observed_values.all():
        blended = sitk.Image(consensus_distance)
    else:
        inside_distance = sitk.SignedMaurerDistanceMap(
            observed,
            insideIsPositive=True,
            squaredDistance=False,
            useImageSpacing=True,
        )
        alpha = sitk.Clamp(
            sitk.Cast(inside_distance / float(blend_width_mm), sitk.sitkFloat32),
            lowerBound=0.0,
            upperBound=1.0,
        )
        alpha *= sitk.Cast(observed, sitk.sitkFloat32)
        blended = source_distance + alpha * (consensus_distance - source_distance)
    output = threshold_organ_distance(blended)
    seam_voxels = _coverage_seam_voxels(output, observed, source_mask)
    return OrganTransferResult(output, blended, seam_voxels)


def final_organ_qc(mask, source, *, residual_seam_voxels=0):
    """Report final liver topology and volume without implicitly repairing it."""
    sitk = backend()
    connected = sitk.ConnectedComponentImageFilter()
    connected.SetFullyConnected(False)
    labels = connected.Execute(mask)
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(labels)
    sizes = sorted(
        (float(statistics.GetPhysicalSize(label)) for label in statistics.GetLabels()),
        reverse=True,
    )
    voxel_volume = float(np.prod(mask.GetSpacing()))
    volume = float(np.count_nonzero(sitk.GetArrayViewFromImage(mask)) * voxel_volume)
    source_volume = float(
        np.count_nonzero(sitk.GetArrayViewFromImage(source)) * voxel_volume
    )
    change = (volume - source_volume) / source_volume if source_volume else float("inf")
    fragments = sizes[1:]
    return FinalOrganQC(
        component_count=len(sizes),
        component_sizes_mm3=tuple(sizes),
        fragment_count=len(fragments),
        fragment_sizes_mm3=tuple(fragments),
        volume_mm3=volume,
        source_volume_mm3=source_volume,
        volume_change_fraction=float(change),
        residual_seam_voxels=int(residual_seam_voxels),
        has_residual_seam=bool(residual_seam_voxels),
        has_fragments=bool(fragments),
    )


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


def _consensus_eligible(metrics, partial, config):
    """Return whether an accepted transform is reliable enough to vote."""
    score = metrics["dice_common_fov" if partial else "dice_full"]
    return np.isfinite(score) and score >= config.minimum_consensus_dice


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
        "common_fov_only" if config.tumor_consensus_method == "staple" else "per_voxel"
    )


def register_cohort(
    table,
    output_dir,
    config=None,
    *,
    pair_registration=register_pair,
    fusion=fuse_tumors,
):
    """Return (cohort, errors), grouping by configured manifest columns.

    This image-processing API performs fresh work. The CLI runner adds group
    scheduling, checkpointing, and resume around this function.
    pair_registration and fusion may be injected by alternative implementations.
    """
    config = config or RegistrationConfig()
    sitk = backend()
    df = prepare_cohort(table, config)
    ids = df.registration_scan_id.tolist()
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

    groups = df.groupby("registration_group_id", sort=True)
    for _, group in groups:
        group_id = group.iloc[0].registration_group_id
        label = group_label(group.iloc[0], config.grouping_columns)
        logger.info("Registration group started: %s, series=%d", label, len(group))
        directory = root / group_id
        group_started = time.perf_counter()
        df.loc[group.index, "registration_started_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        df.loc[group.index, "registration_group_id"] = group_id
        priorities = config.reference_selection_priority
        order = sorted(
            group.index, key=lambda index: reference_rank(df.loc[index], priorities)
        )
        loaded, mask_qc, organ_fields = {}, {}, {}
        for i in order:
            organ_path = df.at[i, config.organ_mask_column]
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
                if "empty" in quality.reasons:
                    df.at[i, "registration_status"] = "skipped"
                    df.at[i, "registration_skip_reason"] = "empty_organ_mask"
                    df.at[i, "consensus_status"] = "skipped"
                    logger.warning(
                        "Registration series skipped: %s, reason=empty organ mask",
                        df.at[i, "registration_scan_label"],
                    )
                    continue
                if quality.status == "invalid" or (
                    quality.partial and not config.accept_partial_organ_masks
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

        def get_organ_field(index):
            if index not in organ_fields:
                organ_fields[index] = organ_distance_field(
                    loaded[index][1],
                    padding_mm=config.organ_distance_field_padding_mm,
                )
            return organ_fields[index]

        transforms, inverse_transforms, pair_results, native_organs = (
            {},
            {},
            {},
            {},
        )
        native_organ_seams = {}
        consensus_eligible = set()
        fireants_reference_cache = {}
        for i, (image, organ) in loaded.items():
            started = time.perf_counter()
            logger.info(
                "Registration organ alignment started: %s",
                df.at[i, "registration_scan_label"],
            )
            try:
                result = None
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
                    if pair_registration is register_pair:
                        result = pair_registration(
                            reference_organ,
                            organ,
                            config,
                            reference,
                            image,
                            fireants_reference_cache,
                        )
                    else:
                        # Preserve the established three-argument injection API.
                        result = pair_registration(reference_organ, organ, config)
                    tx = result.reference_to_scan
                    pair_results[i] = result
                    report = _registration_report(result)
                record_stages(df, i, report)
                metrics = getattr(result, "overlap", None) or overlap_qc(
                    reference_organ, organ, tx
                )
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
                inverse = getattr(result, "scan_to_reference", None)
                if inverse is None:
                    inverse = tx.GetInverse()
                pair_dir = directory / ids[i]
                if config.preserve_source_organ_mask:
                    native_organ = organ
                elif config.organ_consensus_method == "anchor":
                    native_consensus = resample_organ_distance(
                        get_organ_field(ref), image, inverse
                    )
                    known = resample(coverage_image(reference), image, inverse)
                    source_distance = expand_organ_distance(get_organ_field(i), image)
                    transfer = _fill_unobserved(
                        native_consensus,
                        known,
                        source_distance,
                        blend_width_mm=config.organ_coverage_blend_width_mm,
                    )
                    native_organ = transfer.mask
                    native_organ_seams[i] = transfer.residual_seam_voxels
                    df.at[i, "reg_organ_native_path"] = write_artifact(
                        native_organ,
                        pair_dir / "organ_native.nii.gz",
                    )
                    df.at[i, config.organ_mask_column] = df.at[
                        i, "reg_organ_native_path"
                    ]
                transforms[i] = tx
                inverse_transforms[i] = inverse
                if _consensus_eligible(
                    metrics, mask_qc[ref].partial or mask_qc[i].partial, config
                ):
                    consensus_eligible.add(i)
                if (
                    config.preserve_source_organ_mask
                    or config.organ_consensus_method == "anchor"
                ):
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
        if (
            not config.preserve_source_organ_mask
            and config.organ_consensus_method != "anchor"
        ):
            organ_contributors = [i for i in transforms if i in consensus_eligible]
            contributor_labels = [
                df.at[i, "registration_scan_label"] for i in organ_contributors
            ]
            logger.info(
                "Registration organ consensus started: organ_consensus=%s, "
                "contributors=%d [%s]",
                config.organ_consensus_method,
                len(organ_contributors),
                " | ".join(contributor_labels),
            )
            try:
                aligned_fields, organ_coverages = [], []
                fusion_reference = _organ_consensus_domain(
                    reference,
                    organ_contributors,
                    loaded,
                    inverse_transforms,
                    mask_qc,
                    padding_mm=(
                        config.organ_distance_field_padding_mm
                        + config.maximum_elastic_displacement_mm
                        + max(reference.GetSpacing())
                    ),
                )
                for i in organ_contributors:
                    organ = loaded[i][1]
                    aligned_fields.append(
                        resample_organ_distance(
                            get_organ_field(i), fusion_reference, transforms[i]
                        )
                    )
                    organ_coverages.append(
                        resample(
                            coverage_image(organ),
                            fusion_reference,
                            transforms[i],
                        )
                    )
                fused_organ = fuse_organs(
                    aligned_fields,
                    organ_coverages,
                    organ_consensus=config.organ_consensus_method,
                )
            except (RuntimeError, ValueError, TypeError) as exc:
                df.loc[list(transforms), "registration_status"] = "failed"
                df.loc[list(transforms), "registration_confidence"] = "failed"
                error(ref, "organ_consensus", exc)
                df.loc[group.index, "consensus_status"] = "no_reference"
                df.loc[group.index, "registration_elapsed_seconds"] = (
                    time.perf_counter() - group_started
                )
                _record_consensus_inputs(df, group.index, [], config)
                publish_group_log(df, group.index, errors, config, directory)
                continue

            failed_transfers = []
            for i in list(transforms):
                try:
                    image, source_organ = loaded[i]
                    inverse = inverse_transforms[i]
                    native_consensus = resample_organ_distance(
                        fused_organ.distance_field, image, inverse
                    )
                    native_coverage = resample(fused_organ.coverage, image, inverse)
                    source_distance = expand_organ_distance(get_organ_field(i), image)
                    transfer = _fill_unobserved(
                        native_consensus,
                        native_coverage,
                        source_distance,
                        blend_width_mm=config.organ_coverage_blend_width_mm,
                    )
                    native_organ = transfer.mask
                    native_organ_seams[i] = transfer.residual_seam_voxels
                    pair_dir = directory / ids[i]
                    df.at[i, "reg_organ_native_path"] = write_artifact(
                        native_organ, pair_dir / "organ_native.nii.gz"
                    )
                    df.at[i, config.organ_mask_column] = df.at[
                        i, "reg_organ_native_path"
                    ]
                    native_organs[i] = native_organ
                except (RuntimeError, ValueError, TypeError) as exc:
                    failed_transfers.append(i)
                    df.at[i, "registration_status"] = "failed"
                    df.at[i, "registration_confidence"] = "failed"
                    error(i, "organ_consensus_transfer", exc)
            for i in failed_transfers:
                transforms.pop(i, None)
                inverse_transforms.pop(i, None)
                pair_results.pop(i, None)
            if ref not in transforms:
                df.loc[group.index, "consensus_status"] = "no_reference"
                df.loc[group.index, "registration_elapsed_seconds"] = (
                    time.perf_counter() - group_started
                )
                _record_consensus_inputs(df, group.index, [], config)
                publish_group_log(df, group.index, errors, config, directory)
                continue
        for i, native_organ in native_organs.items():
            quality = final_organ_qc(
                native_organ,
                loaded[i][1],
                residual_seam_voxels=native_organ_seams.get(i, 0),
            )
            record_final_organ_qc(df, i, quality)
        tumors = {}
        # Read all available inputs so tumor Dice can be reported independently
        # from the selected consensus policy. Use the final QC-aware reference
        # order for fallback anchors, then record excluded scans as before.
        tumor_order = active_order + [i for i in order if i not in loaded]
        for i in tumor_order:
            if i not in transforms:
                df.at[i, "tumor_consensus_input_status"] = "excluded_" + str(
                    df.at[i, "registration_status"]
                )
                continue
            path = (
                df.at[i, config.tumor_mask_column]
                if config.tumor_mask_column in df
                else None
            )
            if _missing_file(path):
                df.at[i, "tumor_consensus_input_status"] = "skipped_missing"
                continue
            try:
                tumor = read_mask(path, loaded[i][0])
                if not np.count_nonzero(sitk.GetArrayViewFromImage(tumor)):
                    df.at[i, "tumor_consensus_input_status"] = "skipped_empty"
                    continue
                tumors[i] = tumor
                df.at[i, "tumor_consensus_input_status"] = (
                    "contributed"
                    if i in consensus_eligible
                    else "excluded_minimum_consensus_dice"
                )
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

        contributors = [i for i in tumors if i in consensus_eligible]
        if config.tumor_consensus_method == "anchor":
            contributors = contributors[:1]
        for i in tumors:
            if (
                i not in contributors
                and df.at[i, "tumor_consensus_input_status"] == "contributed"
            ):
                df.at[i, "tumor_consensus_input_status"] = "excluded_anchor_policy"
        _record_consensus_inputs(df, group.index, contributors, config)
        failed = df.loc[group.index, "registration_status"].isin(
            ["failed", "low_confidence", "invalid_organ_mask"]
        )
        df.loc[failed.index[failed], "consensus_status"] = "registration_failed"
        df.loc[list(transforms), "consensus_status"] = "no_tumor_input"
        if contributors:
            contributor_labels = [
                df.at[i, "registration_scan_label"] for i in contributors
            ]
            logger.info(
                "Registration tumor consensus started: tumor_consensus=%s, "
                "contributors=%d [%s]",
                config.tumor_consensus_method,
                len(contributors),
                " | ".join(contributor_labels),
            )
            try:
                masks, coverages = [], []
                for i in contributors:
                    tumor = tumors[i]
                    masks.append(resample(tumor, reference, transforms[i]))
                    coverages.append(
                        resample(coverage_image(tumor), reference, transforms[i])
                    )
                fused = fusion(
                    masks,
                    coverages,
                    tumor_consensus=config.tumor_consensus_method,
                    threshold=config.consensus_probability_threshold,
                )
                for i, tx in transforms.items():
                    try:
                        inverse = inverse_transforms[i]
                        image = loaded[i][0]
                        pair_dir = directory / ids[i]
                        if config.tumor_consensus_method == "anchor":
                            composed = sitk.CompositeTransform(3)
                            composed.AddTransform(transforms[contributors[0]])
                            composed.AddTransform(inverse)
                            native = resample(tumors[contributors[0]], image, composed)
                        else:
                            native = resample(fused.mask, image, inverse)
                        native_coverage = resample(fused.coverage, image, inverse)
                        native = sitk.And(native, native_coverage)
                        if config.clip_tumor_consensus_to_organ:
                            native = sitk.And(native, native_organs[i])
                        df.at[i, "reg_tumor_native_path"] = write_artifact(
                            native, pair_dir / "tumor_native.nii.gz"
                        )
                        df.at[i, config.tumor_mask_column] = df.at[
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
