"""Physical-space organ registration, independent of tumor masks."""

from dataclasses import dataclass, field
import time
from itertools import permutations, product
import numpy as np
from scipy import ndimage

from .config import REGISTRATION_STAGES

DICE_TOLERANCE = 1e-6
MI_TOLERANCE = 1e-8


def backend():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError(
            "Registration requires optional dependencies: pip install 'imperandi[registration]'"
        ) from exc
    return sitk


def resample(image, reference, transform, *, label=True):
    sitk = backend()
    return sitk.Resample(
        image,
        reference,
        transform,
        sitk.sitkNearestNeighbor if label else sitk.sitkLinear,
        0,
        sitk.sitkUInt8 if label else sitk.sitkFloat32,
    )


def _mask_roi(mask, padding_mm):
    """Crop a nonempty mask to its padded physical-space bounding box."""
    sitk = backend()
    labels = sitk.LabelShapeStatisticsImageFilter()
    labels.Execute(mask)
    if not labels.HasLabel(1):
        raise ValueError("Cannot build a distance field from an empty organ mask")
    bounds = labels.GetBoundingBox(1)
    start = np.array(bounds[:3], dtype=int)
    stop = start + np.array(bounds[3:], dtype=int)
    padding = np.ceil(float(padding_mm) / np.array(mask.GetSpacing())).astype(int)
    start = np.maximum(0, start - padding)
    stop = np.minimum(np.array(mask.GetSize()), stop + padding)
    return sitk.RegionOfInterest(
        mask,
        [int(value) for value in stop - start],
        [int(value) for value in start],
    )


def organ_distance_field(mask, *, padding_mm):
    """Return an unclamped physical signed-distance field for organ transfer.

    The padded crop contains every zero crossing while avoiding a full-volume
    distance transform. This field is deliberately separate from the bounded
    optimizer distance map below.
    """
    sitk = backend()
    padding = max(float(padding_mm), 2.0 * max(mask.GetSpacing()))
    cropped = _mask_roi(mask, padding)
    outside = sitk.Clamp(
        sitk.SignedMaurerDistanceMap(
            cropped,
            insideIsPositive=False,
            squaredDistance=False,
            useImageSpacing=True,
        ),
        lowerBound=0.0,
    )
    complement = sitk.Cast(cropped == 0, sitk.sitkUInt8)
    inside = sitk.Clamp(
        sitk.SignedMaurerDistanceMap(
            complement,
            insideIsPositive=False,
            squaredDistance=False,
            useImageSpacing=True,
        ),
        lowerBound=0.0,
    )
    # A single SignedMaurer map is zero at foreground boundary voxel centers.
    # Subtracting complementary distances puts zero between foreground and
    # background centers and avoids systematic erosion after interpolation.
    distances = sitk.Cast(outside - inside, sitk.sitkFloat32)
    maximum = float(sitk.GetArrayViewFromImage(distances).max())
    outside = max(maximum + max(mask.GetSpacing()), max(mask.GetSpacing()))
    distances.SetMetaData("imperandi_outside_distance_mm", repr(outside))
    return distances


def _organ_distance_outside_value(field):
    key = "imperandi_outside_distance_mm"
    if field.HasMetaDataKey(key):
        return float(field.GetMetaData(key))
    maximum = float(backend().GetArrayViewFromImage(field).max())
    return max(maximum + max(field.GetSpacing()), max(field.GetSpacing()))


def resample_organ_distance(field, reference, transform):
    """Linearly transfer an organ field and leave thresholding to its consumer."""
    sitk = backend()
    return sitk.Resample(
        field,
        reference,
        transform,
        sitk.sitkLinear,
        _organ_distance_outside_value(field),
        sitk.sitkFloat32,
    )


def expand_organ_distance(field, reference):
    """Paste a cropped native field back onto its unchanged native lattice."""
    sitk = backend()
    if any(
        not np.allclose(a, b, atol=1e-6, rtol=0)
        for a, b in [
            (field.GetSpacing(), reference.GetSpacing()),
            (field.GetDirection(), reference.GetDirection()),
        ]
    ):
        raise ValueError("Organ distance crop does not match its native lattice")
    output = sitk.Image(reference.GetSize(), sitk.sitkFloat32)
    output.CopyInformation(reference)
    output += _organ_distance_outside_value(field)
    destination = reference.TransformPhysicalPointToIndex(field.GetOrigin())
    return sitk.Paste(
        output,
        field,
        field.GetSize(),
        [0] * field.GetDimension(),
        destination,
    )


def threshold_organ_distance(field):
    """Threshold an organ field exactly once on its destination grid."""
    sitk = backend()
    return sitk.Cast(field <= 0.0, sitk.sitkUInt8)


def read_image(path):
    sitk = backend()
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Registration requires scalar 3-D images")
    if not np.all(
        np.isfinite(image.GetOrigin() + image.GetSpacing() + image.GetDirection())
    ):
        raise ValueError("Nonfinite image geometry")
    if min(image.GetSpacing()) <= 0:
        raise ValueError("Image spacing must be positive")
    return image


def read_mask(path, image):
    """Require matching native geometry; do not guess a mask's coordinate system."""
    sitk = backend()
    mask = read_image(path)
    if mask.GetSize() != image.GetSize() or any(
        not np.allclose(a, b, atol=1e-5, rtol=0)
        for a, b in [
            (mask.GetOrigin(), image.GetOrigin()),
            (mask.GetSpacing(), image.GetSpacing()),
            (mask.GetDirection(), image.GetDirection()),
        ]
    ):
        raise ValueError("Mask does not match its native image geometry")
    values = sitk.GetArrayViewFromImage(mask)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Mask contains invalid values")
    return sitk.Cast(mask > 0, sitk.sitkUInt8)


@dataclass(frozen=True)
class OrganMaskQC:
    status: str
    volume_mm3: float
    bbox_index: tuple[int, ...]
    boundary_contact: tuple[tuple[bool, bool], ...]
    largest_component_fraction: float
    reasons: tuple[str, ...]

    @property
    def partial(self) -> bool:
        return self.status == "partial"


def _has_straight_side(values):
    """Detect a broad, abrupt end face along any image axis.

    A cut face occupies at least a quarter of the largest cross-section and
    retains at least 90% of the adjacent inward section. Requiring both avoids
    treating small, voxelized rounded tips as cuts. This heuristic detects
    image-axis-aligned cuts, including images with oblique physical geometry.
    """
    for axis in range(values.ndim):
        areas = values.sum(axis=tuple(i for i in range(values.ndim) if i != axis))
        occupied = np.flatnonzero(areas)
        if len(occupied) < 3:
            continue
        for edge, inward in (
            (occupied[0], occupied[0] + 1),
            (occupied[-1], occupied[-1] - 1),
        ):
            if areas[edge] >= max(4, 0.25 * areas.max()) and (
                areas[edge] >= 0.9 * areas[inward]
            ):
                return True
    return False


def assess_organ_mask(mask, config) -> OrganMaskQC:
    """Assess a geometry-validated binary mask without modifying it."""
    values = backend().GetArrayViewFromImage(mask) > 0
    indices = np.argwhere(values)
    count = len(indices)
    volume = float(count * np.prod(mask.GetSpacing()))
    if not count:
        return OrganMaskQC("invalid", volume, (), (), 0.0, ("empty",))
    start = indices.min(axis=0)[::-1]
    stop = indices.max(axis=0)[::-1] + 1
    spacing = np.asarray(mask.GetSpacing())
    low = start * spacing <= config.partial_mask_boundary_margin_mm
    high = (
        np.asarray(mask.GetSize()) - stop
    ) * spacing <= config.partial_mask_boundary_margin_mm
    contact = tuple((bool(a), bool(b)) for a, b in zip(low, high))
    labels, _ = ndimage.label(values)  # Conservative face-connected components.
    sizes = np.bincount(labels.ravel())[1:]
    fraction = float(sizes.max() / count)
    reasons = []
    if count < 4:
        reasons.append("insufficient_foreground")
    if fraction < config.minimum_largest_component_fraction:
        reasons.append("fragmented")
    if values.all():
        reasons.append("fills_entire_fov")
    status = "invalid" if reasons else "partial" if any(low | high) else "complete"
    if status != "invalid" and _has_straight_side(labels == sizes.argmax() + 1):
        status = "partial"
        reasons.append("straight_side")
    return OrganMaskQC(
        status,
        volume,
        tuple(int(v) for v in (*start, *(stop - start))),
        contact,
        fraction,
        tuple(reasons),
    )


def coverage_image(image):
    """All voxels of an acquisition are observed, regardless of segmentation."""
    sitk = backend()
    coverage = sitk.Image(image.GetSize(), sitk.sitkUInt8) + 1
    coverage.CopyInformation(image)
    return coverage


def overlap_qc(fixed, moving, transform) -> dict[str, float]:
    """Report full/reference-grid Dice and Dice within transformed common FOV."""
    sitk = backend()
    a = sitk.GetArrayViewFromImage(fixed) > 0
    warped = resample(moving, fixed, transform)
    b = sitk.GetArrayViewFromImage(warped) > 0
    observed = resample(coverage_image(moving), fixed, transform)
    support = sitk.GetArrayViewFromImage(observed) > 0
    common_count = int(a[support].sum() + b[support].sum())
    intersection = int(np.count_nonzero(a & b & support))
    return {
        "dice_full": float(2 * intersection / max(1, a.sum() + b.sum())),
        "dice_common_fov": float(2 * intersection / max(1, common_count)),
        "common_fov_fraction": float(support.mean()),
        "common_foreground_voxels": common_count,
    }


def registration_confidence(metrics, partial: bool, config) -> str:
    """Require actual observed foreground and overlap, never volume alone."""
    score = metrics["dice_common_fov" if partial else "dice_full"]
    if (
        not np.isfinite(
            [score, metrics["common_fov_fraction"], metrics["common_foreground_voxels"]]
        ).all()
        or score < config.minimum_accepted_organ_dice
        or metrics["common_fov_fraction"] < config.minimum_common_field_of_view_fraction
        or metrics["common_foreground_voxels"] < 4
    ):
        return "low_confidence"
    return "ok_partial_coverage" if partial else "ok"


def dice(fixed, moving, transform):
    sitk = backend()
    a = sitk.GetArrayViewFromImage(fixed) > 0
    warped = resample(moving, fixed, transform)
    b = sitk.GetArrayViewFromImage(warped) > 0
    return float(2 * np.count_nonzero(a & b) / max(1, a.sum() + b.sum()))


def distance_map(mask, *, padding_mm, band_mm):
    """Build a bounded signed distance map around a foreground mask."""
    sitk = backend()
    cropped = _mask_roi(mask, padding_mm)
    distances = sitk.SignedMaurerDistanceMap(
        cropped, insideIsPositive=False, squaredDistance=False, useImageSpacing=True
    )
    return sitk.Clamp(distances, lowerBound=-float(band_mm), upperBound=float(band_mm))


def boundary_band(mask, *, band_mm):
    """Return a physical-width shell on both sides of an organ boundary."""
    sitk = backend()
    distances = sitk.SignedMaurerDistanceMap(
        mask,
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=True,
    )
    return sitk.Cast(sitk.Abs(distances) <= float(band_mm), sitk.sitkUInt8)


def _moments(mask):
    sitk = backend()
    indices = np.argwhere(sitk.GetArrayViewFromImage(mask) > 0)[:, ::-1]
    if len(indices) < 4:
        raise ValueError("Organ mask needs at least four foreground voxels")
    indices = indices[:: max(1, len(indices) // 50000)]
    points = (indices * mask.GetSpacing()) @ np.array(mask.GetDirection()).reshape(
        3, 3
    ).T + mask.GetOrigin()
    center = points.mean(axis=0)
    values, axes = np.linalg.eigh(np.cov(points.T))
    return center, values, axes


def _rotation_angle_degrees(rotation):
    """Return the principal angle of a proper 3-D rotation matrix."""
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def initialize_pca(fixed, moving, max_rotation_degrees=45.0):
    """Return a PCA mapping, excluding rotations beyond the configured limit."""
    sitk = backend()
    cf, vf, af = _moments(fixed)
    cm, vm, am = _moments(moving)
    rotations = [np.eye(3)]
    # Nearly equal eigenvalues do not establish a stable anatomical orientation.
    if all(np.min(np.diff(v)) > 0.01 * max(v[-1], 1e-8) for v in (vf, vm)):
        for perm in permutations(range(3)):
            for signs in product((-1, 1), repeat=3):
                rotation = (am[:, perm] * signs) @ af.T
                if (
                    np.linalg.det(rotation) > 0
                    and _rotation_angle_degrees(rotation)
                    <= float(max_rotation_degrees) + 1e-6
                ):
                    rotations.append(rotation)
    candidates = [sitk.Euler3DTransform()]
    for rotation in rotations:
        tx = sitk.Euler3DTransform()
        tx.SetCenter(tuple(cf))
        tx.SetMatrix(rotation.ravel().tolist(), 1e-6)
        tx.SetTranslation(tuple(cm - cf))
        candidates.append(tx)
    factors = [max(1, n // 48) for n in fixed.GetSize()]
    coarse = sitk.Shrink(fixed, factors)
    return max(candidates, key=lambda tx: dice(coarse, moving, tx))


def _bspline_displacement(transform, domain):
    """Sample a B-spline residual on its fixed-space optimization domain."""
    sitk = backend()
    return sitk.TransformToDisplacementField(
        transform,
        sitk.sitkVectorFloat64,
        domain.GetSize(),
        domain.GetOrigin(),
        domain.GetSpacing(),
        domain.GetDirection(),
    )


def _physical_qc_grid(domain, control_spacing_mm):
    """Return a coarse grid whose sampling density is defined in millimetres."""
    sitk = backend()
    lengths = (np.asarray(domain.GetSize()) - 1) * np.asarray(domain.GetSpacing())
    target_spacing = max(
        max(domain.GetSpacing()), min(10.0, float(control_spacing_mm) / 4.0)
    )
    size = []
    spacing = []
    for voxel_count, native_spacing, length in zip(
        domain.GetSize(), domain.GetSpacing(), lengths
    ):
        if voxel_count <= 1:
            size.append(1)
            spacing.append(float(native_spacing))
            continue
        intervals = max(1, int(np.ceil(length / target_spacing)))
        size.append(intervals + 1)
        spacing.append(float(length / intervals))
    coarse = sitk.Image(size, sitk.sitkFloat32)
    coarse.SetOrigin(domain.GetOrigin())
    coarse.SetSpacing(spacing)
    coarse.SetDirection(domain.GetDirection())
    return coarse


def _residual_jacobian(field):
    """Differentiate displacement components in the image-axis basis.

    Use identity direction on this calculation-only image, so the result is
    independent of whether an ITK version applies direction cosines itself.
    For orthonormal D, det(I + D.T @ du/dx @ D) = det(I + du/dx).
    """
    sitk = backend()
    direction = np.asarray(field.GetDirection()).reshape(3, 3)
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError("Elastic field direction must be orthonormal")
    local = sitk.GetImageFromArray(
        sitk.GetArrayViewFromImage(field) @ direction, isVector=True
    )
    local.SetSpacing(field.GetSpacing())
    return sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(local))


def _elastic_field_qc(field, config):
    """Measure displacement and Jacobian limits on a sampled residual field."""
    vectors = backend().GetArrayViewFromImage(field)
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite elastic displacement field")
    displacement = np.linalg.norm(vectors, axis=-1)
    jacobian = _residual_jacobian(field)
    if not np.isfinite(jacobian).all():
        raise ValueError("Nonfinite elastic Jacobian determinant")
    p95 = float(np.percentile(displacement, 95))
    maximum = float(displacement.max())
    jacobian_min = float(jacobian.min())
    jacobian_max = float(jacobian.max())
    nonpositive = int(np.count_nonzero(jacobian <= 0))
    reasons = []
    if p95 > config.maximum_elastic_displacement_p95_mm:
        reasons.append("p95_displacement_exceeds_limit")
    if maximum > config.maximum_elastic_displacement_mm:
        reasons.append("maximum_displacement_exceeds_limit")
    if nonpositive:
        reasons.append("folding")
    if jacobian_min < config.minimum_elastic_jacobian_determinant:
        reasons.append("jacobian_below_plausible_range")
    if jacobian_max > config.maximum_elastic_jacobian_determinant:
        reasons.append("jacobian_above_plausible_range")
    return {
        "displacement_p95_mm": p95,
        "displacement_max_mm": maximum,
        "maximum_allowed_displacement_p95_mm": (
            config.maximum_elastic_displacement_p95_mm
        ),
        "maximum_allowed_displacement_mm": config.maximum_elastic_displacement_mm,
        "jacobian_min": jacobian_min,
        "jacobian_max": jacobian_max,
        "jacobian_nonpositive_voxels": nonpositive,
        "minimum_allowed_jacobian": config.minimum_elastic_jacobian_determinant,
        "maximum_allowed_jacobian": config.maximum_elastic_jacobian_determinant,
        "deformation_qc_passed": not reasons,
        "deformation_qc_reasons": reasons,
    }


def elastic_deformation_qc(transform, domain, config):
    """Measure and validate the B-spline residual deformation."""
    residual = transform.GetNthTransform(transform.GetNumberOfTransforms() - 1)
    if residual.GetName() != "BSplineTransform":
        raise ValueError("Elastic residual must be a BSplineTransform")
    field = _bspline_displacement(residual, domain)
    return field, _elastic_field_qc(field, config)


def _inverse_field_domain(domain, field):
    """Cover both the original grid and its displaced physical-space support."""
    sitk = backend()
    direction = np.asarray(domain.GetDirection()).reshape(3, 3)
    vectors = sitk.GetArrayViewFromImage(field)
    local_vectors = vectors @ direction
    spacing = np.asarray(domain.GetSpacing())
    low = (
        np.floor(
            np.minimum(0.0, local_vectors.reshape(-1, 3).min(axis=0)) / spacing
        ).astype(int)
        - 1
    )
    high = (
        np.ceil(
            np.maximum(0.0, local_vectors.reshape(-1, 3).max(axis=0)) / spacing
        ).astype(int)
        + 1
    )
    size = np.asarray(domain.GetSize(), dtype=int) + high - low
    inverse_domain = sitk.Image([int(value) for value in size], sitk.sitkFloat32)
    inverse_domain.SetOrigin(
        domain.TransformContinuousIndexToPhysicalPoint([float(value) for value in low])
    )
    inverse_domain.SetSpacing(domain.GetSpacing())
    inverse_domain.SetDirection(domain.GetDirection())
    return inverse_domain, low, high


def _expanded_displacement_field(domain, field):
    """Pad a field to its inverse domain without an artificial zero boundary."""
    sitk = backend()
    inverse_domain, low, high = _inverse_field_domain(domain, field)
    vectors = sitk.GetArrayFromImage(field)
    pad_width = [
        (-int(low[2]), int(high[2])),
        (-int(low[1]), int(high[1])),
        (-int(low[0]), int(high[0])),
        (0, 0),
    ]
    expanded = sitk.GetImageFromArray(
        np.pad(vectors, pad_width, mode="edge"), isVector=True
    )
    expanded.CopyInformation(inverse_domain)
    return expanded


def _error_statistics(values):
    if not len(values):
        raise ValueError("Elastic round-trip validation has no valid samples")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite elastic inverse round-trip error")
    return {
        "p95_mm": float(np.percentile(values, 95)),
        "p99_mm": float(np.percentile(values, 99)),
        "max_mm": float(values.max()),
    }


def _round_trip_error(field, inverse_field, anatomical_roi=None):
    """Return errors where interpolation is defined, globally and in the ROI."""
    sitk = backend()
    residual = sitk.DisplacementFieldTransform(sitk.Image(field))
    warped = sitk.Resample(
        inverse_field, field, residual, sitk.sitkLinear, 0, sitk.sitkVectorFloat64
    )
    support = sitk.Image(inverse_field.GetSize(), sitk.sitkFloat32) + 1.0
    support.CopyInformation(inverse_field)
    sampled_support = sitk.Resample(
        support, field, residual, sitk.sitkLinear, 0.0, sitk.sitkFloat32
    )
    valid = sitk.GetArrayViewFromImage(sampled_support) >= 1.0 - 1e-6
    error = sitk.GetArrayFromImage(warped)
    error += sitk.GetArrayViewFromImage(field)
    error = np.linalg.norm(error, axis=-1)
    if anatomical_roi is None:
        roi = np.ones(error.shape, dtype=bool)
    else:
        roi = sitk.GetArrayViewFromImage(anatomical_roi) > 0
        if roi.shape != error.shape:
            raise ValueError("Anatomical ROI does not match round-trip field geometry")
    valid_roi = valid & roi
    return {
        "full_grid": _error_statistics(error[valid]),
        "valid_anatomical_roi": _error_statistics(error[valid_roi]),
        "full_grid_valid_voxels": int(np.count_nonzero(valid)),
        "full_grid_voxels": int(valid.size),
        "valid_anatomical_roi_voxels": int(np.count_nonzero(valid_roi)),
    }


def _record_round_trip_statistics(diagnostics, prefix, statistics):
    for region in ("full_grid", "valid_anatomical_roi"):
        for statistic, value in statistics[region].items():
            diagnostics[f"{prefix}_round_trip_{region}_{statistic}"] = value
    diagnostics[f"{prefix}_round_trip_full_grid_valid_voxels"] = statistics[
        "full_grid_valid_voxels"
    ]
    diagnostics[f"{prefix}_round_trip_full_grid_voxels"] = statistics[
        "full_grid_voxels"
    ]
    diagnostics[f"{prefix}_round_trip_valid_anatomical_roi_voxels"] = statistics[
        "valid_anatomical_roi_voxels"
    ]


def invert_mask_elastic(
    transform, domain, diagnostics=None, field=None, anatomical_roi=None
):
    """Numerically invert the residual and exactly invert the linear stage."""
    sitk = backend()
    diagnostics = {} if diagnostics is None else diagnostics
    linear = transform.GetNthTransform(0)
    residual = transform.GetNthTransform(1)
    field = _bspline_displacement(residual, domain) if field is None else field
    diagnostics["validation_phase"] = "inversion"
    inversion_started = time.perf_counter()
    try:
        # This SimpleITK inverter inherits its output domain from the input;
        # expand the input first so displaced boundary support is represented.
        expanded_field = _expanded_displacement_field(domain, field)
        diagnostics.update(
            inverse_domain_size=list(expanded_field.GetSize()),
            inverse_domain_origin=list(expanded_field.GetOrigin()),
            inverse_domain_spacing=list(expanded_field.GetSpacing()),
        )
        inverter = sitk.InvertDisplacementFieldImageFilter()
        inverter.SetMaximumNumberOfIterations(50)
        inverter.SetMeanErrorToleranceThreshold(0.0)
        inverter.SetMaxErrorToleranceThreshold(0.01)
        inverter.SetEnforceBoundaryCondition(False)
        inverse_field = inverter.Execute(expanded_field)
    finally:
        diagnostics["inversion_elapsed_seconds"] = (
            time.perf_counter() - inversion_started
        )
    if not np.isfinite(sitk.GetArrayViewFromImage(inverse_field)).all():
        raise ValueError("Nonfinite inverse elastic displacement field")
    diagnostics.update(
        inverse_max_error_norm=float(inverter.GetMaxErrorNorm()),
        inverse_mean_error_norm=float(inverter.GetMeanErrorNorm()),
        validation_phase="round_trip",
    )
    round_trip_started = time.perf_counter()
    try:
        if anatomical_roi is None:
            anatomical_roi = sitk.Image(domain.GetSize(), sitk.sitkUInt8) + 1
            anatomical_roi.CopyInformation(domain)
        forward_statistics = _round_trip_error(field, inverse_field, anatomical_roi)
        inverse_residual = sitk.DisplacementFieldTransform(sitk.Image(inverse_field))
        inverse_roi = sitk.Resample(
            anatomical_roi,
            inverse_field,
            inverse_residual,
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        reverse_statistics = _round_trip_error(inverse_field, field, inverse_roi)
        _record_round_trip_statistics(
            diagnostics, "forward_then_inverse", forward_statistics
        )
        _record_round_trip_statistics(
            diagnostics, "inverse_then_forward", reverse_statistics
        )
        # Retain these keys for consumers of the original diagnostics contract.
        forward_error = forward_statistics["valid_anatomical_roi"]["max_mm"]
        reverse_error = reverse_statistics["valid_anatomical_roi"]["max_mm"]
        tolerance = 0.5 * min(domain.GetSpacing())
        diagnostics.update(
            inverse_round_trip_max_mm=forward_error,
            forward_round_trip_max_mm=reverse_error,
            inverse_tolerance_mm=tolerance,
        )
        if max(forward_error, reverse_error) > tolerance:
            raise ValueError("Elastic inverse round-trip error exceeds tolerance")
    finally:
        diagnostics["round_trip_validation_elapsed_seconds"] = (
            time.perf_counter() - round_trip_started
        )
    inverse = sitk.CompositeTransform(3)
    inverse.AddTransform(sitk.DisplacementFieldTransform(inverse_field))
    inverse.AddTransform(linear.GetInverse())
    diagnostics["validation_phase"] = "complete"
    return inverse


@dataclass
class TransformResult:
    reference_to_scan: object
    stage: str
    dice_before: float
    dice_after: float
    warnings: list[str]
    stages: dict = field(default_factory=dict)
    stage_transforms: dict = field(default_factory=dict, repr=False)
    scan_to_reference: object | None = field(default=None, repr=False)
    confidence: str = "ok"
    overlap: dict = field(default_factory=dict)
    organ_volume_ratio: float | None = None


class RegistrationRejected(ValueError):
    """A rejected pair retains stage diagnostics for QC."""

    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


def _select_candidate(
    best,
    score,
    stage,
    candidate,
    candidate_score,
    candidate_stage,
    stages,
    minimum_improvement,
):
    """Accept a candidate only when it clears its minimum Dice improvement."""
    if not np.isfinite(candidate_score):
        raise ValueError("Nonfinite registration Dice")
    detail = stages[candidate_stage]
    improvement = candidate_score - score
    detail.update(
        dice=candidate_score,
        input_dice=score,
        dice_improvement=improvement,
        minimum_required_dice_improvement=minimum_improvement,
        selected=False,
    )
    if candidate_score < score - DICE_TOLERANCE:
        detail.update(status="rejected_worse_dice", fallback_stage=stage)
        return best, score, stage
    if improvement < minimum_improvement - DICE_TOLERANCE:
        detail.update(status="rejected_insufficient_improvement", fallback_stage=stage)
        return best, score, stage
    stages[stage]["selected"] = False
    detail.update(status="evaluated", selected=True)
    return candidate, candidate_score, candidate_stage


def _mi_candidate_rejection(
    candidate_dice,
    dice_guard_reference,
    input_mi,
    candidate_mi,
    minimum_mi_improvement,
    maximum_dice_decrease,
):
    if not np.isfinite(
        [candidate_dice, dice_guard_reference, input_mi, candidate_mi]
    ).all():
        raise ValueError("Nonfinite MI-stage evaluation")
    if candidate_dice < dice_guard_reference - maximum_dice_decrease - DICE_TOLERANCE:
        return "rejected_worse_dice"
    improvement = candidate_mi - input_mi
    if improvement < max(minimum_mi_improvement, MI_TOLERANCE):
        return "rejected_insufficient_mi_improvement"
    return None


def _select_mi_candidate(
    best,
    score,
    stage,
    candidate,
    candidate_dice,
    candidate_stage,
    stages,
    *,
    dice_guard_reference,
    input_mi,
    candidate_mi,
    minimum_mi_improvement,
    maximum_dice_decrease,
    dice_guard_metric,
):
    """Accept an MI improvement only when anatomical Dice stays within tolerance."""
    improvement = candidate_mi - input_mi
    detail = stages[candidate_stage]
    detail.update(
        dice=candidate_dice,
        input_dice=score,
        dice_improvement=candidate_dice - score,
        dice_guard_reference=dice_guard_reference,
        maximum_allowed_dice_decrease=maximum_dice_decrease,
        input_mutual_information=input_mi,
        mutual_information=candidate_mi,
        mutual_information_improvement=improvement,
        minimum_required_mi_improvement=minimum_mi_improvement,
        selection_metric="mattes_mutual_information",
        dice_guard_metric=dice_guard_metric,
        selected=False,
    )
    rejection = _mi_candidate_rejection(
        candidate_dice,
        dice_guard_reference,
        input_mi,
        candidate_mi,
        minimum_mi_improvement,
        maximum_dice_decrease,
    )
    if rejection is not None:
        detail.update(status=rejection, fallback_stage=stage)
        return best, score, stage
    stages[stage]["selected"] = False
    detail.update(status="evaluated", selected=True)
    return candidate, candidate_dice, candidate_stage


def _record_stage_failure(stages, warnings, name, exc, score, fallback):
    warnings.append(f"{name}: {exc}")
    stages[name].update(
        status="failed",
        error=str(exc),
        input_dice=score,
        selected=False,
        fallback_stage=fallback,
    )


def _linear_transform(initial, *, affine):
    sitk = backend()
    if affine:
        transform = sitk.AffineTransform(3)
        transform.SetCenter(initial.GetCenter())
        transform.SetMatrix(initial.GetMatrix())
        transform.SetTranslation(initial.GetTranslation())
    else:
        transform = sitk.Euler3DTransform(initial)
    return transform


def _validate_linear_transform(transform):
    if not np.isfinite(transform.GetParameters()).all():
        raise ValueError("Nonfinite linear transform")
    matrix = np.array(transform.GetMatrix()).reshape(3, 3)
    scales = np.linalg.svd(matrix, compute_uv=False)
    if np.linalg.det(matrix) <= 0 or min(scales) < 0.5 or max(scales) > 2:
        raise ValueError("Implausible linear transform")
    transform.GetInverse()


def _mask_linear_refine(fixed_dm, moving_dm, initial, config, *, affine):
    """Align liver signed-distance maps with a rigid or affine transform."""
    sitk = backend()
    transform = _linear_transform(initial, affine=affine)
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMeanSquares()
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        1.0, 0.001, config.maximum_optimizer_iterations
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(transform, inPlace=True)
    registration.Execute(fixed_dm, moving_dm)
    _validate_linear_transform(transform)
    return transform, registration


def mask_rigid_refine(fixed_dm, moving_dm, initial, config):
    """Rigidly align liver signed-distance maps."""
    return _mask_linear_refine(
        fixed_dm,
        moving_dm,
        initial,
        config,
        affine=False,
    )


def mask_affine_refine(fixed_dm, moving_dm, initial, config):
    """Affinely align liver signed-distance maps."""
    return _mask_linear_refine(
        fixed_dm,
        moving_dm,
        initial,
        config,
        affine=True,
    )


def mi_refine(
    fixed_image,
    moving_image,
    fixed_organ,
    moving_organ,
    initial,
    config,
    *,
    affine=False,
):
    """Refine a linear transform with boundary-band Mattes mutual information."""
    sitk = backend()
    transform = _linear_transform(initial, affine=affine)
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricFixedMask(
        boundary_band(fixed_organ, band_mm=config.organ_boundary_band_half_width_mm)
    )
    registration.SetMetricMovingMask(
        boundary_band(moving_organ, band_mm=config.organ_boundary_band_half_width_mm)
    )
    registration.SetMetricSamplingStrategy(registration.REGULAR)
    registration.SetMetricSamplingPercentage(0.2)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        1.0, 0.001, config.maximum_optimizer_iterations
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(transform, inPlace=True)
    registration.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32),
        sitk.Cast(moving_image, sitk.sitkFloat32),
    )
    _validate_linear_transform(transform)
    return transform, registration


def mutual_information_score(
    fixed_image,
    moving_image,
    fixed_organ,
    moving_organ,
    transform,
    config,
):
    """Return deterministic boundary-band Mattes MI, with higher being better."""
    sitk = backend()
    evaluator = sitk.ImageRegistrationMethod()
    evaluator.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    evaluator.SetMetricFixedMask(
        boundary_band(fixed_organ, band_mm=config.organ_boundary_band_half_width_mm)
    )
    evaluator.SetMetricMovingMask(
        boundary_band(moving_organ, band_mm=config.organ_boundary_band_half_width_mm)
    )
    evaluator.SetInterpolator(sitk.sitkLinear)
    evaluator.SetInitialTransform(transform)
    # SimpleITK minimizes negative Mattes MI. Negate it so positive deltas mean
    # better intensity correspondence in stage diagnostics and selection.
    value = -float(
        evaluator.MetricEvaluate(
            sitk.Cast(fixed_image, sitk.sitkFloat32),
            sitk.Cast(moving_image, sitk.sitkFloat32),
        )
    )
    if not np.isfinite(value):
        raise ValueError("Nonfinite mutual information")
    return value


def mask_elastic_refine(
    fixed_dm, moving_dm, initial, config, optimization_diagnostics=None
):
    """Refine an affine mapping with a coarse signed-distance B-spline."""
    sitk = backend()
    optimization_diagnostics = (
        {} if optimization_diagnostics is None else optimization_diagnostics
    )
    lengths = (np.asarray(fixed_dm.GetSize()) - 1) * np.asarray(fixed_dm.GetSpacing())
    control_spacing = config.elastic_control_point_spacing_mm
    mesh_size = [max(1, int(round(length / control_spacing))) for length in lengths]
    bspline = sitk.BSplineTransformInitializer(fixed_dm, mesh_size, order=3)
    band = float(config.organ_boundary_band_half_width_mm)
    aligned_moving_dm = sitk.Resample(
        moving_dm,
        fixed_dm,
        initial,
        sitk.sitkLinear,
        band,
        sitk.sitkFloat32,
    )
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMeanSquares()
    registration.SetMetricFixedMask(
        sitk.Cast(sitk.Abs(fixed_dm) < band, sitk.sitkUInt8)
    )
    registration.SetMetricMovingMask(
        sitk.Cast(sitk.Abs(aligned_moving_dm) < band, sitk.sitkUInt8)
    )
    registration.SetMetricSamplingStrategy(registration.REGULAR)
    registration.SetMetricSamplingPercentage(0.25)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescentLineSearch(
        learningRate=0.25,
        numberOfIterations=config.elastic_optimizer_iterations,
        convergenceMinimumValue=1e-4,
        convergenceWindowSize=5,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(bspline, inPlace=True)
    qc_domain = _physical_qc_grid(fixed_dm, control_spacing)
    initial_parameters = tuple(bspline.GetParameters())
    best_parameters = initial_parameters
    best_metric = float("inf")
    callback_error = None
    stop_requested = False
    stop_succeeded = False
    checks = 0
    valid_checkpoints = 0
    stop_iteration = None
    stop_reasons = []

    def inspect_residual():
        nonlocal best_parameters, best_metric, callback_error
        nonlocal stop_requested, stop_succeeded, checks, valid_checkpoints
        nonlocal stop_iteration, stop_reasons
        if stop_requested:
            return
        try:
            checks += 1
            coarse_field = _bspline_displacement(bspline, qc_domain)
            qc = _elastic_field_qc(coarse_field, config)
            metric = float(registration.GetMetricValue())
            if qc["deformation_qc_passed"]:
                if np.isfinite(metric) and metric < best_metric:
                    best_metric = metric
                    best_parameters = tuple(bspline.GetParameters())
                    valid_checkpoints += 1
                return
            stop_requested = True
            stop_iteration = int(registration.GetOptimizerIteration())
            stop_reasons = list(qc["deformation_qc_reasons"])
            stop_succeeded = bool(registration.StopRegistration())
        except (RuntimeError, ValueError) as exc:
            callback_error = exc
            stop_requested = True
            stop_iteration = int(registration.GetOptimizerIteration())
            stop_reasons = ["optimizer_qc_callback_failed"]
            stop_succeeded = bool(registration.StopRegistration())

    registration.AddCommand(sitk.sitkIterationEvent, inspect_residual)
    registration.Execute(
        sitk.Cast(fixed_dm, sitk.sitkFloat32),
        aligned_moving_dm,
    )
    # SimpleITK transform mutation is unsafe while Execute is active.
    if stop_requested:
        bspline.SetParameters(best_parameters)
    optimization_diagnostics.update(
        optimizer_qc_grid_size=list(qc_domain.GetSize()),
        optimizer_qc_grid_spacing_mm=list(qc_domain.GetSpacing()),
        optimizer_qc_checks=checks,
        optimizer_valid_checkpoints=valid_checkpoints,
        optimizer_best_valid_metric=(
            None if not np.isfinite(best_metric) else best_metric
        ),
        optimizer_qc_stop_requested=stop_requested,
        optimizer_qc_stop_succeeded=stop_succeeded,
        optimizer_qc_stop_iteration=stop_iteration,
        optimizer_qc_stop_reasons=stop_reasons,
        optimizer_checkpoint_restored=stop_requested,
    )
    if callback_error is not None:
        raise ValueError("Elastic optimizer QC callback failed") from callback_error
    if stop_requested and not stop_succeeded:
        raise RuntimeError("SimpleITK optimizer did not honor callback stop request")
    if not np.isfinite(bspline.GetParameters()).all():
        raise ValueError("Nonfinite elastic transform")
    composite = sitk.CompositeTransform(3)
    composite.AddTransform(initial)
    composite.AddTransform(bspline)
    return composite, registration


def register_pair(
    fixed_organ,
    moving_organ,
    config,
    fixed_image=None,
    moving_image=None,
):
    """Run the staged liver-first registration cascade.

    Every candidate is accepted only when liver Dice improves. Once the selected
    Dice reaches ``early_stop_organ_dice``, later stages are explicitly skipped.
    Images default to the masks for callers using the low-level mask-only API.
    """
    sitk = backend()
    fixed_image = fixed_organ if fixed_image is None else fixed_image
    moving_image = moving_organ if moving_image is None else moving_image
    fixed_qc = assess_organ_mask(fixed_organ, config)
    moving_qc = assess_organ_mask(moving_organ, config)
    if "invalid" in (fixed_qc.status, moving_qc.status):
        raise ValueError("Invalid organ mask for registration")
    partial = fixed_qc.partial or moving_qc.partial
    if partial and not config.accept_partial_organ_masks:
        raise ValueError("Partial organ masks are disabled")

    overlaps = {}

    def score_transform(tx, name):
        if not partial:
            return dice(fixed_organ, moving_organ, tx)
        metrics = overlap_qc(fixed_organ, moving_organ, tx)
        overlaps[name] = metrics
        if (
            metrics["common_fov_fraction"]
            < config.minimum_common_field_of_view_fraction
            or metrics["common_foreground_voxels"] < 4
        ):
            return 0.0
        return metrics["dice_common_fov"]

    identity = sitk.Euler3DTransform()
    before = score_transform(identity, "baseline")
    best, score, stage = identity, before, "baseline"
    peak_dice = before
    warnings = []
    stage_transforms = {"baseline": identity}
    stages = {name: {"dice": None, "status": "not_run"} for name in REGISTRATION_STAGES}
    stages["baseline"].update(dice=before, status="evaluated", selected=True)
    initializer_stage = "geometry" if partial else "pca"
    pipeline = [
        initializer_stage,
        "mask_rigid",
        "mask_affine",
        "mi_affine",
        "mask_elastic",
    ]

    def minimum_stage_improvement(name):
        return config.minimum_stage_dice_improvement[name]

    def minimum_mi_improvement(name):
        return config.minimum_stage_mi_improvement[name]

    dice_metric = "dice_common_fov" if partial else "dice_full"

    if partial:
        stages["pca"].update(status="skipped_partial_coverage", selected=False)
    else:
        stages["geometry"].update(status="skipped_complete_organ", selected=False)

    def reached_target():
        return score >= config.early_stop_organ_dice

    def skip_remaining(after):
        for name in pipeline[pipeline.index(after) + 1 :]:
            if stages[name]["status"] == "not_run":
                stages[name].update(
                    status="skipped_early_stop",
                    input_dice=score,
                    required_dice=config.early_stop_organ_dice,
                    selected=False,
                    fallback_stage=stage,
                )

    if reached_target():
        stages["baseline"]["early_stop"] = True
        for name in pipeline:
            stages[name].update(
                status="skipped_early_stop",
                input_dice=score,
                required_dice=config.early_stop_organ_dice,
                selected=False,
                fallback_stage=stage,
            )

    if not reached_target():
        started = time.perf_counter()
        try:
            if partial:
                initial = sitk.CenteredTransformInitializer(
                    fixed_organ,
                    moving_organ,
                    sitk.Euler3DTransform(),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY,
                )
            else:
                initial = initialize_pca(
                    fixed_organ,
                    moving_organ,
                    config.maximum_pca_rotation_degrees,
                )
            initial_score = score_transform(initial, initializer_stage)
            stage_transforms[initializer_stage] = initial
            best, score, stage = _select_candidate(
                best,
                score,
                stage,
                initial,
                initial_score,
                initializer_stage,
                stages,
                minimum_stage_improvement(initializer_stage),
            )
            peak_dice = max(peak_dice, score)
        except (RuntimeError, ValueError) as exc:
            _record_stage_failure(
                stages, warnings, initializer_stage, exc, score, stage
            )
        finally:
            stages[initializer_stage]["elapsed_seconds"] = time.perf_counter() - started
        if reached_target():
            stages[initializer_stage]["early_stop"] = True
            skip_remaining(initializer_stage)

    fixed_dm = moving_dm = None
    if not reached_target():
        started = time.perf_counter()
        try:
            fixed_dm = distance_map(
                fixed_organ,
                padding_mm=config.distance_map_crop_padding_mm,
                band_mm=config.organ_boundary_band_half_width_mm,
            )
            moving_dm = distance_map(
                moving_organ,
                padding_mm=config.distance_map_crop_padding_mm,
                band_mm=config.organ_boundary_band_half_width_mm,
            )
            tx, reg = mask_rigid_refine(fixed_dm, moving_dm, best, config)
            stages["mask_rigid"].update(
                optimizer_stop=reg.GetOptimizerStopConditionDescription(),
                optimizer_iteration=int(reg.GetOptimizerIteration()),
                metric="mean_squares_signed_distance",
            )
            candidate = score_transform(tx, "mask_rigid")
            stage_transforms["mask_rigid"] = tx
            best, score, stage = _select_candidate(
                best,
                score,
                stage,
                tx,
                candidate,
                "mask_rigid",
                stages,
                minimum_stage_improvement("mask_rigid"),
            )
            peak_dice = max(peak_dice, score)
        except (RuntimeError, ValueError) as exc:
            _record_stage_failure(stages, warnings, "mask_rigid", exc, score, stage)
        finally:
            stages["mask_rigid"]["elapsed_seconds"] = time.perf_counter() - started
        if reached_target():
            stages["mask_rigid"]["early_stop"] = True
            skip_remaining("mask_rigid")

    if not reached_target():
        started = time.perf_counter()
        try:
            if fixed_dm is None:
                fixed_dm = distance_map(
                    fixed_organ,
                    padding_mm=config.distance_map_crop_padding_mm,
                    band_mm=config.organ_boundary_band_half_width_mm,
                )
            if moving_dm is None:
                moving_dm = distance_map(
                    moving_organ,
                    padding_mm=config.distance_map_crop_padding_mm,
                    band_mm=config.organ_boundary_band_half_width_mm,
                )
            tx, reg = mask_affine_refine(fixed_dm, moving_dm, best, config)
            stages["mask_affine"].update(
                optimizer_stop=reg.GetOptimizerStopConditionDescription(),
                optimizer_iteration=int(reg.GetOptimizerIteration()),
                metric="mean_squares_signed_distance",
            )
            candidate = score_transform(tx, "mask_affine")
            stage_transforms["mask_affine"] = tx
            best, score, stage = _select_candidate(
                best,
                score,
                stage,
                tx,
                candidate,
                "mask_affine",
                stages,
                minimum_stage_improvement("mask_affine"),
            )
            peak_dice = max(peak_dice, score)
        except (RuntimeError, ValueError) as exc:
            _record_stage_failure(stages, warnings, "mask_affine", exc, score, stage)
        finally:
            stages["mask_affine"]["elapsed_seconds"] = time.perf_counter() - started
        if reached_target():
            stages["mask_affine"]["early_stop"] = True
            skip_remaining("mask_affine")

    # Temporarily disable MI-affine while retaining its implementation below.
    if stages["mi_affine"]["status"] == "not_run":
        stages["mi_affine"].update(
            status="skipped_disabled",
            input_dice=score,
            selected=False,
            fallback_stage=stage,
        )
    elif stages["mi_affine"]["status"] == "not_run":
        started = time.perf_counter()
        try:
            input_mi = mutual_information_score(
                fixed_image,
                moving_image,
                fixed_organ,
                moving_organ,
                best,
                config,
            )
            tx, reg = mi_refine(
                fixed_image,
                moving_image,
                fixed_organ,
                moving_organ,
                best,
                config,
                affine=True,
            )
            stages["mi_affine"].update(
                optimizer_stop=reg.GetOptimizerStopConditionDescription(),
                optimizer_iteration=int(reg.GetOptimizerIteration()),
                metric="mattes_mutual_information",
            )
            candidate = score_transform(tx, "mi_affine")
            candidate_mi = mutual_information_score(
                fixed_image,
                moving_image,
                fixed_organ,
                moving_organ,
                tx,
                config,
            )
            stage_transforms["mi_affine"] = tx
            best, score, stage = _select_mi_candidate(
                best,
                score,
                stage,
                tx,
                candidate,
                "mi_affine",
                stages,
                dice_guard_reference=peak_dice,
                input_mi=input_mi,
                candidate_mi=candidate_mi,
                minimum_mi_improvement=minimum_mi_improvement("mi_affine"),
                maximum_dice_decrease=config.maximum_mi_stage_dice_decrease,
                dice_guard_metric=dice_metric,
            )
            peak_dice = max(peak_dice, score)
        except (RuntimeError, ValueError) as exc:
            _record_stage_failure(stages, warnings, "mi_affine", exc, score, stage)
        finally:
            stages["mi_affine"]["elapsed_seconds"] = time.perf_counter() - started
        if reached_target():
            stages["mi_affine"]["early_stop"] = True
            skip_remaining("mi_affine")

    selected_inverse = None
    if (
        stages["mask_elastic"]["status"] == "not_run"
        and not config.enable_elastic_stage
    ):
        stages["mask_elastic"].update(
            status="skipped_disabled",
            input_dice=score,
            selected=False,
            fallback_stage=stage,
        )
    elif stages["mask_elastic"]["status"] == "not_run":
        started = time.perf_counter()
        detail = stages["mask_elastic"]
        detail.update(
            optimization_elapsed_seconds=0.0,
            field_qc_elapsed_seconds=0.0,
            inversion_elapsed_seconds=0.0,
            round_trip_validation_elapsed_seconds=0.0,
        )
        try:
            input_mi = mutual_information_score(
                fixed_image,
                moving_image,
                fixed_organ,
                moving_organ,
                best,
                config,
            )
            if fixed_dm is None:
                fixed_dm = distance_map(
                    fixed_organ,
                    padding_mm=config.distance_map_crop_padding_mm,
                    band_mm=config.organ_boundary_band_half_width_mm,
                )
            if moving_dm is None:
                moving_dm = distance_map(
                    moving_organ,
                    padding_mm=config.distance_map_crop_padding_mm,
                    band_mm=config.organ_boundary_band_half_width_mm,
                )
            optimization_diagnostics = {}
            optimization_started = time.perf_counter()
            try:
                tx, reg = mask_elastic_refine(
                    fixed_dm,
                    moving_dm,
                    best,
                    config,
                    optimization_diagnostics,
                )
            finally:
                detail["optimization_elapsed_seconds"] = (
                    time.perf_counter() - optimization_started
                )
                detail.update(optimization_diagnostics)
            candidate = score_transform(tx, "mask_elastic")
            candidate_mi = mutual_information_score(
                fixed_image,
                moving_image,
                fixed_organ,
                moving_organ,
                tx,
                config,
            )
            stage_transforms["mask_elastic"] = tx
            detail.update(
                optimizer_stop=reg.GetOptimizerStopConditionDescription(),
                optimizer_iteration=int(reg.GetOptimizerIteration()),
                optimizer_iteration_limit=config.elastic_optimizer_iterations,
                metric="mean_squares_signed_distance",
                algorithm="BSplineTransform",
                control_point_spacing_mm=config.elastic_control_point_spacing_mm,
                boundary_band_half_width_mm=(config.organ_boundary_band_half_width_mm),
                input_mutual_information=input_mi,
                mutual_information=candidate_mi,
                mutual_information_improvement=candidate_mi - input_mi,
            )
            field_qc_started = time.perf_counter()
            try:
                field, deformation_qc = elastic_deformation_qc(tx, fixed_dm, config)
            finally:
                detail["field_qc_elapsed_seconds"] = (
                    time.perf_counter() - field_qc_started
                )
            detail.update(deformation_qc)
            minimum_improvement = minimum_stage_improvement("mask_elastic")
            if deformation_qc["deformation_qc_passed"]:
                improvement = candidate - score
                if (
                    np.isfinite(candidate)
                    and candidate >= score - DICE_TOLERANCE
                    and improvement >= minimum_improvement - DICE_TOLERANCE
                ):
                    selected_inverse = invert_mask_elastic(
                        tx,
                        fixed_dm,
                        detail,
                        field,
                        sitk.Cast(fixed_dm <= 0, sitk.sitkUInt8),
                    )
                best, score, stage = _select_candidate(
                    best,
                    score,
                    stage,
                    tx,
                    candidate,
                    "mask_elastic",
                    stages,
                    minimum_improvement,
                )
            else:
                if not np.isfinite(candidate):
                    raise ValueError("Nonfinite registration Dice")
                detail.update(
                    dice=candidate,
                    input_dice=score,
                    dice_improvement=candidate - score,
                    minimum_required_dice_improvement=minimum_improvement,
                    selection_metric=dice_metric,
                    status="rejected_deformation_qc",
                    selected=False,
                    fallback_stage=stage,
                )
            peak_dice = max(peak_dice, score)
        except (RuntimeError, ValueError) as exc:
            selected_inverse = None
            _record_stage_failure(
                stages,
                warnings,
                "mask_elastic",
                exc,
                score,
                stage,
            )
        finally:
            stages["mask_elastic"]["elapsed_seconds"] = time.perf_counter() - started
    selected_stage = "identity" if stage == "baseline" else stage
    result = TransformResult(
        best,
        selected_stage,
        before,
        score,
        warnings,
        stages,
        stage_transforms,
        selected_inverse if selected_inverse is not None else best.GetInverse(),
    )
    for name, tx in stage_transforms.items():
        if name not in overlaps:
            overlaps[name] = overlap_qc(fixed_organ, moving_organ, tx)
        stages[name].update(overlaps[name])
        stages[name].setdefault("selection_metric", dice_metric)
    result.overlap = overlaps[stage]
    result.organ_volume_ratio = fixed_qc.volume_mm3 / moving_qc.volume_mm3
    result.confidence = registration_confidence(result.overlap, partial, config)
    return result
