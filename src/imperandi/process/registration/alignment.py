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
    labels = sitk.LabelShapeStatisticsImageFilter()
    labels.Execute(mask)
    if not labels.HasLabel(1):
        raise ValueError("Cannot register an empty organ mask")
    bounds = labels.GetBoundingBox(1)
    start = np.array(bounds[:3], dtype=int)
    stop = start + np.array(bounds[3:], dtype=int)
    padding = np.ceil(float(padding_mm) / np.array(mask.GetSpacing())).astype(int)
    start = np.maximum(0, start - padding)
    stop = np.minimum(np.array(mask.GetSize()), stop + padding)
    cropped = sitk.RegionOfInterest(
        mask,
        [int(value) for value in stop - start],
        [int(value) for value in start],
    )
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


def elastic_refine(fixed_dm, moving_dm, initial, config, *, smoothing_sigma_mm=1.0):
    """Refine a fixed-to-moving mapping with Diffeomorphic Demons.

    SimpleITK resampling transforms map output points to input points. The
    returned composite therefore maps the reference (``fixed``) space into the
    scan (``moving``) space.
    """
    sitk = backend()
    # Demons requires a shared grid. Prewarp by the selected linear transform
    # and estimate a residual deformation in reference space. Outside the
    # moving crop, use the positive distance band (background), not zero.
    aligned_dm = sitk.Resample(
        moving_dm,
        fixed_dm,
        initial,
        sitk.sitkLinear,
        float(config.organ_boundary_band_half_width_mm),
        sitk.sitkFloat32,
    )
    registration = sitk.DiffeomorphicDemonsRegistrationFilter()
    registration.SetNumberOfIterations(config.maximum_optimizer_iterations)
    registration.SetSmoothDisplacementField(True)
    registration.SetStandardDeviations(
        [smoothing_sigma_mm / spacing for spacing in fixed_dm.GetSpacing()]
    )
    field = registration.Execute(sitk.Cast(fixed_dm, sitk.sitkFloat32), aligned_dm)
    if not np.isfinite(sitk.GetArrayViewFromImage(field)).all():
        raise ValueError("Nonfinite elastic displacement field")
    if not np.isfinite(initial.GetParameters()).all():
        raise ValueError("Nonfinite linear transform")
    field = _extend_residual(field)

    composite = sitk.CompositeTransform(3)
    composite.AddTransform(initial)
    # Composite transforms apply the last transform first: initial(residual(p)).
    composite.AddTransform(sitk.DisplacementFieldTransform(field))
    return composite, registration


def _extend_residual(field):
    """Keep the crop unchanged and add a smooth transition to zero outside it.

    Displacement transforms are identity outside their domain. Extending the
    edge values into a tapered collar avoids a jump at a cropped field's edge.
    The resulting field must still pass the Jacobian and inverse checks.
    """
    sitk = backend()
    values = sitk.GetArrayViewFromImage(field)
    displacement = float(np.linalg.norm(values, axis=-1).max())
    spacing = np.asarray(field.GetSpacing())
    width_mm = max(3 * displacement, 2 * max(spacing))
    padding = np.ceil(width_mm / spacing).astype(int)
    padded = np.pad(values, [(p, p) for p in padding[::-1]] + [(0, 0)], mode="edge")
    for axis, width in enumerate(padding[::-1]):
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(width) / width)
        weights = np.ones(padded.shape[axis])
        weights[:width], weights[-width:] = ramp, ramp[::-1]
        shape = [1] * 4
        shape[axis] = len(weights)
        padded *= weights.reshape(shape)
    extended = sitk.GetImageFromArray(padded, isVector=True)
    extended.SetSpacing(field.GetSpacing())
    extended.SetDirection(field.GetDirection())
    extended.SetOrigin(field.TransformIndexToPhysicalPoint([-int(p) for p in padding]))
    return extended


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


def _round_trip_error(field, inverse_field):
    """Maximum residual composition error in mm at field voxel centers."""
    sitk = backend()
    residual = sitk.DisplacementFieldTransform(sitk.Image(field))
    warped = sitk.Resample(
        inverse_field, field, residual, sitk.sitkLinear, 0, sitk.sitkVectorFloat64
    )
    error = sitk.GetArrayFromImage(warped)
    error += sitk.GetArrayViewFromImage(field)
    return float(np.linalg.norm(error, axis=-1).max())


def invert_elastic(transform, fixed, diagnostics=None):
    """Validate and invert the linear-plus-Demons reference-to-scan mapping."""
    sitk = backend()

    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics["validation_phase"] = "jacobian"
    linear = transform.GetNthTransform(0)
    residual = sitk.DisplacementFieldTransform(transform.GetNthTransform(1))
    field = residual.GetDisplacementField()
    matrix = np.asarray(linear.GetMatrix()).reshape(3, 3)
    determinant = float(np.linalg.det(matrix))
    if (
        not np.isfinite(linear.GetParameters()).all()
        or not np.isfinite(determinant)
        or determinant <= 0
    ):
        raise ValueError("Invalid elastic linear component")
    values = sitk.GetArrayViewFromImage(field)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite elastic displacement field")
    for axis in range(3):
        if np.any(np.take(values, [0, -1], axis=axis) != 0):
            raise ValueError(
                "Elastic field does not transition to identity at its boundary"
            )
    # det(J(linear o residual)) = det(linear) * det(J(residual)). Avoid
    # differentiating a sampled rigid rotation at image borders.
    jacobian = determinant * _residual_jacobian(field)
    diagnostics.update(
        jacobian_min=float(jacobian.min()),
        jacobian_nonpositive_voxels=int(np.count_nonzero(jacobian <= 0)),
    )
    if not np.isfinite(jacobian).all() or diagnostics["jacobian_nonpositive_voxels"]:
        raise ValueError("Elastic transform contains a fold")
    del jacobian, values
    # Invert the residual on its own grid, preserving its direction cosines,
    # and invert the linear part exactly. For T = linear(residual(p)),
    # T^-1 = residual^-1(linear^-1(p)).
    diagnostics["validation_phase"] = "inversion"
    inverter = sitk.InvertDisplacementFieldImageFilter()
    inverter.SetMaximumNumberOfIterations(100)
    inverter.SetMeanErrorToleranceThreshold(0.0)
    inverter.SetMaxErrorToleranceThreshold(0.01)
    inverter.SetEnforceBoundaryCondition(True)
    inverse_field = inverter.Execute(field)
    if not np.isfinite(sitk.GetArrayViewFromImage(inverse_field)).all():
        raise ValueError("Nonfinite inverse elastic displacement field")
    diagnostics.update(
        inverse_max_error_norm=float(inverter.GetMaxErrorNorm()),
        inverse_mean_error_norm=float(inverter.GetMeanErrorNorm()),
        validation_phase="round_trip",
    )
    # Check both compositions rather than accepting a finite but inaccurate
    # inverse. Limit error to half the smallest reference voxel, accounting
    # conservatively for amplification by the linear transform in scan space.
    tolerance = 0.5 * min(fixed.GetSpacing())
    forward_error = _round_trip_error(field, inverse_field)
    reverse_error = _round_trip_error(inverse_field, field)
    reverse_error *= float(np.linalg.svd(matrix, compute_uv=False).max())
    diagnostics.update(
        inverse_round_trip_max_mm=forward_error,
        forward_round_trip_max_mm=reverse_error,
        inverse_tolerance_mm=tolerance,
    )
    if (
        not np.isfinite([forward_error, reverse_error]).all()
        or max(forward_error, reverse_error) > tolerance
    ):
        raise ValueError("Elastic inverse round-trip error exceeds tolerance")
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


def mask_rigid_refine(fixed_dm, moving_dm, initial, config):
    """Rigidly align liver signed-distance maps."""
    sitk = backend()
    transform = _linear_transform(initial, affine=False)
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMeanSquares()
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        1.0, 0.001, config.maximum_optimizer_iterations
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([2, 1])
    registration.SetSmoothingSigmasPerLevel([1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(transform, inPlace=True)
    registration.Execute(fixed_dm, moving_dm)
    _validate_linear_transform(transform)
    return transform, registration


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
    registration.SetShrinkFactorsPerLevel([2, 1])
    registration.SetSmoothingSigmasPerLevel([1, 0])
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


def mi_elastic_refine(
    fixed_image,
    moving_image,
    fixed_organ,
    moving_organ,
    fixed_domain,
    initial,
    config,
):
    """Refine a fixed-to-moving mapping with boundary-band MI and a B-spline.

    The optimized B-spline is sampled as a residual displacement field so the
    same boundary taper, topology validation, and numerical inversion used by
    the elastic transform contract apply before the candidate can be selected.
    """
    sitk = backend()
    lengths = (np.asarray(fixed_domain.GetSize()) - 1) * np.asarray(
        fixed_domain.GetSpacing()
    )
    control_spacing = config.elastic_control_point_spacing_mm
    mesh_size = [max(1, int(round(length / control_spacing))) for length in lengths]
    bspline = sitk.BSplineTransformInitializer(fixed_domain, mesh_size, order=3)
    # Optimize the residual in fixed space. Prewarping avoids exposing the
    # preceding linear stage as the registration metric's moving transform,
    # which some ITK builds reject when optimizing a local-support transform.
    aligned_moving_image = resample(moving_image, fixed_image, initial, label=False)
    aligned_moving_organ = resample(moving_organ, fixed_organ, initial)
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricFixedMask(
        boundary_band(fixed_organ, band_mm=config.organ_boundary_band_half_width_mm)
    )
    registration.SetMetricMovingMask(
        boundary_band(
            aligned_moving_organ,
            band_mm=config.organ_boundary_band_half_width_mm,
        )
    )
    registration.SetMetricSamplingStrategy(registration.REGULAR)
    registration.SetMetricSamplingPercentage(0.2)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1.0,
        numberOfIterations=config.maximum_optimizer_iterations,
        convergenceMinimumValue=1e-3,
        convergenceWindowSize=5,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([2, 1])
    registration.SetSmoothingSigmasPerLevel([1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(bspline, inPlace=True)
    registration.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32),
        aligned_moving_image,
    )
    if not np.isfinite(bspline.GetParameters()).all():
        raise ValueError("Nonfinite elastic transform")
    field = sitk.TransformToDisplacementField(
        bspline,
        sitk.sitkVectorFloat64,
        fixed_domain.GetSize(),
        fixed_domain.GetOrigin(),
        fixed_domain.GetSpacing(),
        fixed_domain.GetDirection(),
    )
    field = _extend_residual(field)
    composite = sitk.CompositeTransform(3)
    composite.AddTransform(initial)
    composite.AddTransform(sitk.DisplacementFieldTransform(field))
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
        "mi_affine",
        "mi_elastic",
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

    if stages["mi_affine"]["status"] == "not_run" and not config.enable_affine_stage:
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
    if stages["mi_elastic"]["status"] == "not_run" and not config.enable_elastic_stage:
        stages["mi_elastic"].update(
            status="skipped_disabled",
            input_dice=score,
            selected=False,
            fallback_stage=stage,
        )
    elif stages["mi_elastic"]["status"] == "not_run":
        started = time.perf_counter()
        try:
            if fixed_dm is None:
                fixed_dm = distance_map(
                    fixed_organ,
                    padding_mm=config.distance_map_crop_padding_mm,
                    band_mm=config.organ_boundary_band_half_width_mm,
                )
            input_mi = mutual_information_score(
                fixed_image,
                moving_image,
                fixed_organ,
                moving_organ,
                best,
                config,
            )
            tx, reg = mi_elastic_refine(
                fixed_image,
                moving_image,
                fixed_organ,
                moving_organ,
                fixed_dm,
                best,
                config,
            )
            candidate = score_transform(tx, "mi_elastic")
            candidate_mi = mutual_information_score(
                fixed_image,
                moving_image,
                fixed_organ,
                moving_organ,
                tx,
                config,
            )
            stage_transforms["mi_elastic"] = tx
            stages["mi_elastic"].update(
                optimizer_stop=reg.GetOptimizerStopConditionDescription(),
                optimizer_iteration=int(reg.GetOptimizerIteration()),
                metric="mattes_mutual_information",
                algorithm="BSplineTransform",
            )
            if (
                _mi_candidate_rejection(
                    candidate,
                    peak_dice,
                    input_mi,
                    candidate_mi,
                    minimum_mi_improvement("mi_elastic"),
                    config.maximum_mi_stage_dice_decrease,
                )
                is None
            ):
                selected_inverse = invert_elastic(tx, fixed_organ, stages["mi_elastic"])
            best, score, stage = _select_mi_candidate(
                best,
                score,
                stage,
                tx,
                candidate,
                "mi_elastic",
                stages,
                dice_guard_reference=peak_dice,
                input_mi=input_mi,
                candidate_mi=candidate_mi,
                minimum_mi_improvement=minimum_mi_improvement("mi_elastic"),
                maximum_dice_decrease=config.maximum_mi_stage_dice_decrease,
                dice_guard_metric=dice_metric,
            )
            peak_dice = max(peak_dice, score)
        except (RuntimeError, ValueError) as exc:
            selected_inverse = None
            _record_stage_failure(stages, warnings, "mi_elastic", exc, score, stage)
        finally:
            stages["mi_elastic"]["elapsed_seconds"] = time.perf_counter() - started
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
