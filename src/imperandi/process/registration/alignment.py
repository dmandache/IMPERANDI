"""Physical-space organ registration, independent of tumor masks."""

from dataclasses import dataclass, field
import time
from itertools import permutations, product
import numpy as np
from scipy import ndimage

from .config import REGISTRATION_STAGES

DICE_TOLERANCE = 1e-6


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
    low = start * spacing <= config.boundary_margin_mm
    high = (np.asarray(mask.GetSize()) - stop) * spacing <= config.boundary_margin_mm
    contact = tuple((bool(a), bool(b)) for a, b in zip(low, high))
    labels, _ = ndimage.label(values)  # Conservative face-connected components.
    sizes = np.bincount(labels.ravel())[1:]
    fraction = float(sizes.max() / count)
    reasons = []
    if count < 4:
        reasons.append("insufficient_foreground")
    if fraction < config.min_largest_component_fraction:
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
        or score < config.min_confidence_dice
        or metrics["common_fov_fraction"] < config.min_common_fov_fraction
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


def initialize_pca(fixed, moving):
    """Return reference-to-moving rigid mapping; score proper PCA rotations."""
    sitk = backend()
    cf, vf, af = _moments(fixed)
    cm, vm, am = _moments(moving)
    rotations = [np.eye(3)]
    # Nearly equal eigenvalues do not establish a stable anatomical orientation.
    if all(np.min(np.diff(v)) > 0.01 * max(v[-1], 1e-8) for v in (vf, vm)):
        for perm in permutations(range(3)):
            for signs in product((-1, 1), repeat=3):
                rotation = (am[:, perm] * signs) @ af.T
                if np.linalg.det(rotation) > 0:
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


def elastic_refine(fixed_dm, moving_dm, initial, config):
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
        float(config.distance_band_mm),
        sitk.sitkFloat32,
    )
    registration = sitk.DiffeomorphicDemonsRegistrationFilter()
    registration.SetNumberOfIterations(config.iterations)
    registration.SetSmoothDisplacementField(True)
    registration.SetStandardDeviations(
        [
            config.demons_smoothing_sigma_mm / spacing
            for spacing in fixed_dm.GetSpacing()
        ]
    )
    field = registration.Execute(sitk.Cast(fixed_dm, sitk.sitkFloat32), aligned_dm)
    if not np.isfinite(sitk.GetArrayViewFromImage(field)).all():
        raise ValueError("Nonfinite elastic displacement field")

    composite = sitk.CompositeTransform(3)
    composite.AddTransform(initial)
    # Composite transforms apply the last transform first: initial(residual(p)).
    composite.AddTransform(sitk.DisplacementFieldTransform(field))
    if not np.isfinite(composite.GetParameters()).all():
        raise ValueError("Nonfinite elastic transform")
    return composite, registration


def invert_elastic(transform, fixed):
    """Validate and invert the linear-plus-Demons reference-to-scan mapping."""
    sitk = backend()

    # Validate topology before accepting the deformation. Folded fields cannot
    # be safely inverted and can create anatomically invalid transferred masks.
    field = sitk.TransformToDisplacementField(
        transform,
        sitk.sitkVectorFloat64,
        fixed.GetSize(),
        fixed.GetOrigin(),
        fixed.GetSpacing(),
        fixed.GetDirection(),
    )
    jacobian = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(field))
    if not np.isfinite(jacobian).all() or np.min(jacobian) <= 0:
        raise ValueError("Elastic transform contains a fold")
    # Invert the residual on its own grid, preserving its direction cosines,
    # and invert the linear part exactly. For T = linear(residual(p)),
    # T^-1 = residual^-1(linear^-1(p)).
    linear = transform.GetNthTransform(0)
    residual = sitk.DisplacementFieldTransform(transform.GetNthTransform(1))
    inverse_field = sitk.InvertDisplacementField(
        residual.GetDisplacementField(),
        maximumNumberOfIterations=100,
        enforceBoundaryCondition=False,
    )
    if not np.isfinite(sitk.GetArrayViewFromImage(inverse_field)).all():
        raise ValueError("Nonfinite inverse elastic displacement field")
    inverse = sitk.CompositeTransform(3)
    inverse.AddTransform(sitk.DisplacementFieldTransform(inverse_field))
    inverse.AddTransform(linear.GetInverse())
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
    best, score, stage, candidate, candidate_score, candidate_stage, stages
):
    """Accept improvement; otherwise retain the previous transform and annotate QC."""
    if not np.isfinite(candidate_score):
        raise ValueError("Nonfinite registration Dice")
    detail = stages[candidate_stage]
    detail.update(dice=candidate_score, input_dice=score, selected=False)
    if candidate_score < score - DICE_TOLERANCE:
        detail.update(status="rejected_worse_dice", fallback_stage=stage)
        return best, score, stage
    if candidate_score <= score + DICE_TOLERANCE:
        detail.update(status="rejected_no_improvement", fallback_stage=stage)
        return best, score, stage
    stages[stage]["selected"] = False
    detail.update(status="evaluated", selected=True)
    return candidate, candidate_score, candidate_stage


def _record_stage_failure(stages, warnings, name, exc, score, fallback):
    warnings.append(f"{name}: {exc}")
    stages[name].update(
        status="failed",
        error=str(exc),
        input_dice=score,
        selected=False,
        fallback_stage=fallback,
    )


def linear_refine(fixed_dm, moving_dm, initial, config, *, affine=False):
    """Optimize a copy of the selected linear transform and validate its geometry."""
    sitk = backend()
    if affine:
        transform = sitk.AffineTransform(3)
        transform.SetCenter(initial.GetCenter())
        transform.SetMatrix(initial.GetMatrix())
        transform.SetTranslation(initial.GetTranslation())
    else:
        transform = sitk.Euler3DTransform(initial)
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMeanSquares()
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(1.0, 0.001, config.iterations)
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([2, 1])
    registration.SetSmoothingSigmasPerLevel([1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(transform, inPlace=True)
    registration.Execute(fixed_dm, moving_dm)
    if not np.isfinite(transform.GetParameters()).all():
        raise ValueError("Nonfinite linear transform")
    matrix = np.array(transform.GetMatrix()).reshape(3, 3)
    scales = np.linalg.svd(matrix, compute_uv=False)
    if np.linalg.det(matrix) <= 0 or min(scales) < 0.5 or max(scales) > 2:
        raise ValueError("Implausible linear transform")
    transform.GetInverse()
    return transform, registration


def register_pair(fixed_organ, moving_organ, config):
    sitk = backend()
    fixed_qc = assess_organ_mask(fixed_organ, config)
    moving_qc = assess_organ_mask(moving_organ, config)
    if "invalid" in (fixed_qc.status, moving_qc.status):
        raise ValueError("Invalid organ mask for registration")
    partial = fixed_qc.partial or moving_qc.partial
    if partial and not config.allow_partial_organs:
        raise ValueError("Partial organ masks are disabled")

    overlaps = {}

    def score_transform(tx, name):
        if not partial:
            return dice(fixed_organ, moving_organ, tx)
        metrics = overlap_qc(fixed_organ, moving_organ, tx)
        overlaps[name] = metrics
        if (
            metrics["common_fov_fraction"] < config.min_common_fov_fraction
            or metrics["common_foreground_voxels"] < 4
        ):
            return 0.0
        return metrics["dice_common_fov"]

    identity = sitk.Euler3DTransform()
    before = score_transform(identity, "baseline")
    best, score, stage = identity, before, "baseline"
    warnings = []
    stage_transforms = {"baseline": identity}
    stages = {name: {"dice": None, "status": "not_run"} for name in REGISTRATION_STAGES}
    stages["baseline"].update(dice=before, status="evaluated", selected=True)
    if partial:
        stages["pca"].update(status="skipped_partial_coverage", selected=False)
    for initial_stage in ["geometry"] if partial else ["geometry", "pca"]:
        started = time.perf_counter()
        try:
            if initial_stage == "geometry":
                initial = sitk.CenteredTransformInitializer(
                    fixed_organ,
                    moving_organ,
                    sitk.Euler3DTransform(),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY,
                )
            else:
                initial = initialize_pca(fixed_organ, moving_organ)
            initial_score = score_transform(initial, initial_stage)
            stage_transforms[initial_stage] = initial
            best, score, stage = _select_candidate(
                best, score, stage, initial, initial_score, initial_stage, stages
            )
        except (RuntimeError, ValueError) as exc:
            _record_stage_failure(stages, warnings, initial_stage, exc, score, stage)
        finally:
            stages[initial_stage]["elapsed_seconds"] = time.perf_counter() - started
    fixed_dm = distance_map(
        fixed_organ,
        padding_mm=config.crop_padding_mm,
        band_mm=config.distance_band_mm,
    )
    moving_dm = distance_map(
        moving_organ,
        padding_mm=config.crop_padding_mm,
        band_mm=config.distance_band_mm,
    )
    for name in ["rigid", "affine"] if config.affine else ["rigid"]:
        if name == "affine" and score < config.affine_min_dice:
            stages[name].update(
                status="skipped_low_dice",
                input_dice=score,
                required_dice=config.affine_min_dice,
                selected=False,
                fallback_stage=stage,
                reason=(
                    f"Best pre-affine Dice {score:.4f} is below the affine "
                    f"threshold {config.affine_min_dice:.4f}"
                ),
            )
            continue
        started = time.perf_counter()
        try:
            tx, reg = linear_refine(
                fixed_dm, moving_dm, best, config, affine=name == "affine"
            )
            stages[name].update(
                optimizer_stop=reg.GetOptimizerStopConditionDescription(),
                optimizer_iteration=int(reg.GetOptimizerIteration()),
            )
            candidate = score_transform(tx, name)
            stage_transforms[name] = tx
            best, score, stage = _select_candidate(
                best, score, stage, tx, candidate, name, stages
            )
        except (RuntimeError, ValueError) as exc:
            _record_stage_failure(stages, warnings, name, exc, score, stage)
        finally:
            stages[name]["elapsed_seconds"] = time.perf_counter() - started
    selected_inverse = None
    if config.elastic and score < config.elastic_min_dice:
        stages["elastic"].update(
            status="skipped_low_dice",
            input_dice=score,
            required_dice=config.elastic_min_dice,
            selected=False,
            fallback_stage=stage,
        )
    elif config.elastic:
        started = time.perf_counter()
        try:
            candidate, registration = elastic_refine(fixed_dm, moving_dm, best, config)
            candidate_score = score_transform(candidate, "elastic")
            stage_transforms["elastic"] = candidate
            stages["elastic"].update(
                dice=candidate_score,
                input_dice=score,
                selected=False,
                optimizer_stop=(
                    "maximum_iterations"
                    if registration.GetElapsedIterations() >= config.iterations
                    else "rms_convergence"
                ),
                optimizer_iteration=int(registration.GetElapsedIterations()),
                metric=float(registration.GetMetric()),
                rms_change=float(registration.GetRMSChange()),
                algorithm="DiffeomorphicDemonsRegistrationFilter",
            )
            candidate_inverse = None
            if candidate_score > score + DICE_TOLERANCE:
                candidate_inverse = invert_elastic(candidate, fixed_organ)
            best, score, stage = _select_candidate(
                best, score, stage, candidate, candidate_score, "elastic", stages
            )
            if stage == "elastic":
                selected_inverse = candidate_inverse
        except (RuntimeError, ValueError) as exc:
            _record_stage_failure(stages, warnings, "elastic", exc, score, stage)
        finally:
            stages["elastic"]["elapsed_seconds"] = time.perf_counter() - started
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
        stages[name]["selection_metric"] = "dice_common_fov" if partial else "dice_full"
    result.overlap = overlaps[stage]
    result.organ_volume_ratio = fixed_qc.volume_mm3 / moving_qc.volume_mm3
    result.confidence = registration_confidence(result.overlap, partial, config)
    if score < config.min_dice:
        result.confidence = "failed"
        raise RegistrationRejected(f"Organ overlap rejected: Dice={score:.4f}", result)
    return result
