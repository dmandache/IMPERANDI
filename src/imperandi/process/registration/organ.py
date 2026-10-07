"""Physical-space organ registration, independent of tumor masks."""

from dataclasses import dataclass, field
import time
from itertools import permutations, product
import numpy as np

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
    return sitk.Clamp(
        distances, lowerBound=-float(band_mm), upperBound=float(band_mm)
    )


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


def elastic_refine(fixed, moving, fixed_dm, moving_dm, initial, config):
    """Refine a fixed-to-moving mapping with a cubic B-spline deformation.

    SimpleITK resampling transforms map output points to input points. The
    returned composite therefore maps the reference (``fixed``) space into the
    scan (``moving``) space.
    """
    sitk = backend()
    physical_lengths = [
        max(1.0, (size - 1) * spacing)
        for size, spacing in zip(fixed_dm.GetSize(), fixed_dm.GetSpacing())
    ]
    mesh_size = [
        max(1, int(round(length / config.bspline_ctrl_spacing_mm)))
        for length in physical_lengths
    ]
    bspline = sitk.BSplineTransformInitializer(fixed_dm, mesh_size, order=3)
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMeanSquares()
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1.0,
        numberOfIterations=config.iterations,
        convergenceMinimumValue=1e-3,
        convergenceWindowSize=5,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([2, 1])
    registration.SetSmoothingSigmasPerLevel([1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetMovingInitialTransform(initial)
    registration.SetInitialTransform(bspline, inPlace=True)
    registration.Execute(fixed_dm, moving_dm)

    composite = sitk.CompositeTransform(3)
    composite.AddTransform(initial)
    composite.AddTransform(bspline)
    if not np.isfinite(composite.GetParameters()).all():
        raise ValueError("Nonfinite elastic transform")
    return composite, registration


def invert_elastic(transform, fixed, moving):
    """Validate and numerically invert an elastic transform on the moving grid."""
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
    jacobian = sitk.GetArrayViewFromImage(
        sitk.DisplacementFieldJacobianDeterminant(field)
    )
    if not np.isfinite(jacobian).all() or np.min(jacobian) <= 0:
        raise ValueError("Elastic transform contains a fold")
    inverse_filter = sitk.InverseDisplacementFieldImageFilter()
    inverse_filter.SetReferenceImage(moving)
    inverse_filter.SetSubsamplingFactor(4)
    inverse_field = inverse_filter.Execute(field)
    return sitk.DisplacementFieldTransform(inverse_field)


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


class RegistrationRejected(ValueError):
    """A rejected pair retains stage diagnostics for QC."""

    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


def _select_candidate(
    best, score, stage, candidate, candidate_score, candidate_stage, stages
):
    """Accept improvement; otherwise retain the previous transform and annotate QC."""
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


def register_pair(fixed_organ, moving_organ, config):
    sitk = backend()
    identity = sitk.Euler3DTransform()
    before = dice(fixed_organ, moving_organ, identity)
    best, score, stage = identity, before, "baseline"
    warnings = []
    stage_transforms = {"baseline": identity}
    stages = {
        "baseline": {"dice": before, "status": "evaluated", "selected": True},
        "pca": {"dice": None, "status": "not_run"},
        "rigid": {"dice": None, "status": "not_run"},
        "affine": {"dice": None, "status": "not_run"},
        "elastic": {"dice": None, "status": "not_run"},
    }
    started_pca = time.perf_counter()
    try:
        initial = initialize_pca(fixed_organ, moving_organ)
    except (RuntimeError, ValueError) as exc:
        warnings.append(f"pca: {exc}")
        stages["pca"].update(
            status="failed",
            selected=False,
            fallback_stage="baseline",
            error=str(exc),
            elapsed_seconds=time.perf_counter() - started_pca,
        )
    else:
        pca_score = dice(fixed_organ, moving_organ, initial)
        stages["pca"].update(
            dice=pca_score,
            status="evaluated",
            selected=False,
            elapsed_seconds=time.perf_counter() - started_pca,
        )
        stage_transforms["pca"] = initial
        best, score, stage = _select_candidate(
            identity, before, "baseline", initial, pca_score, "pca", stages
        )
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
        if name == "rigid":
            tx = sitk.Euler3DTransform(best)
        else:
            tx = sitk.AffineTransform(3)
            tx.SetCenter(best.GetCenter())
            tx.SetMatrix(best.GetMatrix())
            tx.SetTranslation(best.GetTranslation())
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMeanSquares()
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(1.0, 0.001, config.iterations)
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel([2, 1])
        reg.SetSmoothingSigmasPerLevel([1, 0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        reg.SetInitialTransform(tx, inPlace=True)
        started = time.perf_counter()
        try:
            reg.Execute(fixed_dm, moving_dm)
            stages[name].update(
                optimizer_stop=reg.GetOptimizerStopConditionDescription(),
                optimizer_iteration=int(reg.GetOptimizerIteration()),
            )
            matrix = np.array(tx.GetMatrix()).reshape(3, 3)
            scales = np.linalg.svd(matrix, compute_uv=False)
            if (
                not np.isfinite(tx.GetParameters()).all()
                or np.linalg.det(matrix) <= 0
                or min(scales) < 0.5
                or max(scales) > 2
            ):
                raise ValueError("Implausible or nonfinite transform")
            tx.GetInverse()
            candidate = dice(fixed_organ, moving_organ, tx)
            stage_transforms[name] = tx
            best, score, stage = _select_candidate(
                best, score, stage, tx, candidate, name, stages
            )
        except (RuntimeError, ValueError) as exc:
            warnings.append(f"{name}: {exc}")
            stages[name].update(
                status="failed",
                error=str(exc),
                input_dice=score,
                selected=False,
                fallback_stage=stage,
            )
        finally:
            stages[name]["elapsed_seconds"] = time.perf_counter() - started
    selected_inverse = None
    if config.elastic:
        started = time.perf_counter()
        try:
            candidate, registration = elastic_refine(
                fixed_organ, moving_organ, fixed_dm, moving_dm, best, config
            )
            candidate_score = dice(fixed_organ, moving_organ, candidate)
            stage_transforms["elastic"] = candidate
            stages["elastic"].update(
                dice=candidate_score,
                input_dice=score,
                selected=False,
                optimizer_stop=registration.GetOptimizerStopConditionDescription(),
                optimizer_iteration=int(registration.GetOptimizerIteration()),
            )
            candidate_inverse = None
            if candidate_score > score + DICE_TOLERANCE:
                candidate_inverse = invert_elastic(
                    candidate, fixed_organ, moving_organ
                )
            best, score, stage = _select_candidate(
                best, score, stage, candidate, candidate_score, "elastic", stages
            )
            if stage == "elastic":
                selected_inverse = candidate_inverse
        except (RuntimeError, ValueError) as exc:
            warnings.append(f"elastic: {exc}")
            stages["elastic"].update(
                status="failed",
                error=str(exc),
                input_dice=score,
                selected=False,
                fallback_stage=stage,
            )
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
    if score < config.min_dice:
        raise RegistrationRejected(f"Organ overlap rejected: Dice={score:.4f}", result)
    return result
