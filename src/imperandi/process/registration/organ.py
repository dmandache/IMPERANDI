"""Physical-space PCA and linear organ registration, independent of tumor masks."""

from dataclasses import dataclass
from itertools import permutations, product
import numpy as np


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


@dataclass
class TransformResult:
    reference_to_scan: object
    stage: str
    dice_before: float
    dice_after: float
    warnings: list[str]


def register_pair(fixed_organ, moving_organ, config):
    sitk = backend()
    initial = initialize_pca(fixed_organ, moving_organ)
    before = dice(fixed_organ, moving_organ, sitk.Transform(3, sitk.sitkIdentity))
    best = initial
    score = dice(fixed_organ, moving_organ, best)
    stage = "pca"
    warnings = []
    fixed_dm = sitk.SignedMaurerDistanceMap(
        fixed_organ, insideIsPositive=False, squaredDistance=False, useImageSpacing=True
    )
    moving_dm = sitk.SignedMaurerDistanceMap(
        moving_organ,
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=True,
    )
    for name in (["rigid", "affine"] if config.affine else ["rigid"]):
        if name == "rigid":
            tx = sitk.Euler3DTransform(initial)
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
        try:
            reg.Execute(fixed_dm, moving_dm)
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
            if candidate > score + 1e-6:
                best, score, stage = tx, candidate, name
        except (RuntimeError, ValueError) as exc:
            warnings.append(f"{name}: {exc}")
    if before >= score - 1e-6:
        best, score, stage = sitk.Euler3DTransform(), before, "identity"
    if score < config.min_dice:
        raise ValueError(f"Organ overlap rejected: Dice={score:.4f}")
    return TransformResult(best, stage, before, score, warnings)
