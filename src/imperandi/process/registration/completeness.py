"""Mask completeness heuristics and overlap restricted to observed image space.

Boundary contact suggests truncation, not proof of anatomical completeness.
Axes refer to image index axes (x, y, z), including oblique acquisitions.
"""

from dataclasses import dataclass
import numpy as np
from scipy import ndimage


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


def assess_organ_mask(mask, config) -> OrganMaskQC:
    """Assess a geometry-validated binary mask without modifying it."""
    from .organ import backend

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
    from .organ import backend

    sitk = backend()
    coverage = sitk.Image(image.GetSize(), sitk.sitkUInt8) + 1
    coverage.CopyInformation(image)
    return coverage


def overlap_qc(fixed, moving, transform) -> dict[str, float]:
    """Report full/reference-grid Dice and Dice within transformed common FOV."""
    from .organ import backend, resample

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
        score < config.min_confidence_dice
        or metrics["common_fov_fraction"] < config.min_common_fov_fraction
        or metrics["common_foreground_voxels"] < 4
    ):
        return "low_confidence"
    return "ok_partial_coverage" if partial else "ok"
