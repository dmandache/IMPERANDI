"""Fusion interface: binary masks and observed support on one common grid."""

from dataclasses import dataclass
import numpy as np
from .organ import backend


@dataclass
class ConsensusResult:
    mask: object
    coverage: object
    probability: object | None


def fuse_tumors(masks, coverages, *, method, threshold=0.5):
    """Anchor is first input. Missing masks must be omitted, empty masks retained.

    Voting uses strict > threshold; STAPLE uses >= threshold. Unknown spatial
    coverage is excluded from every estimator, and returned separately.
    """
    sitk = backend()
    if method not in {"anchor", "majority", "intersection", "union", "staple"}:
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
    if method == "anchor":
        masks, coverages = masks[:1], coverages[:1]
    values = np.stack([sitk.GetArrayFromImage(m) > 0 for m in masks])
    support = np.logical_and.reduce([sitk.GetArrayFromImage(c) > 0 for c in coverages])
    if not support.any():
        raise ValueError("Tumor inputs have no common observed support")
    probability = None
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
        probability = values.mean(axis=0).astype(np.float32)
        binary = (
            probability > threshold
            if method == "majority"
            else probability >= threshold
        )
    elif method == "intersection":
        binary = values.all(axis=0)
    else:
        binary = values.any(axis=0)

    def image(array, dtype):
        out = sitk.GetImageFromArray(array.astype(dtype))
        out.CopyInformation(reference)
        return out

    return ConsensusResult(
        image(binary & support, np.uint8),
        image(support, np.uint8),
        image(probability * support, np.float32) if probability is not None else None,
    )
