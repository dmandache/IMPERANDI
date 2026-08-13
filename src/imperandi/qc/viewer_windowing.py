"""Modality-aware intensity windowing shared by the QC viewers."""

import numpy as np


def normalize_modality(value) -> str:
    """Return a canonical modality label used by viewer controls."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        labels = {normalize_modality(item) for item in value}
        labels.discard("")
        if len(labels) == 1:
            return labels.pop()
        return "|".join(sorted(labels))
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ""
    text = str(value).strip().upper()
    return "MR" if text == "MRI" else text


def percentile_window(volume, lower=1.0, upper=99.0) -> tuple[float, float]:
    """Compute finite voxel bounds for an MRI percentile window."""
    lower = float(lower)
    upper = float(upper)
    if not 0 <= lower < upper <= 100:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100.")

    finite = np.asarray(volume)[np.isfinite(volume)]
    if finite.size == 0:
        return 0.0, 1.0

    low, high = np.percentile(finite, [lower, upper]).astype(float)
    if low == high:
        high = low + 1.0
    return low, high
