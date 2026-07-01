from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

DEFAULT_ISOTROPIC_RESOLUTION_MM = 1.0


def validate_isotropic_resolution(value):
    """Validate and return a positive finite display resolution in millimetres."""
    resolution = float(value)
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("isotropic_resolution_mm must be a positive finite number")
    return resolution


def load_oriented_nifti(file_path, orientation="LAS"):
    """Load NIfTI data and voxel spacing after applying viewer orientation."""
    img = nib.load(Path(file_path).resolve())
    data = img.get_fdata()
    affine = img.affine

    current_ornt = nib.orientations.io_orientation(affine)
    if orientation == "RAS":
        new_ornt = np.array([[0, 1], [1, 1], [2, 1]])
    elif orientation == "LAS":
        new_ornt = np.array([[0, -1], [1, 1], [2, 1]])
    else:
        raise ValueError("orientation must be 'RAS' or 'LAS'")

    transform = nib.orientations.ornt_transform(current_ornt, new_ornt)
    oriented = nib.orientations.apply_orientation(data, transform)

    zooms = np.asarray(img.header.get_zooms()[: data.ndim], dtype=float)
    axis_order = np.argsort(transform[:, 0])
    oriented_zooms = tuple(abs(float(zooms[i])) for i in axis_order)
    return oriented, oriented_zooms


def resample_to_isotropic(data, spacing, resolution_mm, order=1):
    """Resample 3D data to isotropic voxel spacing for display."""
    resolution = validate_isotropic_resolution(resolution_mm)
    spacing = np.asarray(spacing, dtype=float)
    if data.ndim != 3 or spacing.shape[0] != 3:
        return data
    if np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        return data

    factors = spacing / resolution
    if np.allclose(factors, 1.0, rtol=0.0, atol=1e-6):
        return data
    return zoom(data, factors, order=order)


def load_nifti_isotropic(
    file_path,
    orientation="LAS",
    resolution_mm=DEFAULT_ISOTROPIC_RESOLUTION_MM,
    order=1,
):
    """Load, orient, and resample a NIfTI volume for isotropic display."""
    data, spacing = load_oriented_nifti(file_path, orientation=orientation)
    return resample_to_isotropic(data, spacing, resolution_mm, order=order)
