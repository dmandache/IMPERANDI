"""I/O and mask postprocessing helpers for segmentation workflows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes
from skimage.measure import label, regionprops
from skimage.morphology import ball

logger = logging.getLogger(__name__)


def load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]:
    """Return image data, affine matrix and voxel sizes (zoom)."""
    img = nib.load(str(path))
    return img.get_fdata(), img.affine, img.header.get_zooms()


def save_nifti(
    data: np.ndarray, affine: np.ndarray, out_path: Path, *, dtype=np.uint8
) -> None:
    """Write *data* to *out_path* as a NIfTI-1 file with *dtype*."""
    img = nib.Nifti1Image(data.astype(dtype, copy=False), affine)
    nib.save(img, str(out_path))


def compute_struct_elem(zooms: Tuple[float, ...], radius_mm: float = 5.0) -> np.ndarray:
    """Create a spherical structuring element with *radius_mm* in real units."""
    radii_vox = [max(1, int(round(radius_mm / z))) for z in zooms]
    return ball(max(radii_vox))


def clean_and_merge_masks(
    dir_path: Path,
    mask_files: List[str],
    *,
    output_name: str,
    radius_mm: float = 5.0,
    verbose: bool = False,
    close: bool = True,
    fill_holes: bool = True,
    largest_cc: bool = True,
) -> bool:
    """Merge masks and optionally apply morphological cleanup."""
    masks: Dict[str, np.ndarray] = {}
    ref_affine: np.ndarray | None = None
    voxel_zooms: Tuple[float, ...] | None = None

    for fname in mask_files:
        src = dir_path / fname
        if not src.exists():
            logger.warning("Mask missing - skipping merge: %s", src)
            continue

        data, affine, zooms = load_nifti(src)
        if ref_affine is None:
            ref_affine, voxel_zooms = affine, zooms
        elif not np.allclose(ref_affine, affine):
            logger.error("Affine mismatch for %s - aborting merge.", src.name)
            return False
        masks[fname] = data > 0

    if not masks:
        logger.error("No valid masks found to merge in %s", dir_path)
        return False

    if len({m.shape for m in masks.values()}) > 1:
        logger.error("Mask shape mismatch in %s - aborting merge.", dir_path)
        return False

    merged = np.logical_or.reduce(list(masks.values()))
    if close:
        merged = binary_closing(
            merged, structure=compute_struct_elem(voxel_zooms, radius_mm)
        )
    if fill_holes:
        merged = binary_fill_holes(merged)

    if largest_cc:
        labeled, n_cc = label(merged, return_num=True)
        if n_cc > 1:
            largest = max(regionprops(labeled), key=lambda r: r.area)
            merged = labeled == largest.label
            if verbose:
                logger.info(
                    "%s : kept largest CC (%s voxels) out of %s",
                    dir_path,
                    largest.area,
                    n_cc,
                )
        elif verbose:
            logger.info("%s : single connected component", dir_path)

    save_nifti(merged, ref_affine, dir_path / output_name)

    for fname, mask in masks.items():
        save_nifti(mask & merged, ref_affine, dir_path / fname)

    return True


def _mask_key_from_filename(filename: str) -> str:
    """Return mask key from a NIfTI filename."""
    if filename.endswith(".nii.gz"):
        return filename[: -len(".nii.gz")]
    if filename.endswith(".nii"):
        return filename[: -len(".nii")]
    return Path(filename).stem


def mask_column_for_output_file(path: Path) -> str:
    """Map output file path to DataFrame mask column name."""
    return f"mask_{_mask_key_from_filename(path.name)}"


def discover_segmentation_outputs(output_dir: Path, source_nifti: Path) -> List[Path]:
    """Discover all produced NIfTI masks in output directory, excluding the source volume."""
    masks: List[Path] = []
    source_resolved = source_nifti.resolve()
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if not (lower_name.endswith(".nii.gz") or lower_name.endswith(".nii")):
            continue
        try:
            if path.resolve() == source_resolved:
                continue
        except Exception:
            pass
        masks.append(path)
    return sorted(masks, key=lambda p: p.name)


def snapshot_segmentation_outputs(
    output_dir: Path, source_nifti: Path
) -> Dict[str, Tuple[int, int]]:
    """Snapshot discovered output files as ``{filename: (mtime_ns, size)}``."""
    snapshot: Dict[str, Tuple[int, int]] = {}
    for path in discover_segmentation_outputs(output_dir, source_nifti):
        stat = path.stat()
        snapshot[path.name] = (int(stat.st_mtime_ns), int(stat.st_size))
    return snapshot


def diff_changed_outputs(
    before: Dict[str, Tuple[int, int]], after: Dict[str, Tuple[int, int]]
) -> List[str]:
    """Return output filenames that are new or modified between snapshots."""
    changed: List[str] = []
    for name, after_sig in after.items():
        if name not in before or before[name] != after_sig:
            changed.append(name)
    return sorted(changed)


def register_output_key_map(
    key_to_output: Dict[str, str], output_files: List[Path], warnings: List[str] | None = None
) -> None:
    """Register discovered output files in ``stem -> filename`` map."""
    for path in output_files:
        key = _mask_key_from_filename(path.name)
        if key in key_to_output and key_to_output[key] != path.name:
            msg = (
                f"Segmentation output key collision for '{key}': "
                f"{key_to_output[key]} -> {path.name}. Last writer wins."
            )
            logger.warning(msg)
            if warnings is not None:
                warnings.append(msg)
        key_to_output[key] = path.name
