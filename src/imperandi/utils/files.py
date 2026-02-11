"""Filesystem and validation utilities for temporary files and imaging assets.

The definitions in this module are part of the Imperandi codebase and are
intended to be reused by higher-level workflows and CLI entry points.
"""

from pathlib import Path
import logging
import shutil
import os
import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


def make_temp_dir(temp_dir="/data/scratch/bdr220003/temp/"):
    """Make temp dir.

    Args:
        temp_dir: Directory location used for reading or writing artifacts.
    """
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    is_empty = not any(temp_dir.iterdir())
    if not is_empty:
        shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)


def dir_is_empty(temp_dir):
    """Perform is empty.

    Args:
        temp_dir (Any): Directory path used for input or output data.

    Returns:
        Any: Result of `dir_is_empty`.
    """
    temp_dir = Path(temp_dir)
    is_empty = not any(temp_dir.iterdir())
    return is_empty


def empty_dir(temp_dir):
    """Empty dir.

    Args:
        temp_dir: Directory location used for reading or writing artifacts.
    """
    if not dir_is_empty(temp_dir):
        shutil.rmtree(temp_dir)


def copy_files_to_temp_dir(paths, temp_dir="/data/scratch/bdr220003/temp/"):
    """Copy files TO temp dir.

    Args:
        paths: Filesystem path used by this routine.
        temp_dir: Directory location used for reading or writing artifacts.
    """
    if dir_is_empty(temp_dir):
        logger.info("Temp directory %s is empty", temp_dir)
    else:
        logger.info("Temp directory %s not empty", temp_dir)
    check_permissions(temp_dir)

    temp_dir = Path(temp_dir)
    for src_path in paths:
        src_path = Path(src_path)
        dst_path = temp_dir / src_path.name
        shutil.copy(src_path, dst_path)


def check_file(file_path):
    # Specify the file path
    """Check file.

    Args:
        file_path: Filesystem path used by this routine.
    """
    file_path = Path(file_path)

    # Check if the file exists
    if file_path.exists():
        logger.info("File %s exists", file_path)

        # Check if the file is readable
        if file_path.is_file() and file_path.stat().st_mode & 0o400:
            logger.info("File is readable")
        else:
            logger.info("File is not readable")

        # Check if the file is writable
        if file_path.is_file() and file_path.stat().st_mode & 0o200:
            logger.info("File is writable")
        else:
            logger.info("File is not writable")

        # Check if the file is executable
        if file_path.is_file() and file_path.stat().st_mode & 0o100:
            logger.info("File is executable")
        else:
            logger.info("File is not executable")
    else:
        logger.info("File %s does not exist", file_path)


def check_permissions(directory):
    # Create a Path object
    """Check permissions.

    Args:
        directory (Any): Directory path used for input or output data.
    """
    directory_path = Path(directory)

    # Check if the directory exists
    if not directory_path.exists():
        logger.info("Directory '%s' does not exist.", directory)
        return

    # Check read permission
    if os.access(directory, os.R_OK):
        logger.info("Read permission is granted on '%s'.", directory)
    else:
        logger.info("No read permission on '%s'.", directory)

    # Check write permission
    if os.access(directory, os.W_OK):
        logger.info("Write permission is granted on '%s'.", directory)
    else:
        logger.info("No write permission on '%s'.", directory)


def load_nifti_file(nifti_path):
    """
    Load a NIfTI file and return the image data and pixel spacing.
    """
    # Load the NIfTI file
    nifti_img = nib.load(nifti_path)

    # Extract the binary mask (3D image data)
    mask = nifti_img.get_fdata()

    # Extract the pixel spacing (voxel dimensions) from the affine matrix
    affine = nifti_img.affine
    # Voxel dimensions can be derived from the affine matrix (scaling factors)
    dx = np.linalg.norm(affine[0, :3])  # Spacing along the x-axis
    dy = np.linalg.norm(affine[1, :3])  # Spacing along the y-axis
    dz = np.linalg.norm(affine[2, :3])  # Spacing along the z-axis

    pixel_spacing = (dx, dy, dz)

    return mask, pixel_spacing


def is_valid_nifti(path):
    """Return whether valid NIfTI.

    Args:
        path (Any): Filesystem path consumed by this operation.

    Returns:
        Any: True when the condition is satisfied; otherwise False.
    """
    try:
        nib.load(path).get_fdata()
        return True
    except Exception as e:
        logger.error("Error reading nifti %s: ⚠️ %s !", path, e)
        return False
