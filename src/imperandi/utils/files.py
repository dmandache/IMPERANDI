from pathlib import Path
import shutil
import os
import nibabel as nib
import numpy as np


def make_temp_dir(temp_dir="/data/scratch/bdr220003/temp/"):
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    is_empty = not any(temp_dir.iterdir())
    if not is_empty:
        shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)


def dir_is_empty(temp_dir):
    temp_dir = Path(temp_dir)
    is_empty = not any(temp_dir.iterdir())
    return is_empty


def empty_dir(temp_dir):
    if not dir_is_empty(temp_dir):
        shutil.rmtree(temp_dir)


def copy_files_to_temp_dir(paths, temp_dir="/data/scratch/bdr220003/temp/"):
    if dir_is_empty(temp_dir):
        print(f"Temp directory {temp_dir} is empty")
    else:
        print(f"Temp directory {temp_dir} not empty")
    check_permissions(temp_dir)

    temp_dir = Path(temp_dir)
    for src_path in paths:
        src_path = Path(src_path)
        dst_path = temp_dir / src_path.name
        shutil.copy(src_path, dst_path)


def check_file(file_path):
    # Specify the file path
    file_path = Path(file_path)

    # Check if the file exists
    if file_path.exists():
        print(f"File {file_path} exists")

        # Check if the file is readable
        if file_path.is_file() and file_path.stat().st_mode & 0o400:
            print("File is readable")
        else:
            print("File is not readable")

        # Check if the file is writable
        if file_path.is_file() and file_path.stat().st_mode & 0o200:
            print("File is writable")
        else:
            print("File is not writable")

        # Check if the file is executable
        if file_path.is_file() and file_path.stat().st_mode & 0o100:
            print("File is executable")
        else:
            print("File is not executable")
    else:
        print(f"File {file_path} does not exist")


def check_permissions(directory):
    # Create a Path object
    directory_path = Path(directory)

    # Check if the directory exists
    if not directory_path.exists():
        print(f"Directory '{directory}' does not exist.")
        return

    # Check read permission
    if os.access(directory, os.R_OK):
        print(f"Read permission is granted on '{directory}'.")
    else:
        print(f"No read permission on '{directory}'.")

    # Check write permission
    if os.access(directory, os.W_OK):
        print(f"Write permission is granted on '{directory}'.")
    else:
        print(f"No write permission on '{directory}'.")


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
    try:
        nib.load(path).get_fdata()
        return True
    except Exception as e:
        print(f"Error reading nifti {path}: ⚠️ {e} !")
        return False
