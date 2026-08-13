import sys
from pathlib import Path

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import nibabel as nib
import numpy as np

from imperandi.utils import files


def test_make_temp_dir_creates_missing_dir(tmp_path):
    temp_dir = tmp_path / "temp"

    files.make_temp_dir(temp_dir)

    assert temp_dir.exists()
    assert files.dir_is_empty(temp_dir)


def test_make_temp_dir_cleans_existing_contents(tmp_path):
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    (temp_dir / "stale.txt").write_text("old")

    files.make_temp_dir(temp_dir)

    assert temp_dir.exists()
    assert list(temp_dir.iterdir()) == []


def test_empty_dir_removes_only_non_empty_dir(tmp_path):
    non_empty = tmp_path / "non_empty"
    non_empty.mkdir()
    (non_empty / "x.txt").write_text("x")
    files.empty_dir(non_empty)
    assert not non_empty.exists()

    empty = tmp_path / "empty"
    empty.mkdir()
    files.empty_dir(empty)
    assert empty.exists()


def test_copy_files_to_temp_dir_copies_all_sources(tmp_path):
    src1 = tmp_path / "a.txt"
    src2 = tmp_path / "b.txt"
    src1.write_text("alpha")
    src2.write_text("beta")
    temp_dir = tmp_path / "dest"

    files.copy_files_to_temp_dir([src1, src2], temp_dir=temp_dir)

    assert (temp_dir / "a.txt").read_text() == "alpha"
    assert (temp_dir / "b.txt").read_text() == "beta"


def test_check_file_logs_missing_and_existing(caplog, tmp_path):
    missing = tmp_path / "missing.txt"
    with caplog.at_level("INFO"):
        files.check_file(missing)
    assert any("does not exist" in rec.message for rec in caplog.records)

    caplog.clear()
    existing = tmp_path / "exists.txt"
    existing.write_text("ok")
    with caplog.at_level("INFO"):
        files.check_file(existing)
    assert any("exists" in rec.message for rec in caplog.records)


def test_check_permissions_logs_missing_directory(caplog, tmp_path):
    missing_dir = tmp_path / "no_such_dir"

    with caplog.at_level("INFO"):
        files.check_permissions(str(missing_dir))

    assert any("does not exist" in rec.message for rec in caplog.records)


def test_load_nifti_file_returns_mask_and_spacing(tmp_path):
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    nifti_path = tmp_path / "sample.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(nifti_path))

    mask, pixel_spacing = files.load_nifti_file(str(nifti_path))

    assert mask.shape == data.shape
    assert pixel_spacing == (2.0, 3.0, 4.0)


def test_is_valid_nifti_true_and_false(tmp_path):
    valid_path = tmp_path / "valid.nii.gz"
    nib.save(
        nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4)),
        str(valid_path),
    )
    assert files.is_valid_nifti(str(valid_path))

    invalid_path = tmp_path / "invalid.nii.gz"
    invalid_path.write_text("not a nifti file")
    assert not files.is_valid_nifti(str(invalid_path))
