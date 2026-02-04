import sys
from pathlib import Path

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from imperandi.process import convert as convert_module


def test_convert_list_str_to_list_valid():
    s = "['a', 'b', 'c']"
    out = convert_module.convert_list_str_to_list(s)
    assert isinstance(out, list)
    assert out == ["a", "b", "c"]


def test_convert_list_str_to_list_invalid():
    s = "not a list"
    out = convert_module.convert_list_str_to_list(s)
    assert out == s


def make_series(tmp_path, output_dir, create_nifti=False):
    # create a dummy series_dir with files
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    f1 = series_dir / "img1.dcm"
    f1.write_text("dummy")

    # fields required by process_single_volume
    row = pd.Series(
        {
            "series_dir": str(series_dir),
            "dicom_path": [str(f1)],
            "volume_ordinal_in_series": 1,
            "series_id": "S1",
            "patient_key": "P1",
            "study_id": "ST1",
            "Modality": "CT",
        }
    )

    if create_nifti:
        export_dir = (
            Path(output_dir)
            / row["patient_key"]
            / row["study_id"]
            / row["series_id"]
            / row["Modality"]
        )
        export_dir.mkdir(parents=True, exist_ok=True)
        nif = export_dir / "scan.nii.gz"
        nif.write_text("nifti")

    return row


def test_process_single_volume_skips_existing(tmp_path, monkeypatch):
    out_root = tmp_path / "out"
    out_root.mkdir()

    row = make_series(tmp_path, out_root, create_nifti=True)

    # ensure is_valid_nifti returns True for the created file
    monkeypatch.setattr(convert_module, "is_valid_nifti", lambda p: True)

    k, export_path, error = convert_module.process_single_volume(
        0, row, out_root, verbose=False
    )
    assert error is None
    assert export_path is not None
    assert Path(export_path).exists()


def test_process_single_volume_handles_conversion_error(tmp_path, monkeypatch):
    out_root = tmp_path / "out"
    out_root.mkdir()

    row = make_series(tmp_path, out_root, create_nifti=False)

    # make is_valid_nifti return False so conversion is attempted
    monkeypatch.setattr(convert_module, "is_valid_nifti", lambda p: False)

    # monkeypatch dicom2nifti internals to raise an exception during conversion
    class DummyErr(Exception):
        pass

    def raise_err(*args, **kwargs):
        raise DummyErr("conversion failed")

    monkeypatch.setattr(
        convert_module.dicom2nifti.common, "read_dicom_directory", raise_err
    )

    k, export_path, error_row = convert_module.process_single_volume(
        0, row, out_root, verbose=False
    )
    assert export_path is None
    assert error_row is not None
    assert "error" in error_row
