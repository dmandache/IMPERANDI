import sys
import io
import tarfile
import zipfile
import argparse
from pathlib import Path

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from unittest.mock import MagicMock

from imperandi.process import convert as convert_module
from imperandi.utils.archive_io import ArchiveSession, encode_archive_uri


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

    series_dir = tmp_path / "series"
    series_dir.mkdir()
    (series_dir / "img1.dcm").write_text("dummy")

    # Create the expected output directory and file
    export_dir = out_root / "P1" / "ST1" / "S1"
    export_dir.mkdir(parents=True, exist_ok=True)
    nifti_file = export_dir / "scan.nii.gz"
    nifti_file.write_text("fake nifti")

    row = pd.Series(
        {
            "series_dir": str(series_dir),
            "dicom_path": [str(series_dir / "img1.dcm")],
            "volume_ordinal_in_series": 1,
            "series_id": "S1",
            "patient_key": "P1",
            "study_id": "ST1",
            "Modality": "CT",
        }
    )

    # Mock is_valid_nifti to return True
    monkeypatch.setattr(convert_module, "is_valid_nifti", lambda p: True)

    k, export_path, error = convert_module.process_single_volume(
        0, row, out_root, verbose=False
    )

    assert error is None
    assert export_path == nifti_file


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


def test_process_single_volume_successful_conversion(tmp_path, monkeypatch):
    out_root = tmp_path / "out"
    out_root.mkdir()

    series_dir = tmp_path / "series"
    series_dir.mkdir()
    (series_dir / "img1.dcm").write_text("dummy")

    row = pd.Series(
        {
            "series_dir": str(series_dir),
            "dicom_path": [str(series_dir / "img1.dcm")],
            "volume_ordinal_in_series": 1,
            "series_id": "S1",
            "patient_key": "P1",
            "study_id": "ST1",
            "Modality": "CT",
        }
    )

    # Mock the dicom2nifti functions
    monkeypatch.setattr(
        convert_module.dicom2nifti.common,
        "read_dicom_directory",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        convert_module.dicom2nifti.convert_dicom,
        "dicom_array_to_nifti",
        MagicMock(),
    )
    monkeypatch.setattr(convert_module, "is_valid_nifti", lambda p: True)

    k, export_path, error = convert_module.process_single_volume(
        0, row, out_root, verbose=False
    )

    assert error is None
    assert export_path is not None
    assert "P1" in str(export_path)


def _make_archive_with_dicom(tmp_path: Path) -> tuple[Path, str]:
    inner_tar = tmp_path / "inner.tar.gz"
    with tarfile.open(inner_tar, "w:gz") as tf:
        payload = b"DICM_FAKE"
        info = tarfile.TarInfo(name="patientA/study1/series1/img1.dcm")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    outer_zip = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer_zip, "w") as zf:
        zf.write(inner_tar, arcname="nested/inner.tar.gz")

    return outer_zip, "nested/inner.tar.gz"


def test_materialize_archive_dicom_paths_replaces_uri(tmp_path):
    outer_zip, nested_entry = _make_archive_with_dicom(tmp_path)
    uri = encode_archive_uri(
        outer_zip, [nested_entry, "patientA/study1/series1/img1.dcm"]
    )
    df = pd.DataFrame(
        [
            {
                "dicom_path": [uri],
                "series_id": "S1",
                "study_id": "ST1",
                "patient_key": "P1",
            }
        ]
    )

    with ArchiveSession(cache_dir=tmp_path / ".cache", keep_cache=True) as session:
        out, df_err = convert_module.materialize_archive_dicom_paths(df, session)

    assert df_err.empty
    assert len(out) == 1
    assert isinstance(out.loc[0, "dicom_path"], list)
    local_path = Path(out.loc[0, "dicom_path"][0])
    assert local_path.exists()
    assert local_path.name == "img1.dcm"


def test_materialize_archive_dicom_paths_emits_error_for_missing_member(tmp_path):
    outer_zip = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer_zip, "w"):
        pass

    missing_uri = encode_archive_uri(outer_zip, ["missing.dcm"])
    df = pd.DataFrame(
        [
            {
                "dicom_path": missing_uri,
                "series_id": "S1",
                "study_id": "ST1",
                "patient_key": "P1",
            }
        ]
    )

    with ArchiveSession(cache_dir=tmp_path / ".cache", keep_cache=True) as session:
        out, df_err = convert_module.materialize_archive_dicom_paths(df, session)

    assert out.empty
    assert not df_err.empty
    assert "error" in df_err.columns


def test_main_resume_uses_checkpoint_state(tmp_path, monkeypatch):
    csv_path = tmp_path / "dicom_index.csv"
    pd.DataFrame(
        [
            {
                "dicom_path": ["a.dcm"],
                "series_id": "S1",
                "study_id": "ST1",
                "patient_key": "P1",
                "Modality": "CT",
            }
        ]
    ).to_csv(csv_path, index=False)

    work_sizes = []

    def fake_convert(work_df, output_dir, verbose, num_workers, on_result):
        work_sizes.append(len(work_df))
        for i in range(len(work_df)):
            on_result(i, None, {"error": "x", "_source_idx": int(work_df.iloc[i]["_source_idx"])}, "failed")
        return work_df, pd.DataFrame()

    monkeypatch.setattr(convert_module, "convert_dicom_to_nifti_parallel", fake_convert)
    monkeypatch.setattr(convert_module, "materialize_archive_dicom_paths", lambda df, session: (df, pd.DataFrame()))
    monkeypatch.setattr(convert_module, "report_volumes", lambda *_: None)
    monkeypatch.setattr(convert_module, "report_change", lambda *_: None)

    args = argparse.Namespace(
        csv_path=[str(csv_path)],
        output_dir=str(tmp_path / "nifti"),
        csv_path_out=str(tmp_path / "out.csv"),
        error_csv_path=str(tmp_path / "err.csv"),
        verbose=False,
        dry_run=False,
        num_workers=1,
        archive_max_depth=3,
        archive_cache_dir=None,
        keep_archive_cache=False,
        resume=False,
        strict_resume=False,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        manifest=None,
    )
    convert_module.main(args)
    assert work_sizes[-1] == 1

    args.resume = True
    convert_module.main(args)
    assert work_sizes == [1]
