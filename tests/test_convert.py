import sys
import io
import tarfile
import zipfile
import argparse
import logging
from pathlib import Path

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest
from unittest.mock import MagicMock

from imperandi.process import convert as convert_module
from imperandi.utils.archive_io import ArchiveSession, encode_archive_uri
from imperandi.utils.run_state import build_checkpoint_paths, load_state


def test_convert_list_str_to_list_valid():
    s = "['a', 'b', 'c']"
    out = convert_module.convert_list_str_to_list(s)
    assert isinstance(out, list)
    assert out == ["a", "b", "c"]


def test_convert_list_str_to_list_invalid():
    s = "not a list"
    out = convert_module.convert_list_str_to_list(s)
    assert out == s


def test_normalize_convert_args_defaults_output_dir_to_project_nifti(tmp_path, caplog):
    csv_path = tmp_path / "dicom_index_clean.csv"
    csv_path.write_text("patient_key\n")
    args = convert_module.build_parser().parse_args([str(csv_path)])

    with caplog.at_level(logging.WARNING, logger=convert_module.__name__):
        normalized = convert_module.normalize_convert_args(args)

    assert normalized.output_dir == str(tmp_path / "NIFTI")
    assert "No output_dir supplied" in caplog.text
    assert str(tmp_path / "NIFTI") in caplog.text


def test_normalize_convert_args_keeps_explicit_output_dir_without_warning(
    tmp_path, caplog
):
    csv_path = tmp_path / "dicom_index_clean.csv"
    csv_path.write_text("patient_key\n")
    output_dir = tmp_path / "custom_nifti"
    args = convert_module.build_parser().parse_args([str(csv_path), str(output_dir)])

    with caplog.at_level(logging.WARNING, logger=convert_module.__name__):
        normalized = convert_module.normalize_convert_args(args)

    assert normalized.output_dir == str(output_dir)
    assert "No output_dir supplied" not in caplog.text


@pytest.mark.parametrize(
    ("base_level", "expected_level"),
    [
        (logging.CRITICAL, logging.ERROR),
        (logging.ERROR, logging.WARNING),
        (logging.WARNING, logging.INFO),
        (logging.INFO, logging.DEBUG),
        (logging.DEBUG, logging.DEBUG),
        (logging.NOTSET, logging.DEBUG),
    ],
)
def test_lower_log_level_one_step(base_level, expected_level):
    assert convert_module._lower_log_level_one_step(base_level) == expected_level


def test_configure_dicom2nifti_convert_logger_one_level_lower():
    base_logger = logging.getLogger("imperandi.process.convert.test_base")
    target_logger = logging.getLogger("dicom2nifti.convert_dicom")
    old_base_level = base_logger.level
    old_target_level = target_logger.level

    try:
        base_logger.setLevel(logging.INFO)
        convert_module._configure_dicom2nifti_convert_logger(base_logger)
        assert target_logger.level == logging.DEBUG

        base_logger.setLevel(logging.WARNING)
        convert_module._configure_dicom2nifti_convert_logger(base_logger)
        assert target_logger.level == logging.INFO
    finally:
        base_logger.setLevel(old_base_level)
        target_logger.setLevel(old_target_level)


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
            on_result(
                i,
                None,
                {"error": "x", "_source_idx": int(work_df.iloc[i]["_source_idx"])},
                "failed",
            )
        return work_df, pd.DataFrame()

    monkeypatch.setattr(convert_module, "convert_dicom_to_nifti_parallel", fake_convert)
    monkeypatch.setattr(
        convert_module,
        "materialize_archive_dicom_paths",
        lambda df, session: (df, pd.DataFrame()),
    )
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


def test_main_resume_reprocesses_when_error_checkpoint_path_changes(
    tmp_path, monkeypatch
):
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
            on_result(
                i,
                None,
                {"error": "x", "_source_idx": int(work_df.iloc[i]["_source_idx"])},
                "failed",
            )
        return work_df, pd.DataFrame()

    monkeypatch.setattr(convert_module, "convert_dicom_to_nifti_parallel", fake_convert)
    monkeypatch.setattr(
        convert_module,
        "materialize_archive_dicom_paths",
        lambda df, session: (df, pd.DataFrame()),
    )
    monkeypatch.setattr(convert_module, "report_volumes", lambda *_: None)
    monkeypatch.setattr(convert_module, "report_change", lambda *_: None)

    args = argparse.Namespace(
        csv_path=[str(csv_path)],
        output_dir=str(tmp_path / "nifti"),
        csv_path_out=str(tmp_path / "out.csv"),
        error_csv_path=str(tmp_path / "err1.csv"),
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

    args.resume = True
    args.error_csv_path = str(tmp_path / "err2.csv")
    convert_module.main(args)

    assert work_sizes == [1, 1]


def test_main_uses_volume_id_as_checkpoint_source_idx(tmp_path, monkeypatch):
    csv_path = tmp_path / "dicom_index.csv"
    pd.DataFrame(
        [
            {
                "volume_id": "vol-a",
                "dicom_path": ["a.dcm"],
                "series_id": "S1",
                "study_id": "ST1",
                "patient_key": "P1",
                "Modality": "CT",
            }
        ]
    ).to_csv(csv_path, index=False)

    seen_source_ids = []

    def fake_convert(work_df, output_dir, verbose, num_workers, on_result):
        seen_source_ids.extend(work_df["_source_idx"].tolist())
        for i in range(len(work_df)):
            on_result(i, Path(output_dir) / "scan.nii.gz", None, "converted")
        return work_df, pd.DataFrame()

    monkeypatch.setattr(convert_module, "convert_dicom_to_nifti_parallel", fake_convert)
    monkeypatch.setattr(
        convert_module,
        "materialize_archive_dicom_paths",
        lambda df, session: (df, pd.DataFrame()),
    )
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

    assert seen_source_ids == ["vol-a"]
    paths = build_checkpoint_paths(args.csv_path_out, args.error_csv_path, "convert")
    state = load_state(paths.state_path)
    assert state["completed_indices"] == ["vol-a"]


def test_main_preserves_foreign_columns_from_existing_output(tmp_path, monkeypatch):
    csv_path = tmp_path / "dicom_index.csv"
    out_path = tmp_path / "out.csv"
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
    pd.DataFrame([{"series_id": "S1", "foreign_col": "keep"}]).to_csv(
        out_path, index=False
    )

    def fake_convert(work_df, output_dir, verbose, num_workers, on_result):
        for i in range(len(work_df)):
            on_result(i, Path(output_dir) / "scan.nii.gz", None, "ok")
        return work_df, pd.DataFrame()

    monkeypatch.setattr(convert_module, "convert_dicom_to_nifti_parallel", fake_convert)
    monkeypatch.setattr(
        convert_module,
        "materialize_archive_dicom_paths",
        lambda df, session: (df, pd.DataFrame()),
    )
    monkeypatch.setattr(convert_module, "report_volumes", lambda *_: None)
    monkeypatch.setattr(convert_module, "report_change", lambda *_: None)

    args = argparse.Namespace(
        csv_path=[str(csv_path)],
        output_dir=str(tmp_path / "nifti"),
        csv_path_out=str(out_path),
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
    out_df = pd.read_csv(out_path)
    assert "foreign_col" in out_df.columns
    assert out_df.loc[0, "foreign_col"] == "keep"


def test_main_fails_fast_on_unsafe_shared_output_alignment(tmp_path, monkeypatch):
    csv_path = tmp_path / "dicom_index.csv"
    out_path = tmp_path / "out.csv"
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
    pd.DataFrame([{"foreign_col": "x"}, {"foreign_col": "y"}]).to_csv(
        out_path, index=False
    )

    def fake_convert(work_df, output_dir, verbose, num_workers, on_result):
        for i in range(len(work_df)):
            on_result(i, Path(output_dir) / "scan.nii.gz", None, "ok")
        return work_df, pd.DataFrame()

    monkeypatch.setattr(convert_module, "convert_dicom_to_nifti_parallel", fake_convert)
    monkeypatch.setattr(
        convert_module,
        "materialize_archive_dicom_paths",
        lambda df, session: (df, pd.DataFrame()),
    )
    monkeypatch.setattr(convert_module, "report_volumes", lambda *_: None)
    monkeypatch.setattr(convert_module, "report_change", lambda *_: None)

    args = argparse.Namespace(
        csv_path=[str(csv_path)],
        output_dir=str(tmp_path / "nifti"),
        csv_path_out=str(out_path),
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
    with pytest.raises(
        ValueError, match="Cannot safely preserve existing output columns"
    ):
        convert_module.main(args)
