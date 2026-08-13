"""Assertions shared by full dataset pipeline tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def assert_full_pipeline_outputs(work_dir: Path) -> None:
    expected = {
        "dicom_index.csv",
        "dicom_index_clean.csv",
        "nifti_index.csv",
        "nifti_index_radiomics.csv",
    }
    assert expected.issubset(path.name for path in work_dir.iterdir())

    parsed = pd.read_csv(work_dir / "dicom_index.csv")
    cleaned = pd.read_csv(work_dir / "dicom_index_clean.csv")
    converted = pd.read_csv(work_dir / "nifti_index.csv")
    radiomics = pd.read_csv(work_dir / "nifti_index_radiomics.csv")

    assert not parsed.empty
    assert not cleaned.empty
    assert not converted.empty
    assert len(radiomics) == len(converted)
    assert {"patient_key", "study_id", "series_id"}.issubset(cleaned.columns)
    assert {"nifti_path", "phase", "liver_path"}.issubset(converted.columns)
    assert any("_original_" in column for column in radiomics.columns)

    nifti_paths = converted["nifti_path"].dropna().map(Path)
    liver_paths = converted["liver_path"].dropna().map(Path)
    assert not nifti_paths.empty
    assert not liver_paths.empty
    assert all(path.is_file() for path in nifti_paths)
    assert all(path.is_file() for path in liver_paths)
