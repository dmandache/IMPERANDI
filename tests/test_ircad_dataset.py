import sys
import argparse
from pathlib import Path

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ast import literal_eval

import pandas as pd
import pytest
from pandarallel import pandarallel

from imperandi.ingest import parse as parse_module
from imperandi.ingest import clean as clean_module
from imperandi.utils import files as files_module

pytestmark = pytest.mark.slow


def _normalize_to_repo_rel(path_str: str, repo_root: Path) -> str:
    if path_str is None:
        return ""
    p = str(path_str).replace("\\", "/")
    repo = repo_root.as_posix().rstrip("/")
    if p.lower().startswith((repo + "/").lower()):
        return p[len(repo) + 1 :]
    return p


def _resolve_repo_path(path_str: str, repo_root: Path) -> Path:
    p = str(path_str).replace("\\", "/")
    path = Path(p)
    if path.is_absolute():
        return path
    return repo_root / p


def _build_parse_args(root_path: Path, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        root_path=str(root_path),
        output_dir=str(output_dir),
        manifest=None,
        tags="",
        force_dicom_read=False,
        id_source="auto",
        patient_key_from="PatientName",
        study_id_from="StudyInstanceUID",
        series_id_from="SeriesInstanceUID",
        checkpoint_frequency=None,
        num_workers=1,
        verbose=False,
    )


def _normalize_dicom_path_column(df: pd.DataFrame) -> pd.DataFrame:
    if "dicom_path" not in df.columns:
        return df

    def _normalize_cell(val):
        if isinstance(val, list):
            return sorted(map(str, val))
        if isinstance(val, str):
            try:
                parsed = literal_eval(val)
            except (ValueError, SyntaxError):
                return val
            if isinstance(parsed, list):
                return sorted(map(str, parsed))
        return val

    out = df.copy()
    out["dicom_path"] = out["dicom_path"].apply(_normalize_cell)
    return out


def test_ircad_parse_matches_golden(tmp_path, ircad_dicom_root, ircad_reference_csv):
    pandarallel.initialize(progress_bar=False, nb_workers=1)

    args = _build_parse_args(ircad_dicom_root, tmp_path)
    parse_module.main(args)

    generated_csv = tmp_path / "dicom_index.csv"
    golden_csv = ircad_reference_csv("dicom_index.csv")

    generated = pd.read_csv(generated_csv)
    golden = pd.read_csv(golden_csv)

    repo_root = Path(__file__).resolve().parents[1]
    generated["dicom_path_norm"] = generated["dicom_path"].map(
        lambda p: _normalize_to_repo_rel(p, repo_root)
    )
    golden["dicom_path_norm"] = golden["dicom_path"].map(
        lambda p: _normalize_to_repo_rel(p, repo_root)
    )

    assert generated.shape[0] == golden.shape[0]
    assert generated["dicom_path_norm"].is_unique
    assert golden["dicom_path_norm"].is_unique
    assert set(generated["dicom_path_norm"]) == set(golden["dicom_path_norm"])

    generated = generated.set_index("dicom_path_norm")
    golden = golden.set_index("dicom_path_norm")

    key_cols = ["patient_key", "study_id", "series_id", "SOPInstanceUID", "Modality"]
    for col in key_cols:
        if col in generated.columns and col in golden.columns:
            pd.testing.assert_series_equal(
                generated[col].sort_index(),
                golden[col].sort_index(),
                check_dtype=False,
                check_names=False,
            )


def test_ircad_clean_matches_golden(tmp_path, ircad_dicom_root, ircad_reference_csv):
    input_csv = ircad_reference_csv("dicom_index.csv")
    output_csv = tmp_path / "dicom_index_clean.csv"

    clean_module.clean_and_save_data(
        [str(input_csv)],
        str(output_csv),
        csv_dict_path=None,
        manifest={},
        volume_min=clean_module.DEFAULT_VOLUME_LOWERBOUND,
        volume_max=clean_module.DEFAULT_VOLUME_UPPERBOUND,
    )

    generated = _normalize_dicom_path_column(pd.read_csv(output_csv))
    golden = _normalize_dicom_path_column(
        pd.read_csv(ircad_reference_csv("dicom_index_clean.csv"))
    )

    generated = generated.sort_values("volume_id").reset_index(drop=True)
    golden = golden.sort_values("volume_id").reset_index(drop=True)

    generated = generated.reindex(sorted(generated.columns), axis=1)
    golden = golden.reindex(sorted(golden.columns), axis=1)

    pd.testing.assert_frame_equal(
        generated,
        golden,
        check_dtype=False,
        check_exact=False,
        rtol=1e-6,
        atol=1e-6,
    )


def test_ircad_nifti_files_readable(ircad_nifti_root, ircad_reference_csv):
    nifti_index = ircad_reference_csv("nifti_index.csv")
    df = pd.read_csv(nifti_index)

    assert "nifti_path" in df.columns
    repo_root = Path(__file__).resolve().parents[1]
    paths = [
        _resolve_repo_path(p, repo_root)
        for p in df["nifti_path"].dropna().astype(str).tolist()
    ]

    missing = [p for p in paths if not p.exists()]
    assert not missing, f"Missing NIfTI files (showing up to 3): {missing[:3]}"

    for path in paths:
        assert files_module.is_valid_nifti(path)
