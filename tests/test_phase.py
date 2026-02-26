import sys
from pathlib import Path

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

import pandas as pd

from imperandi.extract import phase as phase_module


def test_normalize_phase_args_defaults(tmp_path):
    csv_path = tmp_path / "nifti_index.csv"
    csv_path.write_text("nifti_path\n")

    args = argparse.Namespace(
        csv_path_pos=str(csv_path),
        csv_path_opt=None,
        csv_path_out=None,
        error_csv_path=None,
        totalseg_home_dir=None,
        verbose=False,
        dry_run=False,
    )

    out = phase_module.normalize_phase_args(args)

    assert out.csv_path == str(csv_path.resolve())
    assert out.csv_path_out == str(csv_path.resolve())
    assert out.error_csv_path == str(csv_path.parent / "phase_errors.csv")
    assert not hasattr(out, "csv_path_pos")
    assert not hasattr(out, "csv_path_opt")


def test_process_single_volume_success(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    monkeypatch.setattr(phase_module.nib, "load", lambda _: object())

    idx, phase_info, err = phase_module.process_single_volume(
        0,
        {"nifti_path": str(nifti)},
        phase_extractor=lambda _: {"phase": "portal", "probability": 0.9},
    )

    assert idx == 0
    assert err is None
    assert phase_info["totalseg_phase"] == "portal"
    assert phase_info["totalseg_probability"] == 0.9


def test_process_single_volume_missing_file():
    idx, phase_info, err = phase_module.process_single_volume(
        0,
        {"nifti_path": "does/not/exist.nii.gz"},
        phase_extractor=lambda _: {"phase": "portal"},
    )

    assert idx == 0
    assert phase_info is None
    assert "file not found" in err


def test_main_writes_phase_columns_and_error_csv(tmp_path, monkeypatch):
    valid_nifti = tmp_path / "valid.nii.gz"
    valid_nifti.write_text("nifti")
    missing_nifti = tmp_path / "missing.nii.gz"

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame(
        [
            {"nifti_path": str(valid_nifti), "study_id": "s1"},
            {"nifti_path": str(missing_nifti), "study_id": "s2"},
        ]
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(phase_module.nib, "load", lambda _: object())
    monkeypatch.setattr(
        phase_module,
        "_load_phase_extractor",
        lambda: (lambda _: {"phase": "arterial", "confidence": 0.8}),
    )

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        totalseg_home_dir=None,
        verbose=False,
    )

    phase_module.main(args)

    out_df = pd.read_csv(args.csv_path_out)
    assert "totalseg_phase" in out_df.columns
    assert "totalseg_confidence" in out_df.columns
    assert out_df.loc[0, "totalseg_phase"] == "arterial"
    assert pd.isna(out_df.loc[1, "totalseg_phase"])

    err_df = pd.read_csv(args.error_csv_path)
    assert len(err_df) == 1
    assert "file not found" in err_df.loc[0, "error_message"]


def test_main_resume_skips_completed_rows(tmp_path, monkeypatch):
    nifti = tmp_path / "valid.nii.gz"
    nifti.write_text("nifti")
    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": str(nifti)}]).to_csv(csv_path, index=False)

    calls = {"count": 0}

    def fake_process_single_volume(idx, row, *, phase_extractor):
        calls["count"] += 1
        return idx, {"totalseg_phase": "portal"}, None

    monkeypatch.setattr(phase_module, "_load_phase_extractor", lambda: (lambda _: {}))
    monkeypatch.setattr(phase_module, "process_single_volume", fake_process_single_volume)

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        verbose=False,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )
    phase_module.main(args)
    assert calls["count"] == 1

    calls["count"] = 0
    args.resume = True
    phase_module.main(args)
    assert calls["count"] == 0
