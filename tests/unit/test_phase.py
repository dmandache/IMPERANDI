import sys
from pathlib import Path

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import logging

import pandas as pd
import pytest

from imperandi.extract import phase as phase_module


@pytest.mark.parametrize("change", ["rule_phase", "phase_applicability_reason"])
def test_semantic_resume_rechecks_curated_phase_inputs(tmp_path, monkeypatch, change):
    csv_path = tmp_path / "input.csv"
    output = tmp_path / "output.csv"
    rows = pd.DataFrame({
        "volume_id": ["v1", "v2"],
        "Modality": ["CT", "CT"],
        "rule_phase": ["ARTERIAL", "PORTAL_VENOUS"],
        "phase_applicability_reason": [None, None],
    })
    rows.to_csv(csv_path, index=False)
    monkeypatch.setattr(
        phase_module, "_load_phase_extractor",
        lambda: pytest.fail("Metadata-resolved rows must not load the predictor"),
    )
    args = argparse.Namespace(
        csv_path=str(csv_path), csv_path_out=str(output),
        error_csv_path=str(tmp_path / "errors.csv"), manifest="generic",
        verbose=False, force=False, resume=True, strict_resume=False,
        checkpoint_every_rows=1, checkpoint_every_sec=3600,
    )
    phase_module.main(args)
    rows.loc[0, change] = "DELAYED" if change == "rule_phase" else "LOCALIZER"
    rows.to_csv(csv_path, index=False)
    phase_module.main(args)
    result = pd.read_csv(output)
    assert result.loc[1, "phase"] == "PORTAL_VENOUS"
    if change == "rule_phase":
        assert result.loc[0, "phase"] == "DELAYED"
    else:
        assert result.loc[0, "phase_status"] == "NOT_APPLICABLE"
        assert pd.isna(result.loc[0, "phase"])


@pytest.mark.parametrize("failed", [False, True])
def test_semantic_resume_reruns_only_changed_image_without_stale_prediction(
    tmp_path, monkeypatch, failed
):
    csv_path = tmp_path / "input.csv"
    output = tmp_path / "output.csv"
    paths = [tmp_path / "a.nii.gz", tmp_path / "b.nii.gz"]
    for path in paths:
        path.write_bytes(b"original")
    pd.DataFrame({
        "volume_id": ["v1", "v2"], "Modality": ["CT", "CT"],
        "rule_phase": ["OTHER", "OTHER"], "nifti_path": list(map(str, paths)),
    }).to_csv(csv_path, index=False)
    calls = []

    def predict(idx, row, **kwargs):
        calls.append(row["volume_id"])
        changed = paths[0].read_bytes() != b"original"
        if changed and failed:
            return idx, None, "prediction failed"
        return idx, {"totalseg_phase": "delayed" if changed else "portal"}, None

    monkeypatch.setattr(phase_module, "_load_phase_extractor", lambda: object())
    monkeypatch.setattr(phase_module, "process_single_volume", predict)
    args = argparse.Namespace(
        csv_path=str(csv_path), csv_path_out=str(output),
        error_csv_path=str(tmp_path / "errors.csv"), manifest="generic",
        verbose=False, force=False, resume=True, strict_resume=True,
        checkpoint_every_rows=1, checkpoint_every_sec=3600,
    )
    phase_module.main(args)
    assert calls == ["v1", "v2"]
    calls.clear()
    paths[0].write_bytes(b"changed image contents")
    phase_module.main(args)
    assert calls == ["v1"]
    result = pd.read_csv(output)
    assert result.loc[1, "phase"] == "PORTAL_VENOUS"
    assert result.loc[0, "phase"] == ("OTHER" if failed else "DELAYED")
    if failed:
        assert pd.isna(result.loc[0, "totalseg_phase"])


def test_normalize_phase_args_defaults(tmp_path):
    csv_path = tmp_path / "nifti_index.csv"
    csv_path.write_text("nifti_path\n")

    args = argparse.Namespace(
        csv_path_pos=str(csv_path),
        csv_path_opt=None,
        csv_path_out_pos=None,
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
    assert not hasattr(out, "csv_path_out_pos")


def test_normalize_phase_args_accepts_positional_csv_path_out(tmp_path):
    csv_path = tmp_path / "nifti_index.csv"
    csv_path.write_text("nifti_path\n")
    csv_out = tmp_path / "phase_custom.csv"

    args = argparse.Namespace(
        csv_path_pos=str(csv_path),
        csv_path_opt=None,
        csv_path_out_pos=str(csv_out),
        csv_path_out=None,
        error_csv_path=None,
        totalseg_home_dir=None,
        verbose=False,
        dry_run=False,
    )

    out = phase_module.normalize_phase_args(args)

    assert out.csv_path_out == str(csv_out)
    assert not hasattr(out, "csv_path_out_pos")


def test_normalize_phase_args_prefers_flag_csv_path_out(tmp_path):
    csv_path = tmp_path / "nifti_index.csv"
    csv_path.write_text("nifti_path\n")
    csv_out_pos = tmp_path / "phase_pos.csv"
    csv_out_opt = tmp_path / "phase_opt.csv"

    args = argparse.Namespace(
        csv_path_pos=str(csv_path),
        csv_path_opt=None,
        csv_path_out_pos=str(csv_out_pos),
        csv_path_out=str(csv_out_opt),
        error_csv_path=None,
        totalseg_home_dir=None,
        verbose=False,
        dry_run=False,
    )

    out = phase_module.normalize_phase_args(args)

    assert out.csv_path_out == str(csv_out_opt)


def test_process_single_volume_success(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    monkeypatch.setattr(phase_module.nib, "load", lambda _: object())

    idx, phase_info, err = phase_module.process_single_volume(
        0,
        {"nifti_path": str(nifti)},
        phase_extractor=lambda _, quiet=True: {
            "phase": "portal",
            "probability": 0.9,
        },
    )

    assert idx == 0
    assert err is None
    assert phase_info["totalseg_phase"] == "portal"
    assert phase_info["totalseg_probability"] == 0.9


def test_process_single_volume_supports_generator_api(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    monkeypatch.setattr(phase_module.nib, "load", lambda _: object())

    def generator_extractor(_, quiet=True):
        yield {"id": 1, "progress": 2, "status": "Loading data"}
        yield {"id": 4, "progress": 85, "status": "Predicting phase"}
        yield {
            "id": 5,
            "progress": 100,
            "status": "Done",
            "result": {"phase": "portal_venous", "probability": 0.9},
        }

    idx, phase_info, err = phase_module.process_single_volume(
        0,
        {"nifti_path": str(nifti)},
        phase_extractor=generator_extractor,
    )

    assert idx == 0
    assert err is None
    assert phase_info["totalseg_phase"] == "portal_venous"
    assert phase_info["totalseg_probability"] == 0.9


def test_process_single_volume_generator_without_result_fails(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    monkeypatch.setattr(phase_module.nib, "load", lambda _: object())

    def generator_extractor(_, quiet=True):
        yield {"id": 1, "progress": 2, "status": "Loading data"}
        yield {"id": 4, "progress": 85, "status": "Predicting phase"}

    idx, phase_info, err = phase_module.process_single_volume(
        0,
        {"nifti_path": str(nifti)},
        phase_extractor=generator_extractor,
    )

    assert idx == 0
    assert phase_info is None
    assert err == "phase extractor did not return a prediction dictionary"


def test_process_single_volume_missing_file():
    idx, phase_info, err = phase_module.process_single_volume(
        0,
        {"nifti_path": "does/not/exist.nii.gz"},
        phase_extractor=lambda _, quiet=True: {"phase": "portal"},
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
        lambda: lambda _, quiet=True: {"phase": "arterial", "confidence": 0.8},
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
    extractor_loads = {"count": 0}

    def fake_process_single_volume(idx, row, *, phase_extractor, verbose=False):
        calls["count"] += 1
        return idx, {"totalseg_phase": "portal"}, None

    def fake_load_phase_extractor():
        extractor_loads["count"] += 1
        return lambda _: {}

    monkeypatch.setattr(
        phase_module, "_load_phase_extractor", fake_load_phase_extractor
    )
    monkeypatch.setattr(
        phase_module, "process_single_volume", fake_process_single_volume
    )

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(csv_path),
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
    assert extractor_loads["count"] == 1


def test_main_skips_rows_with_existing_totalseg_phase_when_not_forced(
    tmp_path, monkeypatch, caplog
):
    caplog.set_level(logging.INFO, logger=phase_module.__name__)
    nifti = tmp_path / "valid.nii.gz"
    nifti.write_text("nifti")
    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": str(nifti), "totalseg_phase": "portal"}]).to_csv(
        csv_path, index=False
    )

    calls = {"count": 0}
    monkeypatch.setattr(phase_module.nib, "load", lambda _: object())
    monkeypatch.setattr(
        phase_module,
        "_load_phase_extractor",
        lambda: lambda _, quiet=True: {"phase": "arterial"},
    )

    def fake_process_single_volume(idx, row, *, phase_extractor, verbose=False):
        calls["count"] += 1
        return idx, {"totalseg_phase": "arterial"}, None

    monkeypatch.setattr(
        phase_module, "process_single_volume", fake_process_single_volume
    )

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        verbose=False,
        force=False,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )
    phase_module.main(args)

    assert calls["count"] == 0
    out_df = pd.read_csv(args.csv_path_out)
    assert out_df.loc[0, "totalseg_phase"] == "portal"
    assert "1 resumed" in caplog.text


def test_main_force_recomputes_existing_totalseg_phase(tmp_path, monkeypatch):
    nifti = tmp_path / "valid.nii.gz"
    nifti.write_text("nifti")
    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": str(nifti), "totalseg_phase": "portal"}]).to_csv(
        csv_path, index=False
    )

    calls = {"count": 0}
    monkeypatch.setattr(phase_module.nib, "load", lambda _: object())
    monkeypatch.setattr(
        phase_module,
        "_load_phase_extractor",
        lambda: lambda _, quiet=True: {"phase": "arterial"},
    )

    def fake_process_single_volume(idx, row, *, phase_extractor, verbose=False):
        calls["count"] += 1
        return idx, {"totalseg_phase": "arterial"}, None

    monkeypatch.setattr(
        phase_module, "process_single_volume", fake_process_single_volume
    )

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        verbose=False,
        force=True,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )
    phase_module.main(args)

    assert calls["count"] == 1
    out_df = pd.read_csv(args.csv_path_out)
    assert out_df.loc[0, "totalseg_phase"] == "arterial"


def test_main_preserves_foreign_columns_from_existing_output(tmp_path, monkeypatch):
    nifti = tmp_path / "valid.nii.gz"
    nifti.write_text("nifti")
    csv_path = tmp_path / "nifti_index.csv"
    out_path = tmp_path / "out.csv"
    pd.DataFrame([{"nifti_path": str(nifti)}]).to_csv(csv_path, index=False)
    pd.DataFrame([{"nifti_path": str(nifti), "foreign_col": "keep-me"}]).to_csv(
        out_path, index=False
    )

    monkeypatch.setattr(phase_module.nib, "load", lambda _: object())
    monkeypatch.setattr(
        phase_module,
        "_load_phase_extractor",
        lambda: lambda _, quiet=True: {"phase": "portal"},
    )

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(out_path),
        error_csv_path=str(tmp_path / "errors.csv"),
        verbose=False,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )
    phase_module.main(args)

    out_df = pd.read_csv(out_path)
    assert "foreign_col" in out_df.columns
    assert out_df.loc[0, "foreign_col"] == "keep-me"


def test_main_skips_totalsegmentator_when_rules_resolve_phase(
    tmp_path, monkeypatch, caplog
):
    caplog.set_level(logging.INFO, logger=phase_module.__name__)
    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame(
        [
            {
                "nifti_path": str(tmp_path / "not-needed.nii.gz"),
                "rule_phase": "ARTERIAL",
                "rule_phase_confidence": "high",
            }
        ]
    ).to_csv(csv_path, index=False)

    def fail_if_loaded():
        raise AssertionError("TotalSegmentator should not load after a rule match")

    monkeypatch.setattr(phase_module, "_load_phase_extractor", fail_if_loaded)
    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        manifest="generic",
        verbose=False,
        force=False,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )

    phase_module.main(args)

    out_df = pd.read_csv(args.csv_path_out)
    assert out_df.loc[0, "phase"] == "ARTERIAL"
    assert out_df.loc[0, "phase_source"] == "metadata_rules"
    assert "totalseg_phase" not in out_df.columns
    assert "Phase strategy order:" in caplog.text
    assert "selected 0/1 volume(s)" in caplog.text
    assert "metadata_rules (rules) resolved 1 volume(s); 0 unresolved" in caplog.text


def test_main_does_not_run_ct_predictor_for_mri(tmp_path, monkeypatch):
    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame(
        [
            {
                "nifti_path": str(tmp_path / "not-needed.nii.gz"),
                "Modality": "MR",
                "rule_phase": "OTHER",
            }
        ]
    ).to_csv(csv_path, index=False)

    def fail_if_loaded():
        raise AssertionError("The CT phase predictor must not load for MRI")

    monkeypatch.setattr(phase_module, "_load_phase_extractor", fail_if_loaded)
    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        manifest="generic",
        verbose=False,
        force=False,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )

    phase_module.main(args)

    out_df = pd.read_csv(args.csv_path_out)
    assert out_df.loc[0, "phase"] == "OTHER"
    assert out_df.loc[0, "phase_source"] == "fallback"
