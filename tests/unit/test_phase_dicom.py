"""Optional DICOM conversion for rows reaching TotalSegmentator phase prediction."""

import argparse
from pathlib import Path
import zipfile

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import yaml

from imperandi import cli
from imperandi.extract import phase
from imperandi.process import convert
from imperandi.utils.archive_io import encode_archive_uri


def dicom_paths(root):
    root.mkdir()
    paths = [root / "a.dcm", root / "b.dcm"]
    for path in paths:
        path.write_bytes(b"slice")
    # Only the requested volume's files should reach conversion.
    (root / "other.dcm").write_bytes(b"another volume")
    return [str(path) for path in paths]


@pytest.fixture
def conversion(tmp_path, monkeypatch):
    state = {"outputs": [], "directories": [], "payloads": []}
    temporary_directory = convert.tempfile.TemporaryDirectory
    monkeypatch.setattr(convert.tempfile, "tempdir", str(tmp_path))

    def temporary(*args, **kwargs):
        directory = temporary_directory(*args, **kwargs)
        state["directories"].append(Path(directory.name))
        return directory

    def read(directory):
        payloads = [path.read_bytes() for path in sorted(Path(directory).iterdir())]
        state["payloads"].append(payloads)
        return payloads

    def write(payloads, output, reorient_nifti):
        assert reorient_nifti is False
        state["outputs"].append(Path(output))
        image = nib.Nifti1Image(
            np.full((2, 2, 2), sum(map(len, payloads)), dtype=np.int16), np.eye(4)
        )
        nib.save(image, output)

    monkeypatch.setattr(convert.tempfile, "TemporaryDirectory", temporary)
    monkeypatch.setattr(convert.dicom2nifti.common, "read_dicom_directory", read)
    monkeypatch.setattr(
        convert.dicom2nifti.convert_dicom, "dicom_array_to_nifti", write
    )
    return state


@pytest.mark.parametrize("source", ["files", "archive"])
@pytest.mark.parametrize("missing", ["absent", "null", "blank", "file", "directory"])
@pytest.mark.parametrize("generator", [False, True])
def test_missing_nifti_converts_selected_dicom_and_cleans_up(
    tmp_path, conversion, source, missing, generator
):
    paths = dicom_paths(tmp_path / "slices")
    if source == "archive":
        archive = tmp_path / "dicom.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            for path in paths:
                bundle.write(path, f"series/{Path(path).name}")
        paths = [encode_archive_uri(archive, [f"series/{Path(p).name}"]) for p in paths]
    row = {"dicom_path": repr(paths), "series_dir": "obsolete-directory"}
    if missing != "absent":
        row["nifti_path"] = {
            "null": None,
            "blank": " ",
            "file": str(tmp_path / "missing.nii.gz"),
            "directory": str(tmp_path),
        }[missing]
    original = row.copy()

    def predict(image, quiet):
        assert quiet
        assert Path(image.get_filename()).is_file()
        assert image.get_fdata().mean() == 10

        def updates():
            assert Path(image.get_filename()).is_file()
            yield {"result": {"phase": "portal"}}

        return updates() if generator else {"phase": "portal"}

    result = phase.process_single_volume(
        7, row, phase_extractor=predict, force_nifti_conversion=True
    )

    assert result == (7, {"totalseg_phase": "portal"}, None)
    assert conversion["payloads"] == [[b"slice", b"slice"]]
    assert all(not path.exists() for path in conversion["directories"])
    assert row == original


@pytest.mark.parametrize("enabled", [False, True])
def test_conversion_requires_opt_in_and_reuses_available_nifti(
    tmp_path, conversion, enabled
):
    paths = dicom_paths(tmp_path / "slices")
    row = {"dicom_path": paths}
    result = phase.process_single_volume(0, row, phase_extractor=None)
    assert "missing or invalid" in result[2]
    assert not conversion["directories"]

    nifti = tmp_path / "existing.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)), nifti)
    result = phase.process_single_volume(
        0,
        {**row, "nifti_path": str(nifti)},
        phase_extractor=lambda image, quiet: {"phase": "arterial"},
        force_nifti_conversion=enabled,
    )
    assert result == (0, {"totalseg_phase": "arterial"}, None)
    assert not conversion["directories"]
    assert nifti.is_file()


@pytest.mark.parametrize("failure", ["conversion", "invalid_output", "prediction"])
def test_conversion_and_prediction_errors_clean_up(
    tmp_path, conversion, monkeypatch, failure
):
    paths = dicom_paths(tmp_path / "slices")
    if failure == "conversion":

        def fail(*args):
            raise ValueError("bad DICOM")

        monkeypatch.setattr(convert.dicom2nifti.common, "read_dicom_directory", fail)
    elif failure == "invalid_output":
        monkeypatch.setattr(
            convert.dicom2nifti.convert_dicom,
            "dicom_array_to_nifti",
            lambda *a, **k: None,
        )

    def predict(image, quiet):
        raise ValueError("prediction failed")

    _, metadata, error = phase.process_single_volume(
        0, {"dicom_path": paths}, phase_extractor=predict, force_nifti_conversion=True
    )
    assert metadata is None
    assert {
        "conversion": "bad DICOM",
        "invalid_output": "valid nifti",
        "prediction": "prediction failed",
    }[failure] in error
    assert conversion["directories"]
    assert all(not path.exists() for path in conversion["directories"])


@pytest.mark.parametrize("setting", ["disabled", "cli", "manifest"])
def test_phase_command_converts_only_rows_needing_prediction(
    tmp_path, conversion, monkeypatch, setting
):
    paths = dicom_paths(tmp_path / "slices")
    source, output = tmp_path / "input.csv", tmp_path / "output.csv"
    rows = pd.DataFrame(
        {
            "volume_id": ["rules", "ct1", "ct2", "mr", "existing"],
            "Modality": ["CT", "CT", "CT", "MR", "CT"],
            "rule_phase": ["ARTERIAL", "OTHER", "OTHER", "OTHER", "OTHER"],
            "totalseg_phase": [None, None, None, None, "native"],
            "dicom_path": [paths] * 5,
        }
    )
    rows.to_csv(source, index=False)
    loads = []

    def load():
        loads.append(True)
        return lambda image, quiet: {"phase": "portal"}

    monkeypatch.setattr(phase, "_load_phase_extractor", load)
    command = ["phase", str(source), str(output)]
    if setting == "cli":
        command.append("--force-nifti-conversion")
    elif setting == "manifest":
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text(
            yaml.safe_dump(
                {
                    "phase_curation": {
                        "strategies": [
                            {"type": "rules"},
                            {
                                "type": "totalsegmentator",
                                "force_nifti_conversion": True,
                                "mapping": {
                                    "portal": "PORTAL_VENOUS",
                                    "native": "NATIVE",
                                },
                            },
                        ]
                    }
                }
            )
        )
        command += ["--manifest", str(manifest)]

    assert cli.main(command) == 0

    result = pd.read_csv(output).set_index("volume_id")
    predicted_phase = "OTHER" if setting == "disabled" else "PORTAL_VENOUS"
    assert result.loc[rows.volume_id, "phase"].tolist() == [
        "ARTERIAL",
        predicted_phase,
        predicted_phase,
        "OTHER",
        "NATIVE",
    ]
    assert "nifti_path" not in result
    assert result.dicom_path.tolist() == [repr(paths)] * 5
    assert len(conversion["outputs"]) == (0 if setting == "disabled" else 2)
    assert loads == ([] if setting == "disabled" else [True])
    assert all(not path.exists() for path in conversion["directories"])


@pytest.mark.parametrize("failed", [False, True])
def test_resume_reprocesses_changed_dicom_paths_without_stale_prediction(
    tmp_path, conversion, monkeypatch, failed
):
    paths = [dicom_paths(tmp_path / name) for name in ("first", "second")]
    source, output = tmp_path / "input.csv", tmp_path / "output.csv"
    pd.DataFrame(
        {
            "volume_id": ["v1", "v2"],
            "Modality": ["CT", "CT"],
            "rule_phase": ["OTHER", "OTHER"],
            "dicom_path": paths,
        }
    ).to_csv(source, index=False)
    predictions = []

    def predict(image, quiet):
        changed = image.get_fdata().mean() != 10
        predictions.append(changed)
        if changed and failed:
            raise ValueError("prediction failed")
        return {"phase": "delayed" if changed else "portal"}

    monkeypatch.setattr(phase, "_load_phase_extractor", lambda: predict)
    args = argparse.Namespace(
        csv_path=str(source),
        csv_path_out=str(output),
        error_csv_path=str(tmp_path / "errors.csv"),
        manifest="generic",
        verbose=False,
        force=False,
        force_nifti_conversion=True,
        resume=True,
        strict_resume=True,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
    )
    phase.main(args)
    phase.main(args)
    assert predictions == [False, False]
    assert len(conversion["outputs"]) == 2

    # File contents do not affect resume when the CSV path values are unchanged,
    # even with strict_resume enabled.
    completed_output = pd.read_csv(output)
    Path(paths[0][0]).write_bytes(b"changed slice")
    phase.main(args)
    assert predictions == [False, False]
    assert len(conversion["outputs"]) == 2
    pd.testing.assert_frame_equal(pd.read_csv(output), completed_output)

    # A new path value invalidates only its row and must discard old predictions.
    replacement_paths = dicom_paths(tmp_path / "replacement")
    Path(replacement_paths[0]).write_bytes(b"changed slice")
    rows = pd.read_csv(source)
    rows.loc[0, "dicom_path"] = repr(replacement_paths)
    rows.to_csv(source, index=False)

    phase.main(args)

    result = pd.read_csv(output)
    assert predictions == [False, False, True]
    assert len(conversion["outputs"]) == 3
    assert result.dicom_path.tolist() == [repr(replacement_paths), repr(paths[1])]
    assert result.phase.tolist() == ["OTHER" if failed else "DELAYED", "PORTAL_VENOUS"]
    if failed:
        assert pd.isna(result.loc[0, "totalseg_phase"])
    assert all(not path.exists() for path in conversion["directories"])


def test_conversion_setting_defaults_false_and_requires_boolean():
    assert phase.build_parser().parse_args([]).force_nifti_conversion is False
    config = phase.validate_phase_curation(
        {"strategies": [{"type": "totalsegmentator"}]}
    )
    assert config["strategies"][0]["force_nifti_conversion"] is False
    with pytest.raises(ValueError, match="force_nifti_conversion must be a boolean"):
        phase.validate_phase_curation(
            {
                "strategies": [
                    {"type": "totalsegmentator", "force_nifti_conversion": "true"}
                ]
            }
        )
