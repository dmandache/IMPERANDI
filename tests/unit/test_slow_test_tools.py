"""Fast checks for the dataset-backed test tooling."""

from __future__ import annotations

import zipfile
from pathlib import Path

from tests.slow import run_pipeline as pipeline_module
from tests.slow.ircad.download import _safe_extract


def test_ircad_extraction_keeps_only_patient_scan_data(tmp_path):
    archive = tmp_path / "patient.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("3Dircadb1.1/PATIENT_DICOM.zip", b"scan archive")
        output.writestr("3Dircadb1.1/MASKS_DICOM/liver.zip", b"mask archive")

    destination = tmp_path / "input"
    _safe_extract(archive, destination, "3Dircadb1.1")

    assert (destination / "3Dircadb1.1" / "PATIENT_DICOM.zip").is_file()
    assert not (destination / "3Dircadb1.1" / "MASKS_DICOM").exists()


def test_shared_pipeline_runner_executes_every_stage(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    input_dir = dataset_dir / "data" / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "scan.dcm").write_bytes(b"fixture")
    commands = []
    monkeypatch.setattr(pipeline_module, "_run", commands.append)

    work_dir = pipeline_module.run_pipeline(dataset_dir)

    assert [command[3] for command in commands] == list(pipeline_module.STAGES)
    assert work_dir == (dataset_dir / "data" / "work").resolve()


def test_user_facing_pipeline_scripts_show_every_stage_without_runner():
    slow_dir = Path(__file__).resolve().parents[1] / "slow"
    for dataset in ("ircad", "tcga_lihc"):
        for filename in ("pipeline.sh", "pipeline.ps1"):
            script = (slow_dir / dataset / filename).read_text(encoding="utf-8")
            assert "run_pipeline.py" not in script
            for stage in pipeline_module.STAGES:
                assert f"imperandi {stage}" in script
            assert "NumWorkers = 1" in script or 'NUM_WORKERS="${4:-1}"' in script
            for setting in ("SCRIPT_DIR", "INPUT_DIR", "WORK_DIR", "MANIFEST"):
                assert setting in script
