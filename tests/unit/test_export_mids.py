"""Focused tests for the strict CT/MR volumetric MIDS-style exporter."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from imperandi import cli
from imperandi.export.mids.core import _export_config, build_plan
from imperandi.export.mids.identity import IdentityMapper
from imperandi.export.mids.models import MidsExportError


def _volume(path: Path, content: bytes = b"synthetic-nifti") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path)


def _cohort(tmp_path: Path) -> pd.DataFrame:
    images = [
        _volume(tmp_path / "source" / f"image-{index}.nii.gz", bytes([index]))
        for index in range(1, 6)
    ]
    masks = [
        _volume(tmp_path / "source" / f"mask-{index}.nii.gz", bytes([index + 10]))
        for index in range(1, 5)
    ]
    common = {
        "patient_key": "RAW/PATIENT 17",
        "series_id": "1.2.840.series-common",
        "PatientName": "Identifying^Name",
        "PatientBirthDate": "19700101",
        "AccessionNumber": "ACCESS-SECRET",
        "ReferringPhysicianName": "Doctor^Name",
        "BodyPartExamined": "ABDOMEN",
        "exam_stage": "baseline",
        "phase_source": "metadata_rules",
        "phase_confidence": 0.95,
        "segmentation_qc_status": "pass",
        "registration_qc_status": "pass",
        "registration_confidence": 0.9,
    }
    return pd.DataFrame(
        [
            {
                **common,
                "study_id": "1.2.840.exam-one",
                "volume_id": "1.2.840.volume-one",
                "Modality": "CT",
                "phase": "ARTERIAL",
                "nifti_path": images[0],
                "mask_liver": masks[0],
                "consensus_mask_liver": masks[1],
                "liver_original_firstorder_Mean": 11.5,
            },
            {
                **common,
                "study_id": "1.2.840.exam-one",
                "volume_id": "1.2.840.volume-two",
                "Modality": "CT",
                "phase": "ARTERIAL",
                "nifti_path": images[1],
            },
            {
                **common,
                "study_id": "1.2.840.exam-one",
                "volume_id": "1.2.840.volume-three",
                "Modality": "MRI",
                "phase": "PORTAL_VENOUS",
                "mri_sequence": "T2",
                "mri_sequence_confidence": "high",
                "mri_sequence_reason": "curated T2 classification",
                "nifti_path": images[2],
            },
            {
                **common,
                "study_id": "1.2.840.exam-two",
                "volume_id": "1.2.840.volume-four",
                "Modality": "CT",
                "phase": "UNKNOWN-CUSTOM-VALUE",
                "BodyPartExamined": pd.NA,
                "nifti_path": images[3],
                "consensus_mask_liver": masks[2],
                "registration_qc_status": "low_confidence",
                "registration_confidence": 0.1,
            },
            {
                **common,
                "study_id": "1.2.840.exam-three",
                "volume_id": "1.2.840.volume-five",
                "Modality": "US",
                "nifti_path": images[4],
                "mask_liver": masks[3],
            },
        ]
    )


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *extra: str,
    with_key: bool = True,
) -> tuple[int, Path, Path]:
    frame = _cohort(tmp_path)
    csv_path = tmp_path / "cohort.csv"
    output = tmp_path / "mids-dataset"
    frame.to_csv(csv_path, index=False)
    if with_key:
        monkeypatch.setenv("IMPERANDI_MIDS_KEY", "a-test-key-with-at-least-16-bytes")
    code = cli.main(
        [
            "export",
            "mids",
            "--csv_path",
            str(csv_path),
            "--output_dir",
            str(output),
            *extra,
        ]
    )
    return code, csv_path, output


def test_dry_run_reports_plan_and_writes_nothing(tmp_path, monkeypatch, capsys):
    code, _, output = _run(tmp_path, monkeypatch, "--dry-run")
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["status"] == "ready_with_exclusions"
    assert report["counts"] == {
        "images": 4,
        "masks": 2,
        "radiomics_rows": 1,
        "excluded_rows": 2,
        "planned_dataset_files": report["counts"]["planned_dataset_files"],
    }
    assert any(item.get("kind") == "consensus" for item in report["excluded_rows"])
    assert any(
        "Unsupported modality" in reason
        for item in report["excluded_rows"]
        for reason in item["reasons"]
    )
    assert report["resolved_collisions"]
    assert not output.exists()


def test_export_tree_sessions_collisions_privacy_and_derivatives(tmp_path, monkeypatch):
    code, csv_path, output = _run(tmp_path, monkeypatch)
    assert code == 0

    images = sorted(
        path.relative_to(output) for path in output.glob("sub-*/ses-*/mim-*/*.nii.gz")
    )
    assert len(images) == 4
    assert {path.parts[2] for path in images} == {"mim-ct", "mim-mr"}
    assert (
        len({path.parts[1] for path in images}) == 2
    )  # repeated clinical stage, distinct exams
    arterial = [path for path in images if "pc-arterial" in path.name]
    assert len(arterial) == 2
    assert {"run-01", "run-02"} == {
        next(part for part in path.stem.split("_") if part.startswith("run-"))
        for path in arterial
    }
    mr_image = next(path for path in images if path.parts[2] == "mim-mr")
    assert "acq-t2" in mr_image.name
    assert "pc-portal" in mr_image.name
    unknown = next(
        path
        for path in images
        if path.parts[2] == "mim-ct"
        and "UNKNOWN" not in path.name
        and "pc-" not in path.name
    )
    assert "bp-" not in unknown.name

    exported_text = "\n".join(
        path.read_text(errors="ignore")
        for path in output.rglob("*")
        if path.is_file() and not path.name.endswith((".nii", ".gz"))
    )
    for forbidden in (
        "RAW/PATIENT 17",
        "Identifying^Name",
        "19700101",
        "ACCESS-SECRET",
        "Doctor^Name",
        str(csv_path),
        str(tmp_path / "source"),
    ):
        assert forbidden not in exported_text
    assert (
        "UNKNOWN-CUSTOM-VALUE" in exported_text
    )  # retained as metadata, not guessed into a filename

    source_masks = list(output.glob("derivatives/imperandi-segmentation/**/*.nii.gz"))
    consensus_masks = list(
        output.glob("derivatives/imperandi-registration-consensus/**/*.nii.gz")
    )
    assert len(source_masks) == 1
    assert len(consensus_masks) == 1
    assert "desc-source" in source_masks[0].name
    assert "desc-consensus" in consensus_masks[0].name

    source_table = pd.read_csv(
        output / "derivatives/imperandi-segmentation/masks.tsv", sep="\t"
    )
    consensus_table = pd.read_csv(
        output / "derivatives/imperandi-registration-consensus/masks.tsv", sep="\t"
    )
    assert source_table.loc[0, "source_image"] != source_table.loc[0, "mask_path"]
    assert consensus_table.loc[0, "mask_kind"] == "consensus"
    assert (output / source_table.loc[0, "source_image"]).is_file()
    assert (output / consensus_table.loc[0, "mask_path"]).is_file()

    radiomics = pd.read_csv(
        output / "derivatives/imperandi-radiomics/radiomics.tsv", sep="\t"
    )
    assert radiomics.loc[0, "liver_original_firstorder_Mean"] == 11.5
    assert (output / radiomics.loc[0, "source_image"]).is_file()

    participants = pd.read_csv(output / "participants.tsv", sep="\t")
    assert participants.columns.tolist() == ["participant_id", "sex"]
    assert not any(
        column in participants
        for column in ("PatientName", "PatientBirthDate", "nifti_path")
    )
    for scans_path in output.glob("sub-*/ses-*/*_scans.tsv"):
        scans = pd.read_csv(scans_path, sep="\t")
        assert scans["filename"].map(lambda value: not Path(value).is_absolute()).all()
        assert (
            scans["filename"]
            .map(lambda value: (scans_path.parent / value).is_file())
            .all()
        )
        mr_scans = scans[scans["modality"].eq("MR")]
        if not mr_scans.empty:
            assert mr_scans.iloc[0]["mri_sequence_classification"] == "T2"
            assert mr_scans.iloc[0]["mri_sequence_confidence"] == "high"
            assert (
                mr_scans.iloc[0]["mri_sequence_reason"] == "curated T2 classification"
            )


def test_rerun_rejects_existing_output_unless_replace_selected(tmp_path, monkeypatch):
    assert _run(tmp_path, monkeypatch)[0] == 0
    with pytest.raises(MidsExportError, match="Output already exists"):
        _run(tmp_path, monkeypatch)
    assert _run(tmp_path, monkeypatch, "--overwrite", "replace")[0] == 0


def test_id_map_auto_is_dry_run_safe_then_generates_and_reuses_map(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("IMPERANDI_MIDS_KEY", raising=False)
    code, csv_path, output = _run(
        tmp_path,
        monkeypatch,
        "--id-map",
        "auto",
        "--dry-run",
        with_key=False,
    )
    report = json.loads(capsys.readouterr().out)
    map_path = csv_path.with_name(f"{csv_path.stem}_mids_id_map.csv")
    assert code == 0
    assert report["automatic_id_map"]["written"] is False
    assert report["automatic_id_map"]["existing_status"] == "missing"
    assert not map_path.exists()
    assert not output.exists()

    code, _, output = _run(tmp_path, monkeypatch, "--id-map", "auto", with_key=False)
    assert code == 0
    assert output.is_dir()
    generated = pd.read_csv(map_path, dtype=str)
    assert {
        "mapped_patient_key",
        "mapped_study_id",
        "mapped_series_id",
        "mapped_volume_id",
        "participant_label",
        "session_label",
        "image_label",
    }.issubset(generated.columns)
    assert generated["mapped_volume_id"].str.fullmatch(r"\d{4}").all()

    # The same cohort deterministically reuses the protected map.
    assert (
        _run(
            tmp_path,
            monkeypatch,
            "--id-map",
            "auto",
            "--overwrite",
            "replace",
            with_key=False,
        )[0]
        == 0
    )


def test_filename_collision_from_duplicate_volume_identity_is_reported(
    tmp_path, monkeypatch
):
    frame = _cohort(tmp_path).iloc[:2].copy()
    frame.loc[1, "volume_id"] = frame.loc[0, "volume_id"]
    monkeypatch.setenv("IMPERANDI_MIDS_KEY", "a-test-key-with-at-least-16-bytes")
    plan = build_plan(frame, tmp_path / "out", _export_config(None))
    assert plan.collisions
    assert plan.report()["status"] == "invalid"


def test_unsafe_external_labels_and_exam_stage_only_are_rejected(tmp_path):
    config = _export_config(None)["identity"]
    config = {**config, "session": {"source_columns": ["exam_stage"]}}
    with pytest.raises(MidsExportError, match="exam_stage cannot be the sole"):
        IdentityMapper(config, key_file=str(tmp_path / "missing"))

    id_map = tmp_path / "ids.csv"
    pd.DataFrame(
        [
            {
                "patient_key": "p1",
                "study_id": "e1",
                "volume_id": "v1",
                "participant_label": "unsafe/path",
                "session_label": "safe01",
                "image_label": "image01",
            }
        ]
    ).to_csv(id_map, index=False)
    with pytest.raises(MidsExportError, match="Unsafe participant_label"):
        IdentityMapper(_export_config(None)["identity"], id_map_path=str(id_map))


def test_external_map_supplies_all_safe_export_identifiers(tmp_path, monkeypatch):
    frame = _cohort(tmp_path).iloc[:1].copy()
    id_map = tmp_path / "ids.csv"
    pd.DataFrame(
        [
            {
                "patient_key": frame.loc[0, "patient_key"],
                "study_id": frame.loc[0, "study_id"],
                "volume_id": frame.loc[0, "volume_id"],
                "participant_label": "participant01",
                "session_label": "exam01",
                "image_label": "volume01",
            }
        ]
    ).to_csv(id_map, index=False)
    monkeypatch.delenv("IMPERANDI_MIDS_KEY", raising=False)
    plan = build_plan(
        frame,
        tmp_path / "out",
        _export_config(None),
        id_map_path=str(id_map),
    )
    image = next(item for item in plan.files if item.kind == "image")
    assert image.relative_path.parts[:2] == ("sub-participant01", "ses-exam01")
    scans = next(
        rows for path, rows in plan.tables.items() if path.name.endswith("_scans.tsv")
    )
    assert scans[0]["source_image_id"] == "volume01"


def test_missing_key_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.delenv("IMPERANDI_MIDS_KEY", raising=False)
    with pytest.raises(MidsExportError, match="No privacy-safe identity mapping"):
        IdentityMapper(_export_config(None)["identity"])
