"""Synthetic geometry and cohort contracts for registration."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imperandi.process.registration import RegistrationConfig, register_cohort
from imperandi.process.registration.consensus import fuse_tumors
from imperandi.process.registration.organ import register_pair

sitk = pytest.importorskip("SimpleITK")


@pytest.fixture(autouse=True)
def single_thread():
    original = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    yield
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(original)


def image(values):
    img = sitk.GetImageFromArray(np.asarray(values, dtype=np.uint8))
    img.SetSpacing((1.2, 1.7, 2.1))
    img.SetOrigin((15.0, -24.0, 7.0))
    img.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    return img


def organ():
    z, y, x = np.indices((24, 28, 32))
    return image(((x - 15) / 8) ** 2 + ((y - 13) / 6) ** 2 + ((z - 11) / 4) ** 2 < 1)


@pytest.mark.parametrize("affine", [False, True])
def test_physical_translation_and_inverse(affine):
    fixed = organ()
    moving = sitk.Image(fixed)
    shift = np.array([4.0, -3.0, 2.0])
    moving.SetOrigin(tuple(np.array(fixed.GetOrigin()) + shift))
    result = register_pair(
        fixed, moving, RegistrationConfig(affine=affine, iterations=20)
    )
    point = fixed.TransformIndexToPhysicalPoint((15, 13, 11))
    assert np.allclose(
        result.reference_to_scan.TransformPoint(point),
        np.array(point) + shift,
        atol=0.1,
    )
    assert result.dice_after > 0.98
    assert np.allclose(
        result.reference_to_scan.GetInverse().TransformPoint(
            result.reference_to_scan.TransformPoint(point)
        ),
        point,
    )


@pytest.mark.parametrize(
    "method,expected",
    [
        ("anchor", [1, 0, 1, 0]),
        ("majority", [1, 0, 0, 0]),
        ("intersection", [1, 0, 0, 0]),
        ("union", [1, 1, 1, 0]),
    ],
)
def test_fusion_votes_and_ties(method, expected):
    masks = [image(np.array(v).reshape(1, 1, 4)) for v in [[1, 0, 1, 0], [1, 1, 0, 0]]]
    coverage = image(np.ones((1, 1, 4)))
    result = fuse_tumors(masks, [coverage, coverage], method=method)
    assert sitk.GetArrayFromImage(result.mask).ravel().tolist() == expected
    if method == "majority":
        assert sitk.GetArrayFromImage(result.probability).ravel().tolist() == [
            1,
            0.5,
            0.5,
            0,
        ]


@pytest.mark.parametrize(
    "method", ["anchor", "majority", "intersection", "union", "staple"]
)
def test_single_empty_and_coverage(method):
    empty = image(np.zeros((2, 3, 4)))
    coverage = image(np.ones((2, 3, 4)))
    result = fuse_tumors([empty], [coverage], method=method)
    assert not sitk.GetArrayFromImage(result.mask).any()
    with pytest.raises(ValueError, match="support"):
        fuse_tumors([empty], [empty], method=method)


def test_staple_excludes_unknown_voxels():
    a = np.zeros((3, 4, 6))
    a[:, 1:3, 1:4] = 1
    b = a.copy()
    b[0, 1, 1] = 0
    support = np.ones_like(a)
    support[:, :, -1] = 0
    result = fuse_tumors([image(a), image(b)], [image(support)] * 2, method="staple")
    p = sitk.GetArrayFromImage(result.probability)
    assert np.isfinite(p).all()
    assert not p[:, :, -1].any()
    assert p[1, 2, 2] > 0.5


def save_scan(
    tmp_path, name, visit="v1", modality="CT", phase="ARTERIAL", tumor=True, offset=0
):
    directory = tmp_path / name
    directory.mkdir()
    mask = organ()
    mask.SetOrigin(tuple(np.array(mask.GetOrigin()) + [offset, 0, 0]))
    for filename in ["image", "organ", "tumor"]:
        sitk.WriteImage(mask, str(directory / f"{filename}.nii.gz"))
    return {
        "patient_key": "001",
        "study_id": visit,
        "Modality": modality,
        "phase": phase,
        "mri_sequence": "T1",
        "nifti_path": str(directory / "image.nii.gz"),
        "mask_liver": str(directory / "organ.nii.gz"),
        "mask_liver_tumor": str(directory / "tumor.nii.gz") if tumor else None,
    }


def test_cohort_grouping_anchor_transfer_and_native_preservation(tmp_path):
    rows = [
        save_scan(tmp_path, "arterial", tumor=False, offset=4),
        save_scan(tmp_path, "portal", phase="PORTAL_VENOUS"),
        save_scan(tmp_path, "later", visit="v2"),
        save_scan(tmp_path, "mr", modality="MRI"),
    ]
    source = pd.DataFrame(rows)
    out, errors = register_cohort(
        source, tmp_path / "out", RegistrationConfig(iterations=10)
    )
    assert errors.empty
    unchanged = [c for c in source if not c.startswith("mask_")]
    pd.testing.assert_frame_equal(out[unchanged], source[unchanged])
    for column in ["mask_liver", "mask_liver_tumor"]:
        pd.testing.assert_series_equal(
            out[f"source_{column}"], source[column], check_names=False
        )
    assert out.mask_liver.equals(out.reg_organ_native_path)
    assert out.mask_liver_tumor.equals(out.reg_tumor_native_path)
    assert out.registration_group_id.nunique() == 3
    assert out.loc[0, "registration_reference_id"] == out.loc[1, "registration_scan_id"]
    native = sitk.ReadImage(out.loc[0, "reg_tumor_native_path"])
    expected = sitk.ReadImage(rows[0]["mask_liver"])
    assert native.GetOrigin() == expected.GetOrigin()
    assert np.array_equal(
        sitk.GetArrayFromImage(native), sitk.GetArrayFromImage(expected)
    )
    assert not any(c.startswith("mask_") for c in set(out) - set(source))


def test_failure_is_not_identity_success(tmp_path):
    rows = [save_scan(tmp_path, "a", phase="PORTAL_VENOUS"), save_scan(tmp_path, "b")]

    def fail(*args):
        raise ValueError("deliberate failure")

    out, errors = register_cohort(
        pd.DataFrame(rows), tmp_path / "out", pair_registration=fail
    )
    assert out.loc[1, "registration_status"] == "failed"
    assert pd.isna(out.loc[1, "reg_tumor_native_path"])
    assert errors.stage.tolist() == ["organ"]


def test_missing_identity_and_duplicate_rejected(tmp_path):
    row = save_scan(tmp_path, "a")
    with pytest.raises(ValueError, match="Duplicate"):
        register_cohort(pd.DataFrame([row, row]), tmp_path / "out")
    row["study_id"] = None
    with pytest.raises(ValueError, match="identity"):
        register_cohort(pd.DataFrame([row]), tmp_path / "out")


def test_config_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown"):
        RegistrationConfig.from_mapping({"afine": True})


def test_cli(tmp_path):
    from imperandi.cli import main

    rows = [save_scan(tmp_path, "a")]
    csv = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    output = tmp_path / "result.csv"
    assert (
        main(
            [
                "register",
                "--csv_path",
                str(csv),
                "--csv_path_out",
                str(output),
                "--output_dir",
                str(tmp_path / "out"),
                "--method",
                "union",
            ]
        )
        == 0
    )
    out = pd.read_csv(output, dtype={"patient_key": str})
    assert out.patient_key.tolist() == ["001"]
    assert out.consensus_status.tolist() == ["single_contributor"]


def test_pca_rotation_with_oblique_geometry():
    fixed = organ()
    # Change physical coordinates without resampling voxel data, giving an exact
    # rigid ground truth that exercises origin, spacing, and axis orientation.
    transform = sitk.Euler3DTransform()
    center = fixed.TransformIndexToPhysicalPoint((15, 13, 11))
    transform.SetCenter(center)
    transform.SetRotation(0.13, -0.17, 0.24)
    transform.SetTranslation((2.0, -3.0, 1.0))
    moving = sitk.Image(fixed)
    moving.SetOrigin(transform.TransformPoint(fixed.GetOrigin()))
    moving.SetDirection(
        tuple(
            (
                np.array(transform.GetMatrix()).reshape(3, 3)
                @ np.array(fixed.GetDirection()).reshape(3, 3)
            ).ravel()
        )
    )
    result = register_pair(fixed, moving, RegistrationConfig(iterations=20))
    assert result.dice_after > 0.95
    # PCA's shape symmetry permits equivalent 180-degree solutions; foreground
    # placement must still be accurate on the fixed validation grid.
    assert result.dice_after > result.dice_before


def test_probability_consensus_pipeline(tmp_path):
    rows = [save_scan(tmp_path, "a", phase="PORTAL_VENOUS"), save_scan(tmp_path, "b")]
    out, errors = register_cohort(
        pd.DataFrame(rows),
        tmp_path / "out",
        RegistrationConfig(method="majority", iterations=5),
    )
    assert errors.empty
    assert out.consensus_status.tolist() == ["ok", "ok"]
    assert out.reg_tumor_probability_native_path.notna().all()
    ref = sitk.ReadImage(rows[0]["nifti_path"])
    registered = sitk.ReadImage(out.loc[1, "reg_nifti_path"])
    assert registered.GetOrigin() == ref.GetOrigin()
    assert registered.GetPixelID() == sitk.sitkFloat32


def test_invalid_mask_and_missing_tumor(tmp_path):
    rows = [save_scan(tmp_path, "a", tumor=False)]
    out, errors = register_cohort(pd.DataFrame(rows), tmp_path / "out")
    assert errors.empty
    assert out.consensus_status.tolist() == ["no_tumor_input"]
    rows[0]["mask_liver"] = str(tmp_path / "missing.nii.gz")
    out, errors = register_cohort(pd.DataFrame(rows), tmp_path / "out2")
    assert out.registration_status.tolist() == ["failed"]
    assert out.reg_nifti_path.isna().all()
    assert len(errors) == 1


def test_anchor_falls_back_to_moving_mask(tmp_path):
    rows = [
        save_scan(tmp_path, "portal", phase="PORTAL_VENOUS", tumor=False),
        save_scan(tmp_path, "arterial", offset=3),
    ]
    out, errors = register_cohort(
        pd.DataFrame(rows), tmp_path / "out", RegistrationConfig(iterations=10)
    )
    assert errors.empty
    transferred = sitk.ReadImage(out.loc[0, "reg_tumor_native_path"])
    expected = sitk.ReadImage(rows[0]["mask_liver"])
    assert np.array_equal(
        sitk.GetArrayFromImage(transferred), sitk.GetArrayFromImage(expected)
    )


def test_affine_refines_scale():
    fixed = organ()
    moving = sitk.Image(fixed)
    moving.SetSpacing((fixed.GetSpacing()[0] * 1.2, *fixed.GetSpacing()[1:]))
    rigid = register_pair(fixed, moving, RegistrationConfig(iterations=100))
    result = register_pair(
        fixed, moving, RegistrationConfig(affine=True, iterations=100)
    )
    assert result.stage == "affine"
    assert result.reference_to_scan.GetName() == "AffineTransform"
    assert result.dice_after > 0.9
    assert result.dice_after > rigid.dice_after + 0.01


def test_affine_is_skipped_until_overlap_is_sufficient():
    fixed = organ()
    moving = sitk.Image(fixed)
    moving.SetSpacing((fixed.GetSpacing()[0] * 1.2, *fixed.GetSpacing()[1:]))
    result = register_pair(
        fixed,
        moving,
        RegistrationConfig(affine=True, affine_min_dice=0.99, iterations=20),
    )
    assert result.stage != "affine"
    assert result.dice_after < 0.99
    assert result.stages["affine"]["status"] == "skipped_low_dice"
    assert result.stages["affine"]["input_dice"] == result.dice_after
    assert result.stages["affine"]["required_dice"] == 0.99


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_affine_dice_threshold_is_validated(value):
    with pytest.raises(ValueError, match="threshold"):
        RegistrationConfig(affine_min_dice=value)


def test_intersection_ignores_unobserved_background():
    a = image(np.ones((2, 3, 4)))
    b = image(np.ones((2, 3, 4)))
    coverage = np.ones((2, 3, 4))
    coverage[:, :, 3] = 0
    result = fuse_tumors([a, b], [a, image(coverage)], method="intersection")
    mask = sitk.GetArrayFromImage(result.mask)
    assert mask[:, :, :3].all()
    assert not mask[:, :, 3].any()
    assert np.array_equal(sitk.GetArrayFromImage(result.coverage), coverage)


def test_numeric_visit_identifiers(tmp_path):
    row = save_scan(tmp_path, "a")
    row["study_id"] = 1
    out, errors = register_cohort(pd.DataFrame([row]), tmp_path / "out")
    assert errors.empty
    assert out.registration_status.tolist() == ["reference"]


def test_cli_cannot_overwrite_source_with_error_table(tmp_path):
    from argparse import Namespace
    from imperandi.process.registration.register import main

    args = Namespace(
        csv_path=str(tmp_path / "cohort_errors.csv"),
        csv_path_out=str(tmp_path / "cohort.csv"),
    )
    with pytest.raises(ValueError, match="differ"):
        main(args)


def test_native_canonical_masks_preserve_files_and_rerun_sources(tmp_path):
    rows = [
        save_scan(tmp_path, "portal", phase="PORTAL_VENOUS"),
        save_scan(tmp_path, "arterial", offset=4),
    ]
    source = pd.DataFrame(rows)
    originals = {
        path: Path(path).read_bytes()
        for row in rows
        for path in [row["nifti_path"], row["mask_liver"], row["mask_liver_tumor"]]
    }
    first, errors = register_cohort(
        source, tmp_path / "out", RegistrationConfig(iterations=10)
    )
    assert errors.empty
    for i, row in first.iterrows():
        scan = sitk.ReadImage(row.nifti_path)
        for column in ["mask_liver", "mask_liver_tumor"]:
            assert row[column] != rows[i][column]
            assert row[f"source_{column}"] == rows[i][column]
            mask = sitk.ReadImage(row[column])
            assert (
                mask.GetSize(),
                mask.GetOrigin(),
                mask.GetSpacing(),
                mask.GetDirection(),
            ) == (
                scan.GetSize(),
                scan.GetOrigin(),
                scan.GetSpacing(),
                scan.GetDirection(),
            )
    assert all(Path(path).read_bytes() == data for path, data in originals.items())
    # A fresh invocation on the output CSV must not use its fused masks as input.
    first_artifacts = {path: Path(path).read_bytes() for path in first.mask_liver_tumor}
    second, errors = register_cohort(
        first, tmp_path / "out", RegistrationConfig(iterations=10)
    )
    assert errors.empty
    assert second.source_mask_liver.equals(first.source_mask_liver)
    assert second.source_mask_liver_tumor.equals(first.source_mask_liver_tumor)
    assert set(second.mask_liver_tumor).isdisjoint(first.mask_liver_tumor)
    assert all(
        Path(path).read_bytes() == data for path, data in first_artifacts.items()
    )


def test_stage_qc_and_trace_logs(tmp_path, caplog):
    import json
    import logging

    caplog.set_level(logging.INFO)
    rows = [
        save_scan(tmp_path, "portal", phase="PORTAL_VENOUS"),
        save_scan(tmp_path, "arterial", offset=4),
    ]
    out, errors = register_cohort(
        pd.DataFrame(rows), tmp_path / "out", RegistrationConfig(iterations=10)
    )
    assert errors.empty
    qc = pd.read_csv(out.loc[1, "registration_qc_path"])
    moving = qc.set_index("registration_scan_id").loc[
        out.loc[1, "registration_scan_id"]
    ]
    assert 0 <= moving.dice_baseline < moving.dice_pca <= 1
    assert 0 <= moving.dice_rigid <= 1
    assert pd.isna(moving.dice_affine)
    assert moving.affine_status == "not_run"
    assert moving.registration_reference_id == out.loc[0, "registration_scan_id"]
    events = [
        json.loads(line)
        for line in Path(out.loc[1, "registration_log_path"]).read_text().splitlines()
    ]
    assert any(
        event.get("event") == "scan_result"
        and event["registration_scan_label"] == moving.registration_scan_label
        and event["stages"]["rigid"]["dice"] == pytest.approx(moving.dice_rigid)
        for event in events
    )
    assert any(
        "series=2/2" in record.message
        and "phase=ARTERIAL" in record.message
        and "rigid=" in record.message
        for record in caplog.records
    )


def test_logs_identify_groups_with_human_attributes(tmp_path, caplog):
    import json
    import logging

    caplog.set_level(logging.INFO)
    rows = [
        {
            **save_scan(tmp_path, "portal", phase="PORTAL_VENOUS"),
            "patient_id": "PATIENT-A",
            "date": "2024-05-17",
            "visit_order": 2,
            "series_id": "SERIES-SECRET",
        }
    ]
    out, errors = register_cohort(pd.DataFrame(rows), tmp_path / "out")
    assert errors.empty
    expected = (
        "patient_id=PATIENT-A, date=2024-05-17, visit_order=2, visit=v1, modality=CT"
    )
    assert out.loc[0, "registration_group_label"] == expected
    info_messages = [
        record.message for record in caplog.records if record.levelno == logging.INFO
    ]
    assert sum(expected in message for message in info_messages) == 1
    assert not any(
        out.loc[0, "registration_group_id"] in message for message in info_messages
    )
    assert any("series=1/1" in message for message in info_messages)
    assert not any("SERIES-SECRET" in message for message in info_messages)
    assert not any("image.nii.gz" in message for message in info_messages)
    log_text = Path(out.loc[0, "registration_log_path"]).read_text()
    events = [json.loads(line) for line in log_text.splitlines()]
    assert "SERIES-SECRET" not in log_text
    assert "image.nii.gz" not in log_text
    assert all(
        "series_id" not in event and "nifti_path" not in event for event in events
    )
    group_events = [event for event in events if event["event"] == "group_context"]
    scan_events = [event for event in events if event["event"] != "group_context"]
    assert group_events == [
        {
            "event": "group_context",
            "patient_id": "PATIENT-A",
            "date": "2024-05-17",
            "visit_order": "2",
            "visit": "v1",
            "modality": "CT",
            "series_count": 1,
            "registration_reference_label": (
                "series=1/1, phase=PORTAL_VENOUS, sequence=T1"
            ),
            "consensus_method": "anchor",
        }
    ]
    assert len(scan_events) == 1
    assert all("patient_id" not in event for event in scan_events)
    assert all(
        "series=1/1" in event["registration_scan_label"] for event in scan_events
    )
    qc = pd.read_csv(out.loc[0, "registration_qc_path"])
    assert qc.loc[0, "patient_id"] == "PATIENT-A"
    assert qc.loc[0, "date"] == "2024-05-17"
    assert qc.loc[0, "visit_order"] == 2


def test_scan_labels_number_series_within_each_group():
    from imperandi.process.registration.grouping import prepare_cohort

    rows = [
        {
            "patient_key": "001",
            "study_id": "visit-1",
            "Modality": "CT",
            "phase": phase,
            "series_id": f"PRIVATE-{index}",
            "nifti_path": f"/private/location/scan-{index}.nii.gz",
            "mask_liver": f"/private/location/liver-{index}.nii.gz",
        }
        for index, phase in enumerate(
            ["NATIVE", "DELAYED", "OTHER", "ARTERIAL", "OTHER", "PORTAL_VENOUS"]
        )
    ]
    planned = prepare_cohort(pd.DataFrame(rows), RegistrationConfig())
    labels = planned.registration_scan_label.tolist()
    positions = planned.registration_series_number.tolist()
    assert positions[0:2] == [4, 3]
    assert positions[3] == 2
    assert positions[5] == 1
    assert {positions[2], positions[4]} == {5, 6}
    assert all(
        f"series={position}/6" in labels[index]
        for index, position in enumerate(positions)
    )
    assert not any(
        "patient_id=" in label or "PRIVATE" in label or "/private/" in label
        for label in labels
    )


def test_rejected_pair_keeps_qc_and_original_canonical_paths(tmp_path):
    from imperandi.process.registration.organ import (
        RegistrationRejected,
        TransformResult,
    )

    rows = [
        save_scan(tmp_path, "portal", phase="PORTAL_VENOUS"),
        save_scan(tmp_path, "arterial"),
    ]

    def rejected(*args):
        result = TransformResult(
            sitk.Euler3DTransform(),
            "rigid",
            0.1,
            0.2,
            [],
            {
                "baseline": {"dice": 0.1, "status": "evaluated"},
                "pca": {"dice": 0.15, "status": "evaluated"},
                "rigid": {"dice": 0.2, "status": "evaluated"},
                "affine": {"dice": None, "status": "not_run"},
            },
        )
        raise RegistrationRejected("Overlap rejected", result)

    out, errors = register_cohort(
        pd.DataFrame(rows), tmp_path / "out", pair_registration=rejected
    )
    assert out.loc[1, "mask_liver"] == rows[1]["mask_liver"]
    assert out.loc[1, "mask_liver_tumor"] == rows[1]["mask_liver_tumor"]
    qc = pd.read_csv(out.loc[1, "registration_qc_path"]).set_index(
        "registration_scan_id"
    )
    moving = qc.loc[out.loc[1, "registration_scan_id"]]
    assert moving.dice_rigid == 0.2
    assert moving.registration_status == "failed"
    assert "Overlap rejected" in moving.errors
