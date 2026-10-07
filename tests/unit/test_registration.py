"""Synthetic geometry and cohort contracts for registration."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imperandi.process.registration import RegistrationConfig, register_cohort
from imperandi.process.registration import alignment
from imperandi.process.registration.cohort import fuse_organs, fuse_tumors
from imperandi.process.registration.alignment import register_pair
from imperandi.process.registration.reporting import build_qc

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


def organ_field(values):
    mask = image(values)
    field = sitk.GetImageFromArray(
        np.where(np.asarray(values) > 0, -1.0, 1.0).astype(np.float32)
    )
    field.CopyInformation(mask)
    return field


def organ():
    z, y, x = np.indices((24, 28, 32))
    return image(((x - 15) / 8) ** 2 + ((y - 13) / 6) ** 2 + ((z - 11) / 4) ** 2 < 1)


def test_boundary_band_is_physical_shell_around_organ():
    values = np.zeros((21, 21, 21), dtype=np.uint8)
    values[5:16, 5:16, 5:16] = 1
    mask = sitk.GetImageFromArray(values)
    mask.SetSpacing((1.0, 2.0, 3.0))

    band = sitk.GetArrayFromImage(alignment.boundary_band(mask, band_mm=2.1))

    assert band[10, 10, 5]
    assert band[10, 10, 4]
    assert not band[10, 10, 10]
    assert not band[0, 0, 0]


def test_mutual_information_score_is_deterministic():
    fixed = organ()
    config = RegistrationConfig()
    transform = sitk.Euler3DTransform()

    first = alignment.mutual_information_score(
        fixed, fixed, fixed, fixed, transform, config
    )
    second = alignment.mutual_information_score(
        fixed, fixed, fixed, fixed, transform, config
    )

    assert np.isfinite(first)
    assert second == pytest.approx(first, abs=1e-12)


def test_rigid_mi_refinement_remains_available():
    fixed = organ()

    transform, _ = alignment.mi_refine(
        fixed,
        fixed,
        fixed,
        fixed,
        sitk.Euler3DTransform(),
        RegistrationConfig(maximum_optimizer_iterations=1),
    )

    assert transform.GetName() == "Euler3DTransform"


@pytest.mark.parametrize("affine", [False, True])
def test_physical_translation_and_inverse(affine):
    fixed = organ()
    moving = sitk.Image(fixed)
    shift = np.array([4.0, -3.0, 2.0])
    moving.SetOrigin(tuple(np.array(fixed.GetOrigin()) + shift))
    result = register_pair(
        fixed,
        moving,
        RegistrationConfig(enable_affine_stage=affine, maximum_optimizer_iterations=20),
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
    "tumor_consensus,expected",
    [
        ("anchor", [1, 0, 1, 0]),
        ("majority", [1, 0, 0, 0]),
        ("intersection", [1, 0, 0, 0]),
        ("union", [1, 1, 1, 0]),
    ],
)
def test_fusion_votes_and_ties(tumor_consensus, expected):
    masks = [image(np.array(v).reshape(1, 1, 4)) for v in [[1, 0, 1, 0], [1, 1, 0, 0]]]
    coverage = image(np.ones((1, 1, 4)))
    result = fuse_tumors(masks, [coverage, coverage], tumor_consensus=tumor_consensus)
    assert sitk.GetArrayFromImage(result.mask).ravel().tolist() == expected
    if tumor_consensus == "majority":
        assert sitk.GetArrayFromImage(result.probability).ravel().tolist() == [
            1,
            0.5,
            0.5,
            0,
        ]


@pytest.mark.parametrize(
    "organ_consensus,expected",
    [
        ("anchor", [1, 0, 1, 0]),
        ("majority", [1, 0, 0, 0]),
    ],
)
def test_organ_consensus_anchor_and_majority(organ_consensus, expected):
    fields = [
        organ_field(np.array(v).reshape(1, 1, 4))
        for v in [[1, 0, 1, 0], [1, 1, 0, 0]]
    ]
    coverage = image(np.ones((1, 1, 4)))

    result = fuse_organs(fields, [coverage, coverage], organ_consensus=organ_consensus)

    assert sitk.GetArrayFromImage(result.mask).ravel().tolist() == expected
    assert result.mask.GetPixelID() == sitk.sitkUInt8


def test_organ_majority_excludes_unobserved_votes():
    fields = [organ_field([[[1, 1]]]), organ_field([[[0, 0]]])]
    coverages = [image([[[1, 0]]]), image([[[0, 1]]])]

    result = fuse_organs(fields, coverages, organ_consensus="majority")

    assert sitk.GetArrayFromImage(result.mask).ravel().tolist() == [1, 0]
    assert sitk.GetArrayFromImage(result.observation_count).ravel().tolist() == [1, 1]


@pytest.mark.parametrize("partial", [False, True])
def test_geometry_replaces_pca_only_for_partial_organs(monkeypatch, partial):
    fixed = organ()
    if partial:
        values = sitk.GetArrayFromImage(fixed)
        values[:11] = 0
        fixed = image(values)
    moving = sitk.Image(fixed)
    shift = np.array([100.0, -80.0, 60.0])
    moving.SetOrigin(tuple(np.array(fixed.GetOrigin()) + shift))
    calls = []
    initializer = sitk.CenteredTransformInitializer

    def geometry(*args):
        calls.append("geometry")
        assert args[-1] == sitk.CenteredTransformInitializerFilter.GEOMETRY
        return initializer(*args)

    def pca(*args):
        calls.append("pca")
        return initializer(
            *args,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

    monkeypatch.setattr(sitk, "CenteredTransformInitializer", geometry)
    monkeypatch.setattr(alignment, "initialize_pca", pca)
    result = register_pair(
        fixed, moving, RegistrationConfig(maximum_optimizer_iterations=1)
    )
    assert result.dice_before == 0
    assert result.dice_after == pytest.approx(1)
    expected_stage = "geometry" if partial else "pca"
    assert calls == [expected_stage]
    assert result.stage == expected_stage
    assert result.stages[expected_stage]["selected"] is True
    assert result.stages[expected_stage]["early_stop"] is True
    assert result.stages["mask_rigid"]["status"] == "skipped_early_stop"
    assert result.stages["mask_affine"]["status"] == "skipped_early_stop"
    assert result.stages["mi_affine"]["status"] == "skipped_early_stop"
    assert result.stages["mask_elastic"]["status"] == "skipped_early_stop"
    assert "center_of_mass" not in result.stages
    if partial:
        assert result.stages["pca"]["status"] == "skipped_partial_coverage"
    else:
        assert result.stages["geometry"]["status"] == "skipped_complete_organ"
    point = fixed.TransformIndexToPhysicalPoint((15, 13, 11))
    assert np.allclose(result.reference_to_scan.TransformPoint(point), point + shift)
    assert np.allclose(result.scan_to_reference.TransformPoint(point + shift), point)


@pytest.mark.parametrize(
    "tumor_consensus", ["anchor", "majority", "intersection", "union", "staple"]
)
def test_empty_resampled_tumor_preserves_observed_negative_votes(tumor_consensus):
    empty = image(np.zeros((2, 3, 4)))
    coverage = image(np.ones((2, 3, 4)))
    result = fuse_tumors([empty], [coverage], tumor_consensus=tumor_consensus)
    assert not sitk.GetArrayFromImage(result.mask).any()
    with pytest.raises(ValueError, match="support"):
        fuse_tumors([coverage], [empty], tumor_consensus=tumor_consensus)


def test_staple_excludes_unknown_voxels():
    a = np.zeros((3, 4, 6))
    a[:, 1:3, 1:4] = 1
    b = a.copy()
    b[0, 1, 1] = 0
    support = np.ones_like(a)
    support[:, :, -1] = 0
    result = fuse_tumors(
        [image(a), image(b)], [image(support)] * 2, tumor_consensus="staple"
    )
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
        source,
        tmp_path / "out",
        RegistrationConfig(maximum_optimizer_iterations=10),
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


def test_manifest_grouping_columns_control_cross_modality_registration(tmp_path):
    rows = [
        {**save_scan(tmp_path, "ct", modality="CT"), "exam_stage": "baseline"},
        {**save_scan(tmp_path, "mr", modality="MRI"), "exam_stage": "baseline"},
        {**save_scan(tmp_path, "followup"), "exam_stage": "followup"},
    ]
    config = RegistrationConfig(
        grouping_columns=["patient_key", "exam_stage"],
        reference_selection_priority=[{"Modality": ["MR", "CT"]}],
        maximum_optimizer_iterations=5,
    )

    out, errors = register_cohort(pd.DataFrame(rows), tmp_path / "out", config)

    assert errors.empty
    assert out.registration_group_id.nunique() == 2
    assert out.loc[0, "registration_group_id"] == out.loc[1, "registration_group_id"]
    assert out.loc[1, "registration_status"] == "reference"
    assert out.loc[0, "registration_reference_id"] == out.loc[1, "registration_scan_id"]
    assert out.loc[2, "registration_status"] == "reference"


def test_modality_column_is_optional_when_not_configured(tmp_path):
    from imperandi.process.registration.cohort import prepare_cohort

    rows = [save_scan(tmp_path, "a"), save_scan(tmp_path, "b")]
    for row in rows:
        row.pop("Modality")
    planned = prepare_cohort(
        pd.DataFrame(rows),
        RegistrationConfig(
            grouping_columns=["patient_key", "study_id"],
            reference_selection_priority=[{"phase": ["PORTAL_VENOUS", "ARTERIAL"]}],
        ),
    )

    assert planned.registration_group_id.nunique() == 1
    assert planned.registration_group_size.tolist() == [2, 2]


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


def test_tumor_resampling_failure_preserves_organ_registration(monkeypatch, tmp_path):
    from imperandi.process.registration import cohort

    row = save_scan(tmp_path, "scan")

    def fail(*args, **kwargs):
        raise RuntimeError("tumor resampling failed")

    monkeypatch.setattr(cohort, "resample", fail)
    out, errors = register_cohort(
        pd.DataFrame([row]),
        tmp_path / "out",
        RegistrationConfig(preserve_source_organ_mask=True),
    )
    assert out.loc[0, "registration_status"] == "reference"
    assert out.loc[0, "consensus_status"] == "failed"
    assert out.loc[0, "mask_liver_tumor"] == row["mask_liver_tumor"]
    assert errors.stage.tolist() == ["consensus"]
    assert errors.error.tolist() == ["tumor resampling failed"]


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
                "--organ_consensus_method",
                "majority",
                "--tumor_consensus_method",
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
    result = register_pair(
        fixed, moving, RegistrationConfig(maximum_optimizer_iterations=20)
    )
    assert result.dice_after > 0.95
    # Foreground placement remains accurate with the safe-angle PCA candidates.
    assert result.dice_after > result.dice_before


def test_pca_excludes_rotation_beyond_configured_limit():
    fixed = organ()
    transform = sitk.Euler3DTransform()
    center = fixed.TransformIndexToPhysicalPoint((15, 13, 11))
    transform.SetCenter(center)
    transform.SetRotation(0.0, 0.0, np.pi / 2)
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

    limited = alignment.initialize_pca(fixed, moving, max_rotation_degrees=30.0)
    angle = alignment._rotation_angle_degrees(
        np.array(limited.GetMatrix()).reshape(3, 3)
    )

    assert angle <= 30.0 + 1e-6


def test_majority_consensus_keeps_only_native_masks(tmp_path):
    rows = [save_scan(tmp_path, "a", phase="PORTAL_VENOUS"), save_scan(tmp_path, "b")]
    output_dir = tmp_path / "out"
    out, errors = register_cohort(
        pd.DataFrame(rows),
        output_dir,
        RegistrationConfig(
            tumor_consensus_method="majority", maximum_optimizer_iterations=5
        ),
    )
    assert errors.empty
    assert out.consensus_status.tolist() == ["ok", "ok"]
    assert out.reg_organ_native_path.notna().all()
    assert out.reg_tumor_native_path.notna().all()
    for index, row in out.iterrows():
        source = sitk.ReadImage(rows[index]["nifti_path"])
        tumor = sitk.ReadImage(row.reg_tumor_native_path)
        assert tumor.GetOrigin() == source.GetOrigin()
        assert tumor.GetSize() == source.GetSize()
    artifacts = [path.name for path in output_dir.rglob("*") if path.is_file()]
    assert artifacts.count("organ_native.nii.gz") == 2
    assert artifacts.count("tumor_native.nii.gz") == 2
    assert artifacts.count("registration.jsonl") == 1
    assert len(artifacts) == 5
    assert not set(out).intersection(
        {
            "registration_report_path",
            "reg_reference_to_scan_path",
            "reg_scan_to_reference_path",
            "reg_tumor_common_path",
            "reg_tumor_coverage_common_path",
            "reg_tumor_coverage_native_path",
            "reg_tumor_probability_common_path",
            "reg_tumor_probability_native_path",
            "reg_nifti_path",
            "reg_organ_path",
        }
    )


def test_majority_organ_consensus_is_written_to_native_grids(tmp_path):
    rows = [
        save_scan(tmp_path, "portal", phase="PORTAL_VENOUS"),
        save_scan(tmp_path, "arterial"),
    ]
    moving = sitk.ReadImage(rows[1]["mask_liver"])
    values = sitk.GetArrayFromImage(moving)
    values[11, 13, 15] = 0
    altered = sitk.GetImageFromArray(values)
    altered.CopyInformation(moving)
    sitk.WriteImage(altered, rows[1]["mask_liver"])

    out, errors = register_cohort(
        pd.DataFrame(rows),
        tmp_path / "out",
        RegistrationConfig(
            organ_consensus_method="majority", maximum_optimizer_iterations=5
        ),
    )

    assert errors.empty
    assert out.mask_liver.equals(out.reg_organ_native_path)
    for index, path in enumerate(out.reg_organ_native_path):
        consensus = sitk.ReadImage(path)
        assert consensus.GetPixelID() == sitk.sitkUInt8
        assert sitk.GetArrayFromImage(consensus)[11, 13, 15] == 0
        assert out.loc[index, "registration_final_organ_component_count"] == 1
        assert out.loc[index, "registration_final_organ_fragment_count"] == 0
        assert out.loc[index, "registration_final_organ_residual_seam_voxels"] == 0
        assert (
            abs(out.loc[index, "registration_final_organ_volume_change_fraction"])
            < 0.15
        )


def test_invalid_mask_and_missing_tumor(tmp_path):
    rows = [save_scan(tmp_path, "a", tumor=False)]
    out, errors = register_cohort(pd.DataFrame(rows), tmp_path / "out")
    assert errors.empty
    assert out.consensus_status.tolist() == ["no_tumor_input"]
    assert out.tumor_consensus_input_status.tolist() == ["skipped_missing"]
    rows[0]["mask_liver"] = str(tmp_path / "missing.nii.gz")
    out, errors = register_cohort(pd.DataFrame(rows), tmp_path / "out2")
    assert out.registration_status.tolist() == ["skipped"]
    assert out.registration_skip_reason.tolist() == ["missing_organ_mask"]
    assert out.consensus_status.tolist() == ["skipped"]
    assert out.reg_organ_native_path.isna().all()
    assert errors.empty


def test_empty_organ_is_skipped_and_excluded_from_active_group(tmp_path):
    rows = [
        save_scan(tmp_path, "portal", phase="PORTAL_VENOUS"),
        save_scan(tmp_path, "empty", phase="ARTERIAL"),
    ]
    empty = sitk.ReadImage(rows[1]["mask_liver"])
    empty = sitk.Image(empty.GetSize(), sitk.sitkUInt8)
    empty.CopyInformation(sitk.ReadImage(rows[1]["nifti_path"]))
    sitk.WriteImage(empty, rows[1]["mask_liver"])

    out, errors = register_cohort(pd.DataFrame(rows), tmp_path / "out")

    assert errors.empty
    assert out.registration_status.tolist() == ["reference", "skipped"]
    assert pd.isna(out.loc[0, "registration_skip_reason"])
    assert out.loc[1, "registration_skip_reason"] == "empty_organ_mask"
    assert out.loc[1, "consensus_status"] == "skipped"
    assert out.loc[0, "registration_group_size"] == 1
    assert out.loc[0, "registration_series_number"] == 1
    assert pd.isna(out.loc[1, "registration_group_size"])
    assert pd.isna(out.loc[1, "registration_series_number"])


def test_empty_tumor_is_excluded_from_consensus(tmp_path):
    rows = [
        save_scan(tmp_path, "portal", phase="PORTAL_VENOUS"),
        save_scan(tmp_path, "empty-tumor", phase="ARTERIAL"),
    ]
    image = sitk.ReadImage(rows[1]["nifti_path"])
    empty = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    empty.CopyInformation(image)
    sitk.WriteImage(empty, rows[1]["mask_liver_tumor"])

    out, errors = register_cohort(
        pd.DataFrame(rows),
        tmp_path / "out",
        RegistrationConfig(
            tumor_consensus_method="majority", maximum_optimizer_iterations=5
        ),
    )

    assert errors.empty
    assert out.tumor_consensus_input_status.tolist() == [
        "contributed",
        "skipped_empty",
    ]
    assert out.consensus_status.tolist() == ["single_contributor"] * 2


def test_unreadable_organ_mask_remains_an_input_failure(tmp_path):
    row = save_scan(tmp_path, "corrupt")
    Path(row["mask_liver"]).write_text("not a NIfTI image")

    out, errors = register_cohort(pd.DataFrame([row]), tmp_path / "out")

    assert out.registration_status.tolist() == ["failed"]
    assert errors.stage.tolist() == ["input"]


def test_anchor_falls_back_to_moving_mask(tmp_path):
    rows = [
        save_scan(tmp_path, "portal", phase="PORTAL_VENOUS", tumor=False),
        save_scan(tmp_path, "arterial", offset=3),
    ]
    out, errors = register_cohort(
        pd.DataFrame(rows),
        tmp_path / "out",
        RegistrationConfig(maximum_optimizer_iterations=10),
    )
    assert errors.empty
    transferred = sitk.ReadImage(out.loc[0, "reg_tumor_native_path"])
    expected = sitk.ReadImage(rows[0]["mask_liver"])
    assert np.array_equal(
        sitk.GetArrayFromImage(transferred), sitk.GetArrayFromImage(expected)
    )


def test_signed_distance_mask_affine_refines_scale():
    fixed = organ()
    moving = sitk.Image(fixed)
    moving.SetSpacing((fixed.GetSpacing()[0] * 1.2, *fixed.GetSpacing()[1:]))
    result = register_pair(
        fixed,
        moving,
        RegistrationConfig(
            early_stop_organ_dice=1,
            maximum_optimizer_iterations=100,
        ),
    )
    assert result.stage == "mask_affine"
    assert result.dice_after > 0.9
    assert result.dice_after > result.stages["mask_affine"]["input_dice"]
    assert result.stages["mask_affine"]["status"] == "evaluated"
    assert result.stages["mask_affine"]["metric"] == "mean_squares_signed_distance"


def test_mask_affine_runs_between_mask_rigid_and_mi_affine(monkeypatch):
    calls = []
    scores = iter([0.1, 0.2, 0.3, 0.4, 0.4])
    mutual_information = iter([0.1, 0.2])

    class Optimizer:
        def GetOptimizerStopConditionDescription(self):
            return "test"

        def GetOptimizerIteration(self):
            return 1

    def mask_rigid(*args):
        calls.append("mask_rigid")
        return sitk.Euler3DTransform(), Optimizer()

    def mask_affine(*args):
        calls.append("mask_affine")
        return sitk.AffineTransform(3), Optimizer()

    def mi_affine(*args, **kwargs):
        assert kwargs["affine"] is True
        calls.append("mi_affine")
        return sitk.AffineTransform(3), Optimizer()

    monkeypatch.setattr(alignment, "dice", lambda *args: next(scores))
    monkeypatch.setattr(
        alignment,
        "mutual_information_score",
        lambda *args: next(mutual_information),
    )
    monkeypatch.setattr(
        alignment, "initialize_pca", lambda *args: sitk.Euler3DTransform()
    )
    monkeypatch.setattr(alignment, "mask_rigid_refine", mask_rigid)
    monkeypatch.setattr(alignment, "mask_affine_refine", mask_affine)
    monkeypatch.setattr(alignment, "mi_refine", mi_affine)

    result = register_pair(
        organ(),
        organ(),
        RegistrationConfig(enable_affine_stage=True, early_stop_organ_dice=1),
    )

    assert calls == ["mask_rigid", "mask_affine", "mi_affine"]
    assert list(scores) == []
    assert list(mutual_information) == []
    assert result.stage == "mi_affine"


def test_boundary_band_mi_affine_refines_scale():
    fixed = organ()
    moving = sitk.Image(fixed)
    moving.SetSpacing((fixed.GetSpacing()[0] * 1.2, *fixed.GetSpacing()[1:]))
    result = register_pair(
        fixed,
        moving,
        RegistrationConfig(
            enable_affine_stage=True,
            early_stop_organ_dice=1,
            maximum_optimizer_iterations=100,
        ),
    )
    assert result.stage == "mi_affine"
    assert result.dice_after > 0.9
    assert result.dice_after > result.stages["mi_affine"]["input_dice"]
    assert result.stages["mi_affine"]["status"] == "evaluated"


@pytest.mark.parametrize(
    "scores,affine,selected_stage,rejected_stage,fallback_stage,selected_dice",
    [
        ([0.5, 0.8, 0.7, 0.6], False, "pca", "mask_rigid", "pca", 0.8),
        (
            [0.5, 0.8, 0.9, 0.85, 0.7],
            True,
            "mask_rigid",
            "mi_affine",
            "mask_rigid",
            0.9,
        ),
    ],
)
def test_worse_stage_falls_back_to_previous_best(
    monkeypatch,
    scores,
    affine,
    selected_stage,
    rejected_stage,
    fallback_stage,
    selected_dice,
):
    class Optimizer:
        def GetOptimizerStopConditionDescription(self):
            return "test"

        def GetOptimizerIteration(self):
            return 1

    values = iter(scores)
    monkeypatch.setattr(
        alignment,
        "initialize_pca",
        lambda *args: sitk.Euler3DTransform(),
    )
    monkeypatch.setattr(alignment, "dice", lambda *args: next(values))
    monkeypatch.setattr(
        alignment,
        "mask_rigid_refine",
        lambda *args: (sitk.Euler3DTransform(), Optimizer()),
    )
    monkeypatch.setattr(
        alignment,
        "mask_affine_refine",
        lambda *args: (sitk.AffineTransform(3), Optimizer()),
    )
    monkeypatch.setattr(
        alignment,
        "mi_refine",
        lambda *args, **kwargs: (
            sitk.AffineTransform(3)
            if kwargs.get("affine")
            else sitk.Euler3DTransform(),
            Optimizer(),
        ),
    )

    result = alignment.register_pair(
        organ(),
        organ(),
        RegistrationConfig(
            enable_affine_stage=affine,
            early_stop_organ_dice=1,
            maximum_optimizer_iterations=1,
        ),
    )

    assert list(values) == []
    assert result.stage == selected_stage
    assert result.dice_after == selected_dice
    assert result.stages[rejected_stage]["status"] == "rejected_worse_dice"
    assert result.stages[rejected_stage]["fallback_stage"] == fallback_stage
    assert result.stages[rejected_stage]["selected"] is False
    assert result.stages[selected_stage]["selected"] is True


def test_dice_and_mi_stages_use_their_respective_acceptance_criteria(monkeypatch):
    class Optimizer:
        def GetOptimizerStopConditionDescription(self):
            return "test"

        def GetOptimizerIteration(self):
            return 1

    scores = iter([0.5, 0.5005, 0.5015, 0.501, 0.499])
    mutual_information = iter([0.1, 0.2])
    monkeypatch.setattr(alignment, "dice", lambda *args: next(scores))
    monkeypatch.setattr(
        alignment,
        "mutual_information_score",
        lambda *args: next(mutual_information),
    )
    monkeypatch.setattr(
        alignment, "initialize_pca", lambda *args: sitk.Euler3DTransform()
    )
    monkeypatch.setattr(
        alignment,
        "mask_rigid_refine",
        lambda *args: (sitk.Euler3DTransform(), Optimizer()),
    )
    monkeypatch.setattr(
        alignment,
        "mask_affine_refine",
        lambda *args: (sitk.AffineTransform(3), Optimizer()),
    )
    monkeypatch.setattr(
        alignment,
        "mi_refine",
        lambda *args, **kwargs: (
            sitk.AffineTransform(3)
            if kwargs.get("affine")
            else sitk.Euler3DTransform(),
            Optimizer(),
        ),
    )

    result = register_pair(
        organ(),
        organ(),
        RegistrationConfig(
            enable_affine_stage=True,
            early_stop_organ_dice=1,
            minimum_stage_mi_improvement={"mi_affine": 0.001},
        ),
    )

    assert list(scores) == []
    assert list(mutual_information) == []
    assert result.stage == "mi_affine"
    assert result.dice_after == pytest.approx(0.499)
    assert "mi_rigid" not in result.stages
    for name, required in (
        ("pca", 0.001),
        ("mask_rigid", 0.002),
        ("mask_affine", 0.002),
    ):
        assert result.stages[name]["status"] == "rejected_insufficient_improvement"
        assert result.stages[name]["minimum_required_dice_improvement"] == required
    assert result.stages["mi_affine"]["status"] == "evaluated"
    assert result.stages["mi_affine"]["dice_improvement"] == pytest.approx(-0.001)
    assert result.stages["mi_affine"]["mutual_information_improvement"] == 0.1
    assert result.stages["mi_affine"]["minimum_required_mi_improvement"] == 0.001
    assert result.stages["mi_affine"]["dice_guard_reference"] == 0.5


def test_mi_candidate_requires_improvement_and_respects_peak_dice_guard():
    evaluate = alignment._mi_candidate_rejection

    assert evaluate(0.998, 1.0, 0.1, 0.2, 0.0, 0.002) is None
    assert evaluate(0.997, 1.0, 0.1, 0.2, 0.0, 0.002) == "rejected_worse_dice"
    assert (
        evaluate(1.0, 1.0, 0.1, 0.1, 0.0, 0.002)
        == "rejected_insufficient_mi_improvement"
    )


def test_failed_linear_setup_retains_previous_alignment(monkeypatch):
    def unavailable():
        raise RuntimeError("optimizer unavailable")

    scores = iter([0.5, 0.5, 0.5])
    monkeypatch.setattr(alignment, "dice", lambda *args: next(scores))
    monkeypatch.setattr(
        alignment, "initialize_pca", lambda *args: sitk.Euler3DTransform()
    )
    monkeypatch.setattr(sitk, "ImageRegistrationMethod", unavailable)
    result = register_pair(organ(), organ(), RegistrationConfig())
    assert result.dice_after == 0.5
    assert result.stages["mask_rigid"]["status"] == "failed"
    assert result.stages["mask_rigid"]["fallback_stage"] == "baseline"
    assert result.stages["mask_affine"]["status"] == "failed"
    assert result.stages["mask_affine"]["fallback_stage"] == "baseline"
    assert result.warnings == [
        "mask_rigid: optimizer unavailable",
        "mask_affine: optimizer unavailable",
    ]


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_nonfinite_candidate_retains_previous_alignment(monkeypatch, score):
    class Optimizer:
        def GetOptimizerStopConditionDescription(self):
            return "test"

        def GetOptimizerIteration(self):
            return 1

    scores = iter([0.5, score, score, score])
    monkeypatch.setattr(alignment, "dice", lambda *args: next(scores))
    monkeypatch.setattr(
        alignment, "initialize_pca", lambda *args: sitk.Euler3DTransform()
    )
    monkeypatch.setattr(
        alignment,
        "mask_rigid_refine",
        lambda *args: (sitk.Euler3DTransform(), Optimizer()),
    )
    monkeypatch.setattr(
        alignment,
        "mask_affine_refine",
        lambda *args: (sitk.AffineTransform(3), Optimizer()),
    )
    result = register_pair(
        organ(),
        organ(),
        RegistrationConfig(maximum_optimizer_iterations=1),
    )
    assert result.stage == "identity"
    assert result.dice_after == 0.5
    for name in ("pca", "mask_rigid", "mask_affine"):
        assert result.stages[name]["status"] == "failed"
        assert result.stages[name]["fallback_stage"] == "baseline"
    assert result.stages["geometry"]["status"] == "skipped_complete_organ"


@pytest.mark.parametrize("linear", [sitk.Euler3DTransform(), sitk.AffineTransform(3)])
def test_mask_elastic_builds_a_coarse_bspline_residual(linear):
    fixed = organ()
    domain = alignment.distance_map(fixed, padding_mm=10, band_mm=15)

    transform, registration = alignment.mask_elastic_refine(
        domain,
        domain,
        linear,
        RegistrationConfig(elastic_optimizer_iterations=1),
    )
    field, qc = alignment.elastic_deformation_qc(
        transform, domain, RegistrationConfig()
    )
    diagnostics = {}
    inverse = alignment.invert_mask_elastic(transform, domain, diagnostics, field)
    point = fixed.TransformIndexToPhysicalPoint((15, 13, 11))

    assert transform.GetNumberOfTransforms() == 2
    assert transform.GetNthTransform(1).GetName() == "BSplineTransform"
    assert registration.GetOptimizerIteration() == 1
    assert qc["deformation_qc_passed"] is True
    assert diagnostics["validation_phase"] == "complete"
    assert np.allclose(
        inverse.TransformPoint(transform.TransformPoint(point)), point, atol=0.2
    )


def test_mask_elastic_prewarps_distance_map_instead_of_setting_moving_transform(
    monkeypatch,
):
    fixed = organ()
    moving = sitk.Image(fixed)
    shift = (4.0, -3.0, 2.0)
    moving.SetOrigin(tuple(np.asarray(fixed.GetOrigin()) + shift))
    initial = sitk.Euler3DTransform()
    initial.SetTranslation(shift)
    registration_factory = sitk.ImageRegistrationMethod

    class RegistrationWithoutMovingTransform:
        def __init__(self):
            self.registration = registration_factory()

        def SetMovingInitialTransform(self, transform):
            raise AssertionError("Mask elastic must prewarp the moving distance map")

        def __getattr__(self, name):
            return getattr(self.registration, name)

    monkeypatch.setattr(
        sitk,
        "ImageRegistrationMethod",
        RegistrationWithoutMovingTransform,
    )

    transform, _ = alignment.mask_elastic_refine(
        alignment.distance_map(fixed, padding_mm=10, band_mm=15),
        alignment.distance_map(moving, padding_mm=10, band_mm=15),
        initial,
        RegistrationConfig(elastic_optimizer_iterations=1),
    )

    assert transform.GetNthTransform(0).GetName() == "Euler3DTransform"
    assert transform.GetNthTransform(1).GetName() == "BSplineTransform"


def test_mask_elastic_optimizer_stops_and_restores_valid_checkpoint(monkeypatch):
    fixed = organ()
    domain = alignment.distance_map(fixed, padding_mm=10, band_mm=15)
    diagnostics = {}
    observed_parameters = []
    original_displacement = alignment._bspline_displacement

    def record_displacement(transform, reference):
        observed_parameters.append(tuple(transform.GetParameters()))
        return original_displacement(transform, reference)

    def reject_second_update(field, config):
        if len(observed_parameters) == 1:
            return {
                "deformation_qc_passed": True,
                "deformation_qc_reasons": [],
            }
        return {
            "deformation_qc_passed": False,
            "deformation_qc_reasons": ["maximum_displacement_exceeds_limit"],
        }

    monkeypatch.setattr(alignment, "_bspline_displacement", record_displacement)
    monkeypatch.setattr(alignment, "_elastic_field_qc", reject_second_update)

    transform, _ = alignment.mask_elastic_refine(
        domain,
        domain,
        sitk.Euler3DTransform(),
        RegistrationConfig(elastic_optimizer_iterations=10),
        diagnostics,
    )

    residual = transform.GetNthTransform(1)
    assert len(observed_parameters) == 2
    assert diagnostics["optimizer_qc_stop_requested"] is True
    assert diagnostics["optimizer_qc_stop_succeeded"] is True
    assert diagnostics["optimizer_checkpoint_restored"] is True
    assert diagnostics["optimizer_qc_stop_reasons"] == [
        "maximum_displacement_exceeds_limit"
    ]
    assert diagnostics["optimizer_valid_checkpoints"] == 1
    assert residual.GetParameters() == pytest.approx(observed_parameters[0])


def test_mask_elastic_is_the_optional_final_stage_and_retains_inverse(monkeypatch):
    scores = iter([0.5, 0.6, 0.7, 0.75, 0.8])
    forward = sitk.CompositeTransform(3)
    inverse = sitk.TranslationTransform(3)

    class Optimizer:
        def GetOptimizerStopConditionDescription(self):
            return "test"

        def GetOptimizerIteration(self):
            return 1

        def GetElapsedIterations(self):
            return 2

        def GetMetric(self):
            return 0.01

        def GetRMSChange(self):
            return 0.001

    monkeypatch.setattr(alignment, "dice", lambda *args: next(scores))
    monkeypatch.setattr(
        alignment, "initialize_pca", lambda *args: sitk.Euler3DTransform()
    )
    monkeypatch.setattr(
        alignment,
        "mask_rigid_refine",
        lambda *args: (sitk.Euler3DTransform(), Optimizer()),
    )
    monkeypatch.setattr(
        alignment,
        "mask_affine_refine",
        lambda *args: (sitk.AffineTransform(3), Optimizer()),
    )
    monkeypatch.setattr(
        alignment, "mask_elastic_refine", lambda *args: (forward, Optimizer())
    )
    monkeypatch.setattr(
        alignment,
        "elastic_deformation_qc",
        lambda *args: (
            None,
            {
                "deformation_qc_passed": True,
                "deformation_qc_reasons": [],
                "displacement_p95_mm": 2.0,
                "displacement_max_mm": 3.0,
                "jacobian_min": 0.9,
                "jacobian_max": 1.1,
                "jacobian_nonpositive_voxels": 0,
            },
        ),
    )
    monkeypatch.setattr(alignment, "invert_mask_elastic", lambda *args: inverse)

    result = register_pair(
        organ(),
        organ(),
        RegistrationConfig(
            enable_elastic_stage=True,
            early_stop_organ_dice=1,
        ),
    )

    assert list(scores) == []
    assert result.stage == "mask_elastic"
    assert result.reference_to_scan is forward
    assert result.scan_to_reference is inverse
    assert result.stages["mi_affine"]["status"] == "skipped_disabled"
    detail = result.stages["mask_elastic"]
    assert detail["status"] == "evaluated"
    assert detail["algorithm"] == "BSplineTransform"
    assert detail["metric"] == "mean_squares_signed_distance"
    assert detail["minimum_required_dice_improvement"] == 0.005
    assert detail["optimizer_iteration_limit"] == 25
    assert detail["displacement_p95_mm"] == 2.0
    assert detail["optimization_elapsed_seconds"] >= 0
    assert detail["field_qc_elapsed_seconds"] >= 0
    assert detail["inversion_elapsed_seconds"] == 0
    assert detail["round_trip_validation_elapsed_seconds"] == 0


@pytest.mark.parametrize(
    "elastic_dice,qc_passed,inverse_fails,expected_status",
    [
        (0.754, True, False, "rejected_insufficient_improvement"),
        (0.85, False, False, "rejected_deformation_qc"),
        (0.85, True, True, "failed"),
    ],
)
def test_mask_elastic_rejection_or_failure_falls_back_to_affine(
    monkeypatch,
    elastic_dice,
    qc_passed,
    inverse_fails,
    expected_status,
):
    scores = iter([0.5, 0.6, 0.7, 0.75, elastic_dice])

    class Optimizer:
        def GetOptimizerStopConditionDescription(self):
            return "test"

        def GetOptimizerIteration(self):
            return 1

    monkeypatch.setattr(alignment, "dice", lambda *args: next(scores))
    monkeypatch.setattr(
        alignment, "initialize_pca", lambda *args: sitk.Euler3DTransform()
    )
    monkeypatch.setattr(
        alignment,
        "mask_rigid_refine",
        lambda *args: (sitk.Euler3DTransform(), Optimizer()),
    )
    affine = sitk.AffineTransform(3)
    affine.SetTranslation((3.0, -2.0, 1.0))
    monkeypatch.setattr(
        alignment,
        "mask_affine_refine",
        lambda *args: (affine, Optimizer()),
    )
    monkeypatch.setattr(
        alignment,
        "mask_elastic_refine",
        lambda *args: (sitk.CompositeTransform(3), Optimizer()),
    )
    reasons = [] if qc_passed else ["maximum_displacement_exceeds_limit"]
    monkeypatch.setattr(
        alignment,
        "elastic_deformation_qc",
        lambda *args: (
            None,
            {
                "deformation_qc_passed": qc_passed,
                "deformation_qc_reasons": reasons,
            },
        ),
    )

    def reject_inverse(*args):
        exception = (
            ValueError("inverse failed")
            if inverse_fails
            else AssertionError("Rejected elastic candidate must not be inverted")
        )
        raise exception

    monkeypatch.setattr(alignment, "invert_mask_elastic", reject_inverse)

    result = register_pair(
        organ(),
        organ(),
        RegistrationConfig(enable_elastic_stage=True, early_stop_organ_dice=1),
    )

    assert list(scores) == []
    assert result.stage == "mask_affine"
    assert result.reference_to_scan is affine
    point = (12.0, -4.0, 8.0)
    assert result.scan_to_reference.TransformPoint(
        result.reference_to_scan.TransformPoint(point)
    ) == pytest.approx(point)
    assert result.stages["mask_elastic"]["status"] == expected_status
    assert result.stages["mask_elastic"]["fallback_stage"] == "mask_affine"


def test_affine_is_not_gated_by_an_absolute_dice_threshold():
    fixed = organ()
    moving = sitk.Image(fixed)
    moving.SetSpacing((fixed.GetSpacing()[0] * 1.2, *fixed.GetSpacing()[1:]))
    result = register_pair(
        fixed,
        moving,
        RegistrationConfig(
            enable_affine_stage=True,
            early_stop_organ_dice=1,
            maximum_optimizer_iterations=20,
        ),
    )
    assert result.stages["mi_affine"]["status"] != "skipped_low_dice"
    assert result.stages["mi_affine"]["metric"] == "mattes_mutual_information"
    assert result.stages["mi_affine"]["minimum_required_mi_improvement"] == 0.0


def test_intersection_ignores_unobserved_background():
    a = image(np.ones((2, 3, 4)))
    b = image(np.ones((2, 3, 4)))
    coverage = np.ones((2, 3, 4))
    coverage[:, :, 3] = 0
    result = fuse_tumors([a, b], [a, image(coverage)], tumor_consensus="intersection")
    mask = sitk.GetArrayFromImage(result.mask)
    assert mask[:, :, :3].all()
    assert mask[:, :, 3].all()
    assert sitk.GetArrayFromImage(result.coverage).all()


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
        csv_path=str(tmp_path / "register_errors.csv"),
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
        source,
        tmp_path / "out",
        RegistrationConfig(maximum_optimizer_iterations=10),
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
        first,
        tmp_path / "out",
        RegistrationConfig(maximum_optimizer_iterations=10),
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
    config = RegistrationConfig(maximum_optimizer_iterations=10)
    out, errors = register_cohort(pd.DataFrame(rows), tmp_path / "out", config)
    assert errors.empty
    qc = build_qc(out, errors, config)
    moving = qc.set_index("registration_scan_id").loc[
        out.loc[1, "registration_scan_id"]
    ]
    assert 0 <= moving.dice_baseline < moving.dice_pca <= 1
    assert pd.isna(moving.dice_mask_rigid)
    assert pd.isna(moving.dice_mask_affine)
    assert pd.isna(moving.dice_mi_affine)
    assert pd.isna(moving.dice_mask_elastic)
    assert "dice_mi_rigid" not in qc.columns
    assert moving.mask_rigid_status == "skipped_early_stop"
    assert moving.mask_affine_status == "skipped_early_stop"
    assert moving.mi_affine_status == "skipped_early_stop"
    assert moving.mask_elastic_status == "skipped_early_stop"
    assert "mi_rigid_status" not in qc.columns
    assert moving.registration_reference_id == out.loc[0, "registration_scan_id"]
    events = [
        json.loads(line)
        for line in Path(out.loc[1, "registration_log_path"]).read_text().splitlines()
    ]
    assert any(
        event.get("event") == "scan_result"
        and event["registration_scan_label"] == moving.registration_scan_label
        and event["stages"]["pca"]["dice"] == pytest.approx(moving.dice_pca)
        and "geometry" not in event["stages"]
        for event in events
    )
    assert any(
        "series=2/2" in record.message
        and "phase=ARTERIAL" in record.message
        and "pca=" in record.message
        and "geometry=" not in record.message
        for record in caplog.records
    )


def test_default_logging_keeps_geometry_when_attempted():
    from imperandi.process.registration.reporting import _stages_for_log

    attempted = {"geometry": {"dice": None, "status": "failed"}}
    skipped = {"geometry": {"dice": None, "status": "skipped_complete_organ"}}

    assert "geometry" in _stages_for_log(attempted)
    assert "geometry" not in _stages_for_log(skipped)


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
    expected = "patient_key=001, study_id=v1, Modality=CT"
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
            "grouping_columns": ["patient_key", "study_id", "Modality"],
            "group_values": {
                "patient_key": "001",
                "study_id": "v1",
                "Modality": "CT",
            },
            "series_count": 1,
            "registration_reference_label": (
                "series=1/1, phase=PORTAL_VENOUS, sequence=T1"
            ),
            "organ_consensus": "anchor",
            "tumor_consensus": "anchor",
        }
    ]
    assert len(scan_events) == 1
    assert all("patient_id" not in event for event in scan_events)
    assert all(
        "series=1/1" in event["registration_scan_label"] for event in scan_events
    )
    qc = build_qc(out, errors, RegistrationConfig())
    assert qc.loc[0, "patient_id"] == "PATIENT-A"
    assert qc.loc[0, "date"] == "2024-05-17"
    assert qc.loc[0, "visit_order"] == 2


def test_scan_labels_number_series_within_each_group():
    from imperandi.process.registration.cohort import prepare_cohort

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
    from imperandi.process.registration.alignment import (
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
            "mask_rigid",
            0.1,
            0.2,
            [],
            {
                "baseline": {"dice": 0.1, "status": "evaluated"},
                "pca": {"dice": 0.15, "status": "evaluated"},
                "geometry": {"dice": None, "status": "not_run"},
                "mask_rigid": {"dice": 0.2, "status": "evaluated"},
                "mask_affine": {"dice": None, "status": "not_run"},
                "mi_affine": {"dice": None, "status": "not_run"},
                "mask_elastic": {"dice": None, "status": "not_run"},
            },
        )
        raise RegistrationRejected("Overlap rejected", result)

    out, errors = register_cohort(
        pd.DataFrame(rows), tmp_path / "out", pair_registration=rejected
    )
    assert out.loc[1, "mask_liver"] == rows[1]["mask_liver"]
    assert out.loc[1, "mask_liver_tumor"] == rows[1]["mask_liver_tumor"]
    qc = build_qc(out, errors, RegistrationConfig()).set_index("registration_scan_id")
    moving = qc.loc[out.loc[1, "registration_scan_id"]]
    assert moving.dice_mask_rigid == 0.2
    assert moving.registration_status == "failed"
    assert "Overlap rejected" in moving.errors


@pytest.mark.parametrize("keep", [False, True])
def test_keep_source_segmentation_maps_tumor_to_source_organ(tmp_path, keep):
    rows = [
        save_scan(tmp_path, "portal", phase="PORTAL_VENOUS"),
        save_scan(tmp_path, "arterial", tumor=False, offset=4),
    ]
    source_organ = sitk.ReadImage(rows[1]["mask_liver"])
    values = sitk.GetArrayFromImage(source_organ)
    values[11, 13, 15] = 0
    altered = sitk.GetImageFromArray(values)
    altered.CopyInformation(source_organ)
    sitk.WriteImage(altered, rows[1]["mask_liver"])
    original_bytes = Path(rows[1]["mask_liver"]).read_bytes()
    out, errors = register_cohort(
        pd.DataFrame(rows),
        tmp_path / "out",
        RegistrationConfig(
            maximum_optimizer_iterations=10,
            preserve_source_organ_mask=keep,
        ),
    )
    assert errors.empty
    assert out.consensus_status.eq("single_contributor").all()
    assert Path(rows[1]["mask_liver"]).read_bytes() == original_bytes
    if keep:
        assert out.mask_liver.tolist() == [r["mask_liver"] for r in rows]
        assert out.reg_organ_native_path.isna().all()
        assert not list((tmp_path / "out").rglob("organ_native.nii.gz"))
    else:
        assert out.mask_liver.equals(out.reg_organ_native_path)
    native = sitk.ReadImage(out.loc[1, "reg_tumor_native_path"])
    assert native.GetOrigin() == source_organ.GetOrigin()
    assert sitk.GetArrayFromImage(native)[11, 13, 15] == (0 if keep else 1)
    assert sitk.GetArrayFromImage(native).any()
    if keep:
        rerun, errors = register_cohort(
            out,
            tmp_path / "rerun",
            RegistrationConfig(
                maximum_optimizer_iterations=10,
                preserve_source_organ_mask=True,
            ),
        )
        assert errors.empty
        assert rerun.mask_liver.tolist() == out.source_mask_liver.tolist()


def test_preserve_source_organ_mask_requires_boolean():
    with pytest.raises(ValueError, match="preserve_source_organ_mask"):
        RegistrationConfig.from_mapping({"preserve_source_organ_mask": "true"})


def test_preserve_source_organ_mask_bypasses_organ_consensus(monkeypatch, tmp_path):
    from imperandi.process.registration import cohort

    row = save_scan(tmp_path, "scan", tumor=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("organ consensus must not run")

    monkeypatch.setattr(cohort, "fuse_organs", forbidden)

    out, errors = register_cohort(
        pd.DataFrame([row]),
        tmp_path / "out",
        RegistrationConfig(
            organ_consensus_method="majority",
            preserve_source_organ_mask=True,
            maximum_optimizer_iterations=1,
        ),
    )

    assert errors.empty
    assert out.loc[0, "mask_liver"] == row["mask_liver"]
    assert pd.isna(out.loc[0, "reg_organ_native_path"])
