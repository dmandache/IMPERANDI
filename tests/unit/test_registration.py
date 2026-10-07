"""Synthetic geometry and cohort contracts for registration."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imperandi.process.registration import RegistrationConfig, register_cohort
from imperandi.process.registration import alignment
from imperandi.process.registration.cohort import fuse_tumors
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


@pytest.mark.parametrize("partial", [False, True])
def test_geometry_precedes_pca_and_min_dice_decision(monkeypatch, partial):
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
        return sitk.Euler3DTransform()

    monkeypatch.setattr(sitk, "CenteredTransformInitializer", geometry)
    monkeypatch.setattr(alignment, "initialize_pca", pca)
    result = register_pair(
        fixed, moving, RegistrationConfig(iterations=1, min_dice=0.95)
    )
    assert result.dice_before == 0
    assert result.dice_after == pytest.approx(1)
    assert calls == (["geometry"] if partial else ["geometry", "pca"])
    assert result.stage == "geometry"
    assert result.stages["geometry"]["selected"] is True
    assert "center_of_mass" not in result.stages
    if partial:
        assert result.stages["pca"]["status"] == "skipped_partial_coverage"
    point = fixed.TransformIndexToPhysicalPoint((15, 13, 11))
    assert np.allclose(result.reference_to_scan.TransformPoint(point), point + shift)
    assert np.allclose(result.scan_to_reference.TransformPoint(point + shift), point)


@pytest.mark.parametrize(
    "method", ["anchor", "majority", "intersection", "union", "staple"]
)
def test_empty_resampled_tumor_preserves_observed_negative_votes(method):
    empty = image(np.zeros((2, 3, 4)))
    coverage = image(np.ones((2, 3, 4)))
    result = fuse_tumors([empty], [coverage], method=method)
    assert not sitk.GetArrayFromImage(result.mask).any()
    with pytest.raises(ValueError, match="support"):
        fuse_tumors([coverage], [empty], method=method)


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


def test_tumor_resampling_failure_preserves_organ_registration(monkeypatch, tmp_path):
    from imperandi.process.registration import cohort

    row = save_scan(tmp_path, "scan")

    def fail(*args, **kwargs):
        raise RuntimeError("tumor resampling failed")

    monkeypatch.setattr(cohort, "resample", fail)
    out, errors = register_cohort(
        pd.DataFrame([row]),
        tmp_path / "out",
        RegistrationConfig(keep_source_segmentation=True),
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


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_demons_smoothing_is_positive_and_finite(value):
    with pytest.raises(ValueError, match="Demons smoothing sigma"):
        RegistrationConfig(demons_smoothing_sigma_mm=value)


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


def test_majority_consensus_keeps_only_native_masks(tmp_path):
    rows = [save_scan(tmp_path, "a", phase="PORTAL_VENOUS"), save_scan(tmp_path, "b")]
    output_dir = tmp_path / "out"
    out, errors = register_cohort(
        pd.DataFrame(rows),
        output_dir,
        RegistrationConfig(method="majority", iterations=5),
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
        RegistrationConfig(method="majority", iterations=5),
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


@pytest.mark.parametrize(
    "scores,affine,selected_stage,rejected_stage,fallback_stage,selected_dice",
    [
        ([0.8, 0.4, 0.9], False, "rigid", "pca", "baseline", 0.9),
        ([0.5, 0.8, 0.4], False, "pca", "rigid", "pca", 0.8),
        ([0.5, 0.8, 0.9, 0.7], True, "rigid", "affine", "rigid", 0.9),
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
    values = iter([scores[0], scores[0], *scores[1:]])
    monkeypatch.setattr(
        alignment,
        "initialize_pca",
        lambda fixed, moving: sitk.Euler3DTransform(),
    )
    monkeypatch.setattr(alignment, "dice", lambda *args: next(values))

    result = alignment.register_pair(
        organ(),
        organ(),
        RegistrationConfig(
            affine=affine,
            affine_min_dice=0,
            iterations=1,
            min_dice=0,
        ),
    )

    assert list(values) == []
    assert result.stage == selected_stage
    assert result.dice_after == selected_dice
    assert result.stages[rejected_stage]["status"] == "rejected_worse_dice"
    assert result.stages[rejected_stage]["fallback_stage"] == fallback_stage
    assert result.stages[rejected_stage]["selected"] is False
    assert result.stages[selected_stage]["selected"] is True


def test_demons_preserves_initial_transform_on_different_grids(tmp_path):
    fixed = organ()
    moving = sitk.Image(fixed)
    shift = (4.0, -3.0, 2.0)
    moving.SetOrigin(tuple(np.array(fixed.GetOrigin()) + shift))
    initial = sitk.AffineTransform(3)
    initial.SetTranslation(shift)
    fixed_dm = alignment.distance_map(fixed, padding_mm=5, band_mm=15)
    moving_dm = alignment.distance_map(moving, padding_mm=10, band_mm=15)

    transform, demons = alignment.elastic_refine(
        fixed_dm, moving_dm, initial, RegistrationConfig(iterations=5)
    )

    assert demons.GetName() == "DiffeomorphicDemonsRegistrationFilter"
    assert np.allclose(demons.GetStandardDeviations(), 1 / np.array(fixed.GetSpacing()))
    assert alignment.dice(fixed, moving, transform) == 1
    point = fixed.TransformIndexToPhysicalPoint((15, 13, 11))
    assert np.allclose(transform.TransformPoint(point), initial.TransformPoint(point))
    inverse = alignment.invert_elastic(transform, fixed)
    assert np.allclose(
        inverse.TransformPoint(transform.TransformPoint(point)), point, atol=0.1
    )
    path = tmp_path / "demons.h5"
    sitk.WriteTransform(transform, str(path))
    restored = sitk.ReadTransform(str(path))
    assert np.allclose(restored.TransformPoint(point), transform.TransformPoint(point))


def test_failed_linear_setup_retains_previous_alignment(monkeypatch):
    def unavailable():
        raise RuntimeError("optimizer unavailable")

    monkeypatch.setattr(sitk, "ImageRegistrationMethod", unavailable)
    result = register_pair(organ(), organ(), RegistrationConfig())
    assert result.dice_after == 1
    assert result.stages["rigid"]["status"] == "failed"
    assert result.stages["rigid"]["fallback_stage"] == "baseline"
    assert result.warnings == ["rigid: optimizer unavailable"]


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_nonfinite_candidate_retains_previous_alignment(monkeypatch, score):
    scores = iter([1, score, score, score])
    monkeypatch.setattr(alignment, "dice", lambda *args: next(scores))
    monkeypatch.setattr(
        alignment, "initialize_pca", lambda *args: sitk.Euler3DTransform()
    )
    result = register_pair(organ(), organ(), RegistrationConfig(iterations=1))
    assert result.stage == "identity"
    assert result.dice_after == 1
    for name in ("geometry", "pca", "rigid"):
        assert result.stages[name]["status"] == "failed"
        assert result.stages[name]["fallback_stage"] == "baseline"


def test_demons_improves_nonrigid_organ_overlap():
    fixed = organ()
    z, y, x = np.indices((24, 28, 32))
    moving = image(((x - 15) / 9) ** 2 + ((y - 13) / 7) ** 2 + ((z - 11) / 4) ** 2 < 1)
    initial = sitk.Euler3DTransform()
    fixed_dm = alignment.distance_map(fixed, padding_mm=10, band_mm=15)
    moving_dm = alignment.distance_map(moving, padding_mm=10, band_mm=15)
    transform, _ = alignment.elastic_refine(
        fixed_dm, moving_dm, initial, RegistrationConfig(iterations=50)
    )
    assert alignment.dice(fixed, moving, transform) > alignment.dice(
        fixed, moving, initial
    )
    inverse = alignment.invert_elastic(transform, fixed)
    for index in ((15, 13, 11), (20, 13, 11), (15, 17, 11)):
        point = fixed.TransformIndexToPhysicalPoint(index)
        assert np.allclose(
            inverse.TransformPoint(transform.TransformPoint(point)), point, atol=0.2
        )


def test_elastic_stage_can_be_selected_and_retains_inverse(monkeypatch):
    values = iter([0.5, 0.5, 0.6, 0.7, 0.8])
    forward = sitk.CompositeTransform(3)
    inverse = sitk.TranslationTransform(3)

    class Optimizer:
        def GetMetric(self):
            return 0.01

        def GetElapsedIterations(self):
            return 3

        def GetRMSChange(self):
            return 0.001

    monkeypatch.setattr(
        alignment,
        "initialize_pca",
        lambda fixed, moving: sitk.Euler3DTransform(),
    )
    monkeypatch.setattr(alignment, "dice", lambda *args: next(values))
    monkeypatch.setattr(
        alignment,
        "elastic_refine",
        lambda *args: (forward, Optimizer()),
    )
    monkeypatch.setattr(alignment, "invert_elastic", lambda *args: inverse)
    result = register_pair(
        organ(),
        organ(),
        RegistrationConfig(elastic=True, iterations=1, min_dice=0),
    )
    assert result.stage == "elastic"
    assert result.reference_to_scan is forward
    assert result.scan_to_reference is inverse
    assert result.stages["elastic"]["dice"] == 0.8
    assert result.stages["elastic"]["optimizer_iteration"] == 3


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
    config = RegistrationConfig(iterations=10)
    out, errors = register_cohort(pd.DataFrame(rows), tmp_path / "out", config)
    assert errors.empty
    qc = build_qc(out, errors, config)
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
    qc = build_qc(out, errors, RegistrationConfig()).set_index("registration_scan_id")
    moving = qc.loc[out.loc[1, "registration_scan_id"]]
    assert moving.dice_rigid == 0.2
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
        RegistrationConfig(iterations=10, keep_source_segmentation=keep),
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
            RegistrationConfig(iterations=10, keep_source_segmentation=True),
        )
        assert errors.empty
        assert rerun.mask_liver.equals(out.source_mask_liver)


def test_keep_source_segmentation_requires_boolean():
    with pytest.raises(ValueError, match="keep_source_segmentation"):
        RegistrationConfig.from_mapping({"keep_source_segmentation": "true"})
