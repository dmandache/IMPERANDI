"""Synthetic geometry and cohort contracts for registration."""

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
    pd.testing.assert_frame_equal(out[source.columns], source)
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
    from imperandi.process.registration.cli import main

    args = Namespace(
        csv_path=str(tmp_path / "cohort_errors.csv"),
        csv_path_out=str(tmp_path / "cohort.csv"),
    )
    with pytest.raises(ValueError, match="differ"):
        main(args)
