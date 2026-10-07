"""Elastic validation in physical coordinates, including crop boundaries."""

import numpy as np
import pandas as pd
import pytest

from imperandi.process.registration import RegistrationConfig, register_cohort
from imperandi.process.registration import alignment

sitk = pytest.importorskip("SimpleITK")


@pytest.fixture(autouse=True)
def single_thread():
    previous = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    yield
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(previous)


def direction(name):
    if name == "identity":
        return np.eye(3)
    if name == "flipped":
        return np.diag([-1.0, -1.0, 1.0])
    if name == "reflected":
        return np.diag([-1.0, 1.0, 1.0])
    rotation = sitk.Euler3DTransform()
    rotation.SetRotation(0.2, -0.3, 0.4)
    return np.array(rotation.GetMatrix()).reshape(3, 3)


def organ(name="oblique", radii=(8, 6, 4)):
    z, y, x = np.indices((24, 28, 32))
    mask = sitk.GetImageFromArray(
        (
            ((x - 15) / radii[0]) ** 2
            + ((y - 13) / radii[1]) ** 2
            + ((z - 11) / radii[2]) ** 2
            < 1
        ).astype(np.uint8)
    )
    mask.SetSpacing((1.2, 1.7, 2.1))
    mask.SetOrigin((15.0, -24.0, 7.0))
    mask.SetDirection(direction(name).ravel())
    return mask


def vector_field(reference, values=None):
    if values is None:
        values = np.zeros(tuple(reversed(reference.GetSize())) + (3,))
    result = sitk.GetImageFromArray(values, isVector=True)
    result.CopyInformation(reference)
    return result


def composite(field, linear=None):
    return sitk.CompositeTransform(
        [
            linear if linear is not None else sitk.Euler3DTransform(),
            sitk.DisplacementFieldTransform(sitk.Image(field)),
        ]
    )


@pytest.mark.parametrize("orientation", ["identity", "flipped", "reflected", "oblique"])
@pytest.mark.parametrize("angle", [np.pi / 2, np.pi])
@pytest.mark.parametrize("affine", [False, True])
def test_valid_linear_mapping_has_no_elastic_fold(
    orientation, angle, affine, monkeypatch
):
    fixed = organ(orientation)
    rotation = sitk.Euler3DTransform()
    rotation.SetRotation(angle, 0, 0)
    rotation.SetCenter(fixed.TransformIndexToPhysicalPoint((15, 13, 11)))
    rotation.SetTranslation((3.0, -2.0, 1.0))
    linear = rotation
    if affine:
        linear = sitk.AffineTransform(3)
        linear.SetCenter(rotation.GetCenter())
        linear.SetTranslation(rotation.GetTranslation())
        linear.SetMatrix(
            (
                np.array(rotation.GetMatrix()).reshape(3, 3) @ np.diag([1.1, 0.9, 1.2])
            ).ravel()
        )
    transform = composite(vector_field(fixed), linear)

    def no_full_grid(*args, **kwargs):
        raise AssertionError("Do not allocate a full-grid sampled linear transform")

    monkeypatch.setattr(sitk, "TransformToDisplacementField", no_full_grid)
    diagnostics = {}
    inverse = alignment.invert_elastic(transform, fixed, diagnostics)
    assert diagnostics["jacobian_min"] == pytest.approx(
        np.linalg.det(np.array(linear.GetMatrix()).reshape(3, 3))
    )
    assert diagnostics["validation_phase"] == "complete"
    for index in ((0, 0, 0), (15, 13, 11), (31, 27, 23)):
        point = fixed.TransformIndexToPhysicalPoint(index)
        assert inverse.TransformPoint(transform.TransformPoint(point)) == pytest.approx(
            point
        )


@pytest.mark.parametrize("orientation", ["identity", "flipped", "reflected", "oblique"])
@pytest.mark.parametrize("scale", [1.2, -0.2])
def test_residual_jacobian_matches_analytic_physical_scaling(orientation, scale):
    fixed = organ(orientation)
    indices = np.moveaxis(np.indices(tuple(reversed(fixed.GetSize()))), 0, -1)[
        ..., ::-1
    ]
    points = (indices * fixed.GetSpacing()) @ direction(
        orientation
    ).T + fixed.GetOrigin()
    values = np.zeros(points.shape)
    values[..., 0] = (scale - 1) * points[..., 0]
    determinant = alignment._residual_jacobian(vector_field(fixed, values))
    assert np.allclose(determinant[1:-1, 1:-1, 1:-1], scale)


def test_crop_collar_preserves_interior_and_inverts_boundary_displacement():
    fixed = organ()
    values = np.zeros(tuple(reversed(fixed.GetSize())) + (3,))
    values[:] = [1.5, -0.5, 0.25]
    field = vector_field(fixed, values)
    with pytest.raises(ValueError, match="boundary"):
        alignment.invert_elastic(composite(field), fixed)
    extended = alignment._extend_residual(field)
    transform = composite(extended)
    inverse = alignment.invert_elastic(transform, fixed)
    for index in ((0, 0, 0), (15, 13, 11), (31, 27, 23)):
        point = np.array(fixed.TransformIndexToPhysicalPoint(index))
        assert transform.TransformPoint(point) == pytest.approx(point + values[0, 0, 0])
        assert inverse.TransformPoint(transform.TransformPoint(point)) == pytest.approx(
            point, abs=0.15
        )
    assert all(a > b for a, b in zip(extended.GetSize(), fixed.GetSize()))


def test_actual_fold_remains_rejected():
    fixed = organ()
    values = np.zeros(tuple(reversed(fixed.GetSize())) + (3,))
    # du/d(image-x) = -2 along the physical image-x direction in the interior.
    for x in range(10, 22):
        values[6:18, 7:21, x] = (
            -2 * (x - 15) * fixed.GetSpacing()[0] * direction("oblique")[:, 0]
        )
    diagnostics = {}
    with pytest.raises(ValueError, match="fold"):
        alignment.invert_elastic(
            composite(vector_field(fixed, values)), fixed, diagnostics
        )
    assert diagnostics["jacobian_nonpositive_voxels"] > 0
    assert diagnostics["validation_phase"] == "jacobian"


def test_finite_but_wrong_inverse_is_rejected(monkeypatch):
    fixed = organ()
    values = np.zeros(tuple(reversed(fixed.GetSize())) + (3,))
    values[..., 0] = 1.5
    field = alignment._extend_residual(vector_field(fixed, values))

    class WrongInverse:
        def SetMaximumNumberOfIterations(self, value):
            pass

        def SetMeanErrorToleranceThreshold(self, value):
            pass

        def SetMaxErrorToleranceThreshold(self, value):
            pass

        def SetEnforceBoundaryCondition(self, value):
            pass

        def GetMaxErrorNorm(self):
            return 0.0

        def GetMeanErrorNorm(self):
            return 0.0

        def Execute(self, field):
            return vector_field(field)

    monkeypatch.setattr(sitk, "InvertDisplacementFieldImageFilter", WrongInverse)
    diagnostics = {}
    with pytest.raises(ValueError, match="round-trip"):
        alignment.invert_elastic(composite(field), fixed, diagnostics)
    assert diagnostics["inverse_round_trip_max_mm"] >= 1.5
    assert diagnostics["validation_phase"] == "round_trip"


def test_real_elastic_candidate_is_selected_through_cohort(tmp_path):
    rows = []
    for name, radii, phase in [
        ("fixed", (8, 6, 4), "PORTAL_VENOUS"),
        ("moving", (9, 7, 4), "ARTERIAL"),
    ]:
        path = tmp_path / (name + ".nii.gz")
        sitk.WriteImage(organ(radii=radii), str(path))
        rows.append(
            dict(
                patient_key="p",
                study_id="v",
                Modality="CT",
                phase=phase,
                nifti_path=str(path),
                mask_liver=str(path),
                mask_liver_tumor=str(path),
            )
        )
    result, errors = register_cohort(
        pd.DataFrame(rows),
        tmp_path / "out",
        RegistrationConfig(elastic=True, iterations=50),
    )
    assert errors.empty
    assert result.loc[1, "registration_selected_stage"] == "elastic"
    assert result.loc[1, "registration_dice_selected"] > 0.95
    native = sitk.ReadImage(result.loc[1, "reg_tumor_native_path"])
    assert native.GetSize() == organ().GetSize()
    moving = sitk.ReadImage(rows[1]["nifti_path"])
    assert np.allclose(native.GetDirection(), moving.GetDirection(), atol=1e-6, rtol=0)
    assert native.GetSpacing() == pytest.approx(moving.GetSpacing())
    assert native.GetOrigin() == pytest.approx(moving.GetOrigin())
    assert alignment.dice(moving, native, sitk.Euler3DTransform()) > 0.95
