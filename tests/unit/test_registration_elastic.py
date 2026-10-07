"""QC contracts for the lightweight mask-driven B-spline stage."""

import numpy as np
import pytest

from imperandi.process.registration import alignment
from imperandi.process.registration.config import RegistrationConfig

sitk = pytest.importorskip("SimpleITK")


@pytest.fixture(autouse=True)
def single_thread():
    previous = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    yield
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(previous)


def domain():
    image = sitk.Image([20, 18, 16], sitk.sitkFloat32)
    image.SetSpacing((1.2, 1.7, 2.1))
    image.SetOrigin((15.0, -24.0, 7.0))
    rotation = sitk.Euler3DTransform()
    rotation.SetRotation(0.2, -0.3, 0.4)
    image.SetDirection(rotation.GetMatrix())
    return image


def vector_field(reference, values=None):
    if values is None:
        values = np.zeros(tuple(reversed(reference.GetSize())) + (3,))
    field = sitk.GetImageFromArray(values, isVector=True)
    field.CopyInformation(reference)
    return field


def bspline_composite(reference, linear=None):
    residual = sitk.BSplineTransformInitializer(reference, [1, 1, 1], order=3)
    return sitk.CompositeTransform(
        [linear if linear is not None else sitk.Euler3DTransform(), residual]
    )


def qc_for(monkeypatch, values, config=None):
    reference = domain()
    field = vector_field(reference, values)
    monkeypatch.setattr(alignment, "_bspline_displacement", lambda *args: field)
    return alignment.elastic_deformation_qc(
        bspline_composite(reference), reference, config or RegistrationConfig()
    )[1]


def test_small_smooth_deformation_passes_qc(monkeypatch):
    reference = domain()
    values = np.zeros(tuple(reversed(reference.GetSize())) + (3,))
    values[:] = [2.0, -1.0, 0.5]

    qc = qc_for(monkeypatch, values)

    expected = np.linalg.norm(values[0, 0, 0])
    assert qc["deformation_qc_passed"] is True
    assert qc["deformation_qc_reasons"] == []
    assert qc["displacement_p95_mm"] == pytest.approx(expected)
    assert qc["displacement_max_mm"] == pytest.approx(expected)
    assert qc["jacobian_min"] == pytest.approx(1)
    assert qc["jacobian_max"] == pytest.approx(1)
    assert qc["jacobian_nonpositive_voxels"] == 0


def test_p95_displacement_limit_rejects_deformation(monkeypatch):
    reference = domain()
    values = np.zeros(tuple(reversed(reference.GetSize())) + (3,))
    values[..., 0] = 9.0

    qc = qc_for(monkeypatch, values)

    assert qc["deformation_qc_passed"] is False
    assert "p95_displacement_exceeds_limit" in qc["deformation_qc_reasons"]
    assert "maximum_displacement_exceeds_limit" not in qc["deformation_qc_reasons"]


def test_maximum_displacement_limit_rejects_local_outlier(monkeypatch):
    reference = domain()
    values = np.zeros(tuple(reversed(reference.GetSize())) + (3,))
    values[8, 9, 10, 0] = 13.0

    qc = qc_for(monkeypatch, values)

    assert qc["deformation_qc_passed"] is False
    assert "maximum_displacement_exceeds_limit" in qc["deformation_qc_reasons"]
    assert "p95_displacement_exceeds_limit" not in qc["deformation_qc_reasons"]


def test_folding_and_implausible_low_jacobian_are_rejected(monkeypatch):
    reference = domain()
    values = np.zeros(tuple(reversed(reference.GetSize())) + (3,))
    direction = np.asarray(reference.GetDirection()).reshape(3, 3)
    for x in range(reference.GetSize()[0]):
        values[..., x, :] = -2 * x * reference.GetSpacing()[0] * direction[:, 0]

    qc = qc_for(monkeypatch, values)

    assert qc["jacobian_nonpositive_voxels"] > 0
    assert "folding" in qc["deformation_qc_reasons"]
    assert "jacobian_below_plausible_range" in qc["deformation_qc_reasons"]


def test_implausible_high_jacobian_is_rejected(monkeypatch):
    reference = domain()
    values = np.zeros(tuple(reversed(reference.GetSize())) + (3,))
    direction = np.asarray(reference.GetDirection()).reshape(3, 3)
    for x in range(reference.GetSize()[0]):
        values[..., x, :] = 2 * x * reference.GetSpacing()[0] * direction[:, 0]

    qc = qc_for(monkeypatch, values)

    assert qc["jacobian_max"] > 2
    assert "jacobian_above_plausible_range" in qc["deformation_qc_reasons"]


def test_identity_bspline_inverse_preserves_exact_linear_stage():
    reference = domain()
    linear = sitk.AffineTransform(3)
    linear.SetTranslation((4.0, -3.0, 2.0))
    transform = bspline_composite(reference, linear)
    diagnostics = {}

    inverse = alignment.invert_mask_elastic(transform, reference, diagnostics)
    point = reference.TransformIndexToPhysicalPoint((10, 9, 8))

    assert diagnostics["validation_phase"] == "complete"
    assert inverse.TransformPoint(transform.TransformPoint(point)) == pytest.approx(
        point
    )


def test_inaccurate_finite_inverse_is_rejected(monkeypatch):
    reference = domain()
    values = np.zeros(tuple(reversed(reference.GetSize())) + (3,))
    values[..., 0] = 1.5
    field = vector_field(reference, values)

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

        def Execute(self, image):
            return vector_field(image)

    monkeypatch.setattr(sitk, "InvertDisplacementFieldImageFilter", WrongInverse)
    diagnostics = {}

    with pytest.raises(ValueError, match="round-trip"):
        alignment.invert_mask_elastic(
            bspline_composite(reference), reference, diagnostics, field
        )

    assert diagnostics["inverse_round_trip_max_mm"] >= 1.5
    assert diagnostics["validation_phase"] == "round_trip"
