import numpy as np
import pandas as pd
import pytest

sitk = pytest.importorskip("SimpleITK")

from imperandi.process.registration.config import RegistrationConfig  # noqa: E402
from imperandi.process.registration.fireants_backend import (  # noqa: E402
    coarse_image,
    physical_matrix_to_sitk,
    sitk_transform_to_physical_matrix,
)


def _geometry_image(*, direction=None):
    image = sitk.Image([19, 17, 13], sitk.sitkFloat32)
    image.SetOrigin((31.0, -17.0, 8.5))
    image.SetSpacing((0.7, 1.8, 4.2))
    if direction is not None:
        image.SetDirection(direction)
    image[4:13, 3:12, 2:9] = 1.0
    return image


@pytest.mark.parametrize("rigid", [False, True])
def test_physical_transform_conversion_preserves_points_and_inverse(rigid):
    center = (18.0, -2.0, 41.0)
    if rigid:
        original = sitk.Euler3DTransform()
        original.SetRotation(0.13, -0.09, 0.21)
    else:
        original = sitk.AffineTransform(3)
        original.SetMatrix((1.1, 0.05, 0.0, -0.03, 0.9, 0.04, 0.0, 0.02, 1.05))
    original.SetCenter(center)
    original.SetTranslation((4.0, -7.0, 2.5))

    matrix = sitk_transform_to_physical_matrix(original)
    restored = physical_matrix_to_sitk(matrix, center=center, rigid=rigid)
    for point in ((0.0, 0.0, 0.0), (11.2, -8.5, 37.0), (90.0, 14.0, -2.0)):
        expected = original.TransformPoint(point)
        assert np.allclose(restored.TransformPoint(point), expected, atol=1e-5)
        assert np.allclose(
            restored.GetInverse().TransformPoint(expected), point, atol=1e-5
        )


def test_coarse_grid_preserves_anisotropic_oblique_physical_geometry():
    direction = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    image = _geometry_image(direction=direction)
    coarse = coarse_image(image, 3.0, 15.0)

    assert coarse.GetDirection() == image.GetDirection()
    assert coarse.GetOrigin() == image.GetOrigin()
    assert min(coarse.GetSpacing()) >= 3.0
    original_last = image.TransformIndexToPhysicalPoint(
        tuple(value - 1 for value in image.GetSize())
    )
    coarse_last = coarse.TransformIndexToPhysicalPoint(
        tuple(value - 1 for value in coarse.GetSize())
    )
    assert np.allclose(coarse_last, original_last, atol=1e-6)


def test_converted_transform_supports_forward_and_inverse_mask_transfer():
    direction = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    fixed = sitk.Cast(_geometry_image(direction=direction) > 0, sitk.sitkUInt8)
    moving = sitk.Image(fixed)
    shift = np.array([5.0, -9.0, 7.5])
    moving.SetOrigin(tuple(np.asarray(fixed.GetOrigin()) + shift))
    matrix = np.eye(4)
    matrix[:3, 3] = shift
    transform = physical_matrix_to_sitk(matrix, center=(14.0, 3.0, -8.0), rigid=True)

    aligned = sitk.Resample(moving, fixed, transform, sitk.sitkNearestNeighbor)
    returned = sitk.Resample(
        fixed, moving, transform.GetInverse(), sitk.sitkNearestNeighbor
    )
    assert np.array_equal(
        sitk.GetArrayFromImage(aligned), sitk.GetArrayFromImage(fixed)
    )
    assert np.array_equal(
        sitk.GetArrayFromImage(returned), sitk.GetArrayFromImage(moving)
    )


def test_fireants_unavailable_has_explicit_simpleitk_fallback(monkeypatch):
    from imperandi.process.registration import fireants_backend
    from imperandi.process.registration.alignment import mask_rigid_refine

    def unavailable():
        raise fireants_backend.FireANTsUnavailable("test CUDA unavailable")

    monkeypatch.setattr(fireants_backend, "_imports", unavailable)
    fixed = _geometry_image()
    moving = sitk.Image(fixed)
    config = RegistrationConfig(
        mask_registration_backend="fireants",
        fireants_fallback_to_simpleitk=True,
        maximum_optimizer_iterations=1,
    )
    transform, report = mask_rigid_refine(
        fixed, moving, sitk.Euler3DTransform(), config, fireants_state={}
    )

    assert isinstance(transform, sitk.Euler3DTransform)
    assert report.diagnostics["backend"] == "simpleitk"
    assert report.diagnostics["requested_backend"] == "fireants"
    assert report.diagnostics["fallback_reason"] == "test CUDA unavailable"


def test_fireants_unavailable_can_be_strict(monkeypatch):
    from imperandi.process.registration import fireants_backend
    from imperandi.process.registration.alignment import mask_rigid_refine

    monkeypatch.setattr(
        fireants_backend,
        "_imports",
        lambda: (_ for _ in ()).throw(
            fireants_backend.FireANTsUnavailable("test package unavailable")
        ),
    )
    image = _geometry_image()
    config = RegistrationConfig(
        mask_registration_backend="fireants",
        fireants_fallback_to_simpleitk=False,
    )
    with pytest.raises(RuntimeError, match="test package unavailable"):
        mask_rigid_refine(
            image,
            image,
            sitk.Euler3DTransform(),
            config,
            fireants_state={},
        )


def test_fireants_backend_cli_override_is_explicit():
    from imperandi.process.registration.register import build_parser

    args = build_parser().parse_args(["--mask_registration_backend", "fireants"])
    assert args.mask_registration_backend == "fireants"


def test_linear_stage_diagnostics_use_original_grid_candidate_dice():
    from imperandi.process.registration.alignment import dice, register_pair

    fixed = sitk.Cast(_geometry_image() > 0, sitk.sitkUInt8)
    moving = sitk.Image(fixed)
    moving.SetSpacing((0.8, 1.8, 4.2))
    result = register_pair(
        fixed,
        moving,
        RegistrationConfig(
            early_stop_organ_dice=1.0,
            maximum_optimizer_iterations=1,
        ),
    )
    detail = result.stages["mask_rigid"]

    assert detail["backend"] == "simpleitk"
    assert detail["optimization_elapsed_seconds"] >= 0
    assert detail["gpu_peak_memory_allocated_bytes"] is None
    assert detail["transform_direction"] == "fixed_physical_to_moving_physical"
    assert np.isclose(
        detail["dice"],
        dice(fixed, moving, result.stage_transforms["mask_rigid"]),
    )


def test_benchmark_selects_seeded_manifest_group_and_ignores_singletons():
    from tools.benchmark_registration_backends import _select_group

    table = pd.DataFrame(
        [
            {
                "patient_key": patient,
                "nifti_path": f"image-{number}.nii.gz",
                "mask_liver": f"mask-{number}.nii.gz",
            }
            for number, patient in enumerate(["a", "a", "b", "b", "b", "single"])
        ]
    )
    config = RegistrationConfig(grouping_columns=["patient_key"])

    first, first_metadata = _select_group(table, config, seed=17)
    second, second_metadata = _select_group(table, config, seed=17)

    pd.testing.assert_frame_equal(first, second)
    assert first_metadata == second_metadata
    assert first.patient_key.nunique() == 1
    assert first.patient_key.iloc[0] in {"a", "b"}
    assert first_metadata["eligible_group_count"] == 2
    assert first_metadata["selected_group_size"] == len(first)
