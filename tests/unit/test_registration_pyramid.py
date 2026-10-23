"""Configured pyramid schedules are executed by each registration optimizer."""

import numpy as np
import pytest

from imperandi.process.registration import alignment
from imperandi.process.registration.config import RegistrationConfig

sitk = pytest.importorskip("SimpleITK")


@pytest.fixture(autouse=True)
def single_thread():
    original = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    yield
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(original)


@pytest.mark.parametrize(
    "stage", ["mask_rigid", "mask_affine", "mi_rigid", "mi_affine", "mask_elastic"]
)
@pytest.mark.parametrize(
    "factors,sigmas", [([4, 2, 1], [2, 1, 0]), ([4, 2], [2, 1]), ([1], [0])]
)
def test_optimizer_executes_configured_pyramid(monkeypatch, stage, factors, sigmas):
    factory = sitk.ImageRegistrationMethod
    observed = {}
    levels = []

    class ObservedRegistration:
        def __init__(self):
            self.registration = factory()
            self.registration.AddCommand(
                sitk.sitkMultiResolutionIterationEvent,
                lambda: levels.append(self.registration.GetCurrentLevel()),
            )

        def __getattr__(self, name):
            return getattr(self.registration, name)

        def SetShrinkFactorsPerLevel(self, values):
            observed["factors"] = list(values)
            self.registration.SetShrinkFactorsPerLevel(values)

        def SetSmoothingSigmasPerLevel(self, values):
            observed["sigmas"] = list(values)
            self.registration.SetSmoothingSigmasPerLevel(values)

        def SmoothingSigmasAreSpecifiedInPhysicalUnitsOn(self):
            observed["physical_units"] = True
            self.registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    monkeypatch.setattr(sitk, "ImageRegistrationMethod", ObservedRegistration)
    z, y, x = np.indices((24, 28, 32))
    values = ((x - 15) / 8) ** 2 + ((y - 13) / 6) ** 2 + ((z - 11) / 4) ** 2 < 1
    mask = sitk.GetImageFromArray(values.astype(np.uint8))
    mask.SetSpacing((1.2, 1.7, 2.1))
    domain = alignment.distance_map(mask, padding_mm=10, band_mm=15)
    prefix = "elastic" if stage == "mask_elastic" else "linear"
    other = "linear" if prefix == "elastic" else "elastic"
    config = RegistrationConfig.from_mapping(
        {
            f"{prefix}_shrink_factors": factors,
            f"{prefix}_smoothing_sigmas_mm": sigmas,
            f"{other}_shrink_factors": [3, 1],
            f"{other}_smoothing_sigmas_mm": [1, 0],
            "maximum_optimizer_iterations": 1,
            "elastic_optimizer_iterations": 1,
        }
    )
    initial = sitk.Euler3DTransform()
    if stage.startswith("mi_"):
        transform, _ = alignment.mi_refine(
            mask, mask, mask, mask, initial, config, affine=stage == "mi_affine"
        )
    else:
        transform, _ = getattr(alignment, f"{stage}_refine")(
            domain, domain, initial, config
        )

    assert observed == {"factors": factors, "sigmas": sigmas, "physical_units": True}
    assert levels == list(range(len(factors)))
    assert np.isfinite(transform.GetParameters()).all()
