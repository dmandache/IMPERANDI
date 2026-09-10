"""Registration configuration rejects invalid values before loading images."""

import pytest

from imperandi.process.registration.config import RegistrationConfig


@pytest.mark.parametrize("name", ["min_dice", "affine_min_dice", "threshold"])
@pytest.mark.parametrize(
    "value", [True, False, None, "0.5", float("nan"), float("inf")]
)
def test_thresholds_require_finite_numbers(name, value):
    with pytest.raises(ValueError, match=name):
        RegistrationConfig(**{name: value})


@pytest.mark.parametrize("value", [0, 1])
def test_dice_threshold_endpoints_are_valid(value):
    config = RegistrationConfig(min_dice=value, affine_min_dice=value)
    assert config.min_dice == config.affine_min_dice == value
