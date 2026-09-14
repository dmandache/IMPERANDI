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


@pytest.mark.parametrize(
    "priorities",
    [
        None,
        [],
        {"MRI": [{"mri_sequence": ["T1"]}]},
        {"MR": []},
        {"MR": {"mri_sequence": ["T1"]}},
        {"MR": [None]},
        {"MR": [{}]},
        {"MR": [{"mri_sequence": ["T1"], "phase": ["PORTAL_VENOUS"]}]},
        {"MR": [{"": ["T1"]}]},
        {"MR": [{1: ["T1"]}]},
        {"MR": [{"mri_sequence": "T1"}]},
        {"MR": [{"mri_sequence": []}]},
        {"MR": [{"mri_sequence": ["T1", " "]}]},
        {"MR": [{"mri_sequence": ["T1", 2]}]},
        {"MR": [{"mri_sequence": ["T1", " t1 "]}]},
        {"MR": [{"SliceThickness": "minimum"}]},
        {"MR": [{"SliceThickness": True}]},
        {"MR": [{"SliceThickness": None}]},
        {"MR": [{"SliceThickness": "min"}, {"SliceThickness": "max"}]},
    ],
)
def test_invalid_reference_criteria_are_rejected(priorities):
    with pytest.raises(ValueError, match="reference_priority"):
        RegistrationConfig.from_mapping({"reference_priority": priorities})


def test_ordered_reference_criteria_preserve_user_order():
    priorities = {
        "MR": [
            {"mri_sequence": ["T1", "T2"]},
            {"phase": ["PORTAL_VENOUS"]},
            {"PixelSpacingXY": "min"},
            {"SliceThickness": "min"},
            {"registration_organ_volume_mm3": "max"},
        ]
    }
    config = RegistrationConfig.from_mapping({"reference_priority": priorities})
    assert config.reference_priority == priorities
    assert "CT" not in config.reference_priority
    assert RegistrationConfig(reference_priority={}).reference_priority == {}
