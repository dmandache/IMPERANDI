"""Registration configuration rejects invalid values before loading images."""

import pytest

from imperandi.process.registration.config import RegistrationConfig


@pytest.mark.parametrize(
    "name", ["min_dice", "affine_min_dice", "early_stop_dice", "threshold"]
)
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


@pytest.mark.parametrize("value", [0, -0.01, 1.01])
def test_early_stop_dice_requires_positive_probability(value):
    with pytest.raises(ValueError, match="early_stop_dice"):
        RegistrationConfig(early_stop_dice=value)


@pytest.mark.parametrize(
    "priorities",
    [
        None,
        {"MRI": [{"mri_sequence": ["T1"]}]},
        "mri_sequence",
        [None],
        [{}],
        [{"mri_sequence": ["T1"], "phase": ["PORTAL_VENOUS"]}],
        [{"": ["T1"]}],
        [{1: ["T1"]}],
        [{"mri_sequence": "T1"}],
        [{"mri_sequence": []}],
        [{"mri_sequence": ["T1", " "]}],
        [{"mri_sequence": ["T1", 2]}],
        [{"mri_sequence": ["T1", " t1 "]}],
        [{"SliceThickness": "minimum"}],
        [{"SliceThickness": True}],
        [{"SliceThickness": None}],
        [{"SliceThickness": "min"}, {"SliceThickness": "max"}],
    ],
)
def test_invalid_reference_criteria_are_rejected(priorities):
    with pytest.raises(ValueError, match="reference_priority"):
        RegistrationConfig.from_mapping({"reference_priority": priorities})


def test_ordered_reference_criteria_preserve_user_order():
    priorities = [
        {"Modality": ["MR", "CT"]},
        {"mri_sequence": ["T1", "T2"]},
        {"phase": ["PORTAL_VENOUS"]},
        {"PixelSpacingXY": "min"},
        {"SliceThickness": "min"},
        {"registration_organ_volume_mm3": "max"},
    ]
    config = RegistrationConfig.from_mapping({"reference_priority": priorities})
    assert config.reference_priority == priorities
    assert RegistrationConfig(reference_priority=[]).reference_priority == []


@pytest.mark.parametrize(
    "columns",
    [None, [], {}, "patient_key", [""], [1], ["patient_key", "patient_key"]],
)
def test_group_columns_require_unique_nonempty_names(columns):
    with pytest.raises(ValueError, match="group_columns"):
        RegistrationConfig.from_mapping({"group_columns": columns})
