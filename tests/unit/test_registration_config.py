"""Registration configuration rejects invalid values before loading images."""

import pytest

from imperandi.process.registration.config import RegistrationConfig


@pytest.mark.parametrize(
    "name", ["early_stop_organ_dice", "consensus_probability_threshold"]
)
@pytest.mark.parametrize(
    "value", [True, False, None, "0.5", float("nan"), float("inf")]
)
def test_thresholds_require_finite_numbers(name, value):
    with pytest.raises(ValueError, match=name):
        RegistrationConfig(**{name: value})


@pytest.mark.parametrize("value", [0, -0.01, 1.01])
def test_early_stop_dice_requires_positive_probability(value):
    with pytest.raises(ValueError, match="early_stop_organ_dice"):
        RegistrationConfig(early_stop_organ_dice=value)


@pytest.mark.parametrize("value", [-0.01, 180.01, True, None, "45", float("nan")])
def test_pca_rotation_limit_is_bounded(value):
    with pytest.raises(ValueError, match="maximum_pca_rotation_degrees"):
        RegistrationConfig(maximum_pca_rotation_degrees=value)


@pytest.mark.parametrize("value", [0, 45, 180])
def test_pca_rotation_limit_accepts_valid_angles(value):
    config = RegistrationConfig(maximum_pca_rotation_degrees=value)
    assert config.maximum_pca_rotation_degrees == value


def test_minimum_stage_improvement_partial_override_keeps_defaults():
    config = RegistrationConfig(minimum_stage_dice_improvement={"pca": 0.01})
    assert config.minimum_stage_dice_improvement["pca"] == 0.01
    assert config.minimum_stage_dice_improvement["mask_rigid"] == 0.002
    assert config.minimum_stage_dice_improvement["mi_affine"] == 0.003


@pytest.mark.parametrize("value", [None, [], "pca"])
def test_minimum_stage_improvement_requires_mapping(value):
    with pytest.raises(ValueError, match="minimum_stage_dice_improvement"):
        RegistrationConfig(minimum_stage_dice_improvement=value)


@pytest.mark.parametrize("value", [-0.01, 1.01, True, None, "0.1", float("nan")])
def test_minimum_stage_improvement_values_are_probabilities(value):
    with pytest.raises(ValueError, match="minimum_stage_dice_improvement pca"):
        RegistrationConfig(minimum_stage_dice_improvement={"pca": value})


def test_minimum_stage_improvement_rejects_unknown_stages():
    with pytest.raises(ValueError, match="Unknown minimum_stage_dice_improvement"):
        RegistrationConfig(minimum_stage_dice_improvement={"rigid": 0.002})


@pytest.mark.parametrize(
    "name", ["affine_min_dice", "min_dice", "demons_smoothing_sigma_mm"]
)
def test_redundant_settings_are_removed(name):
    with pytest.raises(ValueError, match="Removed registration settings"):
        RegistrationConfig.from_mapping({name: 0.1})


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
    with pytest.raises(ValueError, match="reference_selection_priority"):
        RegistrationConfig.from_mapping({"reference_selection_priority": priorities})


def test_ordered_reference_criteria_preserve_user_order():
    priorities = [
        {"Modality": ["MR", "CT"]},
        {"mri_sequence": ["T1", "T2"]},
        {"phase": ["PORTAL_VENOUS"]},
        {"PixelSpacingXY": "min"},
        {"SliceThickness": "min"},
        {"registration_organ_volume_mm3": "max"},
    ]
    config = RegistrationConfig.from_mapping(
        {"reference_selection_priority": priorities}
    )
    assert config.reference_selection_priority == priorities
    assert (
        RegistrationConfig(reference_selection_priority=[]).reference_selection_priority
        == []
    )


@pytest.mark.parametrize(
    "columns",
    [None, [], {}, "patient_key", [""], [1], ["patient_key", "patient_key"]],
)
def test_grouping_columns_require_unique_nonempty_names(columns):
    with pytest.raises(ValueError, match="grouping_columns"):
        RegistrationConfig.from_mapping({"grouping_columns": columns})


def test_tumor_consensus_method_is_explicit():
    assert (
        RegistrationConfig.from_mapping(
            {"tumor_consensus_method": "union"}
        ).tumor_consensus_method
        == "union"
    )
    with pytest.raises(ValueError, match="Unknown registration settings.*method"):
        RegistrationConfig.from_mapping({"method": "union"})


@pytest.mark.parametrize("method", [None, "union", "staple", True])
def test_invalid_organ_consensus_is_rejected(method):
    with pytest.raises(ValueError, match="Unknown organ consensus"):
        RegistrationConfig(organ_consensus_method=method)


def test_old_setting_names_report_replacements():
    with pytest.raises(ValueError, match="group_columns -> grouping_columns"):
        RegistrationConfig.from_mapping({"group_columns": ["patient_key"]})
