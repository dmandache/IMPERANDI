"""Registration configuration rejects invalid values before loading images."""

import pytest

from imperandi.process.registration.config import RegistrationConfig


def test_registration_and_consensus_dice_defaults_are_distinct():
    config = RegistrationConfig()
    assert config.minimum_accepted_organ_dice == 0.5
    assert config.minimum_consensus_dice == 0.8
    assert [stage.type for stage in config.stages] == ["mask_rigid", "mask_affine"]


def test_output_space_defaults_to_moving_and_rejects_unknown_values():
    assert RegistrationConfig().output_space == "moving"
    assert RegistrationConfig(output_space="reference").output_space == "reference"
    with pytest.raises(ValueError, match="output space"):
        RegistrationConfig(output_space="fixed")


@pytest.mark.parametrize(
    "name",
    [
        "consensus_probability_threshold",
    ],
)
@pytest.mark.parametrize(
    "value", [True, False, None, "0.5", float("nan"), float("inf")]
)
def test_thresholds_require_finite_numbers(name, value):
    with pytest.raises(ValueError, match=name):
        RegistrationConfig(**{name: value})


def test_consensus_dice_cannot_be_below_registration_acceptance():
    with pytest.raises(ValueError, match="minimum_consensus_dice"):
        RegistrationConfig(
            minimum_accepted_organ_dice=0.7,
            minimum_consensus_dice=0.6,
        )


@pytest.mark.parametrize("value", [-0.01, 180.01, True, None, "45", float("nan")])
def test_pca_rotation_limit_is_bounded(value):
    with pytest.raises(ValueError, match="maximum_pca_rotation_degrees"):
        RegistrationConfig(maximum_pca_rotation_degrees=value)


@pytest.mark.parametrize("value", [0, 45, 180])
def test_pca_rotation_limit_accepts_valid_angles(value):
    config = RegistrationConfig(maximum_pca_rotation_degrees=value)
    assert config.maximum_pca_rotation_degrees == value


@pytest.mark.parametrize("value", [0, -1, True, None, "90", float("nan")])
def test_elastic_control_point_spacing_must_be_positive(value):
    with pytest.raises(ValueError, match="elastic_control_point_spacing_mm"):
        RegistrationConfig(elastic_control_point_spacing_mm=value)


@pytest.mark.parametrize(
    "name",
    ["organ_distance_field_padding_mm", "organ_coverage_blend_width_mm"],
)
@pytest.mark.parametrize("value", [0, -1, True, None, "3", float("nan")])
def test_organ_distance_transfer_widths_must_be_positive(name, value):
    with pytest.raises(ValueError, match=name):
        RegistrationConfig(**{name: value})


@pytest.mark.parametrize("value", [0, 101, -1, True, None, 25.0])
def test_elastic_optimizer_iterations_are_short_positive_integer(value):
    with pytest.raises(ValueError, match="elastic_optimizer_iterations"):
        RegistrationConfig(elastic_optimizer_iterations=value)


@pytest.mark.parametrize(
    "name",
    [
        "maximum_elastic_displacement_p95_mm",
        "maximum_elastic_displacement_mm",
        "minimum_elastic_jacobian_determinant",
        "maximum_elastic_jacobian_determinant",
    ],
)
@pytest.mark.parametrize("value", [0, -1, True, None, "1", float("nan")])
def test_elastic_qc_limits_must_be_positive(name, value):
    with pytest.raises(ValueError, match=name):
        RegistrationConfig(**{name: value})


def test_elastic_qc_limit_order_is_validated():
    with pytest.raises(ValueError, match="p95_mm must not exceed"):
        RegistrationConfig(
            maximum_elastic_displacement_p95_mm=13,
            maximum_elastic_displacement_mm=12,
        )
    with pytest.raises(ValueError, match="minimum_elastic_jacobian"):
        RegistrationConfig(
            minimum_elastic_jacobian_determinant=2,
            maximum_elastic_jacobian_determinant=2,
        )


@pytest.mark.parametrize(
    "name",
    [
        "affine_min_dice",
        "elastic_min_dice",
        "min_dice",
        "demons_smoothing_sigma_mm",
    ],
)
def test_redundant_settings_are_removed(name):
    with pytest.raises(ValueError, match=f"Unknown registration settings.*{name}"):
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


@pytest.mark.parametrize(
    "name",
    [
        "group_columns",
        "organ_column",
        "tumor_column",
        "organ_consensus",
        "tumor_consensus",
        "affine",
        "elastic",
        "bspline_ctrl_spacing_mm",
        "early_stop_dice",
        "boundary_margin_mm",
        "allow_partial_organs",
        "min_largest_component_fraction",
        "min_confidence_dice",
        "min_common_fov_fraction",
        "pca_max_rotation_degrees",
        "min_delta",
        "iterations",
        "crop_padding_mm",
        "distance_band_mm",
        "keep_source_segmentation",
        "constrain_tumor_to_organ",
        "threshold",
        "reference_priority",
    ],
)
def test_old_setting_names_are_unknown(name):
    with pytest.raises(ValueError, match=f"Unknown registration settings.*{name}"):
        RegistrationConfig.from_mapping({name: True})


@pytest.mark.parametrize("prefix", ["linear", "elastic"])
def test_pyramid_defaults_preserve_existing_schedule(prefix):
    config = RegistrationConfig()
    assert getattr(config, f"{prefix}_shrink_factors") == [4, 2, 1]
    assert getattr(config, f"{prefix}_smoothing_sigmas_mm") == [2.0, 1.0, 0.0]


@pytest.mark.parametrize("prefix", ["linear", "elastic"])
@pytest.mark.parametrize(
    "factors,sigmas", [([8, 4, 2], [4, 2, 1]), ([1], [0]), ([4, 2], [0, 0])]
)
def test_pyramid_schedules_support_coarse_and_single_level_runs(
    prefix, factors, sigmas
):
    from dataclasses import asdict
    import yaml

    factors, sigmas = list(factors), list(sigmas)
    settings = {
        f"{prefix}_shrink_factors": factors,
        f"{prefix}_smoothing_sigmas_mm": sigmas,
    }
    config = RegistrationConfig.from_mapping(settings)
    assert getattr(config, f"{prefix}_shrink_factors") == factors
    assert getattr(config, f"{prefix}_smoothing_sigmas_mm") == sigmas
    assert (
        RegistrationConfig.from_mapping(yaml.safe_load(yaml.safe_dump(asdict(config))))
        == config
    )
    factors.clear()
    sigmas.clear()
    assert getattr(config, f"{prefix}_shrink_factors")
    assert getattr(config, f"{prefix}_smoothing_sigmas_mm")


@pytest.mark.parametrize("prefix", ["linear", "elastic"])
@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        "4,2,1",
        [True, 2, 1],
        [4.0, 2, 1],
        [4, 2, 0],
        [4, 2, -1],
        [4, 4, 1],
        [1, 2, 4],
        [float("inf"), 2, 1],
        [2**32, 2, 1],
    ],
)
def test_invalid_pyramid_shrink_factors_are_rejected(prefix, value):
    name = f"{prefix}_shrink_factors"
    with pytest.raises(ValueError, match=name):
        RegistrationConfig.from_mapping({name: value})


@pytest.mark.parametrize("prefix", ["linear", "elastic"])
@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        "2,1,0",
        [True, 1, 0],
        ["2", 1, 0],
        [2, 1, -1],
        [0, 1, 2],
        [float("inf"), 1, 0],
        [float("nan"), 1, 0],
    ],
)
def test_invalid_pyramid_smoothing_sigmas_are_rejected(prefix, value):
    name = f"{prefix}_smoothing_sigmas_mm"
    with pytest.raises(ValueError, match=name):
        RegistrationConfig.from_mapping({name: value})


@pytest.mark.parametrize("prefix", ["linear", "elastic"])
def test_pyramid_pairs_must_have_the_same_length(prefix):
    with pytest.raises(
        ValueError, match=f"{prefix}_shrink_factors.*same number of levels"
    ):
        RegistrationConfig.from_mapping({f"{prefix}_shrink_factors": [2, 1]})
