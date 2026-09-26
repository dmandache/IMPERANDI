"""Small, explicit configuration contract for registration backends."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping

TUMOR_CONSENSUS_METHODS = ("anchor", "majority", "intersection", "union", "staple")
ORGAN_CONSENSUS_METHODS = ("anchor", "majority")
MASK_REGISTRATION_BACKENDS = ("simpleitk", "fireants")
REGISTRATION_STAGES = (
    "baseline",
    "pca",
    "geometry",
    "mask_rigid",
    "mask_affine",
    "mi_affine",
    "mask_elastic",
)

DEFAULT_MINIMUM_STAGE_DICE_IMPROVEMENT = {
    "geometry": 0.001,
    "pca": 0.001,
    "mask_rigid": 0.002,
    "mask_affine": 0.002,
    "mask_elastic": 0.005,
}

DEFAULT_MINIMUM_STAGE_MI_IMPROVEMENT = {
    "mi_affine": 0.0,
}

RENAMED_SETTINGS = {
    "group_columns": "grouping_columns",
    "organ_column": "organ_mask_column",
    "tumor_column": "tumor_mask_column",
    "organ_consensus": "organ_consensus_method",
    "tumor_consensus": "tumor_consensus_method",
    "affine": "enable_affine_stage",
    "elastic": "enable_elastic_stage",
    "bspline_ctrl_spacing_mm": "elastic_control_point_spacing_mm",
    "early_stop_dice": "early_stop_organ_dice",
    "boundary_margin_mm": "partial_mask_boundary_margin_mm",
    "allow_partial_organs": "accept_partial_organ_masks",
    "min_largest_component_fraction": "minimum_largest_component_fraction",
    "min_confidence_dice": "minimum_accepted_organ_dice",
    "min_common_fov_fraction": "minimum_common_field_of_view_fraction",
    "pca_max_rotation_degrees": "maximum_pca_rotation_degrees",
    "min_delta": "minimum_stage_dice_improvement",
    "iterations": "maximum_optimizer_iterations",
    "crop_padding_mm": "distance_map_crop_padding_mm",
    "distance_band_mm": "organ_boundary_band_half_width_mm",
    "keep_source_segmentation": "preserve_source_organ_mask",
    "constrain_tumor_to_organ": "clip_tumor_consensus_to_organ",
    "threshold": "consensus_probability_threshold",
    "reference_priority": "reference_selection_priority",
}

REMOVED_SETTINGS = {
    "affine_min_dice": "affine execution is controlled by enable_affine_stage",
    "elastic_min_dice": "use minimum_stage_dice_improvement.mask_elastic",
    "min_dice": "use minimum_accepted_organ_dice",
    "demons_smoothing_sigma_mm": "mask_elastic uses B-spline refinement, not Demons",
}


def _finite_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(value)
    )


@dataclass(frozen=True)
class RegistrationConfig:
    grouping_columns: list[str] = field(
        default_factory=lambda: ["patient_key", "study_id", "Modality"]
    )
    organ_mask_column: str = "mask_liver"
    tumor_mask_column: str = "mask_liver_tumor"
    organ_consensus_method: str = "anchor"
    tumor_consensus_method: str = "anchor"
    enable_affine_stage: bool = False
    enable_elastic_stage: bool = False
    mask_registration_backend: str = "simpleitk"
    fireants_fallback_to_simpleitk: bool = True
    fireants_coarse_spacing_mm: float = 4.0
    fireants_rigid_learning_rate: float = 0.003
    fireants_affine_learning_rate: float = 0.01
    elastic_control_point_spacing_mm: float = 90.0
    elastic_optimizer_iterations: int = 25
    maximum_elastic_displacement_p95_mm: float = 8.0
    maximum_elastic_displacement_mm: float = 12.0
    minimum_elastic_jacobian_determinant: float = 0.5
    maximum_elastic_jacobian_determinant: float = 2.0
    early_stop_organ_dice: float = 0.95
    partial_mask_boundary_margin_mm: float = 1.0
    accept_partial_organ_masks: bool = True
    minimum_largest_component_fraction: float = 0.8
    minimum_accepted_organ_dice: float = 0.5
    minimum_consensus_dice: float = 0.8
    minimum_common_field_of_view_fraction: float = 0.05
    maximum_pca_rotation_degrees: float = 45.0
    minimum_stage_dice_improvement: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_MINIMUM_STAGE_DICE_IMPROVEMENT)
    )
    minimum_stage_mi_improvement: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_MINIMUM_STAGE_MI_IMPROVEMENT)
    )
    maximum_mi_stage_dice_decrease: float = 0.002
    maximum_optimizer_iterations: int = 100
    distance_map_crop_padding_mm: float = 25.0
    organ_boundary_band_half_width_mm: float = 15.0
    organ_distance_field_padding_mm: float = 10.0
    organ_coverage_blend_width_mm: float = 3.0
    preserve_source_organ_mask: bool = False
    clip_tumor_consensus_to_organ: bool = True
    consensus_probability_threshold: float = 0.5
    reference_selection_priority: list[dict] = field(
        default_factory=lambda: [
            {"Modality": ["CT", "MR"]},
            {"phase": ["PORTAL_VENOUS", "ARTERIAL", "DELAYED", "NATIVE"]},
            {"mri_sequence": ["T1", "T2", "DWI"]},
            {"PixelSpacingXY": "min"},
            {"SliceThickness": "min"},
            {"registration_organ_volume_mm3": "max"},
        ]
    )

    def __post_init__(self):
        for name in (
            "preserve_source_organ_mask",
            "accept_partial_organ_masks",
            "enable_affine_stage",
            "enable_elastic_stage",
            "clip_tumor_consensus_to_organ",
            "fireants_fallback_to_simpleitk",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in (
            "minimum_largest_component_fraction",
            "minimum_accepted_organ_dice",
            "minimum_consensus_dice",
            "minimum_common_field_of_view_fraction",
        ):
            value = getattr(self, name)
            if not _finite_number(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.minimum_consensus_dice < self.minimum_accepted_organ_dice:
            raise ValueError(
                "minimum_consensus_dice must be greater than or equal to "
                "minimum_accepted_organ_dice"
            )
        if (
            not _finite_number(self.partial_mask_boundary_margin_mm)
            or self.partial_mask_boundary_margin_mm < 0
        ):
            raise ValueError(
                "partial_mask_boundary_margin_mm must be finite and nonnegative"
            )
        if (
            not _finite_number(self.maximum_pca_rotation_degrees)
            or not 0 <= self.maximum_pca_rotation_degrees <= 180
        ):
            raise ValueError("maximum_pca_rotation_degrees must be in [0, 180]")
        if self.organ_consensus_method not in ORGAN_CONSENSUS_METHODS:
            raise ValueError(f"Unknown organ consensus: {self.organ_consensus_method}")
        if self.tumor_consensus_method not in TUMOR_CONSENSUS_METHODS:
            raise ValueError(f"Unknown tumor consensus: {self.tumor_consensus_method}")
        if self.mask_registration_backend not in MASK_REGISTRATION_BACKENDS:
            raise ValueError(
                f"Unknown mask registration backend: {self.mask_registration_backend}"
            )
        for name in (
            "fireants_coarse_spacing_mm",
            "fireants_rigid_learning_rate",
            "fireants_affine_learning_rate",
        ):
            if not _finite_number(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            type(self.maximum_optimizer_iterations) is not int
            or self.maximum_optimizer_iterations < 1
        ):
            raise ValueError("maximum_optimizer_iterations must be a positive integer")
        if (
            type(self.elastic_optimizer_iterations) is not int
            or not 1 <= self.elastic_optimizer_iterations <= 100
        ):
            raise ValueError("elastic_optimizer_iterations must be in [1, 100]")
        for name in ("early_stop_organ_dice", "consensus_probability_threshold"):
            if not _finite_number(getattr(self, name)):
                raise ValueError(f"{name} must be a finite numeric threshold")
        if not 0 < self.early_stop_organ_dice <= 1:
            raise ValueError("early_stop_organ_dice must be in (0, 1]")
        if not 0 < self.consensus_probability_threshold < 1:
            raise ValueError("Invalid Dice or probability threshold")
        if not isinstance(self.minimum_stage_dice_improvement, Mapping):
            raise ValueError("minimum_stage_dice_improvement must be a mapping")
        unknown_delta_stages = set(self.minimum_stage_dice_improvement) - set(
            DEFAULT_MINIMUM_STAGE_DICE_IMPROVEMENT
        )
        if unknown_delta_stages:
            raise ValueError(
                "Unknown minimum_stage_dice_improvement stages: "
                f"{sorted(unknown_delta_stages)}"
            )
        minimum_improvement = {
            **DEFAULT_MINIMUM_STAGE_DICE_IMPROVEMENT,
            **self.minimum_stage_dice_improvement,
        }
        for name, value in minimum_improvement.items():
            if not _finite_number(value) or not 0 <= value <= 1:
                raise ValueError(
                    f"minimum_stage_dice_improvement {name} must be in [0, 1]"
                )
        object.__setattr__(self, "minimum_stage_dice_improvement", minimum_improvement)
        if not isinstance(self.minimum_stage_mi_improvement, Mapping):
            raise ValueError("minimum_stage_mi_improvement must be a mapping")
        unknown_mi_stages = set(self.minimum_stage_mi_improvement) - set(
            DEFAULT_MINIMUM_STAGE_MI_IMPROVEMENT
        )
        if unknown_mi_stages:
            raise ValueError(
                "Unknown minimum_stage_mi_improvement stages: "
                f"{sorted(unknown_mi_stages)}"
            )
        minimum_mi_improvement = {
            **DEFAULT_MINIMUM_STAGE_MI_IMPROVEMENT,
            **self.minimum_stage_mi_improvement,
        }
        for name, value in minimum_mi_improvement.items():
            if not _finite_number(value) or value < 0:
                raise ValueError(
                    f"minimum_stage_mi_improvement {name} must be nonnegative"
                )
        object.__setattr__(self, "minimum_stage_mi_improvement", minimum_mi_improvement)
        if (
            not _finite_number(self.maximum_mi_stage_dice_decrease)
            or not 0 <= self.maximum_mi_stage_dice_decrease <= 1
        ):
            raise ValueError("maximum_mi_stage_dice_decrease must be in [0, 1]")
        if (
            not _finite_number(self.distance_map_crop_padding_mm)
            or self.distance_map_crop_padding_mm < 0
            or not _finite_number(self.organ_boundary_band_half_width_mm)
            or self.organ_boundary_band_half_width_mm <= 0
        ):
            raise ValueError("Invalid distance-map crop or organ boundary band width")
        for name in (
            "organ_distance_field_padding_mm",
            "organ_coverage_blend_width_mm",
        ):
            if not _finite_number(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not _finite_number(self.elastic_control_point_spacing_mm)
            or self.elastic_control_point_spacing_mm <= 0
        ):
            raise ValueError(
                "elastic_control_point_spacing_mm must be finite and positive"
            )
        for name in (
            "maximum_elastic_displacement_p95_mm",
            "maximum_elastic_displacement_mm",
            "minimum_elastic_jacobian_determinant",
            "maximum_elastic_jacobian_determinant",
        ):
            if not _finite_number(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            self.maximum_elastic_displacement_p95_mm
            > self.maximum_elastic_displacement_mm
        ):
            raise ValueError(
                "maximum_elastic_displacement_p95_mm must not exceed "
                "maximum_elastic_displacement_mm"
            )
        if (
            self.minimum_elastic_jacobian_determinant
            >= self.maximum_elastic_jacobian_determinant
        ):
            raise ValueError(
                "minimum_elastic_jacobian_determinant must be less than "
                "maximum_elastic_jacobian_determinant"
            )
        for name in (self.organ_mask_column, self.tumor_mask_column):
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Column names must be nonempty strings")
        if (
            not isinstance(self.grouping_columns, list)
            or not self.grouping_columns
            or any(
                not isinstance(column, str) or not column.strip()
                for column in self.grouping_columns
            )
            or len(set(self.grouping_columns)) != len(self.grouping_columns)
        ):
            raise ValueError(
                "grouping_columns must be a nonempty list of unique columns"
            )
        if not isinstance(self.reference_selection_priority, list):
            raise ValueError(
                "reference_selection_priority must be an ordered criterion list"
            )
        columns = set()
        for criterion in self.reference_selection_priority:
            if not isinstance(criterion, Mapping) or len(criterion) != 1:
                raise ValueError(
                    "Each reference_selection_priority criterion must contain "
                    "exactly one column"
                )
            column, preference = next(iter(criterion.items()))
            if not isinstance(column, str) or not column.strip():
                raise ValueError(
                    "reference_selection_priority columns must be nonempty strings"
                )
            if column in columns:
                raise ValueError(
                    f"Duplicate reference_selection_priority column: {column}"
                )
            columns.add(column)
            if isinstance(preference, list):
                if not preference or any(
                    not isinstance(value, str) or not value.strip()
                    for value in preference
                ):
                    raise ValueError(
                        "reference_selection_priority "
                        f"{column} requires a nonempty string list"
                    )
                labels = [value.strip().upper() for value in preference]
                if column == "Modality":
                    labels = ["MR" if label == "MRI" else label for label in labels]
                if len(set(labels)) != len(labels):
                    raise ValueError(
                        "Duplicate reference_selection_priority categories for "
                        f"{column}"
                    )
            elif not isinstance(preference, str) or preference not in {"min", "max"}:
                raise ValueError(
                    f"reference_selection_priority {column} must be a categorical list "
                    "(e.g. [T1, T2]), 'min', or 'max'"
                )

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, Mapping):
            raise ValueError("registration must be a mapping")
        removed = set(value) & set(REMOVED_SETTINGS)
        if removed:
            explanations = ", ".join(
                f"{name} ({REMOVED_SETTINGS[name]})" for name in sorted(removed)
            )
            raise ValueError(f"Removed registration settings: {explanations}")
        renamed = set(value) & set(RENAMED_SETTINGS)
        if renamed:
            replacements = ", ".join(
                f"{name} -> {RENAMED_SETTINGS[name]}" for name in sorted(renamed)
            )
            raise ValueError(f"Renamed registration settings: {replacements}")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown registration settings: {sorted(unknown)}")
        return cls(**value)
