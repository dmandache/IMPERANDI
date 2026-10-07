"""Small, explicit configuration contract for registration backends."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping

CONSENSUS_METHODS = ("anchor", "majority", "intersection", "union", "staple")
REGISTRATION_STAGES = (
    "baseline",
    "pca",
    "geometry",
    "mask_rigid",
    "mi_rigid",
    "mi_affine",
)


def _finite_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(value)
    )


@dataclass(frozen=True)
class RegistrationConfig:
    group_columns: list[str] = field(
        default_factory=lambda: ["patient_key", "study_id", "Modality"]
    )
    organ_column: str = "mask_liver"
    tumor_column: str = "mask_liver_tumor"
    method: str = "anchor"
    affine: bool = False
    affine_min_dice: float = 0.9
    early_stop_dice: float = 0.95
    boundary_margin_mm: float = 1.0
    allow_partial_organs: bool = True
    min_largest_component_fraction: float = 0.8
    min_confidence_dice: float = 0.5
    min_common_fov_fraction: float = 0.05
    demons_smoothing_sigma_mm: float = 1.0
    iterations: int = 100
    min_dice: float = 0.1
    crop_padding_mm: float = 25.0
    distance_band_mm: float = 15.0
    keep_source_segmentation: bool = False
    constrain_tumor_to_organ: bool = True
    threshold: float = 0.5
    reference_priority: list[dict] = field(
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
            "keep_source_segmentation",
            "allow_partial_organs",
            "affine",
            "constrain_tumor_to_organ",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in (
            "min_largest_component_fraction",
            "min_confidence_dice",
            "min_common_fov_fraction",
        ):
            value = getattr(self, name)
            if not _finite_number(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not _finite_number(self.boundary_margin_mm) or self.boundary_margin_mm < 0:
            raise ValueError("boundary_margin_mm must be finite and nonnegative")
        if self.method not in CONSENSUS_METHODS:
            raise ValueError(f"Unknown consensus method: {self.method}")
        if type(self.iterations) is not int or self.iterations < 1:
            raise ValueError("iterations must be a positive integer")
        for name in ("min_dice", "affine_min_dice", "early_stop_dice", "threshold"):
            if not _finite_number(getattr(self, name)):
                raise ValueError(f"{name} must be a finite numeric threshold")
        if not 0 < self.early_stop_dice <= 1:
            raise ValueError("early_stop_dice must be in (0, 1]")
        if (
            not 0 <= self.min_dice <= 1
            or not 0 <= self.affine_min_dice <= 1
            or not 0 < self.threshold < 1
        ):
            raise ValueError("Invalid Dice or probability threshold")
        if (
            not _finite_number(self.crop_padding_mm)
            or self.crop_padding_mm < 0
            or not _finite_number(self.distance_band_mm)
            or self.distance_band_mm <= 0
            or not _finite_number(self.demons_smoothing_sigma_mm)
            or self.demons_smoothing_sigma_mm <= 0
        ):
            raise ValueError(
                "Invalid registration crop, distance band or Demons smoothing sigma"
            )
        for name in (self.organ_column, self.tumor_column):
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Column names must be nonempty strings")
        if (
            not isinstance(self.group_columns, list)
            or not self.group_columns
            or any(
                not isinstance(column, str) or not column.strip()
                for column in self.group_columns
            )
            or len(set(self.group_columns)) != len(self.group_columns)
        ):
            raise ValueError("group_columns must be a nonempty list of unique columns")
        if not isinstance(self.reference_priority, list):
            raise ValueError("reference_priority must be an ordered criterion list")
        columns = set()
        for criterion in self.reference_priority:
            if not isinstance(criterion, Mapping) or len(criterion) != 1:
                raise ValueError(
                    "Each reference_priority criterion must contain exactly one column"
                )
            column, preference = next(iter(criterion.items()))
            if not isinstance(column, str) or not column.strip():
                raise ValueError("reference_priority columns must be nonempty strings")
            if column in columns:
                raise ValueError(f"Duplicate reference_priority column: {column}")
            columns.add(column)
            if isinstance(preference, list):
                if not preference or any(
                    not isinstance(value, str) or not value.strip()
                    for value in preference
                ):
                    raise ValueError(
                        f"reference_priority {column} requires a nonempty string list"
                    )
                labels = [value.strip().upper() for value in preference]
                if column == "Modality":
                    labels = ["MR" if label == "MRI" else label for label in labels]
                if len(set(labels)) != len(labels):
                    raise ValueError(
                        f"Duplicate reference_priority categories for {column}"
                    )
            elif not isinstance(preference, str) or preference not in {"min", "max"}:
                raise ValueError(
                    f"reference_priority {column} must be a categorical list "
                    "(e.g. [T1, T2]), 'min', or 'max'"
                )

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, Mapping):
            raise ValueError("registration must be a mapping")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown registration settings: {sorted(unknown)}")
        return cls(**value)
