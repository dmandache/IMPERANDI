"""Small, explicit configuration contract for registration backends."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping

CONSENSUS_METHODS = ("anchor", "majority", "intersection", "union", "staple")
REGISTRATION_STAGES = (
    "baseline",
    "geometry",
    "pca",
    "rigid",
    "affine",
    "elastic",
)
SUPPORTED_MODALITIES = ("CT", "MR")


@dataclass(frozen=True)
class RegistrationConfig:
    visit_column: str = "study_id"
    organ_column: str = "mask_liver"
    tumor_column: str = "mask_liver_tumor"
    method: str = "anchor"
    affine: bool = False
    affine_min_dice: float = 0.9
    elastic: bool = False
    elastic_min_dice: float = 0.7
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
    constrain_tumor_to_organ: bool = True
    threshold: float = 0.5
    reference_priority: dict = field(
        default_factory=lambda: {
            "CT": [
                {"phase": p} for p in ["PORTAL_VENOUS", "ARTERIAL", "DELAYED", "NATIVE"]
            ],
            "MR": [{"mri_sequence": s} for s in ["T1", "T2", "DWI"]],
        }
    )

    def __post_init__(self):
        if type(self.allow_partial_organs) is not bool:
            raise ValueError("allow_partial_organs must be a boolean")
        for name in (
            "elastic_min_dice",
            "min_largest_component_fraction",
            "min_confidence_dice",
            "min_common_fov_fraction",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 < value <= 1
            ):
                raise ValueError(f"{name} must be in (0, 1]")
        if (
            isinstance(self.boundary_margin_mm, bool)
            or not isinstance(self.boundary_margin_mm, (int, float))
            or not isfinite(self.boundary_margin_mm)
            or self.boundary_margin_mm < 0
        ):
            raise ValueError("boundary_margin_mm must be finite and nonnegative")
        if self.method not in CONSENSUS_METHODS:
            raise ValueError(f"Unknown consensus method: {self.method}")
        if (
            type(self.affine) is not bool
            or type(self.elastic) is not bool
            or type(self.constrain_tumor_to_organ) is not bool
        ):
            raise ValueError(
                "affine, elastic and constrain_tumor_to_organ must be booleans"
            )
        if type(self.iterations) is not int or self.iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if (
            not 0 <= self.min_dice <= 1
            or not 0 <= self.affine_min_dice <= 1
            or not 0 < self.threshold < 1
        ):
            raise ValueError("Invalid Dice or probability threshold")
        if (
            isinstance(self.crop_padding_mm, bool)
            or not isinstance(self.crop_padding_mm, (int, float))
            or not isfinite(self.crop_padding_mm)
            or self.crop_padding_mm < 0
            or isinstance(self.distance_band_mm, bool)
            or not isinstance(self.distance_band_mm, (int, float))
            or not isfinite(self.distance_band_mm)
            or self.distance_band_mm <= 0
            or isinstance(self.demons_smoothing_sigma_mm, bool)
            or not isinstance(self.demons_smoothing_sigma_mm, (int, float))
            or not isfinite(self.demons_smoothing_sigma_mm)
            or self.demons_smoothing_sigma_mm <= 0
        ):
            raise ValueError(
                "Invalid registration crop, distance band or Demons smoothing sigma"
            )
        for name in (self.visit_column, self.organ_column, self.tumor_column):
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Column names must be nonempty strings")
        if not isinstance(self.reference_priority, Mapping):
            raise ValueError("reference_priority must be a modality mapping")
        for modality, selectors in self.reference_priority.items():
            if (
                modality not in SUPPORTED_MODALITIES
                or not isinstance(selectors, list)
                or not selectors
            ):
                raise ValueError("Reference priorities require CT/MR selector lists")
            for selector in selectors:
                if (
                    not isinstance(selector, Mapping)
                    or not selector
                    or any(
                        not isinstance(k, str) or not isinstance(v, str)
                        for k, v in selector.items()
                    )
                ):
                    raise ValueError(
                        "Reference selectors must be nonempty string mappings"
                    )

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, Mapping):
            raise ValueError("registration must be a mapping")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown registration settings: {sorted(unknown)}")
        return cls(**value)
