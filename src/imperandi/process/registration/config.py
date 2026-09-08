"""Small, explicit configuration contract for registration backends."""

from dataclasses import dataclass, field
from typing import Mapping

CONSENSUS_METHODS = ("anchor", "majority", "intersection", "union", "staple")
REGISTRATION_STAGES = ("baseline", "pca", "rigid", "affine")
SUPPORTED_MODALITIES = ("CT", "MR")


@dataclass(frozen=True)
class RegistrationConfig:
    visit_column: str = "study_id"
    organ_column: str = "mask_liver"
    tumor_column: str = "mask_liver_tumor"
    method: str = "anchor"
    affine: bool = False
    affine_min_dice: float = 0.9
    iterations: int = 100
    min_dice: float = 0.1
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
        if self.method not in CONSENSUS_METHODS:
            raise ValueError(f"Unknown consensus method: {self.method}")
        if type(self.affine) is not bool:
            raise ValueError("affine must be a boolean")
        if type(self.iterations) is not int or self.iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if (
            not 0 <= self.min_dice <= 1
            or not 0 <= self.affine_min_dice <= 1
            or not 0 < self.threshold < 1
        ):
            raise ValueError("Invalid Dice or probability threshold")
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
