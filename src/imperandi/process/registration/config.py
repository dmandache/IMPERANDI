"""Small, explicit configuration contract for registration backends."""

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Mapping

from imperandi.utils.columns import is_mask_path_column

TUMOR_CONSENSUS_METHODS = (
    "none",
    "anchor",
    "majority",
    "intersection",
    "union",
    "staple",
)
ORGAN_CONSENSUS_METHODS = ("anchor", "majority")
OUTPUT_SPACES = ("moving", "reference")
REGISTRATION_STAGES = (
    "baseline",
    "pca",
    "geometry",
    "mask_rigid",
    "mask_affine",
    "mi_affine",
    "mask_elastic",
    "mi_elastic",
)

STAGE_MODELS = {
    "mask_rigid": "rigid",
    "mask_affine": "affine",
    "mi_affine": "affine",
    "mask_elastic": "elastic",
    "mi_elastic": "elastic",
}


def _finite_number(value):
    try:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and isfinite(value)
        )
    except OverflowError:
        return False


def _validate_pyramid(shrink_factors, smoothing_sigmas_mm, prefix):
    """Validate paired coarse-to-fine schedules before calling SimpleITK."""
    shrink_name = f"{prefix}_shrink_factors"
    sigma_name = f"{prefix}_smoothing_sigmas_mm"
    if (
        not isinstance(shrink_factors, list)
        or not shrink_factors
        or any(
            type(value) is not int or not 1 <= value <= 2**32 - 1
            for value in shrink_factors
        )
    ):
        raise ValueError(
            f"{shrink_name} must be a nonempty list of integers in [1, {2**32 - 1}]"
        )
    if any(a <= b for a, b in zip(shrink_factors, shrink_factors[1:])):
        raise ValueError(f"{shrink_name} must be strictly decreasing (coarse to fine)")
    try:
        valid_sigmas = (
            isinstance(smoothing_sigmas_mm, list)
            and bool(smoothing_sigmas_mm)
            and all(
                _finite_number(value) and value >= 0 for value in smoothing_sigmas_mm
            )
        )
    except OverflowError:
        valid_sigmas = False
    if not valid_sigmas:
        raise ValueError(
            f"{sigma_name} must be a nonempty list of finite nonnegative numbers"
        )
    if len(shrink_factors) != len(smoothing_sigmas_mm):
        raise ValueError(
            f"{shrink_name} and {sigma_name} must have the same number of levels"
        )
    if any(a < b for a, b in zip(smoothing_sigmas_mm, smoothing_sigmas_mm[1:])):
        raise ValueError(f"{sigma_name} must be nonincreasing (coarse to fine)")


def _mapping(value, allowed, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {name} settings: {sorted(unknown, key=str)}")
    return dict(value)


@dataclass(frozen=True)
class RunCondition:
    dice_below: float | None = None


@dataclass(frozen=True)
class MetricConfig:
    roi: str = "organ_band"
    histogram_bins: int = 50
    sampling_strategy: str = "regular"
    sampling_fraction: float = 0.2
    sampling_seed: int = 42

    def __post_init__(self):
        if self.roi not in ("organ_band", "organ"):
            raise ValueError("metric.roi must be organ_band or organ")
        if type(self.histogram_bins) is not int or not 5 <= self.histogram_bins <= 256:
            raise ValueError("metric.histogram_bins must be an integer in [5, 256]")
        if self.sampling_strategy not in ("none", "regular", "random"):
            raise ValueError("Invalid metric.sampling_strategy")
        if (
            not _finite_number(self.sampling_fraction)
            or not 0 < self.sampling_fraction <= 1
        ):
            raise ValueError("metric.sampling_fraction must be in (0, 1]")
        if type(self.sampling_seed) is not int or not 1 <= self.sampling_seed < 2**32:
            raise ValueError("metric.sampling_seed must be an integer in [1, 2**32)")


@dataclass(frozen=True)
class OptimizerConfig:
    iterations: int


@dataclass(frozen=True)
class PyramidConfig:
    shrink_factors: list[int]
    smoothing_sigmas_mm: list[float]


@dataclass(frozen=True)
class AcceptanceConfig:
    minimum_dice_improvement: float | None = None
    minimum_mi_improvement: float | None = None
    maximum_dice_decrease: float | None = None


@dataclass(frozen=True)
class StageConfig:
    type: str
    run_if: RunCondition
    metric: MetricConfig | None
    optimizer: OptimizerConfig
    pyramid: PyramidConfig
    acceptance: AcceptanceConfig

    @property
    def model(self):
        return STAGE_MODELS[self.type]

    @property
    def uses_mi(self):
        return self.type.startswith("mi_")

    @classmethod
    def resolve(cls, value, config):
        if isinstance(value, cls):
            value = asdict(value)
        raw = _mapping(value, cls.__dataclass_fields__, "stage")
        name = raw.get("type")
        if not isinstance(name, str) or name not in STAGE_MODELS:
            raise ValueError(f"Unknown registration stage: {name}")
        elastic, mi = STAGE_MODELS[name] == "elastic", name.startswith("mi_")
        condition = raw.get(
            "run_if", "always" if mi else {"dice_below": 0.90 if elastic else 0.95}
        )
        if condition == "always":
            condition = {}
        condition = _mapping(condition, ("dice_below",), "run_if")
        threshold = condition.get("dice_below")
        if threshold is not None and (
            not _finite_number(threshold) or not 0 < threshold <= 1
        ):
            raise ValueError("run_if.dice_below must be in (0, 1]")
        metric = raw.get("metric")
        if not mi and metric is not None:
            raise ValueError(f"{name} does not accept MI metric settings")
        metric = (
            MetricConfig(
                **_mapping(
                    {} if metric is None else metric,
                    MetricConfig.__dataclass_fields__,
                    "metric",
                )
            )
            if mi
            else None
        )
        optimizer = _mapping(raw.get("optimizer", {}), ("iterations",), "optimizer")
        iterations = optimizer.get(
            "iterations",
            (
                config.elastic_optimizer_iterations
                if elastic
                else config.maximum_optimizer_iterations
            ),
        )
        if (
            type(iterations) is not int
            or iterations < 1
            or (elastic and iterations > 100)
        ):
            raise ValueError(
                "optimizer.iterations must be positive (elastic maximum 100)"
            )
        prefix = "elastic" if elastic else "linear"
        pyramid = _mapping(
            raw.get("pyramid", {}), PyramidConfig.__dataclass_fields__, "pyramid"
        )
        factors = pyramid.get(
            "shrink_factors", getattr(config, f"{prefix}_shrink_factors")
        )
        sigmas = pyramid.get(
            "smoothing_sigmas_mm", getattr(config, f"{prefix}_smoothing_sigmas_mm")
        )
        _validate_pyramid(factors, sigmas, "stage")
        acceptance = _mapping(
            raw.get("acceptance", {}),
            AcceptanceConfig.__dataclass_fields__,
            "acceptance",
        )
        allowed = (
            ("minimum_mi_improvement", "maximum_dice_decrease")
            if mi
            else ("minimum_dice_improvement",)
        )
        for key, val in acceptance.items():
            if key not in allowed and val is not None:
                raise ValueError(f"{name} does not accept acceptance.{key}")
        defaults = (
            {"minimum_mi_improvement": 0.0, "maximum_dice_decrease": 0.002}
            if mi
            else {"minimum_dice_improvement": 0.005 if elastic else 0.002}
        )
        resolved = {**defaults, **{k: v for k, v in acceptance.items() if k in allowed}}
        for key, val in resolved.items():
            if (
                not _finite_number(val)
                or val < 0
                or (key != "minimum_mi_improvement" and val > 1)
            ):
                raise ValueError(f"Invalid acceptance.{key}")
        return cls(
            name,
            RunCondition(threshold),
            metric,
            OptimizerConfig(iterations),
            PyramidConfig(list(factors), list(sigmas)),
            AcceptanceConfig(**resolved),
        )


@dataclass(frozen=True)
class RegistrationConfig:
    grouping_columns: list[str] = field(
        default_factory=lambda: ["patient_key", "study_id", "Modality"]
    )
    organ_mask_column: str = "mask_liver_path"
    tumor_mask_column: str = "mask_liver_tumor_path"
    organ_consensus_method: str = "anchor"
    tumor_consensus_method: str = "anchor"
    output_space: str = "moving"
    transform_additional_masks: bool = False
    stages: list[StageConfig] = field(
        default_factory=lambda: [{"type": "mask_rigid"}, {"type": "mask_affine"}]
    )
    initializer_minimum_dice_improvement: float = 0.001
    elastic_control_point_spacing_mm: float = 90.0
    elastic_optimizer_iterations: int = 25
    maximum_elastic_displacement_p95_mm: float = 8.0
    maximum_elastic_displacement_mm: float = 12.0
    minimum_elastic_jacobian_determinant: float = 0.5
    maximum_elastic_jacobian_determinant: float = 2.0
    partial_mask_boundary_margin_mm: float = 1.0
    accept_partial_organ_masks: bool = True
    minimum_largest_component_fraction: float = 0.8
    minimum_accepted_organ_dice: float = 0.5
    minimum_consensus_dice: float = 0.8
    minimum_common_field_of_view_fraction: float = 0.05
    maximum_pca_rotation_degrees: float = 45.0
    maximum_optimizer_iterations: int = 100
    linear_shrink_factors: list[int] = field(default_factory=lambda: [4, 2, 1])
    linear_smoothing_sigmas_mm: list[float] = field(
        default_factory=lambda: [2.0, 1.0, 0.0]
    )
    elastic_shrink_factors: list[int] = field(default_factory=lambda: [4, 2, 1])
    elastic_smoothing_sigmas_mm: list[float] = field(
        default_factory=lambda: [2.0, 1.0, 0.0]
    )
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
        for prefix in ("linear", "elastic"):
            shrink_name = f"{prefix}_shrink_factors"
            sigma_name = f"{prefix}_smoothing_sigmas_mm"
            factors, sigmas = getattr(self, shrink_name), getattr(self, sigma_name)
            _validate_pyramid(factors, sigmas, prefix)
            # Do not retain mutable lists owned by a caller's manifest.
            object.__setattr__(self, shrink_name, list(factors))
            object.__setattr__(self, sigma_name, list(sigmas))
        for name in (
            "transform_additional_masks",
            "preserve_source_organ_mask",
            "accept_partial_organ_masks",
            "clip_tumor_consensus_to_organ",
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
        if self.output_space not in OUTPUT_SPACES:
            raise ValueError(
                f"Unknown registration output space: {self.output_space}; "
                f"expected one of {OUTPUT_SPACES}"
            )
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
        if (
            not _finite_number(self.consensus_probability_threshold)
            or not 0 < self.consensus_probability_threshold < 1
        ):
            raise ValueError("consensus_probability_threshold must be in (0, 1)")
        if (
            not _finite_number(self.initializer_minimum_dice_improvement)
            or not 0 <= self.initializer_minimum_dice_improvement <= 1
        ):
            raise ValueError("initializer_minimum_dice_improvement must be in [0, 1]")
        if not isinstance(self.stages, list):
            raise ValueError("stages must be an ordered list")
        resolved = [StageConfig.resolve(stage, self) for stage in self.stages]
        names = [stage.type for stage in resolved]
        if len(names) != len(set(names)):
            raise ValueError("Registration stage types must be unique")
        ranks = [
            {"rigid": 0, "affine": 1, "elastic": 2}[stage.model] for stage in resolved
        ]
        if ranks != sorted(ranks):
            raise ValueError(
                "Registration stages must order rigid before affine before elastic"
            )
        object.__setattr__(self, "stages", resolved)
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
            if not is_mask_path_column(name):
                raise ValueError("Mask filepath columns must be named mask_<name>_path")
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
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown registration settings: {sorted(unknown)}")
        return cls(**value)
