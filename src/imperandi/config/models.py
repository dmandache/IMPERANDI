"""Pydantic models for the user-facing IMPERANDI project file."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONTRAST_PHASES = frozenset(
    {"NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED", "HEPATOBILIARY", "OTHER"}
)
CLINICAL_SLOTS = frozenset(
    {
        "CT_NATIVE",
        "CT_ARTERIAL",
        "CT_PORTAL_VENOUS",
        "CT_DELAYED",
        "MR_T2",
        "MR_DWI",
        "MR_T1_NATIVE",
        "MR_T1_ARTERIAL",
        "MR_T1_PORTAL_VENOUS",
        "MR_T1_DELAYED",
        "MR_T1_HEPATOBILIARY",
    }
)


class StrictModel(BaseModel):
    """Base model that rejects misspelled or obsolete configuration fields."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TableFormat(str, Enum):
    CSV = "csv"
    PARQUET = "parquet"


class ProjectConfig(StrictModel):
    name: str
    profile: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("project.name must not be empty")
        return value


class InputConfig(StrictModel):
    sources: list[str]
    archive_depth: int = Field(default=3, ge=0)

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, value: list[str]) -> list[str]:
        normalized = [source.strip() for source in value]
        if not normalized or any(not source for source in normalized):
            raise ValueError("input.sources must contain at least one path or glob")
        if len(set(normalized)) != len(normalized):
            raise ValueError("input.sources must not contain duplicates")
        return normalized


class OutputConfig(StrictModel):
    root: Path
    imaging_root: Path | None = None
    table_format: TableFormat = TableFormat.PARQUET
    publish_formats: list[TableFormat] = Field(
        default_factory=lambda: [TableFormat.PARQUET]
    )

    @model_validator(mode="before")
    @classmethod
    def default_publication_to_table_format(cls, value):
        if isinstance(value, dict) and "publish_formats" not in value:
            value = dict(value)
            value["publish_formats"] = [value.get("table_format", TableFormat.PARQUET)]
        return value

    @field_validator("publish_formats")
    @classmethod
    def unique_formats(cls, value: list[TableFormat]) -> list[TableFormat]:
        if not value:
            raise ValueError("output.publish_formats must not be empty")
        return list(dict.fromkeys(value))


class IdentityFallbackConfig(StrictModel):
    columns: list[str] = Field(default_factory=list)
    on_missing: Literal["error", "keep"] = "error"

    @field_validator("columns")
    @classmethod
    def normalize_columns(cls, value: list[str]) -> list[str]:
        normalized = [column.strip() for column in value]
        if any(not column for column in normalized):
            raise ValueError("identity fallback columns must not be empty")
        return list(dict.fromkeys(normalized))


class IdentitySourceConfig(StrictModel):
    patient_id_columns: list[str] = Field(default_factory=lambda: ["PatientID"])
    namespace_columns: list[str] = Field(
        default_factory=lambda: ["site_id", "IssuerOfPatientID"]
    )
    fallback: IdentityFallbackConfig = Field(default_factory=IdentityFallbackConfig)

    @field_validator("patient_id_columns", "namespace_columns")
    @classmethod
    def normalize_columns(cls, value: list[str]) -> list[str]:
        normalized = [column.strip() for column in value]
        if any(not column for column in normalized):
            raise ValueError("identity source columns must not be empty")
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def require_identity_source(self):
        if not self.patient_id_columns and not self.fallback.columns:
            raise ValueError(
                "identity.source requires patient_id_columns or fallback.columns"
            )
        return self


class IdentityNormalizationConfig(StrictModel):
    strip: bool = True
    case: Literal["preserve", "upper", "lower"] = "upper"
    collapse_whitespace: bool = True


class HmacIdentityConfig(StrictModel):
    secret_env: str
    namespace: str
    prefix: str = "P"
    length: int = Field(default=16, ge=12, le=64)

    @field_validator("secret_env", "namespace")
    @classmethod
    def non_empty_hmac_fields(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("HMAC secret_env and namespace must not be empty")
        return value.strip()


class CanonicalIdentityConfig(StrictModel):
    strategy: Literal["source", "crosswalk", "hmac", "crosswalk_then_hmac"] = "source"
    crosswalk: Path | None = None
    crosswalk_keys: list[str] = Field(
        default_factory=lambda: ["site_id", "dicom_patient_id"]
    )
    crosswalk_value: str = "patient_id"
    hmac: HmacIdentityConfig | None = None

    @model_validator(mode="after")
    def validate_strategy_requirements(self):
        if not self.crosswalk_keys:
            raise ValueError("identity.canonical.crosswalk_keys must not be empty")
        if "crosswalk" in self.strategy and self.crosswalk is None:
            raise ValueError(f"identity strategy {self.strategy!r} requires crosswalk")
        if "hmac" in self.strategy and self.hmac is None:
            raise ValueError(f"identity strategy {self.strategy!r} requires hmac")
        return self

    @field_validator("crosswalk_keys")
    @classmethod
    def normalize_crosswalk_keys(cls, value: list[str]) -> list[str]:
        normalized = [column.strip() for column in value]
        if any(not column for column in normalized):
            raise ValueError("identity crosswalk keys must not be empty")
        return list(dict.fromkeys(normalized))

    @field_validator("crosswalk_value")
    @classmethod
    def normalize_crosswalk_value(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("identity canonical crosswalk_value must not be empty")
        return value


class IdentityValidationConfig(StrictModel):
    source_collision: Literal["error", "flag"] = "error"
    multiple_source_ids_per_patient: Literal["allow", "flag", "error"] = "allow"


class SensitiveIdentityConfig(StrictModel):
    persist_raw_identifiers: Literal["never", "separate_table", "cohort"] = (
        "separate_table"
    )


class IdentityConfig(StrictModel):
    source: IdentitySourceConfig = Field(default_factory=IdentitySourceConfig)
    normalization: IdentityNormalizationConfig = Field(
        default_factory=IdentityNormalizationConfig
    )
    canonical: CanonicalIdentityConfig = Field(default_factory=CanonicalIdentityConfig)
    validation: IdentityValidationConfig = Field(
        default_factory=IdentityValidationConfig
    )
    sensitive_fields: SensitiveIdentityConfig = Field(
        default_factory=SensitiveIdentityConfig
    )


class OntologyKeyConfig(StrictModel):
    match: Literal["exact", "normalized_exact", "numeric_exact"] = "normalized_exact"


class OntologyOutputConfig(StrictModel):
    value_column: str
    target_column: str
    vocabulary: Literal["clinical_slot", "contrast_phase"] | None = None

    @field_validator("value_column", "target_column")
    @classmethod
    def non_empty_columns(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("ontology value and target columns must not be empty")
        return value


class OntologyConfig(StrictModel):
    id: str
    source: Path
    keys: dict[str, OntologyKeyConfig]
    output: OntologyOutputConfig
    unmatched: Literal["keep", "error"] = "keep"
    conflicts: Literal["error", "flag", "first"] = "error"

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("ontology id must not be empty")
        return value

    @field_validator("keys")
    @classmethod
    def validate_keys(cls, value: dict[str, OntologyKeyConfig]):
        if not value:
            raise ValueError("ontology keys must not be empty")
        normalized = {column.strip(): key for column, key in value.items()}
        if any(not column for column in normalized):
            raise ValueError("ontology key columns must not be empty")
        if len(normalized) != len(value):
            raise ValueError("ontology key columns must be unique after trimming")
        return normalized

    @model_validator(mode="after")
    def require_standard_vocabulary(self):
        expected = {
            "phase_ontology": "contrast_phase",
            "slot_ontology": "clinical_slot",
        }.get(self.output.target_column)
        if expected is not None and self.output.vocabulary != expected:
            raise ValueError(
                f"ontology target {self.output.target_column!r} requires "
                f"vocabulary {expected!r}"
            )
        return self


class AnnotationConfig(StrictModel):
    ontologies: list[OntologyConfig] = Field(default_factory=list)
    rule_packs: list[str] = Field(default_factory=list)
    contextual_strategies: list[
        Literal["art_port", "mask_multiart", "generic_dynamic_volume_order"]
    ] = Field(default_factory=list)

    @field_validator("ontologies")
    @classmethod
    def unique_ontology_ids(cls, value: list[OntologyConfig]):
        ids = [ontology.id for ontology in value]
        if len(set(ids)) != len(ids):
            raise ValueError("ontology ids must be unique")
        return value

    @field_validator("rule_packs")
    @classmethod
    def validate_builtin_rule_packs(cls, value: list[str]) -> list[str]:
        value = [reference.strip() for reference in value]
        if any(not reference for reference in value):
            raise ValueError("annotation rule-pack references must not be empty")
        allowed = {"builtin:liver_ct", "builtin:liver_mri"}
        invalid = {
            reference
            for reference in value
            if reference.startswith("builtin:") and reference not in allowed
        }
        if invalid:
            raise ValueError(f"Unknown built-in rule packs: {sorted(invalid)}")
        return list(dict.fromkeys(value))

    @field_validator("contextual_strategies")
    @classmethod
    def unique_contextual_strategies(cls, value):
        return list(dict.fromkeys(value))


class PhaseResolutionConfig(StrictModel):
    precedence: list[str] = Field(
        default_factory=lambda: [
            "phase_ontology",
            "phase_rules_explicit",
            "phase_rules_inferred",
            "phase_image",
        ]
    )
    disagreement: Literal["flag", "error", "ignore"] = "flag"

    @field_validator("precedence")
    @classmethod
    def validate_precedence(cls, value: list[str]) -> list[str]:
        value = [column.strip() for column in value]
        if not value:
            raise ValueError("phase resolution precedence must not be empty")
        if any(not column for column in value):
            raise ValueError("phase resolution precedence contains an empty column")
        if len(set(value)) != len(value):
            raise ValueError("phase resolution precedence must not contain duplicates")
        return value


class PhasePredictionConfig(StrictModel):
    enabled: bool = False
    backend: Literal["totalsegmentator"] = "totalsegmentator"
    modalities: list[Literal["CT"]] = Field(default_factory=lambda: ["CT"])
    scope: Literal["unresolved", "all_eligible", "selected_and_unresolved"] = (
        "unresolved"
    )
    minimum_confidence: float = Field(default=0.6, ge=0, le=1)
    resolution: PhaseResolutionConfig = Field(default_factory=PhaseResolutionConfig)

    @field_validator("modalities")
    @classmethod
    def non_empty_modalities(cls, value):
        if not value:
            raise ValueError("phase_prediction.modalities must not be empty")
        return list(dict.fromkeys(value))


class SelectionConfig(StrictModel):
    required_slots: dict[Literal["CT", "MR"], list[str]] = Field(default_factory=dict)
    precedence: list[str] = Field(
        default_factory=lambda: [
            "slot_ontology",
            "slot_rules_explicit",
            "slot_rules_inferred",
            "slot_image",
        ]
    )
    disagreement: Literal["flag", "error", "ignore"] = "flag"

    @field_validator("required_slots")
    @classmethod
    def validate_required_slots(cls, value):
        normalized = {}
        for modality, slots in value.items():
            cleaned = [slot.strip() for slot in slots]
            if any(not slot for slot in cleaned):
                raise ValueError("selection required slots must not be empty")
            if len(set(cleaned)) != len(cleaned):
                raise ValueError("selection required slots must not contain duplicates")
            invalid = set(cleaned) - CLINICAL_SLOTS
            if invalid:
                raise ValueError(f"Unknown clinical slots: {sorted(invalid)}")
            wrong_modality = [
                slot for slot in cleaned if not slot.startswith(f"{modality}_")
            ]
            if wrong_modality:
                raise ValueError(
                    f"Clinical slots do not match {modality}: {wrong_modality}"
                )
            normalized[modality] = cleaned
        return normalized

    @field_validator("precedence")
    @classmethod
    def validate_precedence(cls, value: list[str]) -> list[str]:
        value = [column.strip() for column in value]
        if not value:
            raise ValueError("selection precedence must not be empty")
        if any(not column for column in value):
            raise ValueError("selection precedence contains an empty column")
        if len(set(value)) != len(value):
            raise ValueError("selection precedence must not contain duplicates")
        return value


class ConversionConfig(StrictModel):
    enabled: bool = False
    workers: int | None = Field(default=None, ge=1)


class SegmentationTaskConfig(StrictModel):
    id: str
    backend: Literal["totalsegmentator"] = "totalsegmentator"
    modality: Literal["CT", "MR"]
    task: str
    output: str
    fetch_output: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "task", "output")
    @classmethod
    def non_empty_names(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("segmentation task id, task, and output must not be empty")
        return value.strip()

    @field_validator("fetch_output")
    @classmethod
    def non_empty_fetch_output(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("segmentation fetch_output must not be empty")
        return value.strip() if value is not None else None

    @field_validator("parameters")
    @classmethod
    def validate_backend_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        normalized = {str(name).strip(): parameter for name, parameter in value.items()}
        if any(not name for name in normalized):
            raise ValueError("segmentation parameter names must not be empty")
        reserved = {"input", "output", "task", "input_path", "output_dir"}
        invalid = reserved & set(normalized)
        if invalid:
            raise ValueError(
                "segmentation parameters cannot override routing arguments: "
                f"{sorted(invalid)}"
            )
        return normalized


class SegmentationConfig(StrictModel):
    enabled: bool = False
    tasks: list[SegmentationTaskConfig] = Field(default_factory=list)
    postprocess: dict[Literal["CT", "MR", "default"], dict[str, Any]] = Field(
        default_factory=dict
    )

    @field_validator("tasks")
    @classmethod
    def unique_task_ids(cls, value: list[SegmentationTaskConfig]):
        ids = [task.id for task in value]
        if len(set(ids)) != len(ids):
            raise ValueError("segmentation task ids must be unique")
        routed_outputs = [(task.modality, task.output) for task in value]
        if len(set(routed_outputs)) != len(routed_outputs):
            raise ValueError(
                "segmentation outputs must be unique within each modality route"
            )
        return value

    @model_validator(mode="after")
    def require_tasks_when_enabled(self):
        if self.enabled and not self.tasks:
            raise ValueError("segmentation.tasks must not be empty when enabled")
        return self


class RegistrationConfig(StrictModel):
    enabled: bool = False
    transform: Literal["rigid", "rigid_affine", "deformable"] = "rigid_affine"
    pairs: Path | None = None

    @model_validator(mode="after")
    def require_pairs_when_enabled(self):
        if self.enabled and self.pairs is None:
            raise ValueError("registration.pairs is required when enabled")
        return self


class RadiomicsConfig(StrictModel):
    enabled: bool = False
    settings: Path | None = None
    slots: list[str] = Field(default_factory=list)
    masks: list[str] = Field(default_factory=list)

    @field_validator("slots", "masks")
    @classmethod
    def unique_non_empty_names(cls, value: list[str]) -> list[str]:
        normalized = [name.strip() for name in value]
        if any(not name for name in normalized):
            raise ValueError("radiomics slot and mask names must not be empty")
        return list(dict.fromkeys(normalized))


class ExecutionConfig(StrictModel):
    workers: int = Field(default=4, ge=1)
    resume: bool = True
    checkpoint_every_rows: int = Field(default=100, ge=1)
    checkpoint_every_seconds: int = Field(default=300, ge=1)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class ImperandiConfig(StrictModel):
    version: Literal[2]
    project: ProjectConfig
    input: InputConfig
    output: OutputConfig
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    annotations: AnnotationConfig = Field(default_factory=AnnotationConfig)
    phase_prediction: PhasePredictionConfig = Field(
        default_factory=PhasePredictionConfig
    )
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    conversion: ConversionConfig = Field(default_factory=ConversionConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    registration: RegistrationConfig = Field(default_factory=RegistrationConfig)
    radiomics: RadiomicsConfig = Field(default_factory=RadiomicsConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
