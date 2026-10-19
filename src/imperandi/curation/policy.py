"""Immutable curation configuration and explicit Python rule registration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from .rule_engine import Rule

BUILTIN_RULES = {
    "CT": ("metadata", "pre_post_native"),
    "MR": (
        "metadata",
        "acquisition_phase",
        "art_port_late",
        "art_port",
        "mask_multiart",
        "native_context",
        "generic_dynamic",
        "pre_post_native",
    ),
}


def _names(value, field, *, allow_empty=False):
    if (
        not isinstance(value, (list, tuple))
        or (not value and not allow_empty)
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{field} must be a list of unique non-empty names.")
    return list(value)


def validate_rule_options(value) -> dict:
    """Validate serializable built-in rule controls used by Python and manifests."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("phase_curation.rule_policy must be a mapping.")
    normalized = {}
    for modality, options in value.items():
        if modality not in BUILTIN_RULES:
            raise ValueError("rule_policy modalities must be CT or MR.")
        if not isinstance(options, Mapping) or set(options) - {
            "disabled",
            "precedence",
            "context_columns",
        }:
            raise ValueError(
                "rule_policy accepts disabled, precedence, and context_columns."
            )
        disabled = _names(options.get("disabled", []), "disabled", allow_empty=True)
        if "metadata" in disabled or set(disabled) - set(BUILTIN_RULES[modality]):
            raise ValueError("Only named contextual built-in rules can be disabled.")
        precedence = _names(
            options.get("precedence", BUILTIN_RULES[modality]), "precedence"
        )
        if set(precedence) != set(BUILTIN_RULES[modality]):
            raise ValueError(
                f"precedence must list every {modality} built-in rule exactly once."
            )
        context_columns = _names(
            options.get("context_columns", []), "context_columns", allow_empty=True
        )
        normalized[modality] = {
            "disabled": disabled,
            "precedence": precedence,
            "context_columns": context_columns,
        }
    return normalized


@dataclass(frozen=True)
class RuleRegistration:
    rule: Rule
    context: str
    precedes: str


@dataclass(frozen=True)
class CurationPolicy:
    """Value-style policy: each ``with_*`` method returns a new policy.

    Manifest data is stored as JSON so callers cannot mutate nested configuration.
    Registered Python rules should be immutable and declare all settings in their
    version string. They are deliberately never imported from YAML.
    """

    _config_json: str
    registrations: tuple[RuleRegistration, ...] = ()

    @classmethod
    def from_manifest(cls, manifest: Mapping | None = None) -> CurationPolicy:
        """Read a full manifest mapping; omission uses the built-in defaults."""
        if manifest is not None and not isinstance(manifest, Mapping):
            raise ValueError("manifest must be a mapping.")
        return cls.from_phase_config(
            None if manifest is None else manifest.get("phase_curation")
        )

    @classmethod
    def from_phase_config(cls, config: Mapping | None = None) -> CurationPolicy:
        from .phase_resolution import validate_phase_curation

        normalized = validate_phase_curation(config)
        normalized["unresolved_labels"] = sorted(normalized["unresolved_labels"])
        return cls(json.dumps(normalized, sort_keys=True))

    @property
    def phase_curation(self) -> dict:
        return json.loads(self._config_json)

    def with_text_columns(
        self, *, modality: str, columns: Sequence[str]
    ) -> CurationPolicy:
        config = self.phase_curation
        modality = "MR" if modality == "MRI" else modality
        config["text_columns"][modality] = _names(columns, "text_columns")
        return replace(self.from_phase_config(config), registrations=self.registrations)

    def with_rule_options(self, *, modality: str, **options) -> CurationPolicy:
        """Set disabled rules, built-in precedence, or additional context columns."""
        config = self.phase_curation
        controls = config.setdefault("rule_policy", {}).setdefault(modality, {})
        controls.update(options)
        return replace(self.from_phase_config(config), registrations=self.registrations)

    def with_rule(
        self,
        rule: Rule,
        *,
        context: str = "compatible_protocol",
        precedes: str = "generic_dynamic",
    ) -> CurationPolicy:
        if context not in {"row", "exam", "compatible_protocol"}:
            raise ValueError("Rule context must be row, exam, or compatible_protocol.")
        if rule.modality not in BUILTIN_RULES:
            raise ValueError("Rule modality must be CT or MR.")
        if not isinstance(rule.name, str) or not rule.name.strip():
            raise ValueError("Rule name must be non-empty.")
        if not isinstance(rule.version, str) or not rule.version.strip():
            raise ValueError("Rule version must be non-empty.")
        _names(rule.input_columns, "rule.input_columns", allow_empty=True)
        _names(rule.requires, "rule.requires", allow_empty=True)
        names = self.precedence(rule.modality)
        if rule.name in names:
            raise ValueError(f"Duplicate curation rule name: {rule.name}")
        if precedes not in names or precedes == "metadata":
            raise ValueError("precedes must name an existing contextual rule.")
        if set(rule.requires) - set(names):
            raise ValueError("Rule dependencies must name existing rules.")
        if not callable(rule.evaluate):
            raise ValueError("Rule must provide evaluate(context).")
        return replace(
            self,
            registrations=(
                *self.registrations,
                RuleRegistration(rule, context, precedes),
            ),
        )

    def options(self, modality: str) -> dict:
        return self.phase_curation.get("rule_policy", {}).get(modality, {})

    def precedence(self, modality: str) -> tuple[str, ...]:
        names = list(self.options(modality).get("precedence", BUILTIN_RULES[modality]))
        for registration in self.registrations:
            if registration.rule.modality == modality:
                names.insert(names.index(registration.precedes), registration.rule.name)
        return tuple(names)

    @property
    def input_columns(self) -> frozenset[str]:
        from .phase_resolution import phase_curation_input_columns

        columns = phase_curation_input_columns(self.phase_curation)
        for registration in self.registrations:
            columns.update(registration.rule.input_columns)
        return frozenset(columns)

    @property
    def fingerprint(self) -> str:
        """Stable policy identity for callers that cache Python curation runs."""
        payload = {
            "engine_version": 1,
            "config": self.phase_curation,
            "rules": [
                {
                    "name": r.rule.name,
                    "class": f"{type(r.rule).__module__}.{type(r.rule).__qualname__}",
                    "version": r.rule.version,
                    "modality": r.rule.modality,
                    "inputs": list(r.rule.input_columns),
                    "requires": list(r.rule.requires),
                    "context": r.context,
                    "precedes": r.precedes,
                }
                for r in self.registrations
            ],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
