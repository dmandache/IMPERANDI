"""Load, resolve, validate, and hash project configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .models import ImperandiConfig

PROFILE_DIR = Path(__file__).with_name("profiles")


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"Configuration root must be a mapping: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge mappings recursively; scalars and lists are replaced explicitly."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_path(value: Path | None, base_dir: Path) -> Path | None:
    if value is None:
        return value
    value = value.expanduser()
    if value.is_absolute():
        return value
    return (base_dir / value).resolve()


def _resolve_relative_paths(config: ImperandiConfig, base_dir: Path) -> None:
    config.input.sources = [
        (
            str(Path(source).expanduser())
            if Path(source).expanduser().is_absolute()
            else str((base_dir / Path(source).expanduser()).resolve())
        )
        for source in config.input.sources
    ]
    config.output.root = _resolve_path(config.output.root, base_dir)  # type: ignore[assignment]
    canonical = config.identity.canonical
    canonical.crosswalk = _resolve_path(canonical.crosswalk, base_dir)
    for ontology in config.annotations.ontologies:
        ontology.source = _resolve_path(ontology.source, base_dir)  # type: ignore[assignment]
    config.annotations.rule_packs = [
        (
            reference
            if reference.startswith("builtin:")
            else str(
                Path(reference).expanduser()
                if Path(reference).expanduser().is_absolute()
                else (base_dir / Path(reference).expanduser()).resolve()
            )
        )
        for reference in config.annotations.rule_packs
    ]
    config.registration.pairs = _resolve_path(config.registration.pairs, base_dir)
    config.radiomics.settings = _resolve_path(config.radiomics.settings, base_dir)


def load_config(path: str | Path) -> ImperandiConfig:
    """Load a project YAML, apply its built-in profile, and validate it."""
    path = Path(path).expanduser().resolve()
    project_data = _read_yaml(path)
    if "version" not in project_data:
        raise ValueError("Project configuration requires explicit version: 2")
    project_section = project_data.get("project")
    profile_name = (
        project_section.get("profile") if isinstance(project_section, dict) else None
    )
    if profile_name:
        profile_path = PROFILE_DIR / f"{profile_name}.yaml"
        if not profile_path.exists():
            raise FileNotFoundError(
                f"Unknown profile {profile_name!r}; expected {profile_path}"
            )
        project_data = _deep_merge(_read_yaml(profile_path), project_data)

    config = ImperandiConfig.model_validate(project_data)
    _resolve_relative_paths(config, path.parent)
    validate_config_resources(config)
    return config


def _require_table(path: Path, label: str):
    from imperandi.io.tables import read_table

    if path.suffix.lower() not in {".csv", ".parquet"}:
        raise ValueError(f"{label} must be a CSV or Parquet table: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        return read_table(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {label} {path}: {exc}") from exc


def validate_config_resources(config: ImperandiConfig) -> None:
    """Validate referenced policy resources before a pipeline starts."""
    canonical = config.identity.canonical
    if canonical.crosswalk is not None:
        crosswalk = _require_table(canonical.crosswalk, "Identity crosswalk")
        required = {*canonical.crosswalk_keys, canonical.crosswalk_value}
        missing = required - set(crosswalk.columns)
        if missing:
            raise ValueError(
                f"Identity crosswalk is missing columns: {sorted(missing)}"
            )
        from imperandi.identity import validate_identity_crosswalk

        validate_identity_crosswalk(config.identity)

    for ontology in config.annotations.ontologies:
        table = _require_table(ontology.source, f"Ontology {ontology.id!r}")
        required = {*ontology.keys, ontology.output.value_column}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(
                f"Ontology {ontology.id!r} is missing columns: {sorted(missing)}"
            )
        vocabulary_name = ontology.output.vocabulary
        if vocabulary_name is not None:
            from imperandi.annotations.ontology import VOCABULARIES

            values = set(table[ontology.output.value_column].dropna().astype(str))
            invalid = values - VOCABULARIES[vocabulary_name]
            if invalid:
                raise ValueError(
                    f"Ontology {ontology.id!r} contains values outside "
                    f"{vocabulary_name}: {sorted(invalid)}"
                )

    from imperandi.annotations.rules import load_rule_pack

    for reference in config.annotations.rule_packs:
        if reference.startswith("builtin:"):
            continue
        path = Path(reference)
        if not path.is_file():
            raise FileNotFoundError(f"Annotation rule pack does not exist: {path}")
        load_rule_pack(path)

    if config.registration.enabled:
        assert config.registration.pairs is not None
        pairs = _require_table(config.registration.pairs, "Registration pair table")
        if "moving_volume_id" not in pairs.columns:
            raise ValueError(
                "Registration pair table is missing column: moving_volume_id"
            )
        if not {"fixed_volume_id", "fixed_nifti_path"} & set(pairs.columns):
            raise ValueError(
                "Registration pair table requires fixed_volume_id or "
                "fixed_nifti_path"
            )

    settings = config.radiomics.settings
    if config.radiomics.enabled and settings is not None:
        if settings.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"Radiomics settings must be YAML: {settings}")
        if not settings.is_file():
            raise FileNotFoundError(f"Radiomics settings do not exist: {settings}")


def resolved_config(config: ImperandiConfig) -> dict[str, Any]:
    return config.model_dump(mode="json", exclude_none=True)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_dependency_hashes(config: ImperandiConfig) -> dict[str, str]:
    """Hash external policy files that affect deterministic output."""
    paths = [
        config.identity.canonical.crosswalk,
        *(ontology.source for ontology in config.annotations.ontologies),
        *(
            Path(reference)
            for reference in config.annotations.rule_packs
            if not reference.startswith("builtin:")
        ),
        config.registration.pairs if config.registration.enabled else None,
        config.radiomics.settings if config.radiomics.enabled else None,
    ]
    dependencies = {}
    for path in paths:
        if path is None:
            continue
        resolved = path.expanduser().resolve()
        dependencies[str(resolved)] = (
            _file_digest(resolved) if resolved.is_file() else "missing"
        )
        schema = resolved.with_suffix(resolved.suffix + ".schema.json")
        if schema.is_file():
            dependencies[str(schema)] = _file_digest(schema)
    return dict(sorted(dependencies.items()))


def config_hash(config: ImperandiConfig) -> str:
    payload = json.dumps(
        {
            "config": resolved_config(config),
            "dependencies": config_dependency_hashes(config),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
