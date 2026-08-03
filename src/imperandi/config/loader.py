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
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
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
    return config


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
        config.registration.pairs,
        config.radiomics.settings,
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
