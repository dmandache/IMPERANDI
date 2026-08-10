import importlib
from pathlib import Path
from typing import Optional

import yaml


YAML_SUFFIXES = (".yaml", ".yml")


def load_manifest(manifest_arg: Optional[str], *, base_path: Path) -> dict:
    """Load a named or explicitly located YAML dataset manifest."""
    base_path = Path(base_path)

    if not manifest_arg:
        return {}

    manifest_path = Path(manifest_arg)
    if not manifest_path.suffix:
        manifest_path = (
            base_path / "datasets_config" / "manifests" / f"{manifest_arg}.yaml"
        )
    elif manifest_path.suffix.lower() not in YAML_SUFFIXES:
        raise ValueError(
            "Manifest files must use YAML "
            f"(accepted: {', '.join(YAML_SUFFIXES)}): {manifest_path}"
        )
    elif not manifest_path.is_file():
        manifest_path = base_path / "datasets_config" / "manifests" / manifest_arg

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest *{manifest_arg}* not found at {manifest_path}"
        )

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must contain a YAML mapping: {manifest_path}")
    return manifest


def resolve_hook(hook_config: dict):
    """Resolve a manifest hook to a callable within the ``imperandi`` package."""
    module_name = hook_config.get("hook_module")
    function_name = hook_config.get("function")
    if not module_name or not function_name:
        return None
    module = importlib.import_module(f"imperandi.{module_name}")
    return getattr(module, function_name)


def resolve_function_path(function_path: str):
    """Resolve ``module:function`` paths relative to the ``imperandi`` package."""
    if ":" not in function_path:
        raise ValueError(
            f"Invalid function path {function_path!r}. Expected 'module:function'."
        )

    module_name, function_name = function_path.split(":", 1)
    if not module_name.startswith("imperandi."):
        module_name = f"imperandi.{module_name}"

    module = importlib.import_module(module_name)
    return getattr(module, function_name)
