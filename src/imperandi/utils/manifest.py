import importlib
from importlib.resources import files
from pathlib import Path
from typing import Optional

import yaml

YAML_SUFFIXES = (".yaml", ".yml")
BUILTIN_CONFIG_PACKAGE = "imperandi.builtin_datasets_config"


def _named_manifest_candidates(manifest_arg: str) -> list[Path]:
    filename = (
        manifest_arg
        if Path(manifest_arg).suffix.lower() in YAML_SUFFIXES
        else f"{manifest_arg}.yaml"
    )
    builtin_path = Path(
        str(files(BUILTIN_CONFIG_PACKAGE).joinpath("manifests", filename))
    )
    return [builtin_path]


def load_manifest(manifest_arg: Optional[str], *, base_path: Path) -> dict:
    """Load a named or explicitly located YAML dataset manifest."""
    if not manifest_arg:
        return {}

    manifest_path = Path(manifest_arg)
    if not manifest_path.suffix:
        candidates = _named_manifest_candidates(manifest_arg)
        manifest_path = next((path for path in candidates if path.is_file()), candidates[-1])
    elif manifest_path.suffix.lower() not in YAML_SUFFIXES:
        raise ValueError(
            "Manifest files must use YAML "
            f"(accepted: {', '.join(YAML_SUFFIXES)}): {manifest_path}"
        )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest *{manifest_arg}* not found at {manifest_path}"
        )

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must contain a YAML mapping: {manifest_path}")
    return manifest


def _import_hook_module(module_name: str):
    """Import an absolute hook module or one relative to ``imperandi``."""
    if module_name.startswith("imperandi."):
        return importlib.import_module(module_name)
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name and not module_name.startswith(f"{exc.name}."):
            raise
        return importlib.import_module(f"imperandi.{module_name}")


def resolve_hook(hook_config: dict):
    """Resolve a manifest hook to a callable."""
    module_name = hook_config.get("hook_module")
    function_name = hook_config.get("function")
    if not module_name or not function_name:
        return None
    module = _import_hook_module(module_name)
    return getattr(module, function_name)


def resolve_function_path(function_path: str):
    """Resolve a ``module:function`` hook reference."""
    if ":" not in function_path:
        raise ValueError(
            f"Invalid function path {function_path!r}. Expected 'module:function'."
        )

    module_name, function_name = function_path.split(":", 1)
    module = _import_hook_module(module_name)
    return getattr(module, function_name)
