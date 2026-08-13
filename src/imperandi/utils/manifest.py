import importlib
from importlib.resources import files
from pathlib import Path
from typing import Optional

import yaml

YAML_SUFFIXES = (".yaml", ".yml")
BUILTIN_CONFIG_PACKAGE = "imperandi.builtin_datasets_config"
PROJECT_CONFIG_DIR = "dataset_configs"


def _project_config_roots(base_path: Path) -> list[Path]:
    """Return existing project-local configuration roots in priority order."""
    candidates = [
        Path.cwd() / PROJECT_CONFIG_DIR,
        Path(base_path) / PROJECT_CONFIG_DIR,
        Path(__file__).resolve().parents[3] / PROJECT_CONFIG_DIR,
    ]
    roots: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir() and resolved not in roots:
            roots.append(resolved)
    return roots


def _named_manifest_candidates(manifest_arg: str, base_path: Path) -> list[Path]:
    filename = (
        manifest_arg
        if Path(manifest_arg).suffix.lower() in YAML_SUFFIXES
        else f"{manifest_arg}.yaml"
    )
    project_paths = [
        root / "manifests" / filename for root in _project_config_roots(base_path)
    ]
    builtin_path = Path(
        str(files(BUILTIN_CONFIG_PACKAGE).joinpath("manifests", filename))
    )
    return [*project_paths, builtin_path]


def load_manifest(manifest_arg: Optional[str], *, base_path: Path) -> dict:
    """Load a named or explicitly located YAML dataset manifest."""
    base_path = Path(base_path)

    if not manifest_arg:
        return {}

    manifest_path = Path(manifest_arg)
    if not manifest_path.suffix:
        candidates = _named_manifest_candidates(manifest_arg, base_path)
        manifest_path = next((path for path in candidates if path.is_file()), candidates[-1])
    elif manifest_path.suffix.lower() not in YAML_SUFFIXES:
        raise ValueError(
            "Manifest files must use YAML "
            f"(accepted: {', '.join(YAML_SUFFIXES)}): {manifest_path}"
        )
    elif not manifest_path.is_file():
        candidates = _named_manifest_candidates(manifest_arg, base_path)
        manifest_path = next((path for path in candidates if path.is_file()), candidates[-1])

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
    """Import an absolute hook module, retaining old manifest compatibility."""
    if module_name.startswith("imperandi.") or module_name.startswith(
        f"{PROJECT_CONFIG_DIR}."
    ):
        return importlib.import_module(module_name)
    if module_name.startswith("datasets_config."):
        suffix = module_name.removeprefix("datasets_config.")
        try:
            return importlib.import_module(f"{PROJECT_CONFIG_DIR}.{suffix}")
        except ModuleNotFoundError:
            return importlib.import_module(f"{BUILTIN_CONFIG_PACKAGE}.{suffix}")
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
    """Resolve ``module:function`` paths relative to the ``imperandi`` package."""
    if ":" not in function_path:
        raise ValueError(
            f"Invalid function path {function_path!r}. Expected 'module:function'."
        )

    module_name, function_name = function_path.split(":", 1)
    module = _import_hook_module(module_name)
    return getattr(module, function_name)
