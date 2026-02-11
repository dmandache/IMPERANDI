"""Manifest-loading helpers for dataset and processing configuration files.

The definitions in this module are part of the Imperandi codebase and are
intended to be reused by higher-level workflows and CLI entry points.
"""

import importlib
import json
from pathlib import Path
from typing import Optional

DEFAULT_MANIFEST_NAME = "generic"


def load_manifest(manifest_arg: Optional[str], *, base_path: Path) -> dict:
    """Load manifest.

    Args:
        manifest_arg (Optional[str]): Input value for manifest arg.
        base_path (Path): Filesystem path consumed by this operation.

    Returns:
        dict: Dictionary of computed fields.

    Raises:
        FileNotFoundError: If an expected input file cannot be found.
    """
    manifest_ref = manifest_arg or DEFAULT_MANIFEST_NAME

    manifest_path = Path(manifest_ref)
    if not manifest_path.suffix:
        manifest_path = (
            base_path / "datasets_config" / "manifests" / f"{manifest_ref}.json"
        )
    elif not manifest_path.is_file():
        manifest_path = base_path / "datasets_config" / "manifests" / manifest_ref

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest *{manifest_ref}* not found at {manifest_path}"
        )

    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_hook(hook_config: dict):
    """Resolve hook.

    Args:
        hook_config (dict): Input value for hook config.

    Returns:
        Any: Resolved hook.
    """
    module_name = hook_config.get("hook_module")
    function_name = hook_config.get("function")
    if not module_name or not function_name:
        return None
    module = importlib.import_module(f"imperandi.{module_name}")
    return getattr(module, function_name)
