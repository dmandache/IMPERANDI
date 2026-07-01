import importlib
import json
from pathlib import Path
from typing import Optional


def load_manifest(manifest_arg: Optional[str], *, base_path: Path) -> dict:
    """Load a named or explicitly located dataset manifest.

    Args:
        manifest_arg: Built-in manifest name, JSON path, or ``None``.
        base_path: Package directory containing ``datasets_config``.

    Returns:
        The decoded manifest, or an empty dictionary when no manifest is given.

    Raises:
        FileNotFoundError: If the resolved manifest does not exist.
    """

    base_path = Path(base_path)

    if not manifest_arg:
        return {}

    manifest_path = Path(manifest_arg)
    if not manifest_path.suffix:
        manifest_path = (
            base_path / "datasets_config" / "manifests" / f"{manifest_arg}.json"
        )
    elif not manifest_path.is_file():
        manifest_path = base_path / "datasets_config" / "manifests" / manifest_arg

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest *{manifest_arg}* not found at {manifest_path}"
        )

    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_hook(hook_config: dict):
    """Resolve a manifest hook to a callable within the ``imperandi`` package.

    Returns ``None`` when either ``hook_module`` or ``function`` is absent.
    Import and attribute errors are intentionally allowed to surface so invalid
    manifests fail clearly.
    """
    module_name = hook_config.get("hook_module")
    function_name = hook_config.get("function")
    if not module_name or not function_name:
        return None
    module = importlib.import_module(f"imperandi.{module_name}")
    return getattr(module, function_name)
