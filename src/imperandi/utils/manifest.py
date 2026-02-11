import importlib
import json
from pathlib import Path
from typing import Optional

DEFAULT_MANIFEST_NAME = "generic"


def load_manifest(manifest_arg: Optional[str], *, base_path: Path) -> dict:
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
    module_name = hook_config.get("hook_module")
    function_name = hook_config.get("function")
    if not module_name or not function_name:
        return None
    module = importlib.import_module(f"imperandi.{module_name}")
    return getattr(module, function_name)
