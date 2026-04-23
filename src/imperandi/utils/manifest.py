import importlib
import json
from pathlib import Path
from typing import Optional


def load_manifest(manifest_arg: Optional[str], *, base_path: Path) -> dict:

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
    module_name = hook_config.get("hook_module")
    function_name = hook_config.get("function")
    if not module_name or not function_name:
        return None
    module = importlib.import_module(f"imperandi.{module_name}")
    return getattr(module, function_name)
