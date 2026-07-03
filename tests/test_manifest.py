import sys
from pathlib import Path

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imperandi.utils.manifest import load_manifest, resolve_function_path


def test_load_generic_manifest_and_hook_resolution():
    base_path = Path(__file__).resolve().parents[1] / "src" / "imperandi"
    manifest = load_manifest("generic", base_path=base_path)

    assert manifest["dataset_name"] == "generic"
    assert manifest["cleaning"]["version"] == 1
    assert isinstance(manifest["cleaning"]["steps"], list)

    clean_hook = resolve_function_path(
        "datasets_config.hooks.generic:standardize_patient_key"
    )
    assert clean_hook("patient_0012_030") == "12-30"
