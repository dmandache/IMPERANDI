import sys
from pathlib import Path

import pytest

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imperandi.utils.manifest import load_manifest, resolve_hook


def test_load_generic_manifest_and_hook_resolution():
    base_path = Path(__file__).resolve().parents[1] / "src" / "imperandi"
    manifest = load_manifest("generic", base_path=base_path)

    assert manifest["dataset_name"] == "generic"
    assert "id_standardization" in manifest

    hook = resolve_hook(manifest["id_standardization"])
    assert hook is not None
    assert hook("patient_0012_030") == "12-30"

    assert [
        strategy["type"] for strategy in manifest["phase_curation"]["strategies"]
    ] == ["rules", "totalsegmentator"]


def test_load_manifest_accepts_only_yaml(tmp_path):
    base_path = Path(__file__).resolve().parents[1] / "src" / "imperandi"
    yaml_path = tmp_path / "site.yaml"
    yaml_path.write_text("dataset_name: site\n", encoding="utf-8")

    assert load_manifest(str(yaml_path), base_path=base_path) == {
        "dataset_name": "site"
    }

    json_path = tmp_path / "site.json"
    json_path.write_text('{"dataset_name": "site"}', encoding="utf-8")
    with pytest.raises(ValueError, match="must use YAML"):
        load_manifest(str(json_path), base_path=base_path)
