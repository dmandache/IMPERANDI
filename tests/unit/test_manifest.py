import sys
from pathlib import Path

import pytest

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from imperandi.utils.manifest import load_manifest, resolve_hook


@pytest.mark.parametrize(
    ("manifest_name", "dataset_name"),
    [
        ("generic", "generic"),
        ("blueprint_manifest_example", "blueprint_example"),
    ],
)
def test_bundled_manifests_are_loadable(manifest_name, dataset_name):
    base_path = Path(__file__).resolve().parents[2] / "src" / "imperandi"

    manifest = load_manifest(manifest_name, base_path=base_path)

    assert manifest["dataset_name"] == dataset_name
    assert manifest["cleaning"]["version"] == 1
    assert manifest["phase_curation"]["strategies"]
    assert manifest["segmentation"]["backend"] == "totalsegmentator"


def test_load_generic_manifest_and_hook_resolution():
    base_path = Path(__file__).resolve().parents[2] / "src" / "imperandi"
    manifest = load_manifest("generic", base_path=base_path)

    assert manifest["dataset_name"] == "generic"
    assert "id_standardization" not in manifest

    hook = resolve_hook(
        {
            "hook_module": "imperandi.builtin_datasets_config.hooks.generic",
            "function": "standardize_patient_key",
        }
    )
    assert hook is not None
    assert hook("patient_0012_030") == "12-30"

    assert [
        strategy["type"] for strategy in manifest["phase_curation"]["strategies"]
    ] == ["rules", "totalsegmentator"]


def test_load_manifest_accepts_only_yaml(tmp_path):
    base_path = Path(__file__).resolve().parents[2] / "src" / "imperandi"
    yaml_path = tmp_path / "site.yaml"
    yaml_path.write_text("dataset_name: site\n", encoding="utf-8")

    assert load_manifest(str(yaml_path), base_path=base_path) == {
        "dataset_name": "site"
    }

    json_path = tmp_path / "site.json"
    json_path.write_text('{"dataset_name": "site"}', encoding="utf-8")
    with pytest.raises(ValueError, match="must use YAML"):
        load_manifest(str(json_path), base_path=base_path)


def test_external_manifest_requires_an_explicit_path(tmp_path, monkeypatch):
    base_path = Path(__file__).resolve().parents[2] / "src" / "imperandi"
    external_manifest = tmp_path / "dataset_configs" / "manifests" / "operandi.yaml"
    external_manifest.parent.mkdir(parents=True)
    external_manifest.write_text("dataset_name: external_operandi\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match=r"Manifest \*operandi\* not found"):
        load_manifest("operandi", base_path=base_path)

    assert load_manifest(str(external_manifest), base_path=base_path) == {
        "dataset_name": "external_operandi"
    }
