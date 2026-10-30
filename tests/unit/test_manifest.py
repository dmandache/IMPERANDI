import sys
from pathlib import Path

import pytest
import yaml

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
    assert manifest["cleaning"]["steps"]
    step_types = [step["type"] for step in manifest["cleaning"]["steps"]]
    assert (
        step_types.index("phase_curation") == step_types.index("modality_curation") + 1
    )
    assert manifest["phase_curation"]["strategies"]
    assert manifest["segmentation"]["backend"] == "totalsegmentator"
    assert manifest["segmentation"]["crop"] is False
    assert manifest["segmentation"]["crop_margin_mm"] == 50.0
    radiomics = manifest["radiomics"]["modalities"]
    assert radiomics["CT"]["pyradiomics"]["setting"]["resegmentRange"] == [-150, 250]
    assert "resegmentRange" not in radiomics["MR"]["pyradiomics"]["setting"]


@pytest.mark.parametrize(
    "manifest_name",
    [
        "generic",
        "blueprint_manifest_example",
    ],
)
def test_registration_manifests_use_ordered_reference_criteria(manifest_name):
    from imperandi.process.registration.config import RegistrationConfig

    manifest = load_manifest(
        manifest_name, base_path=Path(__file__).resolve().parents[2]
    )
    config = RegistrationConfig.from_mapping(manifest["registration"])
    assert config.grouping_columns == ["patient_key", "study_id", "Modality"]
    assert config.output_space == "moving"
    assert config.reference_selection_priority[0] == {"Modality": ["CT", "MR"]}
    assert config.reference_selection_priority[2] == {
        "mri_sequence": ["T1", "T2", "DWI"]
    }
    assert config.reference_selection_priority[-1] == {
        "registration_organ_volume_mm3": "max"
    }


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


@pytest.mark.parametrize(
    "content,line,column,problem",
    [
        ("cleaning:\n\tsteps: []\n", 2, 1, "cannot start any token"),
        ("cleaning: [\n", 2, 1, "expected the node content"),
    ],
)
def test_manifest_syntax_error_reports_path_location_and_problem(
    tmp_path, content, line, column, problem
):
    manifest_path = tmp_path / "invalid.yaml"
    manifest_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError) as error:
        load_manifest(str(manifest_path), base_path=tmp_path)

    message = str(error.value)
    assert "Invalid YAML in manifest" in message
    assert str(manifest_path) in message
    assert f"line {line}, column {column}" in message
    assert problem in message
    assert isinstance(error.value.__cause__, yaml.YAMLError)


@pytest.mark.parametrize("content", ["", "- item\n", "true\n"])
def test_manifest_requires_a_mapping_even_when_yaml_syntax_is_valid(tmp_path, content):
    manifest_path = tmp_path / "invalid.yaml"
    manifest_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="Manifest must contain a YAML mapping"):
        load_manifest(str(manifest_path), base_path=tmp_path)


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
