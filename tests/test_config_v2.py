from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

from imperandi.config import config_hash, load_config, resolved_config
from imperandi.config.models import TableFormat
from imperandi.pipeline.stages.imaging import _segmentation_backend_config


def _write_config(tmp_path: Path, extra: dict | None = None) -> Path:
    data = {
        "version": 2,
        "project": {"name": "test", "profile": "liver_ct_mri"},
        "input": {"sources": ["input.csv"]},
        "output": {"root": "results", "table_format": "csv"},
        "segmentation": {"enabled": False},
    }
    if extra:
        data.update(extra)
    path = tmp_path / "imperandi.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_config_resolves_profile_and_relative_paths(tmp_path):
    config = load_config(_write_config(tmp_path))

    assert config.output.table_format is TableFormat.CSV
    assert config.output.publish_formats == [TableFormat.CSV]
    assert config.output.root == (tmp_path / "results").resolve()
    assert config.input.sources == [str(tmp_path / "input.csv")]
    assert "builtin:liver_ct" in config.annotations.rule_packs
    assert config.segmentation.enabled is False


def test_profile_keeps_ct_and_mr_totalsegmentator_tasks_explicit(tmp_path):
    config = load_config(_write_config(tmp_path))
    context = SimpleNamespace(config=config)

    ct_tasks = _segmentation_backend_config(context, "CT")["tasks"]
    mr_tasks = _segmentation_backend_config(context, "MR")["tasks"]

    assert [task["task"] for task in ct_tasks] == ["total", "liver_lesions"]
    assert [task["task"] for task in mr_tasks] == ["total_mr"]
    assert ct_tasks[0]["output"] == "liver"
    assert mr_tasks[0]["output"] == "liver"
    assert {task.modality for task in config.segmentation.tasks} == {"CT", "MR"}


def test_segmentation_task_requires_one_explicit_modality(tmp_path):
    path = _write_config(
        tmp_path,
        {
            "segmentation": {
                "enabled": True,
                "tasks": [
                    {
                        "id": "ambiguous-liver",
                        "backend": "totalsegmentator",
                        "modalities": ["CT", "MR"],
                        "task": "total",
                        "output": "liver",
                    }
                ],
            }
        },
    )

    with pytest.raises(ValidationError, match="modality"):
        load_config(path)


def test_config_hash_is_stable(tmp_path):
    path = _write_config(tmp_path)
    assert config_hash(load_config(path)) == config_hash(load_config(path))


def test_config_hash_tracks_ontology_contents(tmp_path):
    ontology = tmp_path / "ontology.csv"
    ontology.write_text("ProtocolName,family\nLIVER,dynamic\n", encoding="utf-8")
    path = _write_config(
        tmp_path,
        {
            "annotations": {
                "ontologies": [
                    {
                        "id": "families",
                        "source": "ontology.csv",
                        "keys": {"ProtocolName": {"match": "normalized_exact"}},
                        "output": {
                            "value_column": "family",
                            "target_column": "protocol_family",
                        },
                    }
                ]
            }
        },
    )
    before = config_hash(load_config(path))

    ontology.write_text("ProtocolName,family\nLIVER,routine\n", encoding="utf-8")

    assert config_hash(load_config(path)) != before


def test_config_hash_ignores_resources_for_disabled_heavy_stages(tmp_path):
    settings = tmp_path / "radiomics.yaml"
    settings.write_text("setting:\n  binWidth: 10\n", encoding="utf-8")
    path = _write_config(
        tmp_path,
        {
            "radiomics": {
                "enabled": False,
                "settings": "radiomics.yaml",
            }
        },
    )
    before = config_hash(load_config(path))

    settings.write_text("setting:\n  binWidth: 25\n", encoding="utf-8")

    assert config_hash(load_config(path)) == before


def test_csv_warning_threshold_is_not_project_configuration(tmp_path):
    path = _write_config(
        tmp_path,
        {
            "output": {
                "root": "results",
                "table_format": "csv",
                "csv_warning_threshold_files": 100,
            }
        },
    )
    with pytest.raises(ValidationError, match="csv_warning_threshold_files"):
        load_config(path)


def test_image_phase_backend_rejects_unsupported_mri_route(tmp_path):
    path = _write_config(
        tmp_path,
        {"phase_prediction": {"enabled": True, "modalities": ["MR"]}},
    )

    with pytest.raises(ValidationError, match=r"phase_prediction\.modalities"):
        load_config(path)


def test_resolved_configuration_is_json_serializable(tmp_path):
    resolved = resolved_config(load_config(_write_config(tmp_path)))
    assert resolved["output"]["table_format"] == "csv"


def test_v2_defaults_keep_heavy_stages_disabled_without_a_profile(tmp_path):
    path = tmp_path / "imperandi.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "project": {"name": "safe-defaults"},
                "input": {"sources": ["input.csv"]},
                "output": {"root": "results"},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.conversion.enabled is False
    assert config.segmentation.enabled is False


def test_enabling_segmentation_requires_at_least_one_task(tmp_path):
    path = tmp_path / "imperandi.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "project": {"name": "missing-tasks"},
                "input": {"sources": ["input.csv"]},
                "output": {"root": "results"},
                "segmentation": {"enabled": True},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match=r"segmentation\.tasks"):
        load_config(path)


def test_required_clinical_slots_must_match_their_modality(tmp_path):
    path = _write_config(
        tmp_path,
        {"selection": {"required_slots": {"CT": ["MR_T2"]}}},
    )

    with pytest.raises(ValidationError, match="do not match CT"):
        load_config(path)


def test_v1_project_configuration_is_rejected_explicitly(tmp_path):
    path = _write_config(tmp_path, {"version": 1})

    with pytest.raises(ValidationError, match="version"):
        load_config(path)


def test_project_configuration_requires_an_explicit_version(tmp_path):
    path = _write_config(tmp_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data.pop("version")
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="explicit version"):
        load_config(path)


def test_referenced_rule_pack_is_validated_before_execution(tmp_path):
    path = _write_config(
        tmp_path,
        {"annotations": {"rule_packs": ["missing-rules.yaml"]}},
    )

    with pytest.raises(FileNotFoundError, match="rule pack"):
        load_config(path)


def test_malformed_project_yaml_reports_the_configuration_path(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("project: [unterminated", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid YAML configuration"):
        load_config(path)


def test_identity_crosswalk_values_are_validated_during_config_load(tmp_path):
    crosswalk = tmp_path / "identities.csv"
    crosswalk.write_text(
        "dicom_patient_id,patient_id\n,P-001\n",
        encoding="utf-8",
    )
    path = _write_config(
        tmp_path,
        {
            "identity": {
                "source": {
                    "patient_id_columns": ["PatientID"],
                    "namespace_columns": [],
                },
                "canonical": {
                    "strategy": "crosswalk",
                    "crosswalk": "identities.csv",
                    "crosswalk_keys": ["dicom_patient_id"],
                },
            }
        },
    )

    with pytest.raises(ValueError, match="empty key values"):
        load_config(path)
