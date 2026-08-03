from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from imperandi.config import config_hash, load_config, resolved_config
from imperandi.config.models import TableFormat


def _write_config(tmp_path: Path, extra: dict | None = None) -> Path:
    data = {
        "version": 1,
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
    assert config.output.root == (tmp_path / "results").resolve()
    assert config.input.sources == [str(tmp_path / "input.csv")]
    assert "builtin:liver_ct" in config.annotations.rule_packs
    assert config.segmentation.enabled is False


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
                            "source_column": "family",
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

    with pytest.raises(ValidationError, match="phase_prediction.modalities"):
        load_config(path)


def test_resolved_configuration_is_json_serializable(tmp_path):
    resolved = resolved_config(load_config(_write_config(tmp_path)))
    assert resolved["output"]["table_format"] == "csv"
