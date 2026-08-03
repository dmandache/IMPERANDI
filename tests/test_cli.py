import json
from pathlib import Path

import pytest
import yaml

from imperandi import cli


def _write_config(tmp_path: Path, **overrides) -> Path:
    data = {
        "version": 2,
        "project": {"name": "cli-test", "profile": "liver_ct_mri"},
        "input": {"sources": ["instances.csv"]},
        "output": {
            "root": "results",
            "table_format": "parquet",
            "publish_formats": ["parquet", "csv"],
        },
        "conversion": {"enabled": False},
        "segmentation": {"enabled": False},
        "registration": {"enabled": False},
        "radiomics": {"enabled": False},
    }
    data.update(overrides)
    path = tmp_path / "imperandi.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_cli_init_creates_a_valid_starter_project(tmp_path, capsys):
    path = tmp_path / "project" / "imperandi.yaml"

    assert cli.main(["init", str(path)]) == 0

    created = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert created["version"] == 2
    assert created["project"]["profile"] == "liver_ct_mri"
    assert created["output"]["table_format"] == "parquet"
    assert created["conversion"]["enabled"] is False
    assert created["segmentation"]["enabled"] is False
    assert "csv_warning_threshold_files" not in created["output"]
    assert str(path) in capsys.readouterr().out


def test_cli_init_does_not_overwrite_without_force(tmp_path):
    path = tmp_path / "imperandi.yaml"
    path.write_text("sentinel", encoding="utf-8")

    assert cli.main(["init", str(path)]) == 2
    assert path.read_text(encoding="utf-8") == "sentinel"


def test_cli_validate_resolves_profile_and_stage_graph(tmp_path, capsys):
    path = _write_config(tmp_path)

    assert cli.main(["validate", str(path)]) == 0
    assert "Configuration is valid (sha256:" in capsys.readouterr().out


def test_cli_validate_rejects_product_warning_threshold_in_project_yaml(tmp_path):
    path = _write_config(
        tmp_path,
        output={
            "root": "results",
            "table_format": "csv",
            "csv_warning_threshold_files": 10,
        },
    )

    assert cli.main(["validate", str(path)]) == 2


def test_cli_plan_exposes_fixed_high_level_pipeline(tmp_path, capsys):
    path = _write_config(tmp_path)

    assert cli.main(["plan", str(path)]) == 0

    plan = yaml.safe_load(capsys.readouterr().out)
    assert [stage["name"] for stage in plan["stages"]] == [
        "01_index",
        "02_identity",
        "03_assemble",
        "04_annotate",
        "05_convert",
        "06_predict_phase",
        "07_resolve_select",
        "08_segment",
        "09_register",
        "10_radiomics",
        "11_publish",
    ]
    assert (
        next(stage for stage in plan["stages"] if stage["name"] == "05_convert")["mode"]
        == "pass_through"
    )


def test_cli_config_resolve_prints_effective_configuration(tmp_path, capsys):
    path = _write_config(tmp_path)

    assert cli.main(["config", "resolve", str(path)]) == 0

    resolved = yaml.safe_load(capsys.readouterr().out)
    assert resolved["output"]["table_format"] == "parquet"
    assert resolved["annotations"]["rule_packs"] == [
        "builtin:liver_ct",
        "builtin:liver_mri",
    ]


def test_cli_status_reports_stage_state(tmp_path, capsys):
    stage_dir = tmp_path / "run" / "01_index"
    stage_dir.mkdir(parents=True)
    (stage_dir / "stage.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "artifacts": {"instances": "/tmp/instances.parquet"},
                "metrics": {"rows": 2},
            }
        ),
        encoding="utf-8",
    )

    assert cli.main(["status", str(tmp_path / "run")]) == 0

    states = yaml.safe_load(capsys.readouterr().out)
    assert states[0]["stage"] == "01_index"
    assert states[0]["status"] == "completed"


def test_removed_step_by_step_commands_are_not_public():
    with pytest.raises(SystemExit):
        cli.main(["parse"])


def test_cli_returns_a_clean_error_for_unexpected_backend_exceptions(
    monkeypatch, caplog
):
    def fail(_args):
        raise KeyError("backend-key")

    class FailingParser:
        @staticmethod
        def parse_args(_argv):
            return type(
                "Args",
                (),
                {
                    "log_level": None,
                    "log_file": None,
                    "quiet": False,
                    "_handler": staticmethod(fail),
                },
            )()

    monkeypatch.setattr(cli, "build_parser", FailingParser)

    assert cli.main(["status", "missing"]) == 2
    assert "backend-key" in caplog.text
