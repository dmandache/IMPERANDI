import sys
from pathlib import Path

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd
import pytest

from imperandi.extract import phase as phase_module


@pytest.mark.parametrize("curation_state", ["raw", "cleaned", "partial"])
@pytest.mark.parametrize("custom_text", [False, True])
def test_phase_cli_applies_missing_modality_curation(
    tmp_path, monkeypatch, curation_state, custom_text
):
    import yaml

    from imperandi import cli
    from imperandi.ingest.clean import run_clean_pipeline

    config = {
        "strategies": [{"type": "rules"}, {"type": "totalsegmentator"}],
        "best_candidate_group_columns": ["patient_key", "study_id"],
    }
    rows = pd.DataFrame(
        {
            "volume_id": ["mr", "other", "ct", "t2", "bolus"],
            "series_id": ["s1", "s2", "s3", "s4", "s5"],
            "patient_key": ["p1"] * 5,
            "study_id": ["exam1"] * 5,
            "Modality": ["MRI", "US", "CT", "MR", "CT"],
            "SeriesDescription": [
                "AX T1 portal venous",
                "ultrasound",
                "AX arterial",
                "AX T2",
                "bolus tracking",
            ],
            "ImageType": ["ORIGINAL PRIMARY"] * 5,
            "Rows": [512] * 5,
            "Columns": [512] * 5,
            "n_files": [120] * 5,
            "SliceThickness": [2] * 5,
            "foreign_column": ["keep"] * 5,
        }
    )
    if custom_text:
        rows["CustomText"] = rows["SeriesDescription"]
        rows["SeriesDescription"] = "localizer"
        config["text_columns"] = {"CT": ["CustomText"], "MR": ["CustomText"]}
    expected = run_clean_pipeline(
        rows.copy(),
        [{"type": "modality_curation"}, {"type": "phase_curation"}],
        phase_curation=config,
    ).set_index("volume_id")
    source_rows = rows.copy()
    if curation_state != "raw":
        curated_ids = expected.index if curation_state == "cleaned" else ["ct"]
        source_rows = (
            pd.concat(
                [
                    expected.loc[curated_ids].reset_index(),
                    rows.loc[~rows.volume_id.isin(curated_ids)],
                ],
                ignore_index=True,
            )
            .set_index("volume_id")
            .loc[rows.volume_id]
            .reset_index()
        )
    source = tmp_path / "input.csv"
    output = tmp_path / "output.csv"
    source_rows.to_csv(source, index=False)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"phase_curation": config}))
    monkeypatch.setattr(
        phase_module,
        "_load_phase_extractor",
        lambda: pytest.fail("Modality rules should resolve these rows without images"),
    )

    assert (
        cli.main(["phase", str(source), str(output), "--manifest", str(manifest)]) == 0
    )

    result = pd.read_csv(output).set_index("volume_id")
    assert result.index.tolist() == rows.volume_id.tolist()
    assert result.foreign_column.eq("keep").all()
    for column in (
        "phase",
        "phase_status",
        "phase_source",
        "rule_phase",
        "rule_phase_reason",
    ):
        pd.testing.assert_series_equal(
            result.loc[expected.index, column].fillna(""),
            expected[column].fillna(""),
            check_dtype=False,
        )
    assert result.loc["ct", "phase"] == "ARTERIAL"
    assert result.loc["mr", "phase"] == "PORTAL_VENOUS"
    assert result.loc[["t2", "bolus"], "phase_status"].eq("NOT_APPLICABLE").all()
    assert not (tmp_path / "phase_errors.csv").exists()
    selected = pd.read_csv(tmp_path / "input_curated.csv").set_index("volume_id")
    assert selected.loc["ct", "phase"] == "ARTERIAL"
    assert selected.loc["mr", "phase"] == "PORTAL_VENOUS"


@pytest.mark.parametrize("change", ["CustomText", "Modality"])
def test_phase_resume_rechecks_raw_modality_metadata(tmp_path, monkeypatch, change):
    import yaml

    from imperandi import cli

    source = tmp_path / "input.csv"
    output = tmp_path / "output.csv"
    rows = pd.DataFrame(
        {
            "volume_id": ["v1", "v2"],
            "Modality": ["CT", "CT"],
            "CustomText": ["T2 arterial", "portal venous"],
        }
    )
    rows.to_csv(source, index=False)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "phase_curation": {
                    "strategies": [{"type": "rules"}],
                    "text_columns": {"CT": ["CustomText"], "MR": ["CustomText"]},
                }
            }
        )
    )
    monkeypatch.setattr(
        phase_module,
        "_load_phase_extractor",
        lambda: pytest.fail("Rules-only curation must not load a predictor"),
    )
    command = ["phase", str(source), str(output), "--manifest", str(manifest)]
    assert cli.main(command) == 0
    assert pd.read_csv(output).loc[0, "phase"] == "ARTERIAL"
    rows.loc[0, change] = "delayed" if change == "CustomText" else "MR"
    rows.to_csv(source, index=False)

    assert cli.main(command) == 0

    result = pd.read_csv(output)
    assert result.loc[1, "phase"] == "PORTAL_VENOUS"
    if change == "CustomText":
        assert result.loc[0, "phase"] == "DELAYED"
    else:
        assert result.loc[0, "phase_status"] == "NOT_APPLICABLE"
