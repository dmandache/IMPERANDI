"""Independent modality annotation and canonical phase cleaning steps."""

import pandas as pd
import pytest
import yaml

from imperandi import cli
from imperandi.curation import curate_by_modality
from imperandi.ingest import clean


@pytest.fixture
def cohort():
    return pd.DataFrame(
        {
            "patient_key": ["p1", "p1"],
            "study_id": ["ct-exam", "mr-exam"],
            "series_id": ["ct-series", "mr-series"],
            "volume_id": ["ct", "mr"],
            "Modality": ["CT", "MR"],
            "SeriesDescription": ["AX native", "AX T1 portal venous"],
            "ImageType": ["ORIGINAL PRIMARY", "ORIGINAL PRIMARY"],
            "site_phase": ["pv", "art"],
            "totalseg_phase": ["delayed", "native"],
            "Rows": [512, 512],
            "Columns": [512, 512],
            "n_files": [120, 120],
            "SliceThickness": [2, 2],
        }
    )


def phase_config(ontology_first=True):
    ontology = {
        "type": "ontology",
        "columns": ["site_phase"],
        "mapping": {"pv": "PORTAL_VENOUS", "art": "ARTERIAL"},
    }
    rules = {"type": "rules"}
    return {
        "strategies": [
            *([ontology, rules] if ontology_first else [rules, ontology]),
            {"type": "totalsegmentator"},
        ]
    }


def test_modality_step_does_not_resolve_canonical_phase(cohort, monkeypatch):
    def fail_if_resolved(*args, **kwargs):
        pytest.fail("Modality annotation must leave canonical phase to its own step")

    monkeypatch.setattr(clean, "apply_phase_curation", fail_if_resolved)
    manifest = {"cleaning": {"steps": [{"type": "modality_curation"}]}}
    steps = clean.validate_cleaning_manifest(manifest)

    out = clean.run_clean_pipeline(cohort, steps).set_index("volume_id")

    assert out.loc["ct", "rule_phase"] == "NATIVE"
    assert out.loc["mr", "rule_phase"] == "PORTAL_VENOUS"
    assert out.loc["mr", "mri_sequence"] == "T1"
    assert "phase" not in out
    assert "phase_status" not in out
    assert "phase_source" not in out


@pytest.mark.parametrize(
    "description,family",
    [("T2 HASTE", "single_shot"), ("T2 VISTA", "multi_shot")],
)
def test_t2_acquisition_family_can_filter_cleaning_output(cohort, description, family):
    cohort.loc[cohort.Modality.eq("MR"), "SeriesDescription"] = description
    column = "mri_t2_acquisition_family"
    steps = clean.validate_cleaning_manifest(
        {
            "cleaning": {
                "steps": [
                    {"type": "modality_curation"},
                    {
                        "type": "filter",
                        "kind": "keep",
                        "scope": "row",
                        "rules": [{"column": column, "op": "eq", "value": family}],
                    },
                ]
            }
        }
    )
    assert column not in clean._collect_required_input_columns(steps)
    result = clean.run_clean_pipeline(cohort, steps)
    assert result.volume_id.tolist() == ["mr"]
    assert result[column].tolist() == [family]


@pytest.mark.parametrize("ontology_first", [False, True])
def test_split_curation_matches_full_modality_api(cohort, ontology_first):
    config = phase_config(ontology_first)
    expected = curate_by_modality(cohort, phase_curation=config)["curated_all"]
    manifest = {
        "phase_curation": config,
        "cleaning": {
            "steps": [
                {"type": "modality_curation"},
                {"type": "phase_curation"},
            ]
        },
    }
    steps = clean.validate_cleaning_manifest(manifest)

    modality = clean.run_clean_pipeline(cohort, steps[:1], phase_curation=config)
    result = clean.run_clean_pipeline(modality, steps[1:], phase_curation=config)

    assert "phase" not in modality
    columns = [
        "rule_phase",
        "phase",
        "phase_status",
        "phase_source",
        "phase_reason",
        "phase_confidence",
        "selection_score",
        "selection_slot",
    ]
    pd.testing.assert_frame_equal(result[columns], expected[columns])
    assert result.phase.tolist() == (
        ["PORTAL_VENOUS", "ARTERIAL"] if ontology_first else ["NATIVE", "PORTAL_VENOUS"]
    )
    assert result.rule_phase.tolist() == ["NATIVE", "PORTAL_VENOUS"]


@pytest.mark.parametrize("row_index", [[7, 3, 9], [3, 3, 9]])
def test_phase_step_uses_existing_evidence_without_modality_annotation(row_index):
    rows = pd.DataFrame(
        {
            "patient_key": ["p1"] * 3,
            "study_id": ["exam1"] * 3,
            "series_id": ["s3", "s1", "s2"],
            "volume_id": ["v3", "v1", "v2"],
            "Modality": ["CT", "CT", "MR"],
            "rule_phase": ["OTHER", "ARTERIAL", "PORTAL_VENOUS"],
            "totalseg_phase": ["delayed", None, None],
        },
        index=row_index,
    )
    config = {"strategies": [{"type": "rules"}, {"type": "totalsegmentator"}]}
    steps = clean.validate_cleaning_manifest(
        {
            "phase_curation": config,
            "cleaning": {"steps": [{"type": "phase_curation"}]},
        }
    )

    result = clean.run_clean_pipeline(rows, steps, phase_curation=config)

    assert result.index.tolist() == rows.index.tolist()
    assert result.volume_id.tolist() == rows.volume_id.tolist()
    assert result.phase.tolist() == ["DELAYED", "ARTERIAL", "PORTAL_VENOUS"]
    assert "ct_phase" not in result
    assert "mri_sequence" not in result


@pytest.mark.parametrize("step_type", ["modality_curation", "phase_curation"])
def test_curation_step_loads_configured_sources(step_type):
    config = phase_config()
    config["text_columns"] = {"CT": ["custom_ct"], "MR": ["custom_mr"]}
    steps = clean.validate_cleaning_manifest(
        {
            "phase_curation": config,
            "cleaning": {"steps": [{"type": step_type}]},
        }
    )

    required = clean._collect_required_input_columns(steps, config)

    assert {"site_phase", "totalseg_phase", "custom_ct", "custom_mr"}.issubset(required)


@pytest.mark.parametrize("resolve_phase", [False, True])
def test_clean_cli_exports_only_after_phase_step(tmp_path, cohort, resolve_phase):
    source, output = tmp_path / "input.csv", tmp_path / "output.csv"
    cohort.to_csv(source, index=False)
    manifest = {"cleaning": {"steps": [{"type": "modality_curation"}]}}
    if resolve_phase:
        manifest["phase_curation"] = phase_config()
        manifest["cleaning"]["steps"].append({"type": "phase_curation"})
    config_path = tmp_path / "site.yaml"
    config_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    assert (
        cli.main(
            [
                "clean",
                str(source),
                str(output),
                "--manifest",
                str(config_path),
            ]
        )
        == 0
    )

    result = pd.read_csv(output)
    assert ("phase" in result) == resolve_phase
    for name in (
        "input_curated.csv",
        "qc_input_curated.csv",
        "qc_unresolved_phase.csv",
    ):
        assert (tmp_path / name).exists() == resolve_phase


def test_phase_only_clean_cli_retains_prior_modality_features(tmp_path, cohort):
    config = phase_config()
    annotated = clean.run_clean_pipeline(
        cohort, [{"type": "modality_curation"}], phase_curation=config
    )
    expected = clean.run_clean_pipeline(
        annotated, [{"type": "phase_curation"}], phase_curation=config
    ).set_index("volume_id")
    source, output = tmp_path / "modality.csv", tmp_path / "phase.csv"
    annotated.to_csv(source, index=False)
    manifest = tmp_path / "phase.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "phase_curation": config,
                "cleaning": {"steps": [{"type": "phase_curation"}]},
            }
        ),
        encoding="utf-8",
    )

    assert (
        cli.main(
            [
                "clean",
                str(source),
                str(output),
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )

    result = pd.read_csv(output).set_index("volume_id")
    for column in (
        "phase",
        "rule_phase",
        "phase_reason",
        "mri_perfusion_reason",
        "mri_perfusion_source",
        "selection_score",
        "selection_slot",
    ):
        pd.testing.assert_series_equal(
            result[column].fillna(""), expected[column].fillna(""), check_dtype=False
        )
