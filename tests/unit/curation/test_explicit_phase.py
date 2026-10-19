"""Manifest phase assignments, precedence, and selection priority."""

import copy

import pandas as pd
import pytest
import yaml

from imperandi import cli
from imperandi.curation import phase_resolution as phase_policy
from imperandi.curation.ct.ct_curation import curate_ct, score_ct
from imperandi.curation.selection_export import save_selected_candidates
from imperandi.curation.mri.mri_curation import curate_mri, score_t1
from imperandi.curation.phase_resolution import (
    apply_phase_curation,
    phase_curation_input_columns,
    phase_needs_strategy,
    phase_selection_bonus,
    validate_phase_curation,
)
from imperandi.extract import phase as phase_command
from imperandi.ingest import clean


def applied_bonuses(frame):
    assert "phase_selection_bonus" not in frame.columns
    return frame.apply(phase_selection_bonus, axis=1).tolist()


def config(*rules):
    return {
        "strategies": [
            {
                "type": "explicit",
                "name": "reviewed_assignments",
                "assignments": list(rules),
            },
            {"type": "rules"},
        ]
    }


def arterial_rule(**changes):
    return {
        "name": "reviewed_ph1",
        "phase": "arterial",
        "rules": [
            {"column": "patient_key", "op": "in", "value": ["p1", "p2"]},
            {"column": "SeriesDescription", "op": "icontains", "value": "ph1"},
        ],
        **changes,
    }


def cohort(modality):
    return pd.DataFrame(
        {
            "patient_key": ["p1", "p1"],
            "study_id": ["s1", "s1"],
            "series_id": ["reviewed", "automatic"],
            "volume_id": [f"{modality}_reviewed", f"{modality}_automatic"],
            "Modality": [modality] * 2,
            "SeriesDescription": ["COR T1 portal PH1", "AX T1 arterial"],
            "ImageType": ["ORIGINAL PRIMARY"] * 2,
            "Rows": [256, 512],
            "Columns": [256, 512],
            "SliceThickness": [9, 2],
            "n_files": [30, 120],
            "ReviewGroup": ["reviewed", "automatic"],
        }
    )


def test_multiple_conditions_override_metadata_with_provenance_and_no_input_changes():
    options = config(arterial_rule())
    original_options = copy.deepcopy(options)
    frame = pd.DataFrame(
        {
            "patient_key": ["p1", "p3", "p1"],
            "SeriesDescription": ["AX T1 PH1", "AX T1 PH1", "AX T1 ph2"],
            "rule_phase": ["PORTAL_VENOUS"] * 3,
        },
        index=[7, 7, 2],
    )
    original = frame.copy(deep=True)
    result = apply_phase_curation(frame, options)
    assert result.index.tolist() == [7, 7, 2]
    assert result.phase.tolist() == ["ARTERIAL", "PORTAL_VENOUS", "PORTAL_VENOUS"]
    assert result.phase_source.tolist() == ["reviewed_assignments", "rules", "rules"]
    assert result.phase_strategy.tolist() == ["explicit", "rules", "rules"]
    assert applied_bonuses(result) == [1000, 0, 0]
    assert result.phase_confidence.iloc[0] == "high"
    assert "reviewed_ph1" in result.phase_reason.iloc[0]
    assert "patient_key='p1'" in result.phase_reason.iloc[0]
    assert "SeriesDescription='AX T1 PH1'" in result.phase_reason.iloc[0]
    pd.testing.assert_frame_equal(frame, original)
    assert options == original_options
    normalized = validate_phase_curation(options)
    assert validate_phase_curation(normalized) == normalized


def test_first_matching_rule_and_strategy_order_are_authoritative():
    options = config(arterial_rule(), arterial_rule(phase="delayed"))
    frame = pd.DataFrame(
        [{"patient_key": "p1", "SeriesDescription": "ph1", "rule_phase": "NATIVE"}]
    )
    assert apply_phase_curation(frame, options).phase.tolist() == ["ARTERIAL"]
    options["strategies"].reverse()
    result = apply_phase_curation(frame, options)
    assert result.phase.tolist() == ["NATIVE"]
    assert applied_bonuses(result) == [0]


@pytest.mark.parametrize(
    "op,value,actual,expected",
    [
        ("eq", "p1", "p1", True),
        ("eq", "p1", "P1", False),
        ("eq", 1, "1", False),
        ("ne", "p1", None, False),
        ("ne", "p1", ["p1", "p2"], False),
        ("in", ["p1"], [None, "p1"], True),
        ("in", [], "p1", False),
        ("not_in", ["p1"], "p2", True),
        ("not_in", ["p1"], pd.NA, False),
        ("not_in", ["p1"], ["p1", "p2"], False),
        ("not_in", ["p1"], [None, "p2"], True),
        ("not_in", ["p1"], [], False),
        ("contains", "ph1", "PH1", False),
        ("icontains", "ph1", "AX PH1", True),
        ("icontains", "ph1.", "ph1X", False),
        ("icontains", "ph1", ["AX T1", "PH1"], True),
        ("regex", r"(?i)ph[12]\b", "AX PH2", True),
        ("regex", r"ph[12]\b", "ph10", False),
        ("lt", 10, "9", True),
        ("lte", 10, 10, True),
        ("gt", 10, 11, True),
        ("gte", 10, 10, True),
        ("gt", 10, "bad", False),
        ("gt", 10, float("inf"), False),
        ("is_null", None, None, True),
        ("is_null", None, [None, pd.NA], True),
        ("is_null", None, [], True),
        ("is_null", None, [None, "p1"], False),
        ("not_null", None, [None, "p1"], True),
    ],
)
def test_condition_operators(op, value, actual, expected):
    condition = {"column": "metadata", "op": op}
    if op not in {"is_null", "not_null"}:
        condition["value"] = value
    frame = pd.DataFrame([{"metadata": actual}])
    result = apply_phase_curation(frame, config(arterial_rule(rules=[condition])))
    assert result.phase.tolist() == ["ARTERIAL" if expected else "OTHER"]
    assert applied_bonuses(result) == [1000 if expected else 0]
    steps = clean.validate_cleaning_manifest(
        {
            "cleaning": {
                "steps": [
                    {
                        "type": "filter",
                        "kind": "keep",
                        "scope": "row",
                        "rules": [condition],
                    }
                ]
            }
        }
    )
    filtered = clean._run_filter_step(frame, steps[0])
    assert len(filtered) == int(expected)


def test_or_conditions_allow_a_single_matching_column_and_missing_columns_do_not_match():
    result = apply_phase_curation(
        pd.DataFrame([{"SeriesDescription": "PH1"}, {"SeriesDescription": "ph2"}]),
        config(arterial_rule(logic="or")),
    )
    assert result.phase.tolist() == ["ARTERIAL", "OTHER"]
    result = apply_phase_curation(
        pd.DataFrame([{"SeriesDescription": "PH1"}]),
        config(arterial_rule()),
    )
    assert result.phase.tolist() == ["OTHER"]
    result = apply_phase_curation(
        pd.DataFrame([{"SeriesDescription": "PH1"}]),
        config(arterial_rule(rules=[{"column": "absent", "op": "is_null"}])),
    )
    assert result.phase.tolist() == ["OTHER"]


def test_explicit_rules_skip_prediction_but_keep_phase_applicability():
    options = config(arterial_rule())
    options["strategies"].append({"type": "totalsegmentator"})
    frame = pd.DataFrame(
        {
            "patient_key": ["p1"] * 3,
            "SeriesDescription": ["PH1"] * 3,
            "phase_applicability_reason": [None, "LOCALIZER", None],
            "rule_phase": ["OTHER"] * 3,
        }
    )
    frame.loc[2, "patient_key"] = "p3"
    assert phase_needs_strategy(frame, options, "totalsegmentator").tolist() == [
        False,
        False,
        True,
    ]
    result = apply_phase_curation(frame, options)
    assert result.phase_status.tolist() == ["RESOLVED", "NOT_APPLICABLE", "UNRESOLVED"]
    assert applied_bonuses(result) == [1000, 0, 0]


@pytest.mark.parametrize(
    "rule",
    [
        None,
        {},
        arterial_rule(name=""),
        arterial_rule(phase=""),
        arterial_rule(phase=None),
        arterial_rule(logic="xor"),
        arterial_rule(logic=[]),
        arterial_rule(rules=[]),
        arterial_rule(rules=[None]),
        arterial_rule(rules=[{"column": "x", "op": "unsupported", "value": 1}]),
        arterial_rule(rules=[{"column": " ", "op": "eq", "value": 1}]),
        arterial_rule(rules=[{"column": "x", "op": "eq"}]),
        arterial_rule(rules=[{"column": "x", "op": "in", "value": "p1"}]),
        arterial_rule(rules=[{"column": "x", "op": "in", "value": [["p1"]]}]),
        arterial_rule(rules=[{"column": "x", "op": "icontains", "value": ""}]),
        arterial_rule(rules=[{"column": "x", "op": "regex", "value": "["}]),
        arterial_rule(rules=[{"column": "x", "op": "gt", "value": "10"}]),
        arterial_rule(rules=[{"column": "x", "op": "gt", "value": True}]),
    ],
)
def test_invalid_explicit_rules_are_rejected(rule):
    with pytest.raises(ValueError, match=r"phase_curation.strategies\[0\].assignments"):
        validate_phase_curation(config(rule))


@pytest.mark.parametrize("rules", [None, []])
def test_explicit_strategy_requires_assignments(rules):
    with pytest.raises(ValueError, match="non-empty list of assignments"):
        validate_phase_curation(
            {"strategies": [{"type": "explicit", "assignments": rules}]}
        )


@pytest.mark.parametrize(
    "modality,curate,score", [("CT", curate_ct, score_ct), ("MR", curate_mri, score_t1)]
)
def test_explicit_assignment_wins_selection_and_survives_csv_export(
    tmp_path, monkeypatch, modality, curate, score
):
    options = config(arterial_rule())
    result = curate(cohort(modality), phase_curation=options)
    curated = result["curated"]
    assert curated.phase.tolist() == ["ARTERIAL", "ARTERIAL"]
    assert curated.rule_phase.iloc[0] == "PORTAL_VENOUS"
    assert result["selected_long"].volume_id.tolist() == [f"{modality}_reviewed"]
    reviewed = curated.iloc[0]
    assert score(reviewed) == score(reviewed.drop("phase_strategy")) + 1000
    source = tmp_path / "curated.csv"
    curated.to_csv(source, index=False)
    selected = tmp_path / "selected.csv"
    assert save_selected_candidates(pd.read_csv(source), selected, options) == 1
    exported = pd.read_csv(selected)
    assert exported.volume_id.tolist() == [f"{modality}_reviewed"]
    assert applied_bonuses(exported) == [1000]
    assert exported.phase_source.tolist() == ["reviewed_assignments"]
    assert exported.phase_strategy.tolist() == ["explicit"]
    original_score = exported.selection_score.iloc[0]
    monkeypatch.setattr(phase_policy, "EXPLICIT_PHASE_SELECTION_BONUS", 2000)
    assert phase_selection_bonus(pd.read_csv(source).iloc[0]) == 2000
    assert save_selected_candidates(pd.read_csv(source), selected, options) == 1
    rescored = pd.read_csv(selected)
    assert "phase_selection_bonus" not in rescored
    assert rescored.selection_score.iloc[0] == original_score + 1000


def test_clean_and_phase_commands_load_custom_conditions_and_reset_bonus_on_input_change(
    tmp_path, monkeypatch
):
    rule = arterial_rule()
    rule["rules"].append({"column": "ReviewGroup", "op": "eq", "value": "reviewed"})
    options = config(rule)
    assert "ReviewGroup" in phase_curation_input_columns(options)
    manifest = {
        "phase_curation": options,
        "cleaning": {
            "steps": [{"type": "modality_curation"}, {"type": "phase_curation"}]
        },
    }
    steps = clean.validate_cleaning_manifest(manifest)
    assert "ReviewGroup" in clean._collect_required_input_columns(steps, options)
    source = tmp_path / "input.csv"
    cohort("CT").to_csv(source, index=False)
    manifest_path = tmp_path / "site.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    cleaned = tmp_path / "cleaned.csv"
    assert (
        cli.main(["clean", str(source), str(cleaned), "--manifest", str(manifest_path)])
        == 0
    )
    assert applied_bonuses(pd.read_csv(cleaned)) == [1000, 0]

    options["strategies"].append({"type": "totalsegmentator"})
    manifest_path.write_text(
        yaml.safe_dump({"phase_curation": options}), encoding="utf-8"
    )
    monkeypatch.setattr(
        phase_command,
        "_load_phase_extractor",
        lambda: pytest.fail("Explicit and metadata rules resolve every row"),
    )
    output = tmp_path / "phase.csv"
    arguments = ["phase", str(source), str(output), "--manifest", str(manifest_path)]
    assert cli.main(arguments) == 0
    assert applied_bonuses(pd.read_csv(output)) == [1000, 0]
    changed = cohort("CT")
    changed["ReviewGroup"] = "unreviewed"
    changed.to_csv(source, index=False)
    assert cli.main(arguments) == 0
    result = pd.read_csv(output)
    assert applied_bonuses(result) == [0, 0]
    assert result.phase.tolist() == ["PORTAL_VENOUS", "ARTERIAL"]


def test_explicit_dynamic_anchor_does_not_give_inferred_neighbors_a_bonus():
    frame = pd.DataFrame(
        {
            "patient_key": ["p1"] * 3,
            "study_id": ["s1"] * 3,
            "series_id": ["s0", "s1", "s2"],
            "volume_id": ["v0", "v1", "v2"],
            "Modality": ["MR"] * 3,
            "SeriesDescription": ["AX T1 LAVA dynamic"] * 3,
            "time": ["120000", "120001", "120002"],
        }
    )
    options = config(
        arterial_rule(
            phase="NATIVE",
            rules=[{"column": "volume_id", "op": "eq", "value": "v0"}],
        )
    )
    result = curate_mri(frame, phase_curation=options)["curated"]
    assert result.phase.tolist() == ["NATIVE", "ARTERIAL", "PORTAL_VENOUS"]
    assert applied_bonuses(result) == [1000, 0, 0]


@pytest.mark.parametrize("logic,expected", [("and", [0]), ("or", [0, 1, 2])])
def test_filter_rule_groups_can_be_reused_as_phase_assignments(logic, expected):
    frame = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "patient_key": ["p1", "p2", "p1", "p3"],
            "SeriesDescription": ["PH1", "PH1", "ph2", None],
        },
        index=[7, 7, 2, 2],
    )
    assignment = arterial_rule(logic=logic)
    assignment["rules"][0]["value"] = ["p1"]
    step = {"type": "filter", "kind": "keep", "scope": "row", **assignment}
    steps = clean.validate_cleaning_manifest({"cleaning": {"steps": [step]}})
    filtered = clean._run_filter_step(frame, steps[0])
    result = apply_phase_curation(frame, config(assignment))
    assigned = result.loc[result.phase.eq("ARTERIAL")]
    assert assigned.row_id.tolist() == filtered.row_id.tolist() == expected


@pytest.mark.parametrize("kind", ["keep", "discard"])
def test_keep_null_preserves_all_null_lists(kind):
    frame = pd.DataFrame({"metadata": [[None, pd.NA], ["ph1"], ["ph2"]]})
    step = {
        "type": "filter",
        "kind": kind,
        "scope": "row",
        "logic": "and",
        "keep_null": True,
        "rules": [{"column": "metadata", "op": "eq", "value": "ph1"}],
    }
    out = clean._run_filter_step(frame, step)
    assert out.index.tolist() == ([0, 1] if kind == "keep" else [0, 2])
