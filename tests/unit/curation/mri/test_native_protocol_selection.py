"""Native selection follows the selected dynamic protocol when possible."""

import pandas as pd
import pytest

from imperandi import cli
from imperandi.curation.mri import mri_curation as mc
from imperandi.curation.selection_export import (
    save_selected_candidates,
    selected_qc_path,
)


def volume(identifier, family, phase="NATIVE", **overrides):
    return {
        "patient_key": "p1",
        "study_id": "exam1",
        "date": "2020-01-01",
        "series_id": identifier,
        "volume_id": identifier,
        "Modality": "MR",
        "SeriesDescription": f"AX T1 {family}"
        + (" dynamic" if phase != "NATIVE" else ""),
        "ContrastBolusAgentPhase": "PRE_CONTRAST" if phase == "NATIVE" else phase,
        "ImageType": "ORIGINAL PRIMARY",
        "time": "115900" if phase == "NATIVE" else "120000",
        "Rows": 512,
        "Columns": 512,
        "n_files": 100,
        "SliceThickness": 3,
        **overrides,
    }


def cohort(dynamic_family="LAVA"):
    return pd.DataFrame(
        [
            volume("native_dixon", "mDIXON water"),
            volume("native_lava", "LAVA"),
            volume("dynamic", dynamic_family, "ARTERIAL"),
        ]
    )


def native_id(result):
    selected = result["selected_long"]
    return selected.loc[selected.selection_slot.eq("T1_NATIVE"), "volume_id"].item()


@pytest.mark.parametrize("image_type", ["ORIGINAL PRIMARY", "DERIVED PRIMARY"])
def test_matching_native_precedes_a_higher_scoring_unrelated_native(image_type):
    rows = cohort()
    rows.loc[rows.volume_id.eq("native_lava"), "ImageType"] = image_type

    result = mc.curate_mri(rows)

    curated = result["curated"].set_index("volume_id")
    assert (
        curated.loc["native_lava", "selection_score"]
        < curated.loc["native_dixon", "selection_score"]
    )
    assert curated.loc[["native_lava", "native_dixon"], "phase"].eq("NATIVE").all()
    assert native_id(result) == "native_lava"
    selected = result["selected_long"].set_index("selection_slot")
    assert "matches dynamic protocol" in selected.loc["T1_NATIVE", "selected_candidate"]
    assert (
        "native_dixon" in result["selected_wide"].loc[0, "T1_NATIVE_other_candidates"]
    )
    assert "_native_protocol_match" not in selected


def test_native_matches_selected_dynamic_rather_than_an_unselected_alternative():
    rows = pd.concat(
        [
            cohort(),
            pd.DataFrame(
                [volume("unselected_dynamic", "mDIXON water MPR", "ARTERIAL")]
            ),
        ],
        ignore_index=True,
    )

    result = mc.curate_mri(rows)

    selected = result["selected_long"].set_index("selection_slot")
    assert selected.loc["T1_ARTERIAL", "volume_id"] == "dynamic"
    assert native_id(result) == "native_lava"


@pytest.mark.parametrize("dynamic_family", [None, "VIBE", ""])
def test_native_uses_ordinary_ranking_without_a_matching_dynamic(dynamic_family):
    rows = cohort(dynamic_family or "")
    if dynamic_family is None:
        rows = rows.loc[rows.volume_id.ne("dynamic")]

    result = mc.curate_mri(rows)

    assert native_id(result) == "native_dixon"
    assert (
        not result["selected_long"]
        .selected_candidate.str.contains("matches dynamic protocol")
        .any()
    )


def test_unknown_sequence_families_do_not_establish_a_protocol_match():
    rows = cohort("")
    rows.loc[rows.volume_id.eq("native_lava"), "SeriesDescription"] = "AX T1"

    assert native_id(mc.curate_mri(rows)) == "native_dixon"


@pytest.mark.parametrize(
    "native_description,dynamic_family",
    [("COR T1 LAVA", "LAVA"), ("AX T1 mDIXON fat", "mDIXON water")],
)
def test_conflicting_plane_or_dixon_component_does_not_create_a_match(
    native_description, dynamic_family
):
    rows = pd.DataFrame(
        [
            volume("unrelated_native", "VIBE BH PACE"),
            volume("incompatible_native", "", SeriesDescription=native_description),
            volume("dynamic", dynamic_family, "ARTERIAL"),
        ]
    )

    result = mc.curate_mri(rows)

    assert native_id(result) == "unrelated_native"
    assert (
        not result["selected_long"]
        .selected_candidate.str.contains("matches dynamic protocol")
        .any()
    )


def test_missing_plane_does_not_hide_a_known_matching_family():
    rows = cohort()
    rows.loc[rows.volume_id.eq("native_lava"), "SeriesDescription"] = "T1 LAVA"

    assert native_id(mc.curate_mri(rows)) == "native_lava"


def test_ineligible_matching_native_does_not_replace_a_diagnostic_fallback():
    rows = cohort()
    rows.loc[rows.volume_id.eq("native_lava"), "SeriesDescription"] += " SUB"

    assert native_id(mc.curate_mri(rows)) == "native_dixon"


def test_explicit_native_assignment_retains_priority_over_protocol_preference():
    result = mc.curate_mri(
        cohort(),
        phase_curation={
            "strategies": [
                {
                    "type": "explicit",
                    "assignments": [
                        {
                            "name": "reviewed_native",
                            "phase": "NATIVE",
                            "rules": [
                                {
                                    "column": "volume_id",
                                    "op": "eq",
                                    "value": "native_dixon",
                                }
                            ],
                        }
                    ],
                },
                {"type": "rules"},
            ]
        },
    )

    assert native_id(result) == "native_dixon"


@pytest.mark.parametrize("second_visit", [2, None])
def test_matching_uses_only_dynamics_in_the_configured_candidate_group(second_visit):
    first = cohort("LAVA").assign(visit=1)
    second = cohort("mDIXON water").assign(visit=second_visit)
    second["volume_id"] = "second_" + second.volume_id
    result = mc.curate_mri(
        pd.concat([first, second], ignore_index=True),
        phase_curation={
            "strategies": [{"type": "rules"}],
            "best_candidate_group_columns": ["patient_key", "visit"],
        },
    )
    selected = result["selected_long"]

    assert set(selected.loc[selected.selection_slot.eq("T1_NATIVE"), "volume_id"]) == {
        "native_lava",
        "second_native_dixon",
    }


def test_configured_text_columns_control_protocol_matching():
    rows = cohort("mDIXON water")
    rows["CustomProtocol"] = rows.SeriesDescription
    rows.loc[rows.volume_id.eq("dynamic"), "CustomProtocol"] = "AX T1 LAVA dynamic"

    result = mc.curate_mri(
        rows,
        phase_curation={
            "strategies": [{"type": "rules"}],
            "text_columns": {"MR": ["CustomProtocol"]},
        },
    )

    assert native_id(result) == "native_lava"


def test_matching_natives_retain_existing_tie_breakers_with_duplicate_input_indices():
    early = volume("native_early", "LAVA", time="115800")
    late = volume("native_late", "LAVA", time="115900")
    rows = pd.concat([cohort(), pd.DataFrame([late, early])]).sample(
        frac=1, random_state=4
    )

    assert native_id(mc.curate_mri(rows)) == "native_early"


def test_selection_export_keeps_the_matching_native_and_qc_alternatives(tmp_path):
    curated = mc.annotate_mri(cohort())
    path = tmp_path / "selected.csv"

    assert (
        save_selected_candidates(curated, path, {"strategies": [{"type": "rules"}]})
        == 2
    )

    selected = pd.read_csv(path).set_index("selection_slot")
    assert selected.loc["T1_NATIVE", "volume_id"] == "native_lava"
    qc = pd.read_csv(selected_qc_path(path))
    assert "native_lava" in qc.loc[0, "T1_NATIVE"]
    assert "matches dynamic protocol" in qc.loc[0, "T1_NATIVE"]
    assert "native_dixon" in qc.loc[0, "T1_NATIVE_other_candidates"]


@pytest.mark.parametrize("command", ["clean", "phase"])
def test_cli_exports_a_native_from_the_selected_dynamic_protocol(tmp_path, command):
    import yaml

    source, output = tmp_path / "input.csv", tmp_path / "output.csv"
    cohort().to_csv(source, index=False)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "phase_curation": {"strategies": [{"type": "rules"}]},
                "cleaning": {
                    "steps": [{"type": "modality_curation"}, {"type": "phase_curation"}]
                },
            }
        )
    )

    assert (
        cli.main([command, str(source), str(output), "--manifest", str(manifest)]) == 0
    )

    selected = pd.read_csv(tmp_path / "input_curated.csv").set_index("selection_slot")
    assert selected.loc["T1_NATIVE", "volume_id"] == "native_lava"


def test_finished_phase_resume_regenerates_native_selection(tmp_path):
    import yaml

    source, output = tmp_path / "input.csv", tmp_path / "output.csv"
    cohort().to_csv(source, index=False)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump({"phase_curation": {"strategies": [{"type": "rules"}]}})
    )
    command = ["phase", str(source), str(output), "--manifest", str(manifest)]
    assert cli.main(command) == 0

    selected_path = tmp_path / "input_curated.csv"
    old_selection = pd.read_csv(selected_path)
    old_selection.loc[old_selection.selection_slot.eq("T1_NATIVE"), "volume_id"] = (
        "native_dixon"
    )
    old_selection.to_csv(selected_path, index=False)

    assert cli.main(command) == 0

    selected = pd.read_csv(selected_path).set_index("selection_slot")
    assert selected.loc["T1_NATIVE", "volume_id"] == "native_lava"
