"""Elapsed seconds follow dynamic groups and survive curation exports."""

from datetime import time

import pandas as pd
import pytest
import yaml

from imperandi import cli
from imperandi.curation.export import save_selected_candidates
from imperandi.curation.mri.curate import annotate_mri
from imperandi.ingest.clean import run_clean_pipeline


def dynamic_rows(description="AX T1 LAVA dynamic"):
    return pd.DataFrame(
        {
            "patient_key": ["p1"] * 4,
            "study_id": ["exam1"] * 4,
            "series_id": ["series1"] * 4,
            "volume_id": ["v0", "v1", "v2", "v3"],
            "Modality": ["MR"] * 4,
            "SeriesDescription": [description] * 4,
            "ImageType": ["ORIGINAL PRIMARY"] * 4,
            "time": ["120000", "120100.25", "12:03:00", time(12, 3)],
            "acquisition_order": [0, 1, 2, 3],
        }
    )


@pytest.mark.parametrize(
    "description",
    [
        "AX T1 LAVA dynamic",
        "AX T1 ART-PORT",
        "AX T1 ART/PORT/LATE",
        "AX T1 Multiart",
        "AX T1 Mask+Multiart",
    ],
)
def test_time_delta_uses_seconds_from_first_volume_independent_of_input_order(
    description,
):
    rows = dynamic_rows(description)

    out = annotate_mri(rows.sample(frac=1, random_state=4)).set_index("volume_id")

    assert out.loc[rows.volume_id, "time_delta"].tolist() == [0, 60.25, 180, 180]


@pytest.mark.parametrize(
    "field,value",
    [
        ("patient_key", "p2"),
        ("study_id", "exam2"),
        ("date", "2020-01-02"),
        ("StudyInstanceUID", "study2"),
        ("ImageType", "DERIVED PRIMARY"),
        ("ImageType", None),
        ("SeriesDescription", "AX T1 VIBE dynamic"),
        ("SeriesDescription", "COR T1 LAVA dynamic"),
        ("SeriesDescription", "AX T1 mDIXON fat dynamic"),
    ],
)
def test_distinct_dynamic_groups_have_independent_reference_times(field, value):
    first = dynamic_rows()
    first["date"] = "2020-01-01"
    first["StudyInstanceUID"] = "study1"
    second = first.copy()
    second[field] = value
    second["volume_id"] = "second_" + second.volume_id
    second["time"] = ["130000", "130100.25", "130300", "130300"]
    rows = pd.concat([first, second], ignore_index=True)

    out = annotate_mri(rows.sample(frac=1, random_state=7)).set_index("volume_id")

    assert out.loc[first.volume_id, "time_delta"].tolist() == [0, 60.25, 180, 180]
    assert out.loc[second.volume_id, "time_delta"].tolist() == [0, 60.25, 180, 180]


@pytest.mark.parametrize(
    "image_type",
    [
        r"ORIGINAL\PRIMARY",
        ["original", "primary"],
        ("ORIGINAL", "PRIMARY"),
        "['ORIGINAL', 'PRIMARY']",
    ],
)
def test_equivalent_image_type_encodings_share_the_time_reference(image_type):
    rows = dynamic_rows()
    rows.at[1, "ImageType"] = image_type

    out = annotate_mri(rows)

    assert out.time_delta.tolist() == [0, 60.25, 180, 180]


@pytest.mark.parametrize("missing", [None, "invalid", "246060"])
@pytest.mark.parametrize("position", [0, 1])
def test_missing_current_or_reference_time_is_not_replaced_with_order_numbers(
    missing, position
):
    rows = dynamic_rows()
    rows.loc[position, "time"] = missing

    out = annotate_mri(rows).set_index("volume_id")

    if position == 0:
        assert out.time_delta.isna().all()
    else:
        assert pd.isna(out.loc["v1", "time_delta"])
        assert out.loc[["v0", "v2", "v3"], "time_delta"].tolist() == [0, 180, 180]


def test_absent_time_and_non_dynamic_groups_have_no_time_delta():
    rows = dynamic_rows().drop(columns="time")
    rows["AcquisitionTime"] = ["120000", "120100", "120200", "120300"]
    assert annotate_mri(rows).time_delta.isna().all()
    assert annotate_mri(dynamic_rows("AX T1 VIBE portal")).time_delta.isna().all()


def test_cleaning_retains_empty_time_delta_when_no_times_are_available():
    rows = dynamic_rows().drop(columns="time")

    out = run_clean_pipeline(
        rows, [{"type": "modality_curation"}, {"type": "finalize"}]
    )

    assert "time_delta" in out
    assert out.time_delta.isna().all()


def test_explicit_native_in_same_group_supplies_reference_time():
    rows = dynamic_rows()
    rows.loc[0, "SeriesDescription"] = "AX T1 LAVA pre"

    out = annotate_mri(rows)

    assert out.time_delta.tolist() == [0, 60.25, 180, 180]
    assert out.loc[0, "phase"] == "NATIVE"


def test_water_and_fat_dynamic_groups_have_independent_reference_times():
    water = dynamic_rows("AX T1 mDIXON water dynamic")
    fat = dynamic_rows("AX T1 mDIXON fat dynamic")
    fat["volume_id"] = "fat_" + fat.volume_id
    fat["time"] = ["130000", "130100.25", "130300", "130300"]

    out = annotate_mri(pd.concat([water, fat], ignore_index=True)).set_index(
        "volume_id"
    )

    assert out.loc[water.volume_id, "time_delta"].tolist() == [0, 60.25, 180, 180]
    assert out.loc[fat.volume_id, "time_delta"].tolist() == [0, 60.25, 180, 180]


def test_shared_populated_time_has_zero_delta_without_ordering_evidence():
    rows = dynamic_rows().drop(columns="acquisition_order")
    rows["time"] = "12:00:00"

    out = annotate_mri(rows)

    assert out.time_delta.eq(0).all()


def test_configured_exam_groups_keep_separate_time_references():
    rows = dynamic_rows()
    rows["visit"] = [1, 1, 2, 2]

    out = annotate_mri(
        rows,
        phase_curation={
            "strategies": [{"type": "rules"}],
            "best_candidate_group_columns": ["patient_key", "visit"],
        },
    )

    assert out.time_delta.tolist() == [0, 60.25, 0, 0]


def test_export_preserves_reference_from_full_group_when_given_a_subset(tmp_path):
    out = annotate_mri(dynamic_rows()).iloc[[1]].copy()
    path = tmp_path / "selected.csv"

    save_selected_candidates(out, path, {"strategies": [{"type": "rules"}]})

    assert pd.read_csv(path).loc[0, "time_delta"] == 60.25


@pytest.mark.parametrize("command", ["clean", "phase"])
@pytest.mark.parametrize("precurated", [False, True])
def test_cli_writes_time_delta_in_main_and_selected_tables(
    tmp_path, command, precurated
):
    rows = dynamic_rows()
    if precurated:
        rows = annotate_mri(rows).drop(columns="time_delta")
    source, output = tmp_path / "input.csv", tmp_path / "output.csv"
    rows.to_csv(source, index=False)
    manifest = tmp_path / "site.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "phase_curation": {"strategies": [{"type": "rules"}]},
                "cleaning": {
                    "steps": [{"type": "modality_curation"}, {"type": "phase_curation"}]
                },
            }
        ),
        encoding="utf-8",
    )

    assert (
        cli.main([command, str(source), str(output), "--manifest", str(manifest)]) == 0
    )

    result = pd.read_csv(output).set_index("volume_id")
    assert result.loc[["v0", "v1", "v2", "v3"], "time_delta"].tolist() == [
        0,
        60.25,
        180,
        180,
    ]
    selected = pd.read_csv(tmp_path / "input_curated.csv").set_index("volume_id")
    pd.testing.assert_series_equal(
        selected.time_delta, result.loc[selected.index, "time_delta"]
    )
