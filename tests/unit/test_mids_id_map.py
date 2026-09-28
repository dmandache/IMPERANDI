"""Tests for deterministic external MIDS ID-map generation."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from imperandi import cli
from imperandi.export.mids.id_map import _width, generate_id_map
from imperandi.export.mids.identity import IdentityMapper
from imperandi.export.mids.models import MidsExportError


def cohort() -> pd.DataFrame:
    # Deliberately shuffled: identifiers must follow chronology/acquisition,
    # not input order or lexical raw IDs.
    return pd.DataFrame(
        [
            {
                "patient_key": "PatientA",
                "study_id": "study-late",
                "series_id": "series-second",
                "volume_id": "volume-second",
                "date": "2024-02-01",
                "time": "120000",
                "acquisition_order": 2,
                "volume_ordinal_in_series": 2,
            },
            {
                "patient_key": "PatientB",
                "study_id": "study-first",
                "series_id": "series-first",
                "volume_id": "volume-first",
                "date": "2020-01-01",
                "time": "080000",
                "acquisition_order": 1,
                "volume_ordinal_in_series": 1,
            },
            {
                "patient_key": "PatientA",
                "study_id": "study-early",
                "series_id": "series-early",
                "volume_id": "volume-early",
                "date": "2022-01-01",
                "time": "090000",
                "acquisition_order": 1,
                "volume_ordinal_in_series": 1,
            },
            {
                "patient_key": "PatientA",
                "study_id": "study-late",
                "series_id": "series-first",
                "volume_id": "volume-first",
                "date": "2024-02-01",
                "time": "110000",
                "acquisition_order": 1,
                "volume_ordinal_in_series": 1,
            },
            # Duplicate cohort row for the same source volume is collapsed.
            {
                "patient_key": "PatientA",
                "study_id": "study-late",
                "series_id": "series-first",
                "volume_id": "volume-first",
                "date": "2024-02-01",
                "time": "110000",
                "acquisition_order": 1,
                "volume_ordinal_in_series": 1,
            },
        ]
    )


def test_generate_map_uses_temporal_hierarchical_order_and_fixed_width():
    result, report = generate_id_map(cohort())
    assert len(result) == 4
    assert report["counts"] == {
        "patient": 2,
        "study": 3,
        "series": 4,
        "volume": 4,
    }
    assert report["digits"] == {
        "patient": 4,
        "study": 4,
        "series": 4,
        "volume": 4,
    }
    assert report["sort_columns_used"] == [
        "date",
        "time",
        "acquisition_order",
        "volume_ordinal_in_series",
    ]
    by_volume = result.set_index("volume_id")
    assert by_volume.loc["volume-early", "mapped_patient_key"] == "0002"
    assert by_volume.loc["volume-early", "mapped_study_id"] == "0002"
    assert by_volume.loc["volume-second", "mapped_study_id"] == "0003"
    late = result[result["study_id"].eq("study-late")].set_index("series_id")
    assert late.loc["series-first", "mapped_series_id"] == "0003"
    assert late.loc["series-second", "mapped_series_id"] == "0004"
    assert by_volume.loc["volume-first", "mapped_volume_id"].nunique() == 2
    assert result["participant_label"].equals(result["mapped_patient_key"])
    assert result["session_label"].equals(result["mapped_study_id"])
    assert result["image_label"].equals(result["mapped_volume_id"])


def test_digit_width_expands_beyond_configured_minimum():
    assert _width(1, 4) == 4
    assert _width(9999, 4) == 4
    assert _width(10000, 4) == 5


def test_nullable_integer_order_columns_sort_missing_values_last():
    frame = cohort()
    frame["acquisition_order"] = frame["acquisition_order"].astype("Int64")
    frame.loc[frame["volume_id"].eq("volume-second"), "acquisition_order"] = pd.NA

    result, _ = generate_id_map(frame)

    assert len(result) == 4
    assert set(result["mapped_volume_id"]) == {"0001", "0002", "0003", "0004"}


def test_keep_patient_key_requires_path_safe_preexisting_pseudonym():
    frame = cohort()
    result, _ = generate_id_map(frame, patient_key_mode="keep")
    assert set(result["mapped_patient_key"]) == {"PatientA", "PatientB"}
    unsafe = frame.copy()
    unsafe.loc[0, "patient_key"] = "raw/patient"
    with pytest.raises(MidsExportError, match="Unsafe patient_key"):
        generate_id_map(unsafe, patient_key_mode="keep")


def test_generated_map_is_consumable_by_export_identity_mapper(tmp_path):
    result, _ = generate_id_map(cohort())
    path = tmp_path / "ids.csv"
    result.to_csv(path, index=False)
    mapper = IdentityMapper(
        {
            "subject_column": "patient_key",
            "session": {"source_columns": ["study_id"]},
            "image_source_columns": ["volume_id"],
        },
        id_map_path=str(path),
    )
    row = cohort().iloc[1]
    assert mapper.map_row(row) == ("0001", "0001", "0001")


def test_cli_dry_run_write_and_overwrite_policy(tmp_path, capsys):
    csv_path = tmp_path / "cohort.csv"
    output = tmp_path / "protected" / "ids.csv"
    cohort().to_csv(csv_path, index=False)
    command = [
        "export",
        "mids-id-map",
        "--csv_path",
        str(csv_path),
        "--output_path",
        str(output),
    ]
    assert cli.main([*command, "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["rows"] == 4
    assert not output.exists()

    assert cli.main(command) == 0
    written = pd.read_csv(output, dtype=str)
    assert len(written) == 4
    assert written.loc[0, "mapped_patient_key"] == "0001"
    with pytest.raises(MidsExportError, match="already exists"):
        cli.main(command)
    assert cli.main([*command, "--overwrite", "replace"]) == 0
