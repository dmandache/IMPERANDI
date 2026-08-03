import logging

import pandas as pd

from imperandi.io.tables import (
    CSV_FILE_COUNT_WARNING_THRESHOLD,
    read_table,
    warn_if_csv_is_large,
    write_table,
)


def test_csv_roundtrip_preserves_structured_columns(tmp_path):
    source = pd.DataFrame(
        {
            "volume_id": ["v1", "v2"],
            "paths": [["a.dcm", "b.dcm"], "c.dcm"],
            "meta": [{"x": 1}, None],
        }
    )
    path = write_table(source, tmp_path / "table.csv")
    restored = read_table(path)
    assert restored.loc[0, "paths"] == ["a.dcm", "b.dcm"]
    assert restored.loc[1, "paths"] == "c.dcm"
    assert restored.loc[0, "meta"] == {"x": 1}
    assert path.with_suffix(".csv.schema.json").exists()


def test_csv_roundtrip_preserves_zero_padded_text_ids(tmp_path):
    source = pd.DataFrame({"PatientID": ["001", "010"]})

    restored = read_table(write_table(source, tmp_path / "identities.csv"))

    assert restored["PatientID"].tolist() == ["001", "010"]


def test_external_csv_without_sidecar_preserves_text_lexemes(tmp_path):
    path = tmp_path / "external.csv"
    path.write_text("PatientID,Rows\n001,512\n", encoding="utf-8")

    restored = read_table(path)

    assert restored.loc[0, "PatientID"] == "001"
    assert restored.loc[0, "Rows"] == "512"


def test_parquet_roundtrip(tmp_path):
    source = pd.DataFrame({"volume_id": ["v1"], "value": [2]})
    restored = read_table(write_table(source, tmp_path / "table.parquet"))
    pd.testing.assert_frame_equal(restored, source)


def test_parquet_roundtrip_allows_mixed_scalar_and_list_cells(tmp_path):
    source = pd.DataFrame(
        {"volume_id": ["v1", "v2"], "paths": [["a.dcm", "b.dcm"], "c.dcm"]}
    )
    path = write_table(source, tmp_path / "table.parquet")
    restored = read_table(path)

    assert restored.loc[0, "paths"] == ["a.dcm", "b.dcm"]
    assert restored.loc[1, "paths"] == "c.dcm"
    assert path.with_suffix(".parquet.schema.json").exists()


def test_large_csv_warning_is_internal_and_non_blocking(caplog):
    with caplog.at_level(logging.WARNING):
        warned = warn_if_csv_is_large("csv", CSV_FILE_COUNT_WARNING_THRESHOLD + 1)
    assert warned
    assert "Parquet is recommended" in caplog.text
    assert not warn_if_csv_is_large("parquet", CSV_FILE_COUNT_WARNING_THRESHOLD + 1)
