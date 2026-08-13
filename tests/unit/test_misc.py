import logging

import pandas as pd

from imperandi.utils.misc import report_change


def test_report_change_logs_compact_tables_with_affected_values(caplog):
    previous_df = pd.DataFrame(
        {
            "patient_key": ["p1", "p1", "p2"],
            "study_id": ["s1", "s1", "s2"],
            "series_id": ["sr1", "sr2", "sr3"],
            "date": ["2024-01-02", "2024-01-02", "2024-02-03"],
            "Modality": ["CT", "MR", "CT"],
            "SliceThickness": [6.0, 7.0, 2.0],
        }
    )
    current_df = previous_df[previous_df["patient_key"].eq("p2")].copy()

    with caplog.at_level(logging.WARNING, logger="imperandi.utils.misc"):
        report_change(
            current_df,
            previous_df,
            columns=["Modality", "SliceThickness"],
        )

    assert "1 patients removed in this step (showing 1)" in caplog.text
    assert "patient_key  studies Modality SliceThickness" in caplog.text
    assert "p1        1   CT, MR       6.0, 7.0" in caplog.text
    assert "1 studies removed in this step (showing 1)" in caplog.text
    assert "patient_key       date Modality SliceThickness" in caplog.text
    assert "p1 2024-01-02   CT, MR       6.0, 7.0" in caplog.text
    assert "study_id" not in caplog.text


def test_report_change_caps_tables_and_reports_omitted_entities(caplog):
    previous_df = pd.DataFrame(
        {
            "patient_key": ["p1", "p2", "p3"],
            "study_id": ["s1", "s2", "s3"],
            "series_id": ["sr1", "sr2", "sr3"],
        }
    )
    current_df = previous_df.iloc[0:0].copy()

    with caplog.at_level(logging.WARNING, logger="imperandi.utils.misc"):
        report_change(current_df, previous_df, max_rows=2)

    assert caplog.text.count("… 1 more omitted") == 2
    assert "p3" not in caplog.text


def test_report_change_reports_studies_without_a_date_column(caplog):
    previous_df = pd.DataFrame(
        {
            "patient_key": ["p1", "p1"],
            "study_id": ["s1", "s2"],
            "series_id": ["sr1", "sr2"],
            "reason": ["scout", "diagnostic"],
        }
    )
    current_df = previous_df[previous_df["study_id"].eq("s2")].copy()

    with caplog.at_level(logging.WARNING, logger="imperandi.utils.misc"):
        report_change(current_df, previous_df, col="reason")

    assert "1 studies removed in this step (showing 1)" in caplog.text
    assert "patient_key reason" in caplog.text
    assert "p1  scout" in caplog.text
    assert "study_id" not in caplog.text


def test_clean_pipeline_passes_filter_columns_to_change_report(monkeypatch):
    from imperandi.ingest import clean

    captured = []
    monkeypatch.setattr(clean, "report_volumes", lambda *_: None)
    monkeypatch.setattr(
        clean,
        "report_change",
        lambda current, previous, columns: captured.append(columns),
    )
    df = pd.DataFrame(
        {
            "patient_key": ["p1"],
            "study_id": ["s1"],
            "series_id": ["sr1"],
            "Modality": ["CT"],
            "SeriesDescription": ["scout"],
        }
    )
    step = {
        "type": "filter",
        "kind": "discard",
        "scope": "row",
        "logic": "or",
        "rules": [
            {"column": "Modality", "op": "eq", "value": "PT"},
            {"column": "Modality", "op": "eq", "value": "NM"},
            {"column": "SeriesDescription", "op": "icontains", "value": "scout"},
        ],
    }

    clean.run_clean_pipeline(df, [step])

    assert captured == [["Modality", "SeriesDescription"]]
