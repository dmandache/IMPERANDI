import sys
from datetime import time as dt_time
from pathlib import Path

import pandas as pd
import pytest

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imperandi.utils import datetime as datetime_utils
from imperandi.utils.datetime import to_times


def test_to_times_converts_only_real_clock_time_columns():
    df = pd.DataFrame(
        {
            "StudyTime": ["093015", "111530"],
            "AcquisitionDateTime": ["20240101123045", "20240101131559"],
            "InstanceCreationTime": ["101010.123", "131415.000"],
            "EchoTime": ["500", "750"],
            "ExposureTime": ["1200", "1500"],
            "AcquisitionDuration": ["30", "45"],
        }
    )

    out = to_times(df.copy())

    assert out.loc[0, "StudyTime"] == dt_time(9, 30, 15)
    assert out.loc[1, "AcquisitionDateTime"] == dt_time(13, 15, 59)
    assert out.loc[1, "InstanceCreationTime"] == dt_time(13, 14, 15)

    # Duration-like columns should remain untouched.
    assert out["EchoTime"].tolist() == ["500", "750"]
    assert out["ExposureTime"].tolist() == ["1200", "1500"]
    assert out["AcquisitionDuration"].tolist() == ["30", "45"]


def test_to_times_uses_content_for_custom_time_columns():
    df = pd.DataFrame(
        {
            "ProcedureStartTime": ["08:31:00", "09:41:20"],
            "ScanTime": ["1200", "1300"],
            "TriggerTime": ["500", "600"],
        }
    )

    out = to_times(df.copy())

    assert out.loc[0, "ProcedureStartTime"] == dt_time(8, 31, 0)
    assert out.loc[1, "ProcedureStartTime"] == dt_time(9, 41, 20)
    assert out["ScanTime"].tolist() == ["1200", "1300"]
    assert out["TriggerTime"].tolist() == ["500", "600"]


def test_get_reads_attributes_mappings_and_defaults():
    class Metadata:
        StudyTime = "093000"

    assert datetime_utils._get(None, "StudyTime", "missing") == "missing"
    assert datetime_utils._get(Metadata(), "StudyTime") == "093000"
    assert datetime_utils._get({"StudyTime": "101500"}, "StudyTime") == "101500"
    assert datetime_utils._get(object(), "StudyTime", "missing") == "missing"


@pytest.mark.parametrize(
    ("value", "include_ms", "expected"),
    [
        (None, False, None),
        (float("nan"), False, None),
        ("nan", False, None),
        ("00:00:00", False, None),
        ("09:30:15.123456", False, dt_time(9, 30, 15)),
        ("09:30:15.123456", True, dt_time(9, 30, 15, 123456)),
        ("20240101123045", False, dt_time(12, 30, 45)),
        ("9", False, dt_time(9, 0)),
        ("0930", False, dt_time(9, 30)),
        ("93015", False, dt_time(9, 30, 15)),
        ("101112.5", True, dt_time(10, 11, 12, 500000)),
        ("000000", False, None),
        ("not-a-time", False, None),
        ("250000", False, None),
        ("125960", False, None),
    ],
)
def test_parse_dicom_time_variants(value, include_ms, expected):
    assert datetime_utils._parse_dicom_time(value, include_ms=include_ms) == expected


def test_earliest_acquisition_datetime_handles_lists_literals_and_scalars():
    assert datetime_utils.earliest_acquisition_datetime(None) is None
    assert datetime_utils.earliest_acquisition_datetime(float("nan")) is None
    assert datetime_utils.earliest_acquisition_datetime(
        "['120000', '090000', None]"
    ) == dt_time(9, 0)
    assert datetime_utils.earliest_acquisition_datetime(("bad", None)) is None
    assert datetime_utils.earliest_acquisition_datetime("093015.2") == dt_time(
        9, 30, 15
    )
    assert datetime_utils.earliest_acquisition_datetime(131500) == dt_time(13, 15)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        (float("nan"), []),
        (["090000"], ["090000"]),
        (("090000", "100000"), ["090000", "100000"]),
        ("", []),
        ("not a literal", ["not a literal"]),
        ("['090000', '100000']", ["090000", "100000"]),
        ("123045", [123045]),
        (123045, [123045]),
    ],
)
def test_iter_time_tokens_normalizes_supported_inputs(value, expected):
    assert datetime_utils._iter_time_tokens(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        (float("nan"), False),
        (dt_time(9, 30), True),
        (pd.Timestamp("2024-01-01 09:30:00"), True),
        ("NaT", False),
        ("09:30:00", True),
        ("99:99:99", False),
        ("not-a-time", False),
        ("1234", False),
        ("123045", True),
        ("20240101123045", True),
    ],
)
def test_clock_like_token_detection(value, expected):
    assert datetime_utils._is_clock_like_token(value) is expected


def test_clock_content_and_column_selection_are_conservative():
    assert not datetime_utils._has_clock_time_content(pd.Series(dtype=object))
    assert datetime_utils._has_clock_time_content(pd.Series(["123045", "invalid"]))
    assert not datetime_utils._has_clock_time_content(
        pd.Series(["invalid", "also bad"])
    )

    df = pd.DataFrame(
        {
            "StudyTime": ["093000"],
            "ProcedureStartTime": ["09:31:00"],
            "CustomTime": ["500"],
            "EchoTime": ["123045"],
            "Description": ["09:45:00"],
        }
    )

    assert datetime_utils._select_time_columns(df) == [
        "StudyTime",
        "ProcedureStartTime",
    ]


def test_iso_detection_handles_empty_compact_and_delimited_dates():
    assert not datetime_utils._is_iso_like(pd.Series(dtype=object))
    assert datetime_utils._is_iso_like(pd.Series(["20240131", "2024-02-01"]))
    assert not datetime_utils._is_iso_like(pd.Series(["31/01/2024"]))


def test_infer_dayfirst_covers_parse_winner_and_tie_breakers(monkeypatch):
    assert datetime_utils._infer_dayfirst(pd.Series(dtype=object))

    def dayfirst_wins(values, *, dayfirst, errors):
        del values, errors
        return pd.Series([pd.Timestamp("2024-01-01") if dayfirst else pd.NaT])

    monkeypatch.setattr(datetime_utils.pd, "to_datetime", dayfirst_wins)
    assert datetime_utils._infer_dayfirst(pd.Series(["01/02/2024"]))

    def tied(values, *, dayfirst, errors):
        del values, dayfirst, errors
        return pd.Series([pd.Timestamp("2024-01-01")])

    monkeypatch.setattr(datetime_utils.pd, "to_datetime", tied)
    assert datetime_utils._infer_dayfirst(pd.Series(["13/01/2024"]))
    assert not datetime_utils._infer_dayfirst(pd.Series(["01/13/2024"]))
    assert datetime_utils._infer_dayfirst(pd.Series(["01/02/2024"]))


def test_to_dates_converts_iso_columns_and_ignores_other_columns():
    df = pd.DataFrame(
        {
            "StudyDate": ["20240131", "20240201"],
            "description": ["20240131", "unchanged"],
        }
    )

    result = datetime_utils.to_dates(df.copy())

    assert result["StudyDate"].tolist() == [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-02-01"),
    ]
    assert result["description"].tolist() == ["20240131", "unchanged"]


def test_to_dates_uses_inferred_order_for_non_iso_columns(monkeypatch):
    df = pd.DataFrame({"ReviewDate": ["31/01/2024"]})
    seen = []

    monkeypatch.setattr(datetime_utils, "_infer_dayfirst", lambda _series: True)

    def fake_to_datetime(values, *, dayfirst, format, errors):
        seen.append((dayfirst, format, errors))
        return pd.Series([pd.Timestamp("2024-01-31")], index=values.index)

    monkeypatch.setattr(datetime_utils.pd, "to_datetime", fake_to_datetime)

    result = datetime_utils.to_dates(df)

    assert result.loc[0, "ReviewDate"] == pd.Timestamp("2024-01-31")
    assert seen == [(True, "%Y%m%d", "coerce")]
