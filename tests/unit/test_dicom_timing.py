"""DICOM clocks survive CSV, date/time cleaning, volume ordering and phase curation."""

from datetime import datetime, time, timezone

import pandas as pd
import pytest

from imperandi.curation import common
from imperandi.ingest.clean import (
    add_time,
    compute_acquisition_order,
    read_csv_with_valid_columns,
)
from imperandi.utils.datetime import to_dates, to_times
from imperandi.utils.dicom_timing import dicom_datetime_interval
from imperandi.curation.phase import (
    validate_phase_curation,
    phase_curation_input_columns,
)


@pytest.mark.parametrize(
    "raw,start,end",
    [
        ("20200101", datetime(2020, 1, 1), datetime(2020, 1, 1, 23, 59, 59, 999999)),
        (
            "202001011234",
            datetime(2020, 1, 1, 12, 34),
            datetime(2020, 1, 1, 12, 34, 59, 999999),
        ),
        (
            "20200101123456.123456",
            datetime(2020, 1, 1, 12, 34, 56, 123456),
            datetime(2020, 1, 1, 12, 34, 56, 123456),
        ),
        (
            "20200101133456.123456+0100",
            datetime(2020, 1, 1, 12, 34, 56, 123456, tzinfo=timezone.utc),
            datetime(2020, 1, 1, 12, 34, 56, 123456, tzinfo=timezone.utc),
        ),
        ("202002", datetime(2020, 2, 1), datetime(2020, 2, 29, 23, 59, 59, 999999)),
    ],
)
def test_dicom_dt_precision_and_optional_offset(raw, start, end):
    interval = dicom_datetime_interval(raw)
    assert interval.start == start
    assert interval.end == end


@pytest.mark.parametrize(
    "raw",
    [
        None,
        pd.NaT,
        pd.NA,
        "bad",
        "20200101246000",
        "2020010112.5",
        "20200101120000+0160",
        20200101120000.0,
    ],
)
def test_invalid_or_already_lossy_dt_is_not_treated_as_an_exact_clock(raw):
    assert dicom_datetime_interval(raw) is None


def test_cleaning_preserves_raw_dt_and_fractional_clock_precision():
    raw = "20200101123456.123456"
    frame = pd.DataFrame(
        {
            "AcquisitionDateTime": [raw],
            "StudyDate": ["20200101"],
            "AcquisitionTime": [None],
        }
    )
    cleaned = add_time(to_times(to_dates(frame)))
    assert cleaned.loc[0, "AcquisitionDateTime"] == raw
    assert cleaned.loc[0, "time"] == time(12, 34, 56, 123456)
    assert cleaned.loc[0, "time_source"] == "AcquisitionDateTime"
    assert cleaned.loc[0, "acquisition_datetime"] == "2020-01-01T12:34:56.123456"


def test_csv_readers_preserve_dt_digits_and_fractional_seconds(tmp_path):
    path = tmp_path / "clocks.csv"
    pd.DataFrame(
        {
            "AcquisitionDateTime": ["20200101123456.123456", None],
            "AcquisitionTime": ["001234.000001", None],
        }
    ).to_csv(path, index=False)
    for frame in [
        common.read_csv(path),
        read_csv_with_valid_columns(path, ["AcquisitionDateTime", "AcquisitionTime"]),
    ]:
        assert frame.loc[0, "AcquisitionDateTime"] == "20200101123456.123456"
        assert frame.loc[0, "AcquisitionTime"] == "001234.000001"


def test_ordering_uses_datetime_date_and_subsecond_differences():
    frame = pd.DataFrame(
        {
            "patient_key": ["p"] * 3,
            "study_id": ["s"] * 3,
            "volume_id": ["late", "early", "middle"],
            "date": ["2020-01-01"] * 3,
            "AcquisitionDateTime": [
                "20200102000000.100000",
                "20200101235959.900000",
                "20200102000000.000000",
            ],
        }
    )
    result = compute_acquisition_order(add_time(frame)).set_index("volume_id")
    assert result.loc[["early", "middle", "late"], "acquisition_order"].tolist() == [
        0,
        1,
        2,
    ]
    assert result.loc["late", "delay_since_first_acq_sec"] == pytest.approx(0.2)
    assert result.loc["middle", "delay_since_prev_acq_sec"] == pytest.approx(0.1)


def test_timing_evidence_is_in_semantic_resume_fingerprint_inputs():
    columns = phase_curation_input_columns(None)
    assert {
        "AcquisitionDateTime",
        "ContrastBolusStartTime",
        "TriggerTime",
        "TemporalPositionIdentifier",
        "Manufacturer",
    } <= columns


def test_acquisition_tm_and_injection_tm_keep_precision_and_valid_midnight():
    frame = pd.DataFrame(
        {
            "AcquisitionDate": ["20200101"],
            "AcquisitionTime": ["000025.000001"],
            "ContrastBolusStartTime": ["000000.000000"],
        }
    )
    cleaned = add_time(to_times(to_dates(frame)))
    assert cleaned.loc[0, "AcquisitionTime"] == "000025.000001"
    assert cleaned.loc[0, "ContrastBolusStartTime"] == "000000.000000"
    assert cleaned.loc[0, "time"] == time(0, 0, 25, 1)
    assert (
        to_dates(cleaned.copy()).loc[0, "acquisition_datetime_source"]
        == "AcquisitionTime"
    )


def test_native_and_iso_offset_datetimes_share_the_utc_clock_domain():
    utc = dicom_datetime_interval("20200101120000.000000+0000")
    iso = dicom_datetime_interval("2020-01-01T13:00:00+01:00")
    native = dicom_datetime_interval(
        datetime.fromisoformat("2020-01-01T13:00:00+01:00")
    )
    assert iso.seconds == utc.seconds == native.seconds


def test_mixed_local_and_utc_study_clocks_do_not_invent_delays_or_order():
    frame = pd.DataFrame(
        {
            "patient_key": ["p", "p"],
            "study_id": ["s", "s"],
            "volume_id": ["local", "utc"],
            "date": ["2020-01-01"] * 2,
            "AcquisitionDateTime": [
                "20200101120000.000000",
                "20200101110000.000000+0000",
            ],
        }
    )
    out = compute_acquisition_order(add_time(frame))
    assert out.acquisition_order.isna().all()
    assert out.delay_since_first_acq_sec.isna().all()
    assert out.delay_since_prev_acq_sec.isna().all()
    assert not any(
        column.startswith("_acq") or column == "_mixed_clock_domains" for column in out
    )
