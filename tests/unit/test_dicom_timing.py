"""Raw DICOM metadata stays intact; ordering uses canonical cleaning outputs."""

from datetime import time

import pandas as pd
import pytest

from imperandi.curation import common
from imperandi.curation.phase import phase_curation_input_columns
from imperandi.ingest.clean import (
    add_time,
    compute_acquisition_order,
    read_csv_with_valid_columns,
)

def test_csv_readers_preserve_dt_digits_and_fractional_seconds(tmp_path):
    path = tmp_path / "clocks.csv"
    pd.DataFrame(
        {
            "AcquisitionDateTime": ["20200101123456.123456", None],
            "AcquisitionTime": ["001234.000001", None],
        }
    ).to_csv(path, index=False)
    for frame in (
        common.read_csv(path),
        read_csv_with_valid_columns(path, ["AcquisitionDateTime", "AcquisitionTime"]),
    ):
        assert frame.loc[0, "AcquisitionDateTime"] == "20200101123456.123456"
        assert frame.loc[0, "AcquisitionTime"] == "001234.000001"


def test_time_coalescing_does_not_create_another_acquisition_clock():
    frame = pd.DataFrame({"AcquisitionTime": ["000025.000001"]})
    cleaned = add_time(frame)
    assert cleaned.loc[0, "time"] == time(0, 0, 25, 1)
    assert cleaned.loc[0, "time_source"] == "AcquisitionTime"
    assert "acquisition_datetime" not in cleaned
    assert "acquisition_datetime_source" not in cleaned


def test_canonical_dates_and_times_determine_order_despite_conflicting_raw_clocks():
    frame = pd.DataFrame(
        {
            "patient_key": ["p"] * 3,
            "study_id": ["s"] * 3,
            "volume_id": ["late", "early", "middle"],
            "date": ["2020-01-02", "2020-01-01", "2020-01-02"],
            "time": ["00:00:00.100000", "23:59:59.900000", "00:00:00.000000"],
            "AcquisitionDateTime": ["19000101000000", "21000101000000", "invalid"],
        }
    )
    result = compute_acquisition_order(frame).set_index("volume_id")
    assert result.loc[["early", "middle", "late"], "acquisition_order"].tolist() == [
        0,
        1,
        2,
    ]
    assert result.loc["late", "delay_since_first_acq_sec"] == pytest.approx(0.2)
    assert result.loc["middle", "delay_since_prev_acq_sec"] == pytest.approx(0.1)


def test_phase_fingerprint_uses_canonical_order_without_raw_clock_calibration():
    columns = phase_curation_input_columns(None)
    assert {
        "time",
        "acquisition_order",
        "volume_order_in_series",
        "n_volumes_in_series",
    } <= columns
    assert (
        not {
            "AcquisitionDateTime",
            "ContrastBolusStartTime",
            "TriggerTime",
            "TemporalPositionIdentifier",
        }
        & columns
    )
