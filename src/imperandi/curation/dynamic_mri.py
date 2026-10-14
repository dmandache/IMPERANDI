"""Independent MRI timing evidence; phase delays are built-in curation rules."""

from __future__ import annotations

import math

import pandas as pd

from imperandi.curation import common
from imperandi.utils.dicom_timing import (
    dicom_datetime_interval,
    dicom_time_interval,
    scalar_values,
)

TIMING_INPUTS = {
    "Manufacturer",
    "ManufacturerModelName",
    "SoftwareVersions",
    "SeriesDescription",
    "ProtocolName",
    "StudyDescription",
    "ImageType",
    "Modality",
    "exam_stage",
    "AcquisitionDateTime",
    "AcquisitionDate",
    "AcquisitionTime",
    "FrameReferenceDateTime",
    "FrameAcquisitionDateTime",
    "FrameAcquisitionDuration",
    "ContrastBolusStartTime",
    "ContrastBolusAgentPhase",
    "ContrastBolusAgent",
    "TemporalPositionIdentifier",
    "TemporalPositionIndex",
    "NumberOfTemporalPositions",
    "TemporalResolution",
    "TriggerTime",
    "TemporalPositionTimeOffset",
    "acquisition_datetime",
    "acquisition_datetime_source",
    "time",
    "time_source",
    "date",
    "StudyDate",
    "StudyInstanceUID",
    "study_id",
    "SeriesInstanceUID",
    "series_id",
    "FrameOfReferenceUID",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "PixelSpacing",
    "Rows",
    "Columns",
    "acquisition_id",
    "acquisition_order",
    "acquisition_plane",
    "volume_order_in_series",
    "volume_order_source",
    "n_volumes_in_series",
    "volume_split_method",
    "RepetitionTime",
    "EchoTime",
}


def temporal_position(row):
    def positive_integer(value):
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (ValueError, TypeError):
            return None
        return (
            int(number)
            if math.isfinite(number) and number >= 1 and number.is_integer()
            else None
        )

    for column in ("TemporalPositionIdentifier", "TemporalPositionIndex"):
        values = scalar_values(row.get(column))
        numbers = [positive_integer(value) for value in values]
        if not numbers or any(value is None for value in numbers):
            continue
        if len(set(numbers)) == 1:
            position = numbers[0]
            try:
                count = float(common.first_scalar(row.get("NumberOfTemporalPositions")))
            except (ValueError, TypeError):
                count = float("nan")
            if pd.notna(count) and position > count:
                return None, "invalid temporal position"
            return position, column
    return None, ""


def acquisition_clock(row):
    """Return [start,end] seconds, clock domain and provenance, or no clock."""
    for column in (
        "FrameReferenceDateTime",
        "FrameAcquisitionDateTime",
        "AcquisitionDateTime",
    ):
        if interval := dicom_datetime_interval(row.get(column)):
            low, high = interval.seconds
            low, high = _frame_window(row, low, high, column)
            return low, high, interval.domain, column
    day = "19700101"
    for column in ("AcquisitionDate", "date", "StudyDate"):
        candidate = common.first_scalar(row.get(column))
        if not common.is_missing(candidate):
            day = candidate
            break
    for column in ("AcquisitionTime", "time"):
        if column == "time" and common.stable_text(row.get("time_source")) not in {
            "AcquisitionTime",
            "AcquisitionDateTime",
            "FrameReferenceDateTime",
            "FrameAcquisitionDateTime",
        }:
            continue
        if interval := dicom_time_interval(row.get(column), day):
            return *interval.seconds, interval.domain, column
    # The raw DT was absent, not a partially precise value overwritten by this fallback.
    if common.stable_text(row.get("acquisition_datetime_source")) in {
        "AcquisitionTime",
        "AcquisitionDateTime",
        "FrameReferenceDateTime",
        "FrameAcquisitionDateTime",
    }:
        if interval := dicom_datetime_interval(row.get("acquisition_datetime")):
            return *interval.seconds, interval.domain, "acquisition_datetime"
    return None


def _frame_window(row, low, high, column):
    try:
        duration = (
            float(common.first_scalar(row.get("FrameAcquisitionDuration"))) / 1000
        )
    except (ValueError, TypeError):
        return low, high
    if math.isfinite(duration) and duration > 0:
        # A representative time need not be the midpoint. Bound both sides.
        if column == "FrameReferenceDateTime":
            low -= duration
        high += duration
    return low, high


def injection_clock(row, clock):
    if clock is None or clock[2] != "LOCAL":
        return None  # A local TM cannot be aligned to UTC without offset evidence.
    # Keep the examination date as the bolus reference across midnight.
    day = common.first_scalar(row.get("StudyDate", row.get("date")))
    if common.is_missing(day):
        day = common.first_scalar(row.get("AcquisitionDate"))
    if common.is_missing(day):
        for column in (
            "FrameReferenceDateTime",
            "FrameAcquisitionDateTime",
            "AcquisitionDateTime",
        ):
            interval = dicom_datetime_interval(row.get(column))
            if interval:
                day = interval.start.date()
                break
    if common.is_missing(day):
        day = common.first_scalar(row.get("date", row.get("StudyDate")))
    interval = dicom_time_interval(row.get("ContrastBolusStartTime"), day)
    return interval.seconds if interval else None


def phase_for_interval(clock, injection):
    """Require the entire possible delay interval to fit exactly one window."""
    low, high = clock[0] - injection[1], clock[1] - injection[0]
    if high < 0:
        return "NATIVE", (low, high)
    from imperandi.curation.mri import rules

    fits = [
        rule.label
        for rule in rules.PHASE_RULES
        if any(start <= low and high <= end for start, end in rule.time_ranges)
    ]
    return (fits[0] if len(fits) == 1 else "OTHER"), (low, high)


def phase_windows():
    """Established MRI delay windows; deliberately not configurable in manifests."""
    from imperandi.curation.mri import rules

    return {
        rule.label: rule.time_ranges[0]
        for rule in rules.PHASE_RULES
        if rule.label in {"ARTERIAL", "PORTAL_VENOUS", "DELAYED"}
    }
