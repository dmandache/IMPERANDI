"""DICOM acquisition clocks with retained precision and clock domains.

An incomplete DT/TM represents an interval, not an exact midnight timestamp.
Naive clocks are local scanner time; explicit UTC offsets are normalized only
when comparing other clocks with explicit offsets.
"""

from __future__ import annotations

import calendar
import re
from ast import literal_eval
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd

DICOM_CLOCK_DTYPES = {
    column: "string"
    for column in (
        "AcquisitionDateTime",
        "FrameReferenceDateTime",
        "FrameAcquisitionDateTime",
        "AcquisitionTime",
        "ContrastBolusStartTime",
        "acquisition_datetime",
    )
}


@dataclass(frozen=True)
class ClockInterval:
    start: datetime
    end: datetime

    @property
    def domain(self):
        return "UTC" if self.start.tzinfo else "LOCAL"

    @property
    def seconds(self):
        # Do not let the host timezone reinterpret a scanner's local clock.
        start, end = self.start, self.end
        if start.tzinfo:
            start, end = start.astimezone(timezone.utc), end.astimezone(timezone.utc)
        origin = datetime(1970, 1, 1, tzinfo=timezone.utc if start.tzinfo else None)
        return (start - origin).total_seconds(), (end - origin).total_seconds()


def scalar_values(value):
    if isinstance(value, str) and value.strip().startswith(("[", "(")):
        try:
            value = literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    if isinstance(value, (list, tuple, set)) or hasattr(value, "tolist"):
        values = value.tolist() if hasattr(value, "tolist") else value
        if isinstance(values, (list, tuple, set)):
            return [item for part in values for item in scalar_values(part)]
    return [value]


def _combine(intervals):
    intervals = [interval for interval in intervals if interval is not None]
    if not intervals or len({interval.domain for interval in intervals}) != 1:
        return None
    return ClockInterval(min(i.start for i in intervals), max(i.end for i in intervals))


def dicom_datetime_interval(value):
    """Parse DICOM DT (including partial precision and optional UTC offset)."""
    values = scalar_values(value)
    if len(values) != 1:
        return _combine(dicom_datetime_interval(item) for item in values)
    value = values[0]
    if value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, datetime):
        return ClockInterval(value, value)
    if value is None or isinstance(value, float):
        return None  # Float conversion may already have destroyed DT precision.
    text = str(value).strip()
    match = re.fullmatch(r"(\d{4}(?:\d{2}){0,5})(?:\.(\d{1,6}))?([+-]\d{4})?", text)
    if not match:
        # Canonical timestamps written by ingestion, not permissive free text.
        if re.match(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", text):
            try:
                parsed = datetime.fromisoformat(text)
                return ClockInterval(parsed, parsed)
            except ValueError:
                pass
        return None
    digits, fraction, offset = match.groups()
    if fraction and len(digits) != 14:
        return None
    parts = [int(digits[:4])] + [
        int(digits[i : i + 2]) for i in range(4, len(digits), 2)
    ]
    defaults = [1, 1, 0, 0, 0]
    parts += defaults[len(parts) - 1 :]
    try:
        tz = None
        if offset:
            hours, minutes = int(offset[1:3]), int(offset[3:5])
            if minutes > 59 or hours > 14 or (hours == 14 and minutes):
                return None
            if offset[0] == "-" and (hours > 12 or (hours == 12 and minutes)):
                return None
            tz = timezone(
                timedelta(
                    minutes=(hours * 60 + minutes) * (1 if offset[0] == "+" else -1)
                )
            )
        start = datetime(
            *parts, microsecond=int((fraction or "").ljust(6, "0") or "0"), tzinfo=tz
        )
        if len(digits) == 4:
            stop = start.replace(year=start.year + 1)
        elif len(digits) == 6:
            stop = start + timedelta(
                days=calendar.monthrange(start.year, start.month)[1]
            )
        else:
            step = {
                8: 86400,
                10: 3600,
                12: 60,
                14: 10 ** (-len(fraction)) if fraction else 1,
            }[len(digits)]
            stop = start + timedelta(seconds=step)
        end = stop - timedelta(microseconds=1)
        if tz:
            start, end = start.astimezone(timezone.utc), end.astimezone(timezone.utc)
        return ClockInterval(start, end)
    except (ValueError, OverflowError):
        return None


def dicom_time_interval(value, day):
    """Attach a TM to an acquisition date without inventing missing precision."""
    values = scalar_values(value)
    if len(values) != 1:
        return _combine(dicom_time_interval(item, day) for item in values)
    value = values[0]
    if value is pd.NaT or value is pd.NA or day is pd.NaT or day is pd.NA:
        return None
    try:
        parsed_day = (
            day
            if isinstance(day, date)
            else pd.to_datetime(str(day), errors="raise").date()
        )
    except (ValueError, TypeError):
        return None
    if isinstance(value, time):
        instant = datetime.combine(parsed_day, value)
        return ClockInterval(instant, instant)
    if value is None:
        return None
    text = str(value).strip()
    if ":" in text:
        try:
            return dicom_time_interval(time.fromisoformat(text), parsed_day)
        except ValueError:
            return None
    # CSV numeric TM may have lost a leading zero; leave raw DT strings intact.
    main, _, fraction = text.partition(".")
    if main.isdigit() and len(main) in {1, 3, 5}:
        main = "0" + main
    if len(main) not in {2, 4, 6} or not main.isdigit():
        return None
    return dicom_datetime_interval(
        parsed_day.strftime("%Y%m%d") + main + ("." + fraction if fraction else "")
    )
