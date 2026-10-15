"""Preserve raw DICOM clock strings when reading metadata CSVs."""

from ast import literal_eval

DICOM_CLOCK_DTYPES = {
    column: "string"
    for column in (
        "StudyDate",
        "SeriesDate",
        "AcquisitionDate",
        "ContentDate",
        "InstanceCreationDate",
        "StudyTime",
        "SeriesTime",
        "AcquisitionDateTime",
        "FrameReferenceDateTime",
        "FrameAcquisitionDateTime",
        "AcquisitionTime",
        "ContentTime",
        "InstanceCreationTime",
        "ContrastBolusStartTime",
        "ContrastBolusStopTime",
        "TimezoneOffsetFromUTC",
    )
}


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
