import sys
from datetime import time as dt_time
from pathlib import Path

import pandas as pd

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
