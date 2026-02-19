from __future__ import annotations

import ast
import pandas as pd
import logging
from datetime import time
from typing import Optional, Any
import re

logger = logging.getLogger(__name__)

_CLOCK_TIME_NAME_ALLOWLIST = {
    "studytime",
    "seriestime",
    "acquisitiontime",
    "contenttime",
    "instancecreationtime",
    "acquisitiondatetime",
}

_DURATION_TIME_NAME_TOKENS = (
    "duration",
    "interval",
    "latency",
    "delay",
    "elapsed",
    "repetition",
    "echo",
    "inversion",
    "exposure",
    "trigger",
    "frame",
    "timesince",
)


def _get(meta: Any, key: str, default=None):
    """Get attribute or dict key from pydicom.Dataset or dict-like."""
    if meta is None:
        return default
    if hasattr(meta, key):
        return getattr(meta, key, default)
    if isinstance(meta, dict):
        return meta.get(key, default)
    return default


def _parse_dicom_time(s, include_ms=False) -> Optional[time]:
    """
    DICOM TM can be:
      HH
      HHMM
      HHMMSS
      with optional .ffffff
    Sometimes stored without leading zero (e.g. '93015' for 09:30:15).
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None

    s = str(s).strip()
    if not s or s.lower() == "nan":
        return None

    if ":" in s:
        parsed = pd.to_datetime(s, errors="coerce")
        if pd.notna(parsed):
            parsed_time = parsed.time()
            if not include_ms:
                parsed_time = parsed_time.replace(microsecond=0)
            if (
                parsed_time.hour
                == parsed_time.minute
                == parsed_time.second
                == parsed_time.microsecond
                == 0
            ):
                return None
            return parsed_time

    # Split fractional part
    if "." in s:
        main, frac = s.split(".", 1)
        frac = (frac + "000000")[:6]  # pad/truncate to microseconds
    else:
        main, frac = s, "000000"

    main = main.strip()

    if len(main) >= 8:  # is datetime, keep time only
        main = main[8:14]
    if not main.isdigit():
        return None
    if main == 0:
        logger.debug("zero detected")
        return None

    # If odd length (e.g. '93015'), assume missing leading zero -> left pad
    if len(main) % 2 == 1:
        main = "0" + main

    # Interpret based on number of components provided
    if len(main) <= 2:  # HH
        main = main.zfill(2) + "0000"
    elif len(main) <= 4:  # HHMM
        main = main.zfill(4) + "00"
    else:  # HHMMSS (or longer -> take first 6)
        main = main.zfill(6)[:6]

    try:
        hh = int(main[0:2])
        mm = int(main[2:4])
        ss = int(main[4:6])
        if ss == 60:
            ss = 0
            mm += 1
        if include_ms:
            us = int(frac)
        else:
            us = 0
        if hh == mm == ss == us == 0:
            return None
        return time(hour=hh, minute=mm, second=ss, microsecond=us)
    except Exception:
        return None


def earliest_acquisition_datetime(tm_or_list) -> Optional[time]:
    if tm_or_list is None or (isinstance(tm_or_list, float) and pd.isna(tm_or_list)):
        return None

    # If string, try literal_eval (safe) to recover list or scalar
    if isinstance(tm_or_list, str):
        s = tm_or_list.strip()
        try:
            tm_or_list = ast.literal_eval(s)
        except Exception:
            # not a literal (e.g. "093015.2"), keep as-is
            pass

    # If list/tuple → parse each and take earliest
    if isinstance(tm_or_list, (list, tuple)):
        ts = [_parse_dicom_time(x) for x in tm_or_list]
        ts = [t for t in ts if t is not None]
        return min(ts) if ts else None

    # Scalar case
    return _parse_dicom_time(tm_or_list)


def _normalize_col_name(col_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", col_name.lower())


def _iter_time_tokens(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, (list, tuple)):
        return list(value)

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            return [raw]
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        return [parsed]

    return [value]


def _is_clock_like_token(token) -> bool:
    if token is None or (isinstance(token, float) and pd.isna(token)):
        return False
    if isinstance(token, (time, pd.Timestamp)):
        return True

    s = str(token).strip()
    if not s or s.lower() in {"none", "nan", "nat"}:
        return False

    # Accept explicit separators (e.g., 12:30:45, 2024-01-01 12:30:45).
    if ":" in s:
        return pd.notna(pd.to_datetime(s, errors="coerce"))

    main = s.split(".", 1)[0].strip()
    if not main.isdigit():
        return False

    # Keep fallback strict to avoid misclassifying durations like 500 or 1200.
    if len(main) not in {6, 14}:
        return False

    return _parse_dicom_time(s) is not None


def _has_clock_time_content(series: pd.Series, n_samples: int = 30) -> bool:
    sample = series.dropna().head(n_samples)
    if sample.empty:
        return False

    matches = sample.apply(
        lambda value: any(_is_clock_like_token(token) for token in _iter_time_tokens(value))
    )
    return bool(matches.mean() >= 0.5)


def _select_time_columns(df: pd.DataFrame) -> list[str]:
    selected = []
    for col in df.columns:
        if "time" not in col.lower():
            continue

        normalized = _normalize_col_name(col)

        if normalized in _CLOCK_TIME_NAME_ALLOWLIST or normalized.endswith("datetime"):
            selected.append(col)
            continue

        if any(token in normalized for token in _DURATION_TIME_NAME_TOKENS):
            continue

        if _has_clock_time_content(df[col]):
            selected.append(col)

    return selected


def to_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert time columns to python datetime.time.
    """
    time_cols = _select_time_columns(df)

    logger.info("Detected real-time columns: %s", time_cols)

    for c in time_cols:
        df[c] = df[c].apply(earliest_acquisition_datetime)

    return df


_ISO_REGEX = re.compile(r"^\d{4}([-/]?\d{2}){2}$")


def _is_iso_like(series, n_samples=20):
    s = series.dropna().astype(str).head(n_samples)
    if s.empty:
        return False
    return s.map(lambda x: bool(_ISO_REGEX.match(x))).all()


def _infer_dayfirst(series, n_samples=50):
    """
    Infer whether dayfirst=True or False is more plausible for a date-like Series.
    Returns True or False.
    """
    s = series.dropna().astype(str).head(n_samples)

    if s.empty:
        return True  # default fallback (EU-style)

    parsed_dayfirst = pd.to_datetime(s, dayfirst=True, errors="coerce")
    parsed_monthfirst = pd.to_datetime(s, dayfirst=False, errors="coerce")

    # count valid parses
    n_df = parsed_dayfirst.notna().sum()
    n_mf = parsed_monthfirst.notna().sum()

    # if one clearly wins, pick it
    if n_df != n_mf:
        return n_df > n_mf

    # tie-breaker: check impossible months/days
    # (e.g. day > 12 suggests dayfirst)
    day_vals = s.str.extract(r"(\d{1,2})")[0].astype(float)
    month_vals = s.str.extract(r"\d{1,2}[/-](\d{1,2})")[0].astype(float)

    if (day_vals > 12).any():
        return True
    if (month_vals > 12).any():
        return False

    # fallback
    return True


def to_dates(df):
    date_cols = [c for c in df.columns if "date" in c.lower()]
    logger.info("Detected date columns: %s", date_cols)

    for c in date_cols:
        if _is_iso_like(df[c]):
            # ISO dates: dayfirst irrelevant → silence warnings
            dayfirst = False
            logger.info("%s: ISO format detected -> dayfirst=False", c)
            df[c] = pd.to_datetime(
                df[c], dayfirst=False, format="%Y%m%d", errors="coerce"
            )
        else:
            dayfirst = _infer_dayfirst(df[c])
            logger.info("%s: dayfirst = %s", c, dayfirst)
            df[c] = pd.to_datetime(
                df[c], dayfirst=dayfirst, format="%Y%m%d", errors="coerce"
            )

    return df
