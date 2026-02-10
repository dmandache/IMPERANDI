from __future__ import annotations

import pandas as pd
import logging
from datetime import time
from typing import Optional, Any
import re

logger = logging.getLogger(__name__)


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
    import ast

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


def to_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert time columns to python datetime.time.
    """
    time_cols = [c for c in df.columns if "time" in c.lower()]

    logger.info("Detected time columns: %s", time_cols)

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
