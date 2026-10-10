"""First-acquisition native fallback for mixed pre/postcontrast groups."""

from __future__ import annotations

from collections.abc import Sequence
import re

import numpy as np
import pandas as pd

from . import common, rules
from .common import acquisition_time_seconds as _time_seconds


def _text_values(value):
    if isinstance(value, (list, tuple, set, np.ndarray)):
        for item in value:
            yield from _text_values(item)
    elif not common.is_missing(value):
        yield common.clean_text(value)


def _first_acquisition(rows: pd.DataFrame) -> list:
    """Use a complete chronology; ties are reconstructions of one acquisition."""
    for column in (
        "time",
        "AcquisitionTime",
        "acquisition_order",
        "TemporalPositionIdentifier",
        "AcquisitionNumber",
    ):
        if column not in rows:
            continue
        parser = (
            _time_seconds
            if column in {"time", "AcquisitionTime"}
            else common.safe_float
        )
        values = rows[column].map(parser)
        if values.notna().all() and values.nunique() > 1:
            return list(rows.index[values.eq(values.min())])
    # Volume rank is comparable only inside one series, not across series.
    series_col = next(
        (c for c in ("series_id", "SeriesInstanceUID") if c in rows), None
    )
    if series_col and rows[series_col].map(common.stable_text).nunique() == 1:
        if "volume_order_in_series" in rows:
            values = rows["volume_order_in_series"].map(common.safe_float)
            if values.notna().all() and values.nunique() > 1:
                return list(rows.index[values.eq(values.min())])
    return []


def infer_pre_post_native_acquisitions(
    rows: pd.DataFrame,
    *,
    text_columns: Sequence[str],
    group_columns: Sequence[str],
    phase_column: str,
    native: str = rules.RX_PHASE_NATIVE,
    post: str = rules.RX_PHASE_POST_CONTRAST,
    phase_rules: Sequence[rules.PhaseRule] = rules.PHASE_RULES,
    extra_patterns: Sequence[str] = (),
) -> list[tuple[object, str]]:
    """Rank full compatible groups, including explicitly labelled acquisitions.

    Callers preserve explicit/special phase assignments. Each mixed context must
    occur in one scalar field value, never across fields or list entries.
    """
    groups = {}
    for idx, row in rows.iterrows():
        compatibility = tuple(common.stable_text(row.get(c)) for c in group_columns)
        group = groups.setdefault(compatibility, {"indices": [], "context": None})
        group["indices"].append(idx)
        evidence = common.first_phase_text_column(
            row,
            text_columns,
            phase_rules=phase_rules,
            post=post,
            extra_patterns=extra_patterns,
        )
        for column in ([evidence[0]] if evidence else []):
            for text in _text_values(row.get(column)):
                if rules.has_pre_post_contrast_text(text, native, post):
                    if group["context"] is None:
                        group["context"] = (column, text)

    assignments = {}
    for data in groups.values():
        if data["context"] is None:
            continue
        column, text = data["context"]
        group = rows.loc[list(dict.fromkeys(data["indices"]))]
        first = _first_acquisition(group)
        if any(
            common.norm_label(group.loc[idx].get(phase_column))
            not in {"OTHER", "NATIVE"}
            for idx in first
        ):
            continue
        blocked = False
        for idx in first:
            # Pure postcontrast text in the selected evidence field blocks native
            # inference even when it cannot establish an exact perfusion phase.
            evidence = common.first_phase_text_column(
                group.loc[idx],
                text_columns,
                phase_rules=phase_rules,
                post=post,
                extra_patterns=extra_patterns,
            )
            blocked = blocked or any(
                not rules.has_pre_post_contrast_text(value, native, post)
                and rules.has_post_contrast_text(value, post)
                and not re.search(native, value)
                for field in ([evidence[0]] if evidence else [])
                for value in _text_values(group.loc[idx].get(field))
            )
        if not blocked:
            for idx in first:
                assignments.setdefault(
                    idx,
                    "inferred NATIVE from first acquisition in a mixed pre/postcontrast "
                    f"group; {column}={text!r}",
                )
    return list(assignments.items())
