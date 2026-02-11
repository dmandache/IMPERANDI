"""Dataset hook implementations used by generic dataset manifests.

The definitions in this module are part of the Imperandi codebase and are
intended to be reused by higher-level workflows and CLI entry points.
"""

import re
from typing import Optional


def standardize_patient_key(x: Optional[str]) -> Optional[str]:
    """
    Extract numeric parts from a string, strip leading zeros,
    join with '-', and fall back to original if no numbers are found.

    Examples:
      "liver_13^patient" -> "13"
      "patient_0012_030" -> "12-30"
      "no_numbers_here"  -> "no_numbers_here"
    """
    if x is None:
        return None

    s = str(x)
    nums = re.findall(r"\d+", s)

    if not nums:
        return s

    cleaned = [n.lstrip("0") or "0" for n in nums]
    return "-".join(cleaned)
