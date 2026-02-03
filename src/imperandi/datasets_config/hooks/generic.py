import re

def standardize_patient_key(x: str | None) -> str | None:
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
