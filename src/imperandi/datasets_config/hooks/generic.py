import re

def standardize_patient_key(x: str | None) -> str | None:
    """
    Extract numeric patient ID from strings like:
      - Eg: PatientName in IRCAD Dataset : "liver_13^patient" -> "13"
    Returns original if no integer is found.
    """
    if not x:
        return None

    x = str(x)

    m = re.search(r"\b(\d+)\b", x)
    if not m:
        return 

    return m.group(1)
