"""Behavior shared by CT/MR rules and intentional modality differences."""

import pandas as pd
import pytest

from imperandi.curation.ct.curate import detect_ct_features, detect_ct_phase
from imperandi.curation.mri import curate as mr


@pytest.mark.parametrize(
    "text,ct_phase,mr_phase",
    [
        ("sans injection", "NATIVE", "NATIVE"),
        ("native", "NATIVE", "NATIVE"),
        ("pre contrast", "NATIVE", "NATIVE"),
        ("preinj", "NATIVE", None),
        ("pregad", "OTHER", "NATIVE"),
        ("sansgado", "OTHER", "NATIVE"),
        ("non-injected", "OTHER", "NATIVE"),
        ("arterial", "ARTERIAL", "ARTERIAL"),
        ("artérielle", "ARTERIAL", "ARTERIAL"),
        ("angio", "ARTERIAL", None),
        ("cta", "ARTERIAL", None),
        ("aortic", "ARTERIAL", None),
        ("portal", "PORTAL_VENOUS", "PORTAL_VENOUS"),
        ("portale", "OTHER", None),
        ("port", "OTHER", "PORTAL_VENOUS"),
        ("portalvenous", "PORTAL_VENOUS", "PORTAL_VENOUS"),
        ("veineux", "PORTAL_VENOUS", "PORTAL_VENOUS"),
        ("parenchymateux", "PORTAL_VENOUS", None),
        ("equilibrium", "DELAYED", "DELAYED"),
        ("tard", "OTHER", "DELAYED"),
        ("20 sec", "OTHER", "ARTERIAL"),
        ("35 seconds", "OTHER", "ARTERIAL"),
        ("36 sec", "OTHER", None),
        ("60 sec", "OTHER", "PORTAL_VENOUS"),
        ("1 min 30 sec", "OTHER", "PORTAL_VENOUS"),
        ("1 min 31 sec", "OTHER", "ARTERIAL"),
        ("3min", "DELAYED", "DELAYED"),
        ("10min", "DELAYED", "DELAYED"),
        ("6min", "OTHER", "DELAYED"),
        ("15min", "OTHER", "DELAYED"),
        ("tardif 2 min", "DELAYED", None),
        ("tardif 16 min", "DELAYED", None),
        ("tardif 20 min", "DELAYED", "HEPATOBILIARY"),
        ("2h", "OTHER", "HEPATOBILIARY"),
        ("arterial portal", "ARTERIAL", "PORTAL_VENOUS"),
        ("native arterial portal tardif", "NATIVE", "NATIVE"),
        ("ph1", "OTHER", None),
        ("portalized", "OTHER", None),
        ("xportal", "OTHER", None),
        ("portal2", "OTHER", None),
        ("_PORTAL-W", "PORTAL_VENOUS", "PORTAL_VENOUS"),
        (None, "OTHER", None),
    ],
)
def test_phase_keywords_timing_and_precedence(text, ct_phase, mr_phase):
    row = pd.Series({"SeriesDescription": text})

    assert detect_ct_phase(row)[0] == ct_phase
    assert mr.detect_explicit_phase_from_text(row)[0] == mr_phase


def test_phase_provenance_and_text_column_policies(monkeypatch):
    monkeypatch.setattr(mr, "TEXT_COLS_DEFAULT", ["SeriesDescription", "ProtocolName"])
    row = pd.Series({"SeriesDescription": "arterial", "ProtocolName": "native"})

    assert detect_ct_phase(row) == (
        "NATIVE",
        "matched CT native/non-injected keyword",
        "high",
    )
    assert mr.detect_explicit_phase_from_text(row) == (
        "ARTERIAL",
        "matched explicit arterial keyword",
        "explicit",
        "explicit_text",
    )


@pytest.mark.parametrize(
    "text,ct_localizer,mr_localizer",
    [
        ("Localizer", True, True),
        ("scout", True, True),
        ("survey", True, True),
        ("repérage", False, False),
        ("repèrage", True, True),
        ("topogram", True, True),
        ("surview", True, False),
        ("loc", False, True),
        ("phantom", False, True),
        ("test", False, True),
        ("scouting", False, False),
    ],
)
def test_localizer_vocabulary(text, ct_localizer, mr_localizer):
    row = pd.Series({"SeriesDescription": text})

    assert detect_ct_features(row)["is_localizer"] == ct_localizer
    assert (mr.detect_mri_sequence(row)[0] == "LOCALIZER") == mr_localizer


@pytest.mark.parametrize("text", ["ax", "AXIAL", "tra", "trans", "transverse"])
def test_shared_axial_vocabulary(text):
    row = pd.Series({"SeriesDescription": text})

    assert detect_ct_features(row)["is_axial"]
    assert mr.detect_plane(row) == "AXIAL"
