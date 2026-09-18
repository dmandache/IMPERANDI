"""Shared CT/MR policy, vocabulary, boundaries, and modality extensions."""

import re

import pandas as pd
import pytest

from imperandi.curation import rules
from imperandi.curation.ct import rules as ct_rules
from imperandi.curation.ct.curate import curate_ct, detect_ct_features, detect_ct_phase
from imperandi.curation.mri import curate as mr

SEPARATORS = ["", " ", "_", ".", "+", "-", "/", " _-/+ "]
SHARED_PHASE_NAMES = {
    "NATIVE": [
        "native",
        "natif",
        "sans",
        "sans injection",
        "sans iv",
        "non injecté",
        "non injectée",
        "non injected",
        "non contrast",
        "non contrastée",
        "without contrast",
        "unenhanced",
        "non enhanced",
        "preinj",
        "pré injection",
        "pre contrast",
        "avant injection",
        "ss iv",
        "si",
        "blanc",
        "C-",
    ],
    "ARTERIAL": [
        "art",
        "arterial",
        "arteriel",
        "artériel",
        "arterielle",
        "artérielle",
        "artery",
        "aorte",
        "aortic",
        "aortique",
        "hepatic artery",
        "late arterial",
        "early arterial",
        "multi art",
    ],
    "PORTAL_VENOUS": [
        "port",
        "portal",
        "portale",
        "porte",
        "porto",
        "portovenous",
        "portal venous",
        "vein",
        "veine",
        "venous",
        "veneux",
        "veineux",
        "veineuse",
        "vp",
        "pv",
        "parenchymateux",
        "parenchymal",
        "phase p",
    ],
    "DELAYED": [
        "tard",
        "tardif",
        "tardive",
        "delay",
        "delayed",
        "delai",
        "délai",
        "late",
        "equilibrium",
        "equilibre",
        "équilibre",
        "eq",
        "interstitiel",
        "interstitial",
        "phase d",
    ],
}


def assert_phases(text, ct_phase, mr_phase):
    row = pd.Series({"SeriesDescription": text})
    assert detect_ct_phase(row)[0] == ct_phase
    assert mr.detect_explicit_phase_from_text(row)[0] == mr_phase


@pytest.mark.parametrize("separator", SEPARATORS)
@pytest.mark.parametrize(
    "phase,words",
    [
        (phase, words)
        for phase, aliases in SHARED_PHASE_NAMES.items()
        for words in aliases
    ],
)
def test_shared_phase_vocabulary_across_separators(phase, words, separator):
    assert_phases(separator.join(words.split()).upper(), phase, phase)


@pytest.mark.parametrize(
    "text,phase",
    [
        ("19 sec", None),
        ("20 sec", "ARTERIAL"),
        ("35 seconds", "ARTERIAL"),
        ("36 sec", None),
        ("59 sec", None),
        ("60 sec", "PORTAL_VENOUS"),
        ("90 seconds", "PORTAL_VENOUS"),
        ("91 sec", None),
        ("179 seconds", None),
        ("180 seconds", "DELAYED"),
        ("190sec", "DELAYED"),
        ("235sec", "DELAYED"),
        ("900 seconds", "DELAYED"),
        ("901 seconds", None),
        ("0:20 min", "ARTERIAL"),
        ("0:35 min", "ARTERIAL"),
        ("0:36 min", None),
        ("0.5 min", "ARTERIAL"),
        ("0,5 min", "ARTERIAL"),
        ("0.20 min", None),
        ("0.35 min", "ARTERIAL"),
        ("1 min", "PORTAL_VENOUS"),
        ("1:30 min", "PORTAL_VENOUS"),
        ("1 min 30 sec", "PORTAL_VENOUS"),
        ("1.5 min", "PORTAL_VENOUS"),
        ("1 min 31 sec", None),
        ("1:31 min", None),
        ("1.6 min", None),
        ("2 min 30 sec", None),
        ("2:30 min", None),
        ("3 min", "DELAYED"),
        ("6 min", "DELAYED"),
        ("10 min", "DELAYED"),
        ("15 mn", "DELAYED"),
        ("15 min 1 sec", None),
        ("16 min", None),
        ("0 h 5 min", "DELAYED"),
        ("tardif 2 min", None),
        ("tardif 16 min", None),
        ("20 sec / 70 sec", None),
        ("20 sec / 91 sec", None),
        ("20 sec / 0.5 min", "ARTERIAL"),
        ("native arterial portal", "NATIVE"),
        ("arterial portal", "PORTAL_VENOUS"),
        ("late arterial", "ARTERIAL"),
        ("portal 20 sec", "PORTAL_VENOUS"),
        ("arterial 70 sec", "ARTERIAL"),
        ("20 sec portal", "PORTAL_VENOUS"),
    ],
)
def test_complete_durations_and_phase_precedence(text, phase):
    assert_phases(text, phase or "OTHER", phase)


@pytest.mark.parametrize("separator", SEPARATORS)
@pytest.mark.parametrize(
    "parts,phase",
    [
        (["20", "sec"], "ARTERIAL"),
        (["60", "seconds"], "PORTAL_VENOUS"),
        (["1", "min", "30", "sec"], "PORTAL_VENOUS"),
        (["1", "min", "31", "sec"], None),
        (["10", "minutes"], "DELAYED"),
    ],
)
def test_duration_separators(parts, phase, separator):
    assert_phases(separator.join(parts), phase or "OTHER", phase)


@pytest.mark.parametrize("separator", SEPARATORS[1:])
def test_durations_after_protocol_separators(separator):
    assert_phases(f"abdomen{separator}30sec", "ARTERIAL", "ARTERIAL")
    assert_phases(f"abdomen{separator}30sec{separator}01", "ARTERIAL", "ARTERIAL")
    assert_phases(f"35sec{separator}5min", "OTHER", None)


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "ph1",
        "phase 2",
        "portalized",
        "xportal",
        "portal2",
        "éportal",
        "portalé",
        "préface",
        "native2",
        "arterialized",
        "120sec",
        "160sec",
        "1200min",
        "11920min",
        "1:60 min",
        "5.5.5min",
        "31secxyz",
    ],
)
def test_unknown_terms_and_numeric_substrings_do_not_become_phases(text):
    assert_phases(text, "OTHER", None)


@pytest.mark.parametrize(
    "text,ct_phase,mr_phase",
    [
        ("angio", "ARTERIAL", None),
        ("CTA", "ARTERIAL", None),
        ("CT-A", "ARTERIAL", None),
        ("angiography", "ARTERIAL", None),
        ("native CTA", "NATIVE", "NATIVE"),
        ("pregad", "OTHER", "NATIVE"),
        ("sansgado", "OTHER", "NATIVE"),
        ("masque", "OTHER", "NATIVE"),
        ("mask", "OTHER", "NATIVE"),
        ("hepato-biliary", "OTHER", "HEPATOBILIARY"),
        ("hepatobiliaire", "OTHER", "HEPATOBILIARY"),
        ("HBP", "OTHER", "HEPATOBILIARY"),
        ("20 min", "OTHER", "HEPATOBILIARY"),
        ("120 min", "OTHER", "HEPATOBILIARY"),
        ("2h", "OTHER", "HEPATOBILIARY"),
        ("2 h 30 min", "OTHER", "HEPATOBILIARY"),
        ("150 min", "OTHER", "HEPATOBILIARY"),
        ("tardif 20 min", "OTHER", "HEPATOBILIARY"),
        ("tardif 2h", "OTHER", "HEPATOBILIARY"),
        ("3h", "OTHER", None),
        ("20 min 1 sec", "OTHER", None),
    ],
)
def test_modality_extensions(text, ct_phase, mr_phase):
    assert_phases(text, ct_phase, mr_phase)


def test_phase_provenance_and_text_column_precedence(monkeypatch):
    monkeypatch.setattr(mr, "TEXT_COLS_DEFAULT", ["SeriesDescription", "ProtocolName"])
    row = pd.Series({"SeriesDescription": "arterial", "ProtocolName": "native"})

    assert detect_ct_phase(row) == (
        "ARTERIAL",
        "matched CT arterial keyword",
        "high",
    )
    assert mr.detect_explicit_phase_from_text(row) == (
        "ARTERIAL",
        "matched explicit arterial keyword",
        "explicit",
        "explicit_text",
    )
    row["SeriesDescription"] = "abdomen"
    assert detect_ct_phase(row)[0] == "NATIVE"
    assert mr.detect_explicit_phase_from_text(row)[0] == "NATIVE"


@pytest.mark.parametrize(
    "text",
    [
        "Localizer",
        "localiser",
        "scout",
        "survey",
        "repérage",
        "repèrage",
        "topogram",
        "surview",
        "loc",
        "phantom",
        "test",
        "cal_body",
    ],
)
def test_shared_localizers(text):
    row = pd.Series({"SeriesDescription": text})
    assert detect_ct_features(row)["is_localizer"]
    assert mr.detect_mri_sequence(row)[0] == "LOCALIZER"


@pytest.mark.parametrize("separator", SEPARATORS)
@pytest.mark.parametrize(
    "words",
    [
        "key images",
        "screen save",
        "processed images",
        "multiplanar reconstruction",
        "volume rendering",
        "volume rendered",
        "dose report",
        "soustraction",
        "MIP",
        "MPR",
    ],
)
def test_shared_non_diagnostic_rules(words, separator):
    text = separator.join(words.split())
    assert detect_ct_features(pd.Series({"SeriesDescription": text}))[
        "is_derived_low_value"
    ]
    assert any(
        re.search(pattern, text)
        for pattern in [
            mr.rules.RX_KEY_IMAGES,
            mr.rules.RX_SUBTRACTION,
            mr.rules.RX_MIP_MPR,
            mr.rules.RX_QUANT_OR_REPORT,
        ]
    )


@pytest.mark.parametrize(
    "text,plane",
    [
        ("ax", "AXIAL"),
        ("AXIAL", "AXIAL"),
        ("tra", "AXIAL"),
        ("trans", "AXIAL"),
        ("transverse", "AXIAL"),
        ("coronale", "CORONAL"),
        ("sagittale", "SAGITTAL"),
    ],
)
def test_shared_planes_override_square_matrix_assumption(text, plane):
    row = pd.Series({"SeriesDescription": text, "Rows": 512, "Columns": 512})
    assert detect_ct_features(row)["is_axial"] == (plane == "AXIAL")
    assert mr.detect_plane(row) == plane


@pytest.mark.parametrize(
    "image_type,expected",
    [
        ("ORIGINAL\\PRIMARY", True),
        (["ORIGINAL", "PRIMARY"], True),
        ("NONORIGINAL PRIMARY", False),
        ("ORIGINAL NOTPRIMARY", False),
    ],
)
def test_ct_image_type_uses_whole_tokens(image_type, expected):
    assert (
        detect_ct_features(pd.Series({"ImageType": image_type}))["is_original"]
        == expected
    )


@pytest.mark.parametrize("separator", SEPARATORS)
@pytest.mark.parametrize(
    "words,sequence",
    [
        ("apparent diffusion coefficient", "DWI"),
        ("dw epi", "DWI"),
        ("twist vibe", "T1"),
        ("lava flex", "T1"),
        ("m dixon", "T1"),
        ("q dixon", "T1"),
        ("t1 weighted", "T1"),
        ("3d t1", "T1"),
        ("post contrast", "T1"),
        ("t2 weighted", "T2"),
    ],
)
def test_mr_sequence_separators(words, sequence, separator):
    row = pd.Series({"SeriesDescription": separator.join(words.split())})
    assert mr.detect_mri_sequence(row)[0] == sequence


@pytest.mark.parametrize("separator", SEPARATORS)
@pytest.mark.parametrize(
    "words,component",
    [
        ("in phase", "IN_PHASE"),
        ("out phase", "OPPOSED_PHASE"),
        ("fat fraction", "FAT_FRACTION"),
        ("r2 star", "R2STAR"),
    ],
)
def test_mr_component_separators(words, component, separator):
    row = pd.Series({"SeriesDescription": "DIXON " + separator.join(words.split())})
    assert mr.detect_dixon_component(row)[0] == component


@pytest.mark.parametrize("separator", SEPARATORS)
@pytest.mark.parametrize("arterial", ["art", "arterial", "artériel", "artério"])
def test_mr_dynamic_profile_separators(arterial, separator):
    row = pd.Series({"SeriesDescription": f"T1 {arterial}{separator}portal"})
    assert mr.text_matches_art_port(row)


def test_ct_selection_uses_strengthened_shared_rules():
    descriptions = [
        "Abdomen non-injected",
        "Abdomen 30_sec",
        "Abdomen portale",
        "Abdomen 6-min",
        "Abdomen portal MIP",
        "repérage",
        "key_images",
    ]
    frame = pd.DataFrame(
        [
            {
                "patient_key": "p1",
                "study_id": "s1",
                "volume_id": str(i),
                "SeriesDescription": text,
                "Rows": 512,
                "Columns": 512,
                "ImageType": "ORIGINAL PRIMARY",
                "n_files": 120,
            }
            for i, text in enumerate(descriptions)
        ]
    )
    result = curate_ct(frame)
    assert set(result["selected_long"]["volume_id"]) == {"0", "1", "2", "3", "6"}
    assert set(result["selected_long"]["selection_slot"]) == {
        "CT_NATIVE",
        "CT_ARTERIAL",
        "CT_PORTAL_VENOUS",
        "CT_DELAYED",
        "CT_OTHER",
    }


def test_shared_phase_selection_priorities():
    # The common phases must have the same ranking in both modalities.
    for label in rules.PHASE_PRIORITY:
        assert ct_rules.CT_PHASE_PRIORITY[label] == mr.rules.T1_PHASE_PRIORITY[label]
