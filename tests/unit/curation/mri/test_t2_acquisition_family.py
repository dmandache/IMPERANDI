"""T2 shot-family rules across vendor names and ordered metadata fields."""

import pandas as pd
import pytest

from imperandi.curation.mri import curate as mr


@pytest.mark.parametrize(
    "description",
    [
        "HASTE",
        "SSFSE",
        "eSSFSE",
        "SS-TSE",
        "SSTSE",
        "ss_tse_ax",
        "SSH",
        "single-shot FSE",
        "single-shot TSE",
        "single-shot FSE/TSE",
        "single shot fast spin echo",
        "single shot turbo spin echo",
        "TSE single shot",
        "T2 single-shot",
        "T2 HASTE FSE",
    ],
)
def test_single_shot_protocols(description):
    row = pd.Series({"SeriesDescription": description})
    assert mr.detect_mri_sequence(row)[0] == "T2"
    assert mr.detect_t2_acquisition_family(row) == "single_shot"


@pytest.mark.parametrize(
    "description",
    [
        "TSE",
        "FSE",
        "FRFSE",
        "fast spin echo",
        "turbo spin echo",
        "conventional SE",
        "conventional spin echo",
        "BLADE",
        "fBLADE",
        "PROPELLER",
        "PROP",
        "MultiVane",
        "MultiVane XD",
        "SPACE",
        "CUBE",
        "VISTA",
        "T2 SE",
        "T2 spin echo",
        "T2 multi-shot",
    ],
)
def test_multi_shot_protocols(description):
    row = pd.Series({"SeriesDescription": description})
    assert mr.detect_mri_sequence(row)[0] == "T2"
    assert mr.detect_t2_acquisition_family(row) == "multi_shot"


@pytest.mark.parametrize("separator", [" ", "_", "-", ".", "+", "/", ""])
@pytest.mark.parametrize(
    "words,family",
    [
        ("SS TSE", "single_shot"),
        ("SS FSE", "single_shot"),
        ("single shot FSE", "single_shot"),
        ("single shot turbo spin echo", "single_shot"),
        ("conventional SE", "multi_shot"),
    ],
)
def test_case_and_protocol_separators(words, family, separator):
    description = separator.join(words.upper().split())
    assert (
        mr.detect_t2_acquisition_family(pd.Series({"SeriesDescription": description}))
        == family
    )


@pytest.mark.parametrize(
    "description",
    [
        None,
        "",
        "T2",
        "T2 FS SENSE TE 90 PACE breath hold",
        "T2 MRCP radial",
        "T2 EPI",
        "T2 multishotting singleshotting",
        "T2 hastened xhaste haste2",
        "T2 fse2 xtse tset",
        "T2 spacecraft cubed vistas",
        "T2 éhaste hasteé",
        "single-shot EPI",
        "T1 FSE",
        "T1 SPACE",
        "T1 CUBE",
        "T1 VISTA",
        "T1 MultiVane",
        "DWI SS-TSE",
        "ADC PROPELLER",
        "localizer HASTE",
    ],
)
def test_unresolved_and_non_t2_acquisitions_are_empty(description):
    assert (
        mr.detect_t2_acquisition_family(pd.Series({"SeriesDescription": description}))
        is None
    )


def test_numeric_sequence_fallback_does_not_imply_a_shot_family():
    row = pd.Series({"RepetitionTime": 3000, "EchoTime": 90})
    assert mr.detect_mri_sequence(row)[0] == "T2"
    assert mr.detect_t2_acquisition_family(row) is None


@pytest.mark.parametrize(
    "series,protocol,family",
    [
        ("T2 HASTE", "T2 FSE", "single_shot"),
        ("T2 FSE", "T2 HASTE", "multi_shot"),
        ("T2 FS", "HASTE", "single_shot"),
        (None, ["T2", "SSFSE"], "single_shot"),
        ("T1 VIBE", "T2 HASTE", None),
        ("T2 single", "shot FSE", "multi_shot"),
        ("T2 single", "shot", None),
    ],
)
def test_first_field_with_family_evidence_controls_classification(
    monkeypatch, series, protocol, family
):
    monkeypatch.setattr(mr, "TEXT_COLS_DEFAULT", ["SeriesDescription", "ProtocolName"])
    row = pd.Series({"SeriesDescription": series, "ProtocolName": protocol})
    assert mr.detect_t2_acquisition_family(row) == family


def test_unconfigured_fields_are_ignored():
    row = pd.Series({"SeriesDescription": "T2", "ProtocolName": "HASTE"})
    assert mr.detect_t2_acquisition_family(row) is None


def test_family_survives_curation_and_selected_output():
    frame = pd.DataFrame(
        {
            "patient_key": ["p1", "p2", "p3", "p4"],
            "study_id": ["s1"] * 4,
            "series_id": ["series1"] * 4,
            "volume_id": ["v1", "v2", "v3", "v4"],
            "Modality": ["MR"] * 4,
            "SeriesDescription": ["T2 HASTE", "T2 CUBE", "T2", "T1 FSE"],
            "ProtocolName": ["T2 VISTA", "T2 SS-TSE", "T2", "T1 SPACE"],
        }
    )
    original = frame.copy(deep=True)
    column = "mri_t2_acquisition_family"
    results = mr.curate_mri(frame)
    assert results["curated"][column].iloc[:2].tolist() == ["single_shot", "multi_shot"]
    assert results["curated"][column].iloc[2:].isna().all()
    assert results["t2_candidates"][column].iloc[:2].tolist() == [
        "single_shot",
        "multi_shot",
    ]
    selected = results["selected_long"].set_index("patient_key")
    assert selected.loc["p1", column] == "single_shot"
    assert selected.loc["p2", column] == "multi_shot"
    assert pd.isna(selected.loc["p3", column])

    options = {
        "strategies": [{"type": "rules"}],
        "text_columns": {"MR": ["ProtocolName", "SeriesDescription"]},
    }
    reordered = mr.annotate_mri(frame, phase_curation=options)
    assert reordered[column].iloc[:2].tolist() == ["multi_shot", "single_shot"]
    pd.testing.assert_frame_equal(frame, original)


def test_custom_text_columns_support_family_detection():
    frame = pd.DataFrame(
        {
            "patient_key": ["p1"],
            "study_id": ["s1"],
            "series_id": ["series1"],
            "volume_id": ["v1"],
            "SeriesDescription": ["T2 HASTE"],
            "CustomText": ["T2 VISTA"],
        }
    )
    options = {
        "strategies": [{"type": "rules"}],
        "text_columns": {"MR": ["MissingTag", "CustomText"]},
    }
    out = mr.annotate_mri(frame, phase_curation=options)
    assert out.loc[0, "mri_t2_acquisition_family"] == "multi_shot"


@pytest.mark.parametrize(
    "family,expected_score",
    [("multi_shot", 70), ("single_shot", 60), (None, 50), (pd.NA, 50)],
)
def test_family_score_does_not_stack_legacy_protocol_bonuses(family, expected_score):
    row = pd.Series(
        {
            "plane": "AXIAL",
            "mri_t2_acquisition_family": family,
            "is_t2_motion_robust": True,
            "is_t2_haste_ssfse": True,
            "is_t2_tse_fse": True,
        }
    )
    assert mr.score_t2(row) == expected_score


def test_multi_shot_selection_beats_overlapping_single_shot_protocol_names():
    frame = pd.DataFrame(
        {
            "patient_key": ["p1", "p1"],
            "study_id": ["s1", "s1"],
            "series_id": ["single", "multi"],
            "volume_id": ["single", "multi"],
            "SeriesDescription": ["AX T2 HASTE FSE", "AX T2 SPACE"],
        }
    )
    results = mr.curate_mri(frame)
    assert results["curated"].t2_score.tolist() == [60, 70]
    assert results["selected_long"].volume_id.tolist() == ["multi"]
