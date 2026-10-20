"""Explicit evidence and canonical ordering drive dynamic phase templates."""

import pandas as pd
import pytest

from imperandi.curation.mri import mri_curation as mc
from imperandi.curation.phase_resolution import validate_phase_curation


def volumes(count=4, description="AX T1 VIBE dynamic"):
    return pd.DataFrame(
        [
            {
                "patient_key": "p",
                "study_id": "exam",
                "date": "2020-01-01",
                "series_id": "dyn",
                "volume_id": f"v{i}",
                "Modality": "MR",
                "SeriesDescription": description,
                "TemporalPositionIdentifier": i + 1,
                "volume_order_in_series": i + 1,
                "n_volumes_in_series": count,
                "ImageType": "ORIGINAL PRIMARY",
                "Rows": 128,
                "Columns": 128,
            }
            for i in range(count)
        ]
    )


def annotate(frame, config=None):
    return mc.annotate_mri(frame, phase_curation=config).set_index("volume_id")


@pytest.mark.parametrize(
    "column",
    [
        "AcquisitionDateTime",
        "FrameAcquisitionDateTime",
        "FrameReferenceDateTime",
        "AcquisitionTime",
        "ContrastBolusStartTime",
        "TriggerTime",
        "TemporalPositionIdentifier",
        "TemporalPositionIndex",
        "acquisition_datetime",
    ],
)
def test_raw_timing_metadata_cannot_change_canonical_volume_progression(column):
    frame = volumes()
    expected = annotate(frame)
    frame[column] = ["120400", "120115", "120025", "invalid"]

    result = annotate(frame.sample(frac=1, random_state=4)).loc[expected.index]

    pd.testing.assert_series_equal(result.phase, expected.phase)
    pd.testing.assert_series_equal(result.rule_phase, expected.rule_phase)
    assert "mri_dynamic_timing_source" not in result
    assert "mri_dynamic_timing_reliable" not in result


@pytest.mark.parametrize("setting", ["mri_dynamic", "phase_windows", "profiles"])
def test_manifest_cannot_configure_mri_profiles_or_delay_windows(setting):
    with pytest.raises(ValueError, match="built-in rules"):
        validate_phase_curation({"strategies": [{"type": "rules"}], setting: {}})
    assert "mri_dynamic" not in validate_phase_curation(None)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("PRE_CONTRAST", "NATIVE"),
        ("ARTERIAL", "ARTERIAL"),
        ("PORTAL_VENOUS", "PORTAL_VENOUS"),
        ("DELAYED", "DELAYED"),
    ],
)
def test_standard_dicom_phase_precedes_conflicting_text(raw, expected):
    frame = volumes(1, "AX T1 VIBE portal")
    frame["ContrastBolusAgentPhase"] = raw
    out = annotate(frame)
    assert out.loc["v0", "phase"] == expected
    assert out.loc["v0", "mri_perfusion_source"] == "dicom_phase"


def test_conflicting_standard_phase_is_not_replaced_by_order_or_native_fallback():
    frame = volumes(description="AX T1 VIBE pre/post dynamic Ph1")
    frame["ContrastBolusAgentPhase"] = pd.Series(
        [
            ["PRE_CONTRAST", "ARTERIAL"],
            None,
            None,
            None,
        ]
    )
    out = annotate(frame)
    assert out.loc["v0", "phase"] == "OTHER"
    assert out.loc["v0", "mri_perfusion_source"] == "dicom_phase_conflict"


@pytest.mark.parametrize("clock", ["202001011200", "20200101120025.000000+0000"])
def test_ambiguous_precision_or_unaligned_local_bolus_uses_generic_order(clock):
    frame = volumes()
    frame["AcquisitionDateTime"] = clock
    frame["ContrastBolusStartTime"] = "120000.000000"
    out = annotate(frame)
    assert out.phase.tolist() == ["NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED"]


def test_container_frame_clock_cannot_supply_distinct_volume_delays():
    frame = volumes()
    frame["FrameAcquisitionDateTime"] = "20200101120025.000000"
    frame["ContrastBolusStartTime"] = "120000.000000"
    out = annotate(frame)
    assert out.phase.tolist() == ["NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED"]


def test_missing_ph1_does_not_block_native_fallback_without_valid_native_context():
    frame = volumes(3)
    frame["SeriesDescription"] = [
        "AX T1 VIBE dynamic Ph2",
        "AX T1 VIBE dynamic Ph3",
        "AX T1 VIBE dynamic Ph4",
    ]
    out = annotate(frame)
    assert out.phase.tolist() == ["NATIVE", "ARTERIAL", "PORTAL_VENOUS"]


@pytest.mark.parametrize("native_count", [1, 2])
def test_ontology_natives_are_counted_once_before_generic_progression(native_count):
    frame = volumes(3 + native_count)
    frame["site_phase"] = ["pre"] * native_count + [None] * 3
    config = {
        "strategies": [
            {
                "type": "ontology",
                "columns": ["site_phase"],
                "mapping": {"pre": "NATIVE"},
            },
            {"type": "rules"},
        ]
    }
    out = annotate(frame, config)
    assert out.phase.tolist() == ["NATIVE"] * native_count + [
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]


def test_ontology_first_arterial_prevents_generic_native():
    frame = volumes()
    frame["site_phase"] = ["art", None, None, None]
    out = annotate(
        frame,
        {
            "strategies": [
                {
                    "type": "ontology",
                    "columns": ["site_phase"],
                    "mapping": {"art": "ARTERIAL"},
                },
                {"type": "rules"},
            ]
        },
    )
    assert out.phase.tolist() == ["ARTERIAL", "PORTAL_VENOUS", "DELAYED", "DELAYED"]


@pytest.mark.parametrize("source", ["dicom", "ontology"])
def test_succeeding_authoritative_portal_reserves_its_phase(source):
    frame = volumes()
    frame.loc[0, "SeriesDescription"] = "AX T1 VIBE pre"
    config = None
    if source == "dicom":
        frame["ContrastBolusAgentPhase"] = [None, None, None, "PORTAL_VENOUS"]
    else:
        frame["site_phase"] = [None, None, None, "portal"]
        config = {
            "strategies": [
                {
                    "type": "ontology",
                    "columns": ["site_phase"],
                    "mapping": {"portal": "PORTAL_VENOUS"},
                },
                {"type": "rules"},
            ]
        }

    out = annotate(frame, config)

    assert out.phase.tolist() == [
        "NATIVE",
        "ARTERIAL",
        "ARTERIAL",
        "PORTAL_VENOUS",
    ]
    assert (
        out.loc[["v1", "v2"], "mri_perfusion_source"]
        .eq("dynamic_explicit_anchor")
        .all()
    )


@pytest.mark.parametrize(
    "bad_evidence", [None, "position", "geometry", "clock", "phase_conflict"]
)
def test_phase_sharing_requires_compatible_supplied_acquisition_identity(bad_evidence):
    frame = volumes(2)
    frame["SeriesDescription"] = [
        "AX T1 mDIXON water portal",
        "AX T1 mDIXON fat dynamic",
    ]
    frame["series_id"] = ["water", "fat"]
    frame["n_volumes_in_series"] = 1
    frame["volume_order_in_series"] = 1
    frame["TemporalPositionIdentifier"] = 1
    frame["acquisition_id"] = "validated-acquisition"
    if bad_evidence == "position":
        frame.loc[1, "volume_order_in_series"] = 2
    elif bad_evidence == "geometry":
        frame["FrameOfReferenceUID"] = ["space1", "space2"]
    elif bad_evidence == "clock":
        frame["time"] = ["120000", "120100"]
    elif bad_evidence == "phase_conflict":
        frame["ContrastBolusAgentPhase"] = pd.Series(
            [None, ["PRE_CONTRAST", "ARTERIAL"]]
        )
    out = annotate(frame)
    assert out.loc["v1", "phase"] == (
        "PORTAL_VENOUS" if bad_evidence is None else "OTHER"
    )
    if bad_evidence is None:
        assert out.loc["v1", "mri_perfusion_source"] == "acquisition_phase"


def test_equal_time_alone_does_not_share_phase_between_components():
    frame = volumes(2)
    frame["SeriesDescription"] = [
        "AX T1 mDIXON water portal",
        "AX T1 mDIXON fat dynamic",
    ]
    frame["AcquisitionTime"] = "120000"
    out = annotate(frame)
    assert out.loc["v1", "phase"] == "OTHER"


def test_grasp_and_conventional_vibe_do_not_form_one_dynamic_run():
    frame = volumes()
    frame["SeriesDescription"] = ["AX T1 VIBE GRASP"] * 2 + ["AX T1 VIBE dynamic"] * 2
    out = annotate(frame)
    assert out.phase.eq("OTHER").all()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("AX T1 5min tardif", "DELAYED"),
        ("AX T1 tardif 2h", "DELAYED"),
        ("AX T1 2h tardif", "DELAYED"),
        ("AX T1 3h tardif", "DELAYED"),
        ("AX T1 2h", "HEPATOBILIARY"),
        ("AX T1 3h", "OTHER"),
    ],
)
def test_named_late_precedes_duration_and_duration_only_keeps_hepatobiliary_window(
    text, expected
):
    assert annotate(volumes(1, text)).loc["v0", "phase"] == expected


def test_long_dynamic_without_calibrated_timing_retains_order_fallback():
    frame = volumes(6)
    frame["AcquisitionTime"] = [
        "120025",
        "120030",
        "120032",
        "120115",
        "120400",
        "120410",
    ]
    out = annotate(frame)
    assert out.phase.tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
        "DELAYED",
        "DELAYED",
    ]
