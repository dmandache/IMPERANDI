"""Native context takes precedence over vendor phase numbering."""

import pandas as pd
import pytest

from imperandi.curation.mri import mri_curation as mc


def frame(descriptions):
    return pd.DataFrame(
        [
            dict(
                patient_key="p",
                study_id="exam",
                date="2020-01-01",
                Modality="MR",
                series_id=f"s{i}",
                volume_id=f"v{i}",
                SeriesDescription=description,
                ImageType="ORIGINAL PRIMARY",
                acquisition_order=i,
                volume_order_in_series=1,
                n_volumes_in_series=1,
            )
            for i, description in enumerate(descriptions)
        ]
    )


def annotate(rows, config=None):
    return mc.annotate_mri(rows, phase_curation=config).set_index("volume_id")


@pytest.mark.parametrize(
    "numbers", [[1, 2, 3, 4], [2, 3, 4, 5], [None, 1, 2, 3], [None, 3, None, 7]]
)
@pytest.mark.parametrize("agent", ["", "gado", "primovist"])
def test_numbering_needs_native_context_before_becoming_strictly_postcontrast(numbers, agent):
    rows = frame(
        [
            f"{'ph'+str(number)+'/' if number is not None else ''}AX LAVA dyn 4 {agent}"
            for number in numbers
        ]
    )
    out = annotate(rows.sample(frac=1, random_state=4)).loc[[f"v{i}" for i in range(4)]]
    expected = (
        ["NATIVE", "OTHER", "NATIVE", "OTHER"]
        if numbers.count(None) > 1
        else ["NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED"]
    )
    assert out.phase.tolist() == expected


@pytest.mark.parametrize(
    "native,source",
    [
        ("AX LAVA pre", "explicit_text"),
        ("AX LAVA", "exam_context"),
        ("water: AX LAVA-flex", "exam_context"),
        ("AX LAVA gado", "exam_context"),
    ],
)
def test_compatible_native_is_resolved_before_numbered_dynamics(
    native, source
):
    rows = frame(
        [native, "Ph1/AX LAVA dyn gado", "Ph2/AX LAVA dyn gado", "Ph3/AX LAVA dyn gado"]
    )
    out = annotate(rows)
    assert out.phase.tolist() == ["NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED"]
    assert out.loc["v0", "mri_perfusion_source"] == source
    assert out.phase.eq("NATIVE").sum() == 1


def test_exam_level_injection_description_does_not_exclude_a_separate_native_candidate():
    rows = frame(
        ["AX LAVA", "Ph1/AX LAVA dyn gado", "Ph2/AX LAVA dyn gado", "Ph3/AX LAVA dyn gado"]
    )
    rows["StudyDescription"] = "IRM hépatique avec injection"
    out = annotate(
        rows,
        {
            "strategies": [{"type": "rules"}],
            "text_columns": {"MR": ["SeriesDescription", "StudyDescription"]},
        },
    )
    assert out.phase.tolist() == ["NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED"]
    assert out.loc["v0", "mri_perfusion_source"] == "exam_context"


@pytest.mark.parametrize(
    "other_native,expected", [("AX VIBE pre", "NATIVE"), ("AX VIBE", "OTHER")]
)
def test_unrelated_native_does_not_suppress_protocol_native_fallback(
    other_native, expected
):
    rows = frame(
        [
            other_native,
            "AX LAVA dyn gado",
            "Ph1/AX LAVA dyn gado",
            "Ph3/AX LAVA dyn gado",
        ]
    )
    out = annotate(rows)
    assert out.loc["v0", "phase"] == expected
    assert out.loc[["v1", "v2", "v3"], "phase"].tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
    ]


def test_two_protocol_groups_resolve_their_own_separate_native_candidates():
    rows = frame(
        [
            "AX LAVA",
            "Ph1/AX LAVA dyn gado",
            "Ph2/AX LAVA dyn gado",
            "Ph3/AX LAVA dyn gado",
            "AX DISCO",
            "Ph2/AX DISCO dynamic",
            "Ph3/AX DISCO dynamic",
            "Ph4/AX DISCO dynamic",
        ]
    )
    out = annotate(rows)
    assert out.phase.tolist() == ["NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED"] * 2
    assert out.loc[["v0", "v4"], "mri_perfusion_source"].eq("exam_context").all()


def test_late_separate_candidate_is_not_native_even_when_an_explicit_anchor_exists():
    rows = frame(["AX LAVA dyn gado", "AX LAVA dyn gado", "AX LAVA delayed", "AX LAVA"])
    out = annotate(rows)
    assert out.phase.tolist() == ["NATIVE", "ARTERIAL", "DELAYED", "OTHER"]


def test_explicit_native_without_clock_is_still_counted_once():
    rows = frame(["AX LAVA pre", "Ph2/AX LAVA dyn gado", "Ph3/AX LAVA dyn gado"])
    rows["acquisition_order"] = [None, 1, 2]
    rows["AcquisitionTime"] = [None, "120030", "120115"]
    out = annotate(rows)
    assert out.phase.tolist() == ["NATIVE", "ARTERIAL", "PORTAL_VENOUS"]


def test_named_phases_override_order_while_raw_bolus_clocks_do_not():
    rows = frame(
        ["Ph2/AX LAVA dyn gado", "Ph3/AX LAVA dyn gado", "Ph4/AX LAVA dyn gado"]
    )
    rows["AcquisitionTime"] = ["120025.000000", "120115.000000", "120400.000000"]
    rows["ContrastBolusStartTime"] = "120000.000000"
    out = annotate(rows)
    assert out.phase.tolist() == ["NATIVE", "ARTERIAL", "PORTAL_VENOUS"]
    rows = frame(["AX LAVA arterial", "Ph1/AX LAVA dyn gado", "Ph2/AX LAVA dyn gado"])
    assert annotate(rows).phase.tolist() == ["ARTERIAL", "PORTAL_VENOUS", "DELAYED"]


@pytest.mark.parametrize("post", ["post", "C+"])
def test_clear_postcontrast_acquisition_words_still_exclude_native_fallback(post):
    rows = frame([f"Ph{i}/AX LAVA dynamic {post}" for i in (2, 3, 4)])
    assert annotate(rows).phase.tolist() == ["ARTERIAL", "PORTAL_VENOUS", "DELAYED"]


def test_numbering_does_not_create_phases_without_comparable_acquisition_order():
    rows = frame(
        ["Ph1/AX LAVA dyn gado", "Ph2/AX LAVA dyn gado", "Ph3/AX LAVA dyn gado"]
    ).drop(columns="acquisition_order")
    assert annotate(rows).phase.eq("OTHER").all()


def test_separate_native_and_one_dynamic_with_known_delayed_anchor_meet_minimum():
    rows = frame(["AX LAVA", "Ph3/AX LAVA dynamic", "AX LAVA delayed"])
    out = annotate(rows)
    assert out.phase.tolist() == ["NATIVE", "ARTERIAL", "DELAYED"]
    assert out.loc["v0", "mri_perfusion_source"] == "exam_context"


@pytest.mark.parametrize(
    "phase_text,expected",
    [("arterial", "ARTERIAL"), ("portal", "PORTAL_VENOUS"), ("delayed", "DELAYED")],
)
def test_named_perfusion_phase_supports_native_without_postcontrast_marker(
    phase_text, expected
):
    out = annotate(frame(["AX LAVA", f"AX LAVA {phase_text}"]))
    assert out.phase.tolist() == ["NATIVE", expected]
    assert out.loc["v0", "mri_perfusion_source"] == "exam_context"
    assert out.loc["v0", "mri_perfusion_confidence"] == "fallback"
    assert out.loc["v1", "mri_perfusion_source"] == "explicit_text"


def test_several_named_perfusion_phases_support_one_native_fallback():
    rows = frame(["AX LAVA", "AX LAVA arterial", "AX LAVA portal", "AX LAVA delayed"])
    out = annotate(rows.sample(frac=1, random_state=5)).loc[rows.volume_id]
    assert out.phase.tolist() == ["NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED"]
    assert out.phase.eq("NATIVE").sum() == 1


@pytest.mark.parametrize("source", ["dicom", "ontology"])
@pytest.mark.parametrize("perfusion_description", ["AX LAVA", "Ph1/AX LAVA"])
def test_resolved_metadata_phase_supports_native_without_dynamic_text(
    source, perfusion_description
):
    rows = frame(["AX LAVA", perfusion_description])
    config = None
    if source == "dicom":
        rows["ContrastBolusAgentPhase"] = [None, "PORTAL_VENOUS"]
    else:
        rows["site_phase"] = [None, "portal"]
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
    out = annotate(rows, config)
    assert out.phase.tolist() == ["NATIVE", "PORTAL_VENOUS"]
    assert out.loc["v0", "mri_perfusion_source"] == "exam_context"


@pytest.mark.parametrize("description", ["AX LAVA dynamic", "Ph1/AX LAVA dynamic"])
def test_phase_resolved_from_elapsed_time_supports_native_fallback(description):
    rows = frame(["AX LAVA", description])
    rows["time"] = ["120000", "120115"]
    out = annotate(rows)
    assert out.phase.tolist() == ["NATIVE", "PORTAL_VENOUS"]
    assert out.loc["v0", "mri_perfusion_source"] == "exam_context"
    assert out.loc["v1", "mri_perfusion_source"] == "time_delta"


@pytest.mark.parametrize(
    "descriptions",
    [
        ["AX LAVA", "AX LAVA"],
        ["AX LAVA", "AX LAVA post"],
        ["AX LAVA", "AX LAVA dynamic"],
        ["AX LAVA portal", "AX LAVA"],
        ["AX LAVA", "AX VIBE portal"],
        ["AX LAVA post", "AX LAVA portal"],
        ["AX LAVA MPR", "AX LAVA portal"],
        ["AX LAVA Ph1", "AX LAVA portal"],
    ],
)
def test_permissive_native_fallback_keeps_candidate_and_context_guards(descriptions):
    assert not annotate(frame(descriptions)).phase.eq("NATIVE").any()


@pytest.mark.parametrize("order", [[None, None], [1, 1]])
def test_named_phase_without_comparable_order_does_not_establish_native(order):
    rows = frame(["AX LAVA", "AX LAVA portal"])
    rows["acquisition_order"] = order
    assert annotate(rows).phase.tolist() == ["OTHER", "PORTAL_VENOUS"]


def test_ph1_identifies_matching_unnumbered_protocol_as_native():
    out = annotate(frame(["AX LAVA", "Ph1/AX LAVA"]))
    assert out.phase.tolist() == ["NATIVE", "OTHER"]
    assert "matching protocol without PhN" in out.loc["v0", "mri_perfusion_reason"]


@pytest.mark.parametrize(
    "expected",
    [["NATIVE", "OTHER"], ["NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED"]],
)
def test_numbered_mixed_pre_post_text_keeps_native_fallback_without_native_context(expected):
    rows = frame([f"Ph{i+1}/AX LAVA pre/post dynamic" for i in range(len(expected))])
    assert annotate(rows).phase.tolist() == expected


@pytest.mark.parametrize("number", [1, 2, 10])
@pytest.mark.parametrize("ordered", [True, False])
def test_matching_dynamic_protocol_without_number_is_native_even_without_order(
    number, ordered
):
    rows = frame(["AX LAVA dynamic gado", f"Ph{number}/AX LAVA dynamic gado"])
    if not ordered:
        rows = rows.drop(columns="acquisition_order")
    out = annotate(rows.sample(frac=1, random_state=3)).loc[rows.volume_id]
    assert out.phase.tolist() == ["NATIVE", "OTHER"]
    assert out.loc["v0", "mri_perfusion_source"] == "exam_context"


@pytest.mark.parametrize("description", ["PH2/AX T1", "AX T1 phase_2", "Ax.T1.Ph10"])
def test_numbered_protocol_matching_normalizes_case_separators_and_token_position(
    description,
):
    out = annotate(frame(["AX T1", description]).drop(columns="acquisition_order"))
    assert out.phase.tolist() == ["NATIVE", "OTHER"]


def test_matching_unnumbered_native_anchors_numbered_postcontrast_progression():
    rows = frame(["AX LAVA dynamic"] + [f"Ph{i}/AX LAVA dynamic" for i in (2, 3, 4)])
    out = annotate(rows.sample(frac=1, random_state=7)).loc[rows.volume_id]
    assert out.phase.tolist() == ["NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED"]


@pytest.mark.parametrize(
    "column,value",
    [
        ("study_id", "other-exam"),
        ("ImageType", "DERIVED PRIMARY"),
        ("acquisition_plane", "CORONAL"),
        ("date", "2020-01-02"),
    ],
)
def test_matching_numbered_protocol_cannot_cross_acquisition_boundaries(column, value):
    rows = frame(["AX LAVA dynamic", "Ph2/AX LAVA dynamic"])
    rows.loc[1, column] = value
    assert not annotate(rows).phase.eq("NATIVE").any()


def test_numbered_protocol_match_requires_same_description_not_just_sequence_family():
    out = annotate(frame(["AX LAVA breath hold", "Ph2/AX LAVA dynamic"]))
    assert out.phase.tolist() == ["OTHER", "OTHER"]


@pytest.mark.parametrize("evidence", ["text", "dicom", "profile"])
@pytest.mark.parametrize("valid_native", [False, True])
def test_numbered_native_evidence_is_blocked_only_with_valid_native_context(
    evidence, valid_native
):
    description = "Ph2/AX LAVA pre" if evidence == "text" else "Ph2/AX LAVA"
    if evidence == "profile":
        description = "Ph2/AX LAVA Mask Multiart"
    rows = frame((["AX LAVA pre"] if valid_native else []) + [description])
    if evidence == "dicom":
        rows["ContrastBolusAgentPhase"] = (
            [None, "PRE_CONTRAST"] if valid_native else ["PRE_CONTRAST"]
        )
    assert annotate(rows).phase.tolist() == (
        ["NATIVE", "OTHER"] if valid_native else ["NATIVE"]
    )


@pytest.mark.parametrize("native", ["AX LAVA", "AX LAVA pre"])
def test_protocol_match_precedes_weaker_fallback_and_repeated_natives_count_once(native):
    rows = frame([native, "AX LAVA dynamic", "Ph1/AX LAVA dynamic", "Ph2/AX LAVA dynamic"])
    out = annotate(rows)
    assert out.phase.tolist() == [
        "NATIVE" if native.endswith("pre") else "OTHER",
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
    ]


@pytest.mark.parametrize("series", ["AX LAVA", "AX LAVA post"])
def test_protocol_name_match_respects_higher_priority_phase_field(series):
    rows = frame([series, "AX LAVA"])
    rows["ProtocolName"] = ["AX LAVA", "Ph2/AX LAVA"]
    out = annotate(rows, {
        "strategies": [{"type": "rules"}],
        "text_columns": {"MR": ["SeriesDescription", "ProtocolName"]},
    })
    assert out.phase.tolist() == ["OTHER" if series.endswith("post") else "NATIVE", "OTHER"]


@pytest.mark.parametrize("native", ["AX LAVA pre", "AX VIBE pre"])
def test_numbered_postcontrast_constraint_requires_compatible_native(native):
    rows = frame([native] + [f"Ph{i}/AX LAVA dynamic" for i in (2, 3, 4)])
    out = annotate(rows)
    assert out.phase.tolist() == (
        ["NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED"]
        if "LAVA" in native
        else ["NATIVE", "NATIVE", "ARTERIAL", "PORTAL_VENOUS"]
    )


def test_numbered_native_order_fallback_does_not_become_its_own_postcontrast_proof():
    rows = frame([f"Ph{i}/AX LAVA dynamic" for i in (1, 2, 3)])
    first = annotate(rows)
    again = annotate(first.reset_index())
    assert first.phase.tolist() == ["NATIVE", "ARTERIAL", "PORTAL_VENOUS"]
    assert again.phase.tolist() == first.phase.tolist()
    assert "_numbered_postcontrast" not in first.columns


def test_native_found_from_mixed_context_enables_numbered_postcontrast_constraint():
    rows = frame(["AX LAVA pre/post", "Ph2/AX LAVA Mask Multiart"])
    out = annotate(rows)
    assert out.phase.tolist() == ["NATIVE", "OTHER"]
    assert out.loc["v0", "mri_perfusion_source"] == "group_pre_post_order"
    assert out.loc["v1", "mri_perfusion_source"] == "ordinal_phase_conflict"


@pytest.mark.parametrize("source", ["dicom", "ontology"])
def test_valid_metadata_native_anchors_numbered_postcontrast_progression(source):
    rows = frame(["AX LAVA"] + [f"Ph{i}/AX LAVA dynamic" for i in (2, 3, 4)])
    config = None
    if source == "dicom":
        rows["ContrastBolusAgentPhase"] = ["PRE_CONTRAST", None, None, None]
    else:
        rows["site_phase"] = ["pre", None, None, None]
        config = {
            "strategies": [
                {"type": "ontology", "columns": ["site_phase"], "mapping": {"pre": "NATIVE"}},
                {"type": "rules"},
            ]
        }
    assert annotate(rows, config).phase.tolist() == [
        "NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED"
    ]
