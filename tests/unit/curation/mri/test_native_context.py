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
    "other_native,source",
    [("AX VIBE pre", "explicit_text"), ("AX VIBE", "native_no_perfusion_evidence")],
)
def test_unrelated_native_does_not_suppress_protocol_native_fallback(
    other_native, source
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
    assert out.loc["v0", "phase"] == "NATIVE"
    assert out.loc["v0", "mri_perfusion_source"] == source
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


def test_separate_native_fallback_does_not_require_candidate_before_context():
    rows = frame(["AX LAVA dyn gado", "AX LAVA dyn gado", "AX LAVA delayed", "AX LAVA"])
    out = annotate(rows)
    assert out.phase.tolist() == ["OTHER", "OTHER", "DELAYED", "NATIVE"]
    assert out.loc["v3", "mri_perfusion_source"] == "exam_context"


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
@pytest.mark.parametrize("protocol", ["AX LAVA", "AX T1", "AX T1 SE", "AX T1 MPRAGE"])
def test_named_perfusion_phase_supports_native_without_postcontrast_marker(
    phase_text, expected, protocol
):
    out = annotate(frame([protocol, f"{protocol} {phase_text}"]))
    assert out.phase.tolist() == ["NATIVE", expected]
    assert out.loc["v0", "mri_perfusion_source"] == "exam_context"
    assert out.loc["v0", "mri_perfusion_confidence"] == "fallback"
    assert out.loc["v1", "mri_perfusion_source"] == "explicit_text"


def test_several_named_perfusion_phases_support_one_native_fallback():
    rows = frame(["AX LAVA", "AX LAVA arterial", "AX LAVA portal", "AX LAVA delayed"])
    out = annotate(rows.sample(frac=1, random_state=5)).loc[rows.volume_id]
    assert out.phase.tolist() == ["NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED"]
    assert out.phase.eq("NATIVE").sum() == 1


@pytest.mark.parametrize(
    "image_type",
    [
        ["ORIGINAL", "SECONDARY", "M", "NORM", "DIS2D", "FM3_4", "FIL"],
        "['ORIGINAL', 'SECONDARY', 'M', 'NORM', 'DIS2D', 'FM3_4', 'FIL']",
    ],
)
def test_secondary_image_type_allows_repeated_protocol_natives(image_type):
    rows = frame(["t1 fl2d in_opp tra 5mm"] * 4)
    rows["ImageType"] = [image_type] * len(rows)
    rows["series_id"] = ["fl0", "fl0", "fl1", "fl1"]
    rows["volume_order_in_series"] = [1, 2, 1, 2]
    rows["n_volumes_in_series"] = 2

    out = annotate(rows.sample(frac=1, random_state=3)).loc[rows.volume_id]

    assert out.mri_sequence.eq("T1").all()
    assert not out.is_derived_low_value.any()
    assert out.phase.eq("NATIVE").all()
    assert out.mri_perfusion_source.eq("native_no_perfusion_evidence").all()
    assert out.mri_perfusion_confidence.eq("fallback").all()


def test_secondary_image_type_allows_separate_native_fallback():
    rows = frame(
        [
            "t1 vibe tra fs",
            "t1 vibe tra fs art",
            "t1 vibe tra fs portal",
            "t1 vibe tra fs 5min",
            "t1 vibe tra fs art_sub",
        ]
    )
    rows["ImageType"] = [
        ["ORIGINAL", "SECONDARY", "M", "NORM", "DIS2D", "FM4_4", "FIL"]
        for _ in range(len(rows))
    ]

    result = mc.curate_mri(rows.sample(frac=1, random_state=4))
    out = result["curated"].set_index("volume_id").loc[rows.volume_id]

    assert out.phase.tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
        "ARTERIAL",
    ]
    assert not out.loc["v0", "is_derived_low_value"]
    assert out.loc["v0", "mri_perfusion_source"] == "exam_context"
    assert out.loc["v4", "is_derived_low_value"]
    selected = result["selected_long"].set_index("selection_slot")
    assert selected.loc["T1_NATIVE", "volume_id"] == "v0"
    assert "v4" not in selected.volume_id.tolist()


def test_lava_flex_water_native_matches_perfusion_across_image_types():
    rows = frame(
        [
            "water: ax lava-flex apnee",
            "ax lava portal 1.10min",
            "ax lava tardif 3min",
            "fat: ax lava-flex apnee",
            "inphase: ax lava-flex apnee",
            "outphase: ax lava-flex apnee",
            "ph1/ax lava multi arteriel",
            "ph2/ax lava multi arteriel",
            "ph3/ax lava multi arteriel",
        ]
    )
    rows["ImageType"] = [
        ["DERIVED", "PRIMARY", "DIXON", "WATER"],
        ["ORIGINAL", "PRIMARY", "OTHER"],
        ["ORIGINAL", "PRIMARY", "OTHER"],
        ["DERIVED", "PRIMARY", "DIXON", "FAT"],
        ["ORIGINAL", "PRIMARY", "OTHER", "NONE"],
        ["ORIGINAL", "PRIMARY", "OTHER", "NONE"],
        ["ORIGINAL", "PRIMARY", "OTHER"],
        ["ORIGINAL", "PRIMARY", "OTHER"],
        ["ORIGINAL", "PRIMARY", "OTHER"],
    ]
    rows["acquisition_order"] = [5, 6, 7, 12, 13, 14, 16, 17, 18]

    result = mc.curate_mri(rows.sample(frac=1, random_state=8))
    out = result["curated"].set_index("volume_id").loc[rows.volume_id]

    assert out.phase.tolist() == [
        "NATIVE",
        "PORTAL_VENOUS",
        "DELAYED",
        "OTHER",
        "NATIVE",
        "NATIVE",
        "ARTERIAL",
        "ARTERIAL",
        "ARTERIAL",
    ]
    assert out.loc["v0", "mri_perfusion_source"] == "exam_context"
    assert out.loc["v0", "mri_perfusion_confidence"] == "fallback"
    selected = result["selected_long"].set_index("selection_slot")
    assert selected.loc["T1_NATIVE", "volume_id"] == "v0"


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
        ["AX LAVA post", "AX LAVA portal"],
        ["AX LAVA MPR", "AX LAVA portal"],
        ["AX LAVA Ph1", "AX LAVA portal"],
    ],
)
def test_permissive_native_fallback_keeps_candidate_and_context_guards(descriptions):
    assert not annotate(frame(descriptions)).phase.eq("NATIVE").any()


@pytest.mark.parametrize("order", [[None, None], [1, 1], [2, 1]])
@pytest.mark.parametrize("protocol", ["AX LAVA", "AX T1 SE"])
def test_named_phase_establishes_separate_native_without_order(order, protocol):
    rows = frame([protocol, f"{protocol} portal"])
    rows["acquisition_order"] = order
    out = annotate(rows)
    assert out.phase.tolist() == ["NATIVE", "PORTAL_VENOUS"]
    assert out.loc["v0", "mri_perfusion_source"] == "exam_context"


def test_context_minimum_does_not_block_final_native_fallback():
    rows = frame(["AX T1 SE", "AX T1 SE dynamic", "AX T1 SE dynamic"])
    rows = rows.drop(columns="acquisition_order")
    assert annotate(rows).phase.tolist() == ["NATIVE", "OTHER", "OTHER"]
    out = annotate(rows.iloc[:2])
    assert out.phase.tolist() == ["NATIVE", "OTHER"]
    assert out.loc["v0", "mri_perfusion_source"] == "native_no_perfusion_evidence"


def test_separate_native_counts_shared_acquisition_identity_once_without_order():
    rows = frame(["AX T1 SE", "AX T1 SE dynamic", "AX T1 SE dynamic"])
    rows["acquisition_id"] = ["native", "same-dynamic", "same-dynamic"]
    out = annotate(rows)
    assert out.phase.tolist() == ["NATIVE", "OTHER", "OTHER"]
    assert out.loc["v0", "mri_perfusion_source"] == "native_no_perfusion_evidence"


def test_metadata_classified_t1_supports_separate_native_without_gre_evidence():
    rows = frame(["AX custom sequence", "AX custom sequence portal"])
    rows["RepetitionTime"] = 400
    rows["EchoTime"] = 10
    out = annotate(rows)
    assert out.phase.tolist() == ["NATIVE", "PORTAL_VENOUS"]
    assert out.loc["v0", "mri_perfusion_source"] == "exam_context"


def test_unmatched_t1_protocol_uses_final_native_fallback():
    rows = frame(["AX T1 SE", "AX T1 MPRAGE portal"])
    out = annotate(rows)
    assert out.phase.tolist() == ["NATIVE", "PORTAL_VENOUS"]
    assert out.loc["v0", "mri_perfusion_source"] == "native_no_perfusion_evidence"


def test_separate_native_fallback_resolves_other_t1_protocols_independently():
    rows = frame(["AX T1 SE", "AX T1 MPRAGE", "AX T1 SE portal", "AX T1 MPRAGE portal"])
    assert annotate(rows).phase.tolist() == [
        "NATIVE", "NATIVE", "PORTAL_VENOUS", "PORTAL_VENOUS"
    ]


@pytest.mark.parametrize("protocol", ["AX LAVA", "AX T1", "AX T1 SE", "AX T1 MPRAGE"])
@pytest.mark.parametrize("count", [1, 2, 4])
def test_native_fallback_does_not_require_repeated_protocols(protocol, count):
    rows = frame([protocol] * count)
    out = annotate(rows.sample(frac=1, random_state=6)).loc[rows.volume_id]
    assert out.phase.eq("NATIVE").all()
    assert out.mri_perfusion_source.eq("native_no_perfusion_evidence").all()
    assert out.mri_perfusion_reason.str.contains("without acquisition perfusion").all()
    assert out.mri_perfusion_confidence.eq("fallback").all()


@pytest.mark.parametrize(
    "column,value",
    [("ProtocolName", "Liver"), ("ScanningSequence", "GR"), ("ScanOptions", "FS")],
)
def test_final_native_fallback_does_not_depend_on_shared_generic_metadata(
    column, value
):
    rows = frame(["AX T1 SE", "AX T1 MPRAGE"])
    rows[column] = value
    out = annotate(
        rows,
        {
            "strategies": [{"type": "rules"}],
            "text_columns": {"MR": ["SeriesDescription", column]},
        },
    )
    assert out.phase.tolist() == ["NATIVE", "NATIVE"]
    assert out.phase.tolist() == annotate(rows.drop(columns=column)).phase.tolist()
    assert out.mri_perfusion_source.eq("native_no_perfusion_evidence").all()
    assert not out.mri_perfusion_reason.str.contains(f"{column}=").any()


def test_contextual_protocol_matching_uses_one_priority_acquisition_field():
    rows = frame(["AX T1 SE", "AX T1 MPRAGE"])
    rows["ProtocolName"] = ["Liver", "Liver portal"]
    out = annotate(
        rows,
        {
            "strategies": [{"type": "rules"}],
            "text_columns": {"MR": ["SeriesDescription", "ProtocolName"]},
        },
    )
    assert out.phase.tolist() == ["NATIVE", "PORTAL_VENOUS"]
    assert out.loc["v0", "mri_perfusion_source"] == "native_no_perfusion_evidence"
    assert out.loc["v1", "mri_perfusion_source"] == "explicit_text"


def test_single_metadata_classified_t1_falls_back_to_native():
    rows = frame(["AX custom sequence"]).drop(columns="acquisition_order")
    rows["RepetitionTime"] = 400
    rows["EchoTime"] = 10
    out = annotate(rows)
    assert out.mri_sequence.tolist() == ["T1"]
    assert out.phase.tolist() == ["NATIVE"]
    assert out.loc["v0", "mri_perfusion_source"] == "native_no_perfusion_evidence"


@pytest.mark.parametrize(
    "description",
    [
        "AX T1 post",
        "AX T1 C+",
        "AX T1 dynamic",
        "AX T1 Ph2",
        "AX T1 ART-PORT",
        "AX T1 ART/PORT/LATE",
        "AX T1 Mask Multiart",
        "AX T1 pre/post",
        "AX T1 2s",
        "AX T1 5min 15min",
        "AX T1 MPR",
        "AX T1 sub",
        "AX T1 fat",
        "AX T1 PDFF",
        "AX T1 R2star",
        "AX T2",
        "AX DWI",
        "localizer",
        "key images",
    ],
)
def test_final_native_fallback_preserves_perfusion_evidence_and_exclusions(description):
    out = annotate(frame([description]))
    assert out.loc["v0", "mri_perfusion_source"] != "native_no_perfusion_evidence"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("PRE_CONTRAST", "NATIVE"),
        ("ARTERIAL", "ARTERIAL"),
        ("POST_CONTRAST", "OTHER"),
        (["ARTERIAL", "DELAYED"], "OTHER"),
    ],
)
def test_final_native_fallback_preserves_structured_phase_evidence(raw, expected):
    rows = frame(["AX T1 SE"])
    rows["ContrastBolusAgentPhase"] = pd.Series([raw])
    out = annotate(rows)
    assert out.phase.tolist() == [expected]
    assert out.loc["v0", "mri_perfusion_source"] != "native_no_perfusion_evidence"


@pytest.mark.parametrize("description", ["AX T1 SE", "AX T1 SE gado"])
def test_bare_agent_context_allows_final_native_fallback(description):
    rows = frame([description])
    out = annotate(rows)
    assert out.phase.tolist() == ["NATIVE"]
    assert out.loc["v0", "mri_perfusion_source"] == "native_no_perfusion_evidence"


@pytest.mark.parametrize(
    "context", ["with injection", "with and without contrast", "2s"]
)
def test_final_native_fallback_preserves_configured_study_phase_context(context):
    rows = frame(["AX T1 SE"]).drop(columns="acquisition_order")
    rows["StudyDescription"] = context
    out = annotate(
        rows,
        {
            "strategies": [{"type": "rules"}],
            "text_columns": {"MR": ["SeriesDescription", "StudyDescription"]},
        },
    )
    assert out.phase.tolist() == ["OTHER"]
    assert out.loc["v0", "mri_perfusion_source"] != "native_no_perfusion_evidence"


def test_final_native_fallback_respects_configured_postcontrast_field():
    rows = frame(["AX T1 SE"])
    rows["ProtocolName"] = "AX T1 SE post"
    out = annotate(
        rows,
        {
            "strategies": [{"type": "rules"}],
            "text_columns": {"MR": ["SeriesDescription", "ProtocolName"]},
        },
    )
    assert out.phase.tolist() == ["OTHER"]
    assert out.loc["v0", "mri_perfusion_source"] != "native_no_perfusion_evidence"


def test_final_native_fallback_preserves_earlier_ontology_phase():
    rows = frame(["AX T1 SE"])
    rows["site_phase"] = "portal"
    out = annotate(
        rows,
        {
            "strategies": [
                {
                    "type": "ontology",
                    "columns": ["site_phase"],
                    "mapping": {"portal": "PORTAL_VENOUS"},
                },
                {"type": "rules"},
            ],
        },
    )
    assert out.phase.tolist() == ["PORTAL_VENOUS"]
    assert out.loc["v0", "mri_perfusion_source"] != "native_no_perfusion_evidence"


def test_explicit_native_is_selected_before_native_without_perfusion_evidence():
    result = mc.curate_mri(frame(["AX T1 SE", "AX T1 SE pre"]))
    out = result["curated"].set_index("volume_id")
    assert out.phase.eq("NATIVE").all()
    assert out.loc["v0", "mri_perfusion_source"] == "native_no_perfusion_evidence"
    assert out.loc["v1", "mri_perfusion_source"] == "explicit_text"
    assert result["selected_long"].volume_id.tolist() == ["v1"]


def test_repeated_protocol_labels_all_candidates_before_best_selection():
    rows = frame(["AX T1 SE"] * 2).drop(columns="acquisition_order")
    rows["n_rows_in_volume"] = [80, 120]
    rows["SliceThickness"] = 5
    rows["PixelSpacing"] = "1.0\\1.0"

    result = mc.curate_mri(rows.iloc[::-1])

    assert result["curated"].phase.eq("NATIVE").all()
    selected = result["selected_long"]
    assert selected.selection_slot.tolist() == ["T1_NATIVE"]
    assert selected.volume_id.tolist() == ["v1"]


def test_repeated_protocol_matching_normalizes_case_and_separators():
    out = annotate(frame(["AX_T1_SE", "ax.t1/se"]))
    assert out.phase.tolist() == ["NATIVE", "NATIVE"]


def test_repeated_protocols_are_resolved_independently():
    rows = frame(["AX T1 SE", "AX T1 MPRAGE", "AX T1 SE", "AX T1 MPRAGE"])
    assert annotate(rows).phase.tolist() == ["NATIVE", "NATIVE", "NATIVE", "NATIVE"]


@pytest.mark.parametrize(
    "times,order",
    [
        ([None, None], [None, None]),
        ([None, None], [None, 2]),
        (["120000", "120000"], [1, 1]),
        (["120100", "120000"], [0, 1]),
    ],
)
def test_repeated_protocol_native_assignments_ignore_chronology(times, order):
    rows = frame(["AX T1 SE"] * 2)
    expected = annotate(rows).phase
    rows["time"] = times
    rows["acquisition_order"] = order

    out = annotate(rows.iloc[::-1]).loc[rows.volume_id]

    assert out.phase.eq("NATIVE").any()
    pd.testing.assert_series_equal(out.phase, expected)


@pytest.mark.parametrize("order", [[None, None], [None, 2], [1, 1]])
def test_repeated_protocol_does_not_require_complete_distinct_acquisition_order(order):
    rows = frame(["AX T1 SE"] * 2)
    rows["acquisition_order"] = order
    assert annotate(rows).phase.tolist() == ["NATIVE", "NATIVE"]


def test_repeated_protocol_ignores_series_local_and_input_order():
    rows = frame(["AX T1 SE"] * 2).drop(columns="acquisition_order")
    rows["series_id"] = "same-series"
    rows["volume_order_in_series"] = [1, 2]
    rows["n_volumes_in_series"] = 2
    assert annotate(rows).phase.tolist() == ["NATIVE", "NATIVE"]
    rows["volume_order_source"] = "input_order"
    assert annotate(rows).phase.tolist() == ["NATIVE", "NATIVE"]


def test_repeated_protocol_does_not_use_canonical_time_to_select_one_native():
    rows = frame(["AX T1 SE"] * 2)
    rows["time"] = ["120100", "120000"]
    assert annotate(rows).phase.tolist() == ["NATIVE", "NATIVE"]


@pytest.mark.parametrize(
    "column,value",
    [
        ("patient_key", "other-patient"),
        ("study_id", "other-study"),
        ("StudyInstanceUID", "other-uid"),
        ("date", "2020-01-02"),
        ("acquisition_plane", "CORONAL"),
        ("SeriesDescription", "AX T1 MPRAGE"),
    ],
)
def test_final_native_fallback_is_independent_of_protocol_groups(column, value):
    rows = frame(["AX T1 SE"] * 2)
    rows.loc[1, column] = value
    out = annotate(rows)
    assert out.phase.tolist() == ["NATIVE", "NATIVE"]
    assert out.mri_perfusion_source.eq("native_no_perfusion_evidence").all()


@pytest.mark.parametrize("image_type", ["DERIVED PRIMARY", None])
def test_repeated_protocol_matches_across_image_types(image_type):
    rows = frame(["AX T1 SE"] * 2)
    rows.loc[1, "ImageType"] = image_type
    assert annotate(rows).phase.tolist() == ["NATIVE", "NATIVE"]


@pytest.mark.parametrize(
    "protocol",
    [
        "AX T1 SE post",
        "AX T1 SE MPR",
        "AX T1 SE fat",
        "AX T2",
        "AX DWI",
    ],
)
def test_repeated_protocol_does_not_create_native_for_ineligible_acquisitions(protocol):
    assert not annotate(frame([protocol] * 2)).phase.eq("NATIVE").any()


@pytest.mark.parametrize("ordered", [False, True])
def test_repeated_protocol_preserves_dedicated_mixed_pre_post_inference(ordered):
    rows = frame(["AX T1 SE pre/post"] * 2)
    if not ordered:
        rows = rows.drop(columns="acquisition_order")

    out = annotate(rows)

    assert out.phase.tolist() == (
        ["NATIVE", "OTHER"] if ordered else ["OTHER", "OTHER"]
    )
    if ordered:
        assert out.loc["v0", "mri_perfusion_source"] == "group_pre_post_order"


def test_repeated_protocol_preserves_dicom_conflicts_and_known_phases():
    rows = frame(["AX T1 SE"] * 2)
    rows["ContrastBolusAgentPhase"] = [["ARTERIAL", "DELAYED"], "PORTAL_VENOUS"]
    out = annotate(rows)
    assert out.phase.tolist() == ["OTHER", "PORTAL_VENOUS"]
    assert out.loc["v0", "mri_perfusion_source"] == "dicom_phase_conflict"
    rows["ContrastBolusAgentPhase"] = ["ARTERIAL", None]
    assert annotate(rows).phase.tolist() == ["ARTERIAL", "NATIVE"]


def test_repeated_native_candidates_remain_labelled_with_perfusion_context():
    rows = frame(["AX LAVA", "AX LAVA", "AX LAVA portal"])
    rows["time"] = ["120100", "120000", "115900"]
    rows["acquisition_order"] = [3, 2, 1]
    first = annotate(rows)
    rows["time"] = ["120000", "120100", "120200"]
    rows["acquisition_order"] = [1, 2, 3]
    second = annotate(rows.iloc[::-1])
    assert first.loc["v0", "phase"] == second.loc["v0", "phase"] == "NATIVE"
    assert first.loc["v1", "phase"] == second.loc["v1", "phase"] == "NATIVE"


@pytest.mark.parametrize("repeated", [False, True])
def test_repeated_protocol_prefers_sequence_description_over_generic_protocol(repeated):
    description = "inphase: ax in/out lava-flex"
    rows = frame([description, description if repeated else description + " apnee"])
    rows["ProtocolName"] = "Foie NLY"
    out = annotate(
        rows,
        {
            "strategies": [{"type": "rules"}],
            "text_columns": {"MR": ["SeriesDescription", "ProtocolName"]},
        },
    )
    assert out.phase.tolist() == ["NATIVE", "NATIVE"]
    assert (
        out.phase.tolist() == annotate(rows.drop(columns="ProtocolName")).phase.tolist()
    )
    assert out.mri_perfusion_source.eq("native_no_perfusion_evidence").all()
    assert not out.mri_perfusion_reason.str.contains("ProtocolName=").any()


def test_repeated_protocol_uses_configured_protocol_name_field():
    rows = frame(["unlabelled", "unlabelled"])
    rows["ProtocolName"] = "AX T1 SE"
    out = annotate(rows, {
        "strategies": [{"type": "rules"}],
        "text_columns": {"MR": ["ProtocolName"]},
    })
    assert out.phase.tolist() == ["NATIVE", "NATIVE"]
    assert out.mri_perfusion_source.eq("native_no_perfusion_evidence").all()
    assert annotate(rows).phase.tolist() == ["OTHER", "OTHER"]


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
        ("acquisition_plane", "CORONAL"),
        ("date", "2020-01-02"),
    ],
)
def test_matching_numbered_protocol_cannot_cross_acquisition_boundaries(column, value):
    rows = frame(["AX LAVA dynamic", "Ph2/AX LAVA dynamic"])
    rows.loc[1, column] = value
    assert not annotate(rows).phase.eq("NATIVE").any()


@pytest.mark.parametrize("image_type", ["DERIVED PRIMARY", None])
def test_matching_numbered_protocol_matches_across_image_types(image_type):
    rows = frame(["AX LAVA dynamic", "Ph2/AX LAVA dynamic"])
    rows.loc[1, "ImageType"] = image_type
    assert annotate(rows).phase.tolist() == ["NATIVE", "OTHER"]


def test_final_native_fallback_does_not_supply_numbered_protocol_context():
    out = annotate(frame(["AX LAVA breath hold", "Ph2/AX LAVA dynamic"]))
    assert out.phase.tolist() == ["NATIVE", "OTHER"]
    assert out.loc["v0", "mri_perfusion_source"] == "native_no_perfusion_evidence"


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
        "NATIVE",
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
    ]
    assert out.loc["v0", "mri_perfusion_source"] == (
        "explicit_text" if native.endswith("pre") else "native_no_perfusion_evidence"
    )


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
