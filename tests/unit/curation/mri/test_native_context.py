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
def test_numbering_and_agent_keep_the_postcontrast_dynamic_default(numbers, agent):
    rows = frame(
        [
            f"{'ph'+str(number)+'/' if number is not None else ''}AX LAVA dyn 4 {agent}"
            for number in numbers
        ]
    )
    out = annotate(rows.sample(frac=1, random_state=4)).loc[[f"v{i}" for i in range(4)]]
    assert out.phase.tolist() == ["ARTERIAL", "PORTAL_VENOUS", "DELAYED", "DELAYED"]


@pytest.mark.parametrize(
    "native,source",
    [
        ("AX LAVA pre", "explicit_text"),
        ("AX LAVA", "exam_context"),
        ("water: AX LAVA-flex", "exam_context"),
        ("AX LAVA gado", "exam_context"),
    ],
)
def test_compatible_native_is_resolved_before_numbered_or_unnumbered_dynamics(
    native, source
):
    rows = frame(
        [native, "AX LAVA dyn gado", "Ph1/AX LAVA dyn gado", "Ph3/AX LAVA dyn gado"]
    )
    out = annotate(rows)
    assert out.phase.tolist() == ["NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED"]
    assert out.loc["v0", "mri_perfusion_source"] == source
    assert out.phase.eq("NATIVE").sum() == 1


def test_exam_level_injection_description_does_not_exclude_a_separate_native_candidate():
    rows = frame(
        ["AX LAVA", "AX LAVA dyn gado", "Ph2/AX LAVA dyn gado", "Ph3/AX LAVA dyn gado"]
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
def test_unrelated_native_does_not_change_postcontrast_dynamic_progression(
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
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]


def test_two_protocol_groups_resolve_their_own_separate_native_candidates():
    rows = frame(
        [
            "AX LAVA",
            "AX LAVA dyn gado",
            "Ph2/AX LAVA dyn gado",
            "Ph3/AX LAVA dyn gado",
            "AX DISCO",
            "AX DISCO dynamic",
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
    assert out.phase.tolist() == ["ARTERIAL", "PORTAL_VENOUS", "DELAYED", "OTHER"]


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
    assert out.phase.tolist() == ["ARTERIAL", "PORTAL_VENOUS", "DELAYED"]
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
    "candidate",
    ["AX LAVA post", "AX LAVA C+", "AX LAVA dynamic post", "Ph1/AX LAVA dynamic post"],
)
@pytest.mark.parametrize("arterial_source", ["text", "dicom", "ontology"])
def test_classified_arterial_unblocks_earlier_postcontrast_native(
    candidate, arterial_source
):
    rows = frame([candidate, "AX LAVA arterial", "AX LAVA portal"])
    config = None
    if arterial_source != "text":
        rows.loc[1, "SeriesDescription"] = "AX LAVA"
    if arterial_source == "dicom":
        rows["ContrastBolusAgentPhase"] = [None, "ARTERIAL", None]
    elif arterial_source == "ontology":
        rows["site_phase"] = [None, "arterial", None]
        config = {
            "strategies": [
                {
                    "type": "ontology",
                    "columns": ["site_phase"],
                    "mapping": {"arterial": "ARTERIAL"},
                },
                {"type": "rules"},
            ]
        }
    out = annotate(rows.sample(frac=1, random_state=4), config)
    assert out.loc[["v0", "v1", "v2"], "phase"].tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
    ]
    assert out.loc["v0", "mri_perfusion_source"] == "exam_context"
    assert out.loc["v0", "mri_perfusion_confidence"] == "fallback"
    assert "already classified ARTERIAL" in out.loc["v0", "phase_reason"]


def test_postcontrast_native_exception_needs_only_the_earlier_candidate_and_arterial():
    out = annotate(frame(["AX LAVA post", "AX LAVA arterial"]))
    assert out.loc[["v0", "v1"], "phase"].tolist() == ["NATIVE", "ARTERIAL"]


def test_new_native_is_removed_from_the_postcontrast_delay_reference():
    rows = frame(["Ph1/AX LAVA dynamic post", "AX LAVA arterial", "AX LAVA dynamic"])
    rows["time"] = ["115800", "120000", "120040"]
    out = annotate(rows)
    assert out.loc[["v0", "v1", "v2"], "phase"].tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
    ]
    assert out.loc[["v1", "v2"], "time_delta"].tolist() == [20, 60]
    assert out.loc["v2", "mri_perfusion_source"] == "time_delta"


@pytest.mark.parametrize("rank", [1, 2, None])
def test_postcontrast_native_exception_requires_candidate_strictly_before_arterial(
    rank,
):
    rows = frame(["AX LAVA post", "AX LAVA arterial"])
    rows.loc[0, "acquisition_order"] = rank
    assert annotate(rows).loc["v0", "phase"] == "OTHER"


@pytest.mark.parametrize(
    "changes",
    [
        {"study_id": "other-study"},
        {"SeriesDescription": "AX VIBE post"},
        {"SeriesDescription": "COR LAVA post"},
        {"ImageType": "DERIVED PRIMARY"},
        {"SeriesDescription": "AX LAVA portal post"},
        {"ContrastBolusAgentPhase": ["PRE_CONTRAST", "ARTERIAL"]},
    ],
)
def test_postcontrast_native_exception_preserves_phase_evidence_and_protocol_boundaries(
    changes,
):
    rows = frame(["AX LAVA post", "AX LAVA arterial"])
    for column, value in changes.items():
        if column not in rows:
            rows[column] = pd.Series(None, index=rows.index, dtype="object")
        rows.at[0, column] = value
    assert annotate(rows).loc["v0", "phase"] != "NATIVE"


def test_postcontrast_native_exception_does_not_duplicate_an_existing_native():
    out = annotate(frame(["AX LAVA pre", "AX LAVA post", "AX LAVA arterial"]))
    assert out.phase.tolist() == ["NATIVE", "OTHER", "ARTERIAL"]


@pytest.mark.parametrize("later", ["dynamic post", "portal", "delayed"])
def test_postcontrast_candidate_still_requires_a_classified_arterial(later):
    out = annotate(frame(["AX LAVA post", f"AX LAVA {later}"]))
    assert out.loc["v0", "phase"] == "OTHER"
