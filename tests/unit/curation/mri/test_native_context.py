"""Native context takes precedence over vendor phase numbering."""

import pandas as pd
import pytest

from imperandi.curation.mri import curate as mc


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
def test_numbering_and_agent_alone_do_not_block_native_fallback(numbers, agent):
    rows = frame(
        [
            f"{'ph'+str(number)+'/' if number is not None else ''}AX LAVA dyn 4 {agent}"
            for number in numbers
        ]
    )
    out = annotate(rows.sample(frac=1, random_state=4)).loc[[f"v{i}" for i in range(4)]]
    assert out.phase.tolist() == ["NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED"]


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
