"""Restored protocol inference when independent timing is unavailable."""

import pandas as pd
import pytest

from imperandi.curation.mri import mri_curation as mc


def acquisitions(description, *, same_series=True, clocks=True):
    return pd.DataFrame(
        [
            {
                "patient_key": "P1",
                "study_id": "EXAM1",
                "date": "2020-01-01",
                "series_id": "DYN" if same_series else f"SER{i}",
                "volume_id": f"v{i}",
                "Modality": "MR",
                "SeriesDescription": description,
                "time": f"12000{i}" if clocks else None,
                "time_source": "AcquisitionTime",
                "acquisition_order": i,
                "TemporalPositionIdentifier": i + 1,
                "ImageType": "ORIGINAL PRIMARY",
                "Rows": 512,
                "Columns": 512,
                "n_files": 80,
                "SliceThickness": 3,
            }
            for i in range(4)
        ]
    )


def annotate(df):
    return mc.annotate_mri(df).set_index("volume_id").loc[["v0", "v1", "v2", "v3"]]


@pytest.mark.parametrize(
    "description",
    [
        "AX T1 LAVA dynamic",
        "AX T1 mDIXON water 4D",
        "AX T1 VIBE GRASP",
        "AX T1 TWIST",
        "AX T1 DISCO dynamic",
        "t1 vibe fs multi gado",
        "AX T1 VIBE MULTI_GADOLINIUM",
    ],
)
@pytest.mark.parametrize("same_series", [False, True])
@pytest.mark.parametrize("clocks", [False, True])
def test_four_dynamic_acquisitions_restore_full_phase_sequence(
    description, same_series, clocks
):
    out = annotate(
        acquisitions(description, same_series=same_series, clocks=clocks).sample(
            frac=1, random_state=17
        )
    )
    assert out.mri_perfusion_label.tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]
    assert out.mri_perfusion_confidence.eq("inferred").all()


@pytest.mark.parametrize(
    "description,expected",
    [
        ("AX T1 LAVA dynamic", ["NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED"]),
        ("AX T1 LAVA ART-PORT", ["ARTERIAL", "OTHER", "OTHER", "PORTAL_VENOUS"]),
        ("AX T1 LAVA ART/PORT/LATE", ["ARTERIAL", "OTHER", "OTHER", "DELAYED"]),
        ("AX T1 LAVA Mask+Multiart", ["NATIVE", "ARTERIAL", "ARTERIAL", "ARTERIAL"]),
    ],
)
def test_different_image_types_share_dynamic_progression(description, expected):
    rows = acquisitions(description, same_series=False, clocks=False)
    rows["ImageType"] = [
        "ORIGINAL PRIMARY M",
        "DERIVED PRIMARY M",
        None,
        "ORIGINAL PRIMARY P",
    ]

    out = mc.annotate_mri(rows.sample(frac=1, random_state=11)).set_index("volume_id")

    assert out.loc[rows.volume_id, "phase"].tolist() == expected


@pytest.mark.parametrize(
    "description",
    ["AX T1 LAVA Multiart", "AX T1 VIBE multi arterial", "AX T1 multi-art", "Multiart"],
)
@pytest.mark.parametrize("same_series", [False, True])
def test_explicit_multiart_is_all_arterial(description, same_series):
    out = annotate(acquisitions(description, same_series=same_series, clocks=False))
    assert out.mri_perfusion_label.eq("ARTERIAL").all()


@pytest.mark.parametrize("same_series", [False, True])
def test_mask_multiart_starts_native_then_remains_arterial(same_series):
    out = annotate(
        acquisitions("AX T1 LAVA Mask+Multiart", same_series=same_series, clocks=False)
    )
    assert out.mri_perfusion_label.tolist() == [
        "NATIVE",
        "ARTERIAL",
        "ARTERIAL",
        "ARTERIAL",
    ]


def test_raw_injection_clocks_do_not_override_canonical_order():
    rows = acquisitions("AX T1 LAVA dynamic")
    rows["AcquisitionTime"] = ["120025", "120032", "120115", "120400"]
    rows["ContrastBolusStartTime"] = "120000"
    out = annotate(rows)
    assert out.mri_perfusion_label.tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]


def test_timing_outside_windows_falls_back_to_known_dynamic_order():
    rows = acquisitions("AX T1 LAVA dynamic")
    rows["AcquisitionTime"] = ["120010", "120045", "120140", "120200"]
    rows["ContrastBolusStartTime"] = "120000"
    out = annotate(rows)
    assert out.mri_perfusion_label.tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]


def test_long_gaps_do_not_delete_an_ordered_four_acquisition_template():
    rows = acquisitions("AX T1 LAVA dynamic")
    rows["time"] = ["110000", "120000", "120100", "123000"]
    out = annotate(rows)
    assert out.mri_perfusion_label.tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]


@pytest.mark.parametrize("description", ["AX T1 LAVA dynamic", "t1 vibe fs multi gado"])
def test_no_permanent_blocked_state_survives_in_missing_order_case(description):
    rows = acquisitions(description, same_series=False, clocks=False)
    rows = rows.drop(columns=["acquisition_order", "TemporalPositionIdentifier"])
    out = annotate(rows)
    assert out.mri_perfusion_label.eq("OTHER").all()
    assert out.mri_perfusion_source.eq(mc.GENERIC_DYNAMIC_CONTEXT_PENDING).all()


def test_supplied_local_volume_order_restores_template_without_clocks():
    rows = acquisitions("AX T1 LAVA dynamic", clocks=False)
    rows = rows.drop(columns=["acquisition_order", "TemporalPositionIdentifier"])
    rows["volume_order_in_series"] = [1, 2, 3, 4]
    rows["n_volumes_in_series"] = 4
    out = annotate(rows.sample(frac=1, random_state=3))
    assert out.mri_perfusion_label.tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]
    assert out.mri_dynamic_order_source.eq("volume_order_in_series within series").all()
    assert out.volume_order_source.eq("provided").all()


@pytest.mark.parametrize(
    "description,expected",
    [
        ("Multiart", ["ARTERIAL"] * 4),
        ("Mask+Multiart", ["NATIVE", "ARTERIAL", "ARTERIAL", "ARTERIAL"]),
    ],
)
def test_bare_multiart_names_identify_t1_dynamic_profiles(description, expected):
    out = annotate(acquisitions(description, clocks=False))
    assert out.mri_sequence.eq("T1").all()
    assert out.mri_perfusion_label.tolist() == expected


@pytest.mark.parametrize(
    "count,expected",
    [
        (2, ["NATIVE", "ARTERIAL", "PORTAL_VENOUS"]),
        (3, ["NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED"]),
    ],
)
@pytest.mark.parametrize(
    "native_description,native_source",
    [
        ("AX T1 LAVA pre", "explicit_text"),
        ("AX T1 LAVA", "exam_context"),
    ],
)
def test_separate_native_counts_before_unlabelled_dynamic_progression(
    count, expected, native_description, native_source
):
    rows = acquisitions("AX T1 LAVA dynamic", same_series=False).iloc[:count].copy()
    rows["time"] = ["120000", "120001", "120002"][:count]
    rows["acquisition_order"] += 1
    native = rows.iloc[0].copy()
    native.update(
        {
            "volume_id": "native",
            "series_id": "PRE",
            "SeriesDescription": native_description,
            "time": "115959",
            "acquisition_order": 0,
        }
    )
    out = mc.annotate_mri(
        pd.concat([pd.DataFrame([native]), rows], ignore_index=True).sample(
            frac=1, random_state=8
        )
    ).set_index("volume_id")
    assert (
        out.loc[
            ["native", *[f"v{i}" for i in range(count)]], "mri_perfusion_label"
        ].tolist()
        == expected
    )
    assert out.loc["native", "mri_perfusion_source"] == native_source
    assert out.mri_perfusion_label.eq("NATIVE").sum() == 1
    if native_source == "exam_context":
        assert out.loc["native", "mri_perfusion_confidence"] == "fallback"
        assert (
            out.loc[[f"v{i}" for i in range(count)], "mri_perfusion_reason"]
            .str.contains("native context=native")
            .all()
        )
        assert (
            not out.loc[[f"v{i}" for i in range(count)], "mri_perfusion_source"]
            .eq("dynamic_explicit_anchor")
            .any()
        )


def test_repeated_protocols_can_be_native_alongside_explicit_native():
    rows = acquisitions("AX T1 LAVA", same_series=False).iloc[:3].copy()
    rows.loc[0, "SeriesDescription"] = "AX T1 LAVA pre"
    out = mc.annotate_mri(rows)
    assert out.mri_perfusion_label.tolist() == ["NATIVE", "NATIVE", "NATIVE"]
    assert out.loc[0, "mri_perfusion_source"] == "explicit_text"
    assert (
        out.loc[[1, 2], "mri_perfusion_reason"]
        .str.contains("without acquisition perfusion")
        .all()
    )


@pytest.mark.parametrize("same_series", [False, True])
def test_extra_dynamic_acquisitions_remain_delayed_after_standard_progression(
    same_series,
):
    rows = acquisitions("AX T1 LAVA dynamic", same_series=same_series, clocks=False)
    extra = rows.iloc[-1].copy()
    extra.update(
        {
            "volume_id": "v4",
            "series_id": "DYN" if same_series else "SER4",
            "acquisition_order": 4,
            "TemporalPositionIdentifier": 5,
        }
    )
    out = mc.annotate_mri(
        pd.concat([rows, pd.DataFrame([extra])], ignore_index=True).sample(
            frac=1, random_state=17
        )
    ).set_index("volume_id")
    assert out.loc[[f"v{i}" for i in range(5)], "mri_perfusion_label"].tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
        "DELAYED",
    ]


def test_separate_native_candidate_is_resolved_independently_of_acquisition_order():
    rows = acquisitions("AX T1 LAVA dynamic", same_series=False).iloc[:3].copy()
    late = rows.iloc[-1].copy()
    late.update(
        {
            "volume_id": "late",
            "series_id": "LATE",
            "SeriesDescription": "AX T1 LAVA",
            "time": "130000",
            "acquisition_order": 9,
        }
    )
    out = mc.annotate_mri(
        pd.concat([rows, pd.DataFrame([late])], ignore_index=True)
    ).set_index("volume_id")
    assert out.loc[["v0", "v1", "v2"], "mri_perfusion_label"].tolist() == [
        "OTHER",
        "OTHER",
        "OTHER",
    ]
    assert out.loc["late", "mri_perfusion_label"] == "NATIVE"
    assert out.loc["late", "mri_perfusion_source"] == "exam_context"


def test_unlabelled_candidate_cannot_anchor_a_different_study():
    rows = acquisitions("AX T1 LAVA dynamic", same_series=False).iloc[:3].copy()
    native = rows.iloc[0].copy()
    native.update(
        {
            "volume_id": "native",
            "series_id": "PRE",
            "study_id": "OTHER",
            "SeriesDescription": "AX T1 LAVA",
            "time": "115900",
            "acquisition_order": -1,
        }
    )
    out = mc.annotate_mri(
        pd.concat([pd.DataFrame([native]), rows], ignore_index=True)
    ).set_index("volume_id")
    assert out.loc["native", "mri_perfusion_label"] == "NATIVE"
    assert out.loc["native", "mri_perfusion_source"] == "native_no_perfusion_evidence"
    assert out.loc[["v0", "v1", "v2"], "mri_perfusion_label"].tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
    ]


def test_unspecified_native_dixon_component_can_support_its_own_series():
    rows = acquisitions("AX T1 mDIXON water dynamic").iloc[:3].copy()
    native = rows.iloc[0].copy()
    native.update(
        {
            "volume_id": "native",
            "SeriesDescription": "AX T1 mDIXON",
            "time": "115900",
            "acquisition_order": -1,
        }
    )
    out = mc.annotate_mri(
        pd.concat([pd.DataFrame([native]), rows], ignore_index=True)
    ).set_index("volume_id")
    assert out.loc[["native", "v0", "v1", "v2"], "mri_perfusion_label"].tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]
    assert out.loc["native", "mri_perfusion_source"] == "exam_context"
