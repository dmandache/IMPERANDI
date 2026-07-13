"""
Pytest suite for the simplified MVP MRI curation module.

Run from the directory containing `mri_curation.py` and `mri_curation_rules.py`:

    pytest -q test_mri_curation.py

The tests focus on edge cases that previously caused missed or wrongly selected
T1/T2 liver MRI diagnostic candidates.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
import pytest

from imperandi.curation.mri import curate as mc


# -----------------------------------------------------------------------------
# Test helpers
# -----------------------------------------------------------------------------


def row(
    desc: str | None = None,
    *,
    patient_key: str = "P1",
    study_id: str = "S1",
    series_id: str = "SER1",
    volume_id: str = "V1",
    date: str = "2020-01-01",
    time: str | int | float | None = "120000",
    protocol: str | None = None,
    image_type: str | None = "ORIGINAL\\PRIMARY",
    n_rows: int = 80,
    slice_thickness: float = 5.0,
    pixel_spacing: str = "1.0\\1.0",
    **extra,
) -> dict:
    """Small row factory with realistic default metadata."""
    out = {
        "patient_key": patient_key,
        "study_id": study_id,
        "series_id": series_id,
        "volume_id": volume_id,
        "date": date,
        "time": time,
        "SeriesDescription": desc,
        "ProtocolName": protocol,
        "ImageType": image_type,
        "n_rows_in_volume": n_rows,
        "SliceThickness": slice_thickness,
        "PixelSpacing": pixel_spacing,
    }
    out.update(extra)
    return out


def curate(rows: list[dict]) -> dict[str, pd.DataFrame]:
    return mc.curate_mri(pd.DataFrame(rows))


def selected_for_slot(results: dict[str, pd.DataFrame], slot: str) -> pd.DataFrame:
    return results["selected_long"].loc[
        results["selected_long"]["selection_slot"].eq(slot)
    ]


# -----------------------------------------------------------------------------
# Sequence classification edge cases
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "desc,expected",
    [
        ("T2 FS AX BLADE PACE DOME", "T2"),
        ("AX T2 SPIR TE 90", "T2"),
        ("T1 VIBE DIXON SANS IV CAIPI_W", "T1"),
        ("T1 VIBE DIXON ART-PORT CAIPI_W", "T1"),
        ("4D mDIXON-W", "T1"),
        ("DWI b800", "DWI"),
        ("ADC MAP", "DWI"),
        ("Apparent_Diffusion_Coefficient", "DWI"),
        ("DW-EPI rec-b-1000", "DWI"),
        ("DTI b_50", "DWI"),
        ("DIF rec b 800", "DWI"),
        ("T1 weighted IDEAL pre gad_W", "T1"),
        ("T2 eSSFSE ASPIR", "T2"),
        ("LOC 3 plans", "LOCALIZER"),
        ("Localizer 3 plans", "LOCALIZER"),
        ("KOS", "KEY_IMAGES"),
        ("Processed Images", "KEY_IMAGES"),
    ],
)
def test_sequence_classification_common_protocol_names(desc: str, expected: str):
    out = mc.annotate_mri(pd.DataFrame([row(desc)]))
    assert out.loc[0, "mri_sequence"] == expected


def test_sequence_detection_ignores_protocol_name_when_series_description_missing():
    out = mc.annotate_mri(
        pd.DataFrame([
            row(None, protocol="T2 FS AX BLADE PACE DOME"),
            row(None, protocol="T1 VIBE DIXON SANS IV CAIPI_W", volume_id="V2", series_id="SER2"),
        ])
    )
    assert out["mri_sequence"].tolist() == ["OTHER", "OTHER"]


def test_build_series_text_normalizes_case_whitespace_and_lists():
    series = pd.Series(
        {
            "SeriesDescription": "  AX   T1    DIXON  ",
            "ProtocolName": ["  PORTAL   VENOUS ", "  WATER  "],
            "ImageType": " ORIGINAL\\PRIMARY ",
        }
    )

    assert mc.build_series_text(series) == "ax t1 dixon"


# -----------------------------------------------------------------------------
# Explicit T1 phase labels
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "desc,expected_phase",
    [
        ("T1 VIBE DIXON SANS IV CAIPI_W", "NATIVE"),
        ("AX T1 DIXON SS IV_W", "NATIVE"),
        ("AX T1 DIXON ART_W", "ARTERIAL"),
        ("mDIXON port", "PORTAL_VENOUS"),
        ("AX T1 DIXON VEIN_W", "PORTAL_VENOUS"),
        ("AX T1 DIXON TARD_W", "DELAYED"),
        ("mDIXON tardif", "DELAYED"),
        ("mDIXON tardif 2h", "HEPATOBILIARY"),
        ("mDIXON tardif+2h", "HEPATOBILIARY"),
        ("mDIXON primovist 20 min", "HEPATOBILIARY"),
    ],
)
def test_explicit_t1_phase_labels(desc: str, expected_phase: str):
    out = mc.annotate_mri(pd.DataFrame([row(desc)]))
    assert out.loc[0, "mri_sequence"] == "T1"
    assert out.loc[0, "mri_perfusion_label"] == expected_phase
    assert out.loc[0, "mri_perfusion_source"].startswith("explicit_text")


def test_low_level_phase_regexes_do_not_classify_ordinal_phases():
    assert not re.search(mc.rules.RX_PHASE_NATIVE, "ph1")
    assert not re.search(mc.rules.RX_PHASE_ARTERIAL, "ph2")
    assert not re.search(mc.rules.RX_PHASE_PORTAL, "ph3")
    assert not re.search(mc.rules.RX_PHASE_DELAYED, "ph4")


def test_water_lava_alone_is_not_native():
    out = mc.annotate_mri(pd.DataFrame([row("WATER: Ax LAVA-Flex APNEE")]))

    assert out.loc[0, "mri_sequence"] == "T1"
    assert out.loc[0, "mri_perfusion_label"] == "OTHER"
    assert out.loc[0, "mri_perfusion_confidence"] == "unknown"


def test_post_gado_ordinal_phases_with_explicit_native_are_context_inferred():
    results = curate([
        row("Ax LAVA pre", series_id="SER_PRE", volume_id="VPRE", time="115900"),
        row("Ph1/Ax LAVA Gado MPh Turbo", series_id="SER_PH1", volume_id="V1", time="120000"),
        row("Ph2/Ax LAVA Gado MPh Turbo", series_id="SER_PH2", volume_id="V2", time="120100"),
        row("Ph3/Ax LAVA Gado MPh Turbo", series_id="SER_PH3", volume_id="V3", time="120200"),
    ])
    cur = results["curated"].set_index("SeriesDescription")

    assert cur.loc["Ax LAVA pre", "mri_perfusion_label"] == "NATIVE"
    assert cur.loc["Ax LAVA pre", "mri_perfusion_confidence"] == "explicit"
    assert cur.loc["Ph1/Ax LAVA Gado MPh Turbo", "mri_perfusion_label"] == "ARTERIAL"
    assert cur.loc["Ph2/Ax LAVA Gado MPh Turbo", "mri_perfusion_label"] == "PORTAL_VENOUS"
    assert cur.loc["Ph3/Ax LAVA Gado MPh Turbo", "mri_perfusion_label"] == "DELAYED"
    assert set(cur.loc[
        [
            "Ph1/Ax LAVA Gado MPh Turbo",
            "Ph2/Ax LAVA Gado MPh Turbo",
            "Ph3/Ax LAVA Gado MPh Turbo",
        ],
        "mri_perfusion_confidence",
    ]) == {"inferred"}
    assert set(cur.loc[
        [
            "Ph1/Ax LAVA Gado MPh Turbo",
            "Ph2/Ax LAVA Gado MPh Turbo",
            "Ph3/Ax LAVA Gado MPh Turbo",
        ],
        "mri_perfusion_source",
    ]) == {"ordinal_context"}


def test_missing_native_fallback_uses_water_lava_only_with_dynamic_context():
    results = curate([
        row("WATER: Ax LAVA-Flex APNEE", series_id="SER_WATER", volume_id="VW", time="115900"),
        row("Ph1/Ax LAVA Gado MPh Turbo", series_id="SER_PH1", volume_id="V1", time="120000"),
        row("Ph2/Ax LAVA Gado MPh Turbo", series_id="SER_PH2", volume_id="V2", time="120100"),
        row("Ph3/Ax LAVA Gado MPh Turbo", series_id="SER_PH3", volume_id="V3", time="120200"),
    ])
    cur = results["curated"].set_index("SeriesDescription")

    assert cur.loc["WATER: Ax LAVA-Flex APNEE", "mri_perfusion_label"] == "NATIVE"
    assert cur.loc["WATER: Ax LAVA-Flex APNEE", "mri_perfusion_confidence"] == "fallback"
    assert cur.loc["WATER: Ax LAVA-Flex APNEE", "mri_perfusion_source"] == "exam_context"
    assert cur.loc["Ph1/Ax LAVA Gado MPh Turbo", "mri_perfusion_label"] == "ARTERIAL"
    assert cur.loc["Ph2/Ax LAVA Gado MPh Turbo", "mri_perfusion_label"] == "PORTAL_VENOUS"
    assert cur.loc["Ph3/Ax LAVA Gado MPh Turbo", "mri_perfusion_label"] == "DELAYED"


def test_explicit_portal_overrides_ordinal_phase_text():
    out = mc.annotate_mri(pd.DataFrame([row("Ph1 Ax LAVA portal")]))

    assert out.loc[0, "mri_perfusion_label"] == "PORTAL_VENOUS"
    assert out.loc[0, "mri_perfusion_confidence"] == "explicit"
    assert out.loc[0, "mri_perfusion_source"] == "explicit_text"


def test_post_gado_ph1_is_never_native():
    out = mc.annotate_mri(pd.DataFrame([row("Ph1/Ax LAVA Gado MPh Turbo")]))

    assert out.loc[0, "mri_perfusion_label"] == "ARTERIAL"
    assert out.loc[0, "mri_perfusion_confidence"] == "inferred"
    assert out.loc[0, "mri_perfusion_source"] == "ordinal_context"


def test_hepatobiliary_2h_takes_priority_over_delayed_tardif():
    out = mc.annotate_mri(pd.DataFrame([row("mDIXON tardif 2h")]))
    assert out.loc[0, "mri_perfusion_label"] == "HEPATOBILIARY"
    assert "hepatobiliary" in out.loc[0, "mri_perfusion_reason"].lower()


# -----------------------------------------------------------------------------
# Special dynamic phase inference edge cases
# -----------------------------------------------------------------------------


def test_art_port_two_independent_volumes_first_arterial_second_portal():
    results = curate([
        row("T1 VIBE DIXON ART-PORT CAIPI_W", series_id="SER_ARTPORT", volume_id="V1", time="120000"),
        row("T1 VIBE DIXON ART-PORT CAIPI_W", series_id="SER_ARTPORT", volume_id="V2", time="120100"),
    ])
    cur = results["curated"].sort_values("volume_order_in_series")

    assert cur["n_volumes_in_series"].tolist() == [2, 2]
    assert cur["mri_perfusion_label"].tolist() == ["ARTERIAL", "PORTAL_VENOUS"]
    assert cur["mri_perfusion_source"].tolist() == ["volume_order_art_port", "volume_order_art_port"]


def test_art_port_two_single_volume_series_use_acquisition_order():
    results = curate([
        row(
            "T1 VIBE DIXON ART-PORT CAIPI_W",
            series_id="SER_LATER",
            volume_id="V_LATER",
            time="110000",
            AcquisitionNumber=20,
        ),
        row(
            "T1 VIBE DIXON ART-PORT CAIPI_W",
            series_id="SER_FIRST",
            volume_id="V_FIRST",
            time="120000",
            AcquisitionNumber=10,
        ),
    ])
    cur = results["curated"].sort_values("AcquisitionNumber")

    assert cur["n_volumes_in_series"].tolist() == [1, 1]
    assert cur["mri_perfusion_label"].tolist() == ["ARTERIAL", "PORTAL_VENOUS"]
    assert cur["mri_perfusion_source"].tolist() == [
        "acquisition_order_art_port",
        "acquisition_order_art_port",
    ]


def test_single_volume_art_port_row_defers_to_exam_context():
    candidate = pd.Series({
        **row("T1 VIBE DIXON ART-PORT CAIPI_W"),
        "mri_sequence": "T1",
        "volume_order_in_series": 1,
        "n_volumes_in_series": 1,
    })

    label, reason, confidence, source = mc.detect_t1_perfusion_phase(candidate)

    assert label == "OTHER"
    assert "awaiting exam acquisition context" in reason
    assert confidence == "unknown"
    assert source == mc.ART_PORT_CONTEXT_PENDING


def test_art_port_prefers_computed_acquisition_order_over_dicom_fields():
    results = curate([
        row(
            "T1 VIBE DIXON ART-PORT CAIPI_W",
            series_id="SER_SECOND",
            volume_id="V_SECOND",
            time="100000",
            AcquisitionNumber=10,
            acquisition_order=1,
        ),
        row(
            "T1 VIBE DIXON ART-PORT CAIPI_W",
            series_id="SER_FIRST",
            volume_id="V_FIRST",
            time="110000",
            AcquisitionNumber=20,
            acquisition_order=0,
        ),
    ])
    cur = results["curated"].set_index("series_id")

    assert cur.loc["SER_FIRST", "mri_perfusion_label"] == "ARTERIAL"
    assert cur.loc["SER_SECOND", "mri_perfusion_label"] == "PORTAL_VENOUS"


def test_art_port_pairs_each_dixon_component_across_two_acquisitions():
    rows = []
    for component_index, component in enumerate(["W", "in", "opp", "F"]):
        for acquisition, label in [(1, "FIRST"), (2, "SECOND")]:
            rows.append(row(
                f"T1 VIBE DIXON ART-PORT CAIPI_{component}",
                series_id=f"SER_{component}_{label}",
                volume_id=f"VOL_{component}_{label}",
                time=f"120{acquisition - 1}00",
                AcquisitionNumber=acquisition,
                acquisition_order=(acquisition - 1) * 4 + component_index,
            ))

    results = curate(rows)
    cur = results["curated"]

    assert cur.groupby("mri_perfusion_label").size().to_dict() == {
        "ARTERIAL": 4,
        "PORTAL_VENOUS": 4,
    }
    assert set(cur["mri_perfusion_source"]) == {"acquisition_order_art_port"}
    assert results["selected_long"].set_index("selection_slot").loc[
        "T1_ARTERIAL", "volume_id"
    ] == "VOL_W_FIRST"
    assert results["selected_long"].set_index("selection_slot").loc[
        "T1_PORTAL_VENOUS", "volume_id"
    ] == "VOL_W_SECOND"


def test_art_port_single_volume_falls_back_to_portal_transition():
    out = mc.annotate_mri(pd.DataFrame([row("T1 VIBE DIXON ART-PORT CAIPI_W")]))
    assert out.loc[0, "mri_perfusion_label"] == "PORTAL_VENOUS"
    assert out.loc[0, "mri_perfusion_source"] == "explicit_text_art_port_single"


def test_ssoustration_art_port_is_treated_as_subtraction():
    results = curate([
        row("T1 VIBE DIXON SSOUSTRATION ART-PORT CAIPI_W"),
    ])
    curated = results["curated"].iloc[0]

    assert bool(curated["is_subtraction"])
    assert curated["mri_perfusion_label"] == "OTHER"
    assert curated["selection_slot"] == "T1_OTHER"
    assert results["selected_long"].empty


def test_mask_multiart_first_native_rest_arterial():
    results = curate([
        row("Ax LAVA Mask+Multiart Fluoro", series_id="SER_MULTIART", volume_id="V1", time="120000"),
        row("Ax LAVA Mask+Multiart Fluoro", series_id="SER_MULTIART", volume_id="V2", time="120100"),
        row("Ax LAVA Mask+Multiart Fluoro", series_id="SER_MULTIART", volume_id="V3", time="120200"),
    ])
    cur = results["curated"].sort_values("volume_order_in_series")

    assert cur["mri_sequence"].tolist() == ["T1", "T1", "T1"]
    assert cur["mri_perfusion_label"].tolist() == ["NATIVE", "ARTERIAL", "ARTERIAL"]
    assert set(cur["mri_perfusion_source"]) == {"volume_order_mask_multiart"}


def test_mask_multiart_single_volume_series_use_acquisition_order_per_component():
    rows = []
    for component_index, component in enumerate(["W", "in"]):
        for acquisition, label in [(1, "FIRST"), (2, "SECOND")]:
            rows.append(row(
                f"Ax LAVA Mask+Multiart Fluoro_{component}",
                series_id=f"SER_{component}_{label}",
                volume_id=f"VOL_{component}_{label}",
                AcquisitionNumber=10 - acquisition,
                acquisition_order=(acquisition - 1) * 2 + component_index,
            ))

    results = curate(rows)
    cur = results["curated"].set_index("volume_id")

    for component in ["W", "in"]:
        assert cur.loc[f"VOL_{component}_FIRST", "mri_perfusion_label"] == "NATIVE"
        assert cur.loc[f"VOL_{component}_SECOND", "mri_perfusion_label"] == "ARTERIAL"
    assert set(cur["mri_perfusion_source"]) == {
        "acquisition_order_mask_multiart"
    }


def test_mask_multiart_single_volume_falls_back_to_arterial():
    out = mc.annotate_mri(pd.DataFrame([row("Ax LAVA Mask+Multiart Fluoro_W")]))

    assert out.loc[0, "mri_perfusion_label"] == "ARTERIAL"
    assert out.loc[0, "mri_perfusion_source"] == (
        "explicit_text_mask_multiart_single"
    )


def test_generic_4d_mdixon_volume_order_pre_art_port_delayed():
    results = curate([
        row("4D mDIXON-W", series_id="SER_4D", volume_id="V1", time="120000"),
        row("4D mDIXON-W", series_id="SER_4D", volume_id="V2", time="120100"),
        row("4D mDIXON-W", series_id="SER_4D", volume_id="V3", time="120200"),
        row("4D mDIXON-W", series_id="SER_4D", volume_id="V4", time="120300"),
    ])
    cur = results["curated"].sort_values("volume_order_in_series")

    assert cur["mri_perfusion_label"].tolist() == [
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    ]
    assert set(cur["mri_perfusion_source"]) == {"volume_order"}


# -----------------------------------------------------------------------------
# Explicit/pure phase should beat dynamic-inferred fallback
# -----------------------------------------------------------------------------


def test_explicit_portal_selected_over_inferred_4d_portal_even_if_same_exam():
    results = curate([
        row("4D mDIXON-W", series_id="SER_4D", volume_id="V1", time="120000"),
        row("4D mDIXON-W", series_id="SER_4D", volume_id="V2", time="120100"),
        row("4D mDIXON-W", series_id="SER_4D", volume_id="V3", time="120200"),
        row("mDIXON port", series_id="SER_PORT", volume_id="VP", time="120300"),
    ])

    portal = selected_for_slot(results, "T1_PORTAL_VENOUS")
    assert len(portal) == 1
    assert portal.iloc[0]["SeriesDescription"] == "mDIXON port"
    assert portal.iloc[0]["mri_perfusion_source"] == "explicit_text"


def test_explicit_native_selected_over_inferred_4d_native():
    results = curate([
        row("4D mDIXON-W", series_id="SER_4D", volume_id="V1", time="120000"),
        row("4D mDIXON-W", series_id="SER_4D", volume_id="V2", time="120100"),
        row("4D mDIXON-W", series_id="SER_4D", volume_id="V3", time="120200"),
        row("mDIXON pre", series_id="SER_PRE", volume_id="VPRE", time="115900"),
    ])

    pre = selected_for_slot(results, "T1_NATIVE")
    assert len(pre) == 1
    assert pre.iloc[0]["SeriesDescription"] == "mDIXON pre"
    assert pre.iloc[0]["mri_perfusion_source"] == "explicit_text"


# -----------------------------------------------------------------------------
# Dixon component and scoring edge cases
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "desc,expected_component",
    [
        ("AX T1 DIXON ART_W", "WATER"),
        ("AX T1 DIXON ART_in", "IN_PHASE"),
        ("AX T1 DIXON ART_opp", "OPPOSED_PHASE"),
        ("AX T1 DIXON phase out", "OPPOSED_PHASE"),
        ("AX T1 DIXON ART_F", "FAT"),
        ("mDIXON water only", "WATER"),
        ("mDIXON-All_BH", "DIXON_ALL"),
        ("mDIXON-Quant_BH", "FAT_FRACTION"),
    ],
)
def test_dixon_component_detection(desc: str, expected_component: str):
    out = mc.annotate_mri(pd.DataFrame([row(desc)]))
    assert out.loc[0, "dixon_component"] == expected_component


def test_water_component_scores_higher_than_in_phase_and_fat_for_same_phase():
    results = curate([
        row("AX T1 DIXON ART_W", series_id="SER_W", volume_id="VW"),
        row("AX T1 DIXON ART_in", series_id="SER_IN", volume_id="VIN"),
        row("AX T1 DIXON ART_F", series_id="SER_F", volume_id="VF"),
    ])
    cur = results["curated"].set_index("SeriesDescription")

    assert cur.loc["AX T1 DIXON ART_W", "t1_score"] > cur.loc["AX T1 DIXON ART_in", "t1_score"]
    assert cur.loc["AX T1 DIXON ART_in", "t1_score"] > cur.loc["AX T1 DIXON ART_F", "t1_score"]


# -----------------------------------------------------------------------------
# Selection and grouping edge cases
# -----------------------------------------------------------------------------


def test_best_t2_selects_axial_fatsat_motion_robust_over_plain_t2():
    results = curate([
        row("AX T2 TE 120 SENSE", series_id="SER_T2_PLAIN", volume_id="V1"),
        row("T2 FS AX BLADE PACE DOME", series_id="SER_T2_BEST", volume_id="V2"),
    ])
    t2 = selected_for_slot(results, "T2")

    assert len(t2) == 1
    assert t2.iloc[0]["SeriesDescription"] == "T2 FS AX BLADE PACE DOME"


def test_subtraction_is_not_selected_as_best_arterial_when_original_exists():
    results = curate([
        row("AX T1 DIXON ART_W_SUB", series_id="SER_SUB", volume_id="VSUB"),
        row("AX T1 DIXON ART_W", series_id="SER_ORIG", volume_id="VORIG"),
    ])
    art = selected_for_slot(results, "T1_ARTERIAL")

    assert len(art) == 1
    assert art.iloc[0]["SeriesDescription"] == "AX T1 DIXON ART_W"
    assert "SUB" not in art.iloc[0]["SeriesDescription"]


def test_same_patient_same_date_different_studies_are_not_collapsed():
    results = curate([
        row("T2 FS AX BLADE", study_id="STUDY_A", series_id="SER_A", volume_id="VA", date="2020-01-01"),
        row("AX T2 SPIR", study_id="STUDY_B", series_id="SER_B", volume_id="VB", date="2020-01-01"),
    ])
    t2 = selected_for_slot(results, "T2")

    assert len(t2) == 2
    assert set(t2["study_id"]) == {"STUDY_A", "STUDY_B"}


def test_missing_optional_metadata_does_not_crash():
    minimal = pd.DataFrame([
        {
            "patient_key": "P1",
            "date": "2020-01-01",
            "SeriesDescription": np.nan,
            "ProtocolName": "T1 VIBE DIXON SANS IV CAIPI_W",
        },
        {
            "patient_key": "P1",
            "date": "2020-01-01",
            "SeriesDescription": "T2 FS AX BLADE PACE DOME",
        },
    ])

    results = mc.curate_mri(minimal)
    cur = results["curated"]

    assert len(cur) == 2
    assert set(cur["mri_sequence"]) == {"OTHER", "T2"}
    assert not results["selected_long"].empty


def test_invalid_time_and_pixel_spacing_do_not_crash_selection():
    results = curate([
        row("T2 FS AX BLADE", time="not-a-time", pixel_spacing="bad-spacing"),
        row("T1 VIBE DIXON SANS IV CAIPI_W", series_id="SER_T1", volume_id="VT1", time=None, pixel_spacing=None),
    ])

    assert len(results["curated"]) == 2
    assert {"T2", "T1_NATIVE"}.issubset(set(results["selected_long"]["selection_slot"]))


def test_selected_wide_includes_ranked_other_candidates_per_slot():
    results = curate([
        row("T2 FS AX BLADE", series_id="T2_BEST", volume_id="T2_BEST"),
        row("T2 AX", series_id="T2_SECOND", volume_id="T2_SECOND"),
        row("T2 COR", series_id="T2_THIRD", volume_id="T2_THIRD"),
        row(
            "T1 VIBE DIXON SANS IV CAIPI_W",
            series_id="T1_ONLY",
            volume_id="T1_ONLY",
        ),
    ])

    selected_wide = results["selected_wide"]

    assert selected_wide.loc[0, "T2"].startswith("T2 FS AX BLADE")
    assert selected_wide.loc[0, "T2_other_candidates"].startswith("T2 AX")
    assert "; T2 COR" in selected_wide.loc[0, "T2_other_candidates"]
    assert "T1_NATIVE_other_candidates" in selected_wide.columns
    assert pd.isna(selected_wide.loc[0, "T1_NATIVE_other_candidates"])


def test_selected_wide_handles_an_empty_other_candidates_table():
    selected_wide = curate([
        row("T2 FS AX BLADE", series_id="T2_ONLY", volume_id="T2_ONLY"),
    ])["selected_wide"]

    assert selected_wide.loc[0, "T2"].startswith("T2 FS AX BLADE")
    assert "T2_other_candidates" in selected_wide.columns
    assert pd.isna(selected_wide.loc[0, "T2_other_candidates"])


def test_candidate_display_includes_volume_ordinal_out_of_series_total():
    results = curate([
        row(
            "T2 AX",
            series_id="T2_MULTIVOLUME",
            volume_id="T2_VOLUME_1",
            time="100000",
        ),
        row(
            "T2 AX",
            series_id="T2_MULTIVOLUME",
            volume_id="T2_VOLUME_2",
            time="110000",
        ),
    ])

    selected_wide = results["selected_wide"]

    assert "[vol=1/2, score=" in selected_wide.loc[0, "T2"]
    assert "[vol=2/2, score=" in selected_wide.loc[0, "T2_other_candidates"]
