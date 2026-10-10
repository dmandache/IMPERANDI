"""Ordered text evidence must remain authoritative through curation and scoring."""

import pandas as pd
import pytest

from imperandi.curation.ct.curate import curate_ct
from imperandi.curation.mri.curate import curate_mri


def dataset(modality, series, protocol, study="", count=4):
    prefix = "AX T1 " if modality == "MR" else "AX "
    return pd.DataFrame(
        [
            {
                "patient_key": "P1",
                "study_id": "S1",
                "series_id": "SER1",
                "volume_id": f"v{rank}",
                "date": "2020-01-01",
                "Modality": modality,
                "SeriesDescription": prefix + series,
                "ProtocolName": protocol,
                "StudyDescription": study,
                "ImageType": "ORIGINAL PRIMARY AXIAL",
                "time": f"120{rank}00",
                "acquisition_order": rank,
                "Rows": 512,
                "Columns": 512,
                "n_files": 100,
                "SliceThickness": 3,
            }
            for rank in range(count)
        ]
    )


def curate(df, columns=None):
    modality = df.Modality.iloc[0]
    function = curate_mri if modality == "MR" else curate_ct
    return function(
        df,
        phase_curation={
            "strategies": [{"type": "rules"}],
            "text_columns": {
                modality: columns
                or ["SeriesDescription", "ProtocolName", "StudyDescription"]
            },
        },
    )["curated"].sort_values("volume_id")


@pytest.mark.parametrize("modality", ["CT", "MR"])
@pytest.mark.parametrize("later_native", ["pre contrast", "w/o contrast"])
def test_generic_post_blocks_later_native_throughout_curation(modality, later_native):
    out = curate(dataset(modality, "post", "", later_native))
    assert out.phase.tolist() == ["OTHER"] * 4
    assert out.phase_status.eq("UNRESOLVED").all()
    assert out.phase_text_column.eq("SeriesDescription").all()


@pytest.mark.parametrize("modality", ["CT", "MR"])
def test_uninformative_earlier_fields_allow_phase_fallback(modality):
    out = curate(dataset(modality, "", "", "w/o contrast"))
    assert out.phase.eq("NATIVE").all()
    assert out.phase_text_column.eq("StudyDescription").all()


@pytest.mark.parametrize("modality", ["CT", "MR"])
def test_reordering_controls_generic_evidence_and_explicit_phases(modality):
    out = curate(
        dataset(modality, "post", "", "w/o contrast"),
        ["StudyDescription", "SeriesDescription"],
    )
    assert out.phase.eq("NATIVE").all()
    assert out.phase_text_column.eq("StudyDescription").all()


@pytest.mark.parametrize("earlier", ["native", "arterial", "portal", "delayed"])
@pytest.mark.parametrize(
    "later", ["ART-PORT", "ART PORT LATE", "Mask Multiart", "dynamic"]
)
def test_later_profiles_cannot_replace_earlier_named_phase(earlier, later):
    out = curate(dataset("MR", earlier, later))
    expected = {
        "native": "NATIVE",
        "arterial": "ARTERIAL",
        "portal": "PORTAL_VENOUS",
        "delayed": "DELAYED",
    }[earlier]
    assert out.phase.eq(expected).all()
    assert out.mri_perfusion_source.eq("explicit_text").all()


def test_earlier_dynamic_controls_all_volumes_despite_later_named_phase():
    df = dataset("MR", "dynamic", "portal")
    df.loc[df.volume_id.eq("v3"), "time"] = "120400"
    out = curate(df)
    assert out.phase.tolist() == ["NATIVE", "ARTERIAL", "PORTAL_VENOUS", "DELAYED"]
    assert out.phase_text_column.eq("SeriesDescription").all()


@pytest.mark.parametrize("series,protocol", [("post", "Ph1"), ("Ph1", "post contrast")])
def test_ordinal_context_cannot_be_synthesized_across_columns(series, protocol):
    out = curate(dataset("MR", series, protocol, count=1))
    assert out.phase.tolist() == ["OTHER"]


def test_post_only_dynamics_never_infer_native_or_assume_phase_offsets():
    out = curate(dataset("MR", "Multiphase post", "w/o"))
    assert out.phase.tolist() == ["OTHER"] * 4


@pytest.mark.parametrize("later", ["ART-PORT", "Mask Multiart", "dynamic"])
def test_earlier_named_phase_wins_in_single_volume_acquisition_groups(later):
    df = dataset("MR", "native", later, count=3)
    df.series_id = ["series1", "series2", "series3"]
    out = curate(df)
    assert out.phase.tolist() == ["NATIVE"] * 3
    assert out.mri_perfusion_source.eq("explicit_text").all()


def test_post_only_single_volume_acquisitions_do_not_gain_native_from_exam_order():
    df = dataset("MR", "Multiphase post", "w/o", count=3)
    df.series_id = ["series1", "series2", "series3"]
    out = curate(df)
    assert out.phase.tolist() == ["OTHER"] * 3


@pytest.mark.parametrize("modality", ["CT", "MR"])
def test_later_mixed_context_remains_available_for_uninformative_series(modality):
    out = curate(dataset(modality, "", "", "w/wo", count=2))
    assert out.phase.tolist() == ["NATIVE", "OTHER"]
    assert out.phase_text_column.eq("StudyDescription").all()


@pytest.mark.parametrize("modality", ["CT", "MR"])
def test_invalid_earlier_duration_blocks_later_pure_phase(modality):
    out = curate(dataset(modality, "100 sec", "arterial", count=1))
    assert out.phase.tolist() == ["OTHER"]
    assert out.phase_text_column.tolist() == ["SeriesDescription"]


@pytest.mark.parametrize("modality", ["CT", "MR"])
@pytest.mark.parametrize("later", ["portal", "post contrast"])
def test_earlier_mixed_context_ignores_later_phase_or_native_blocker(modality, later):
    out = curate(dataset(modality, "pre/post", later, count=2))
    assert out.phase.tolist() == ["NATIVE", "OTHER"]
    assert "SeriesDescription=" in out.phase_reason.iloc[0]


@pytest.mark.parametrize("modality", ["CT", "MR"])
def test_later_mixed_context_cannot_replace_earlier_post(modality):
    out = curate(dataset(modality, "post", "", "w/wo", count=2))
    assert out.phase.tolist() == ["OTHER", "OTHER"]


@pytest.mark.parametrize("later", ["PDFF map", "R2* map", "mDIXON fat"])
def test_earlier_dixon_controls_component_and_product_scoring(later):
    out = curate(dataset("MR", "mDIXON water", later, count=1))
    assert out.dixon_component.tolist() == ["WATER"]
    assert not out.is_quant_or_report.any()
    assert not out.is_derived_low_value.any()


@pytest.mark.parametrize(
    "later,flag", [("Subtraction", "is_subtraction"), ("MPR", "is_mip_mpr")]
)
def test_water_component_does_not_suppress_compatible_product_evidence(later, flag):
    out = curate(dataset("MR", "mDIXON water", later, count=1))
    assert out.dixon_component.tolist() == ["WATER"]
    assert out[flag].all()
    assert out.is_derived_low_value.all()


def test_product_scoring_uses_earliest_product_marker():
    out = curate(dataset("MR", "MPR", "Subtraction", count=1))
    assert out.is_mip_mpr.all()
    assert not out.is_subtraction.any()


def test_dixon_quantitative_fallback_and_column_reordering():
    df = dataset("MR", "VIBE", "PDFF map", count=1)
    out = curate(df)
    assert out.dixon_component.tolist() == ["FAT_FRACTION"]
    assert out.is_derived_low_value.all()
    df.SeriesDescription = "AX T1 mDIXON water"
    out = curate(df, ["ProtocolName", "SeriesDescription"])
    assert out.dixon_component.tolist() == ["FAT_FRACTION"]


def test_structured_image_type_remains_independent_of_text_order():
    df = dataset("MR", "mDIXON water", "PDFF map", count=1)
    df.ImageType = "DERIVED PRIMARY F F"
    out = curate(df)
    assert out.dixon_component.tolist() == ["FAT"]
    assert out.dixon_component_source.tolist() == ["image_type"]


def test_structured_quantitative_product_marker_is_not_suppressed_by_text():
    df = dataset("MR", "mDIXON water", "", count=1)
    df.ImageType = "DERIVED PRIMARY QUANT"
    out = curate(df)
    assert out.dixon_component.tolist() == ["WATER"]
    assert out.is_quant_or_report.all()
    assert out.is_derived_low_value.all()


def test_feature_scoring_does_not_mix_competing_fields():
    out = curate(dataset("MR", "FS BH native", "T2 HASTE PACE post", count=1))
    assert out.mri_sequence.tolist() == ["T1"]
    assert out.plane.tolist() == ["AXIAL"]
    assert out.is_breath_hold.all()
    assert not out.is_resp_triggered.any()
    assert out.is_t2_fatsat.all()
    assert not out.is_t2_haste_ssfse.any()
    assert not out.is_dynamic_t1_text.any()


def test_native_fallback_cannot_borrow_gre_from_a_later_sequence_field():
    out = curate(dataset("MR", "spin echo", "VIBE", count=1))
    assert not out.is_3d_gre.any()


@pytest.mark.parametrize("later", ["bolus monitoring", "localizer"])
def test_ct_later_monitoring_cannot_override_earlier_acquisition(later):
    out = curate(dataset("CT", "portal", later, count=1))
    assert out.phase.tolist() == ["PORTAL_VENOUS"]
    assert out.phase_status.tolist() == ["RESOLVED"]
