"""Tests for CT curation."""

import pandas as pd
import pytest

from imperandi.curation import curate_by_modality, split_by_modality
from imperandi.curation.ct.curate import curate_ct


def _base(desc, **extra):
    row = {
        "patient_key": "p1",
        "study_id": "s1",
        "series_id": desc,
        "volume_id": desc,
        "date": "2020-01-01",
        "Modality": "CT",
        "SeriesDescription": desc,
        "ImageType": "ORIGINAL PRIMARY AXIAL",
        "Rows": 512,
        "Columns": 512,
        "SliceThickness": 2.0,
        "n_files": 120,
    }
    row.update(extra)
    return row


def test_ct_phase_classification_and_selection_per_phase():
    df = pd.DataFrame(
        [
            _base("Abdomen sans injection"),
            _base("Abdomen arterial"),
            _base("Abdomen portal venous"),
            _base("Abdomen tardif 5 min"),
        ]
    )
    results = curate_ct(df)

    assert set(results["curated"]["ct_phase"]) == {
        "NATIVE",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    }
    assert set(results["selected_long"]["selection_slot"]) == {
        "CT_NATIVE",
        "CT_ARTERIAL",
        "CT_PORTAL_VENOUS",
        "CT_DELAYED",
    }


def test_ct_derived_is_not_selected_and_best_other_candidate_is_retained():
    df = pd.DataFrame(
        [
            _base("Abdomen portal venous MIP", volume_id="mip"),
            _base("Topogram scout", volume_id="scout"),
            _base("Abdomen portal venous", volume_id="good"),
        ]
    )
    results = curate_ct(df)
    selected = results["selected_long"]

    assert set(selected["volume_id"]) == {"good"}
    assert set(selected["selection_slot"]) == {"CT_PORTAL_VENOUS"}
    assert "mip" not in set(selected["volume_id"])


def test_ct_curation_accepts_grouped_list_valued_rows():
    df = pd.DataFrame(
        [
            _base(
                ["Abdomen arterial", "Abdomen arterial"],
                study_id=["s1"],
                series_id=["series-art"],
                volume_id="vol-art",
                Modality=["CT"],
                ImageType=["ORIGINAL PRIMARY AXIAL"],
                InstanceNumber=[1, 2, 3],
                AcquisitionNumber=[1, 1],
                n_files=[120, 120],
            ),
            _base(
                ["Abdomen portal venous", "Abdomen portal venous"],
                study_id=["s1"],
                series_id=["series-port"],
                volume_id="vol-port",
                Modality=["CT"],
                ImageType=["ORIGINAL PRIMARY AXIAL"],
                InstanceNumber=[1, 2, 3],
                AcquisitionNumber=[2, 2],
                n_files=[120, 120],
            ),
        ]
    )

    results = curate_ct(df)

    assert {"curated", "selected_long", "selected_wide"}.issubset(results)
    assert set(results["curated"]["ct_phase"]) == {"ARTERIAL", "PORTAL_VENOUS"}
    assert set(results["selected_long"]["selection_slot"]) == {
        "CT_ARTERIAL",
        "CT_PORTAL_VENOUS",
    }


@pytest.mark.parametrize(
    "image_type",
    ["DERIVED PRIMARY AXIAL", ["DERIVED", "PRIMARY", "MONOENERGETIC"]],
)
def test_diagnostic_derived_ct_remains_phaseable_and_selectable(image_type):
    result = curate_ct(pd.DataFrame([_base("Abdomen portal venous", ImageType=image_type)]))
    annotated = result["curated"].iloc[0]

    assert pd.isna(annotated["phase_applicability_reason"])
    assert annotated["phase"] == "PORTAL_VENOUS"
    assert annotated["phase_status"] == "RESOLVED"
    assert result["selected_long"]["selection_slot"].tolist() == ["CT_PORTAL_VENOUS"]


@pytest.mark.parametrize("marker", ["MIP", "MPR", "SUBTRACTION", "SECONDARY", "MAP"])
@pytest.mark.parametrize(
    "phase_text,phase",
    [
        ("native", "NATIVE"),
        ("arterial", "ARTERIAL"),
        ("portal", "PORTAL_VENOUS"),
        ("delayed", "DELAYED"),
    ],
)
def test_derived_ct_product_keeps_phase_but_is_not_selected(marker, phase_text, phase):
    result = curate_ct(
        pd.DataFrame(
            [_base(f"Abdomen {phase_text}", ImageType=["DERIVED", "PRIMARY", marker])]
        )
    )
    annotated = result["curated"].iloc[0]

    assert pd.isna(annotated["phase_applicability_reason"])
    assert annotated["phase_status"] == "RESOLVED"
    assert annotated["phase"] == phase
    assert annotated["rule_phase"] == phase
    assert result["selected_long"].empty


@pytest.mark.parametrize("marker", ["MIP", "MPR", "SUBTRACTION", "SECONDARY", "MAP"])
def test_derived_ct_description_keeps_phase_but_is_not_selected(marker):
    result = curate_ct(pd.DataFrame([_base(f"Abdomen portal venous {marker}")]))

    assert result["curated"].iloc[0]["phase"] == "PORTAL_VENOUS"
    assert result["selected_long"].empty


@pytest.mark.parametrize("marker", ["MIP", "MPR", "SUBTRACTION", "SECONDARY", "MAP"])
def test_derived_ct_product_without_phase_is_unresolved_and_not_selected(marker):
    result = curate_ct(pd.DataFrame([_base("Abdomen", ImageType=["DERIVED", marker])]))

    assert result["curated"].iloc[0]["phase_status"] == "UNRESOLVED"
    assert result["selected_long"].empty


def test_derived_ct_without_phase_evidence_remains_unresolved():
    result = curate_ct(
        pd.DataFrame([_base("Abdomen", ImageType="DERIVED PRIMARY AXIAL")])
    )
    annotated = result["curated"].iloc[0]

    assert annotated["phase"] == "OTHER"
    assert annotated["phase_status"] == "UNRESOLVED"


def test_modality_router_accepts_grouped_list_valued_ct_rows():
    df = pd.DataFrame(
        [
            _base(
                ["Abdomen portal venous"],
                study_id=["s1"],
                series_id=["series-port"],
                volume_id="vol-port",
                Modality=["CT"],
                ImageType=["ORIGINAL PRIMARY AXIAL"],
                n_files=[120],
            )
        ]
    )

    ct, mr, other = split_by_modality(df)
    assert len(ct) == 1
    assert mr.empty
    assert other.empty

    results = curate_by_modality(df)
    assert results["ct"] is not None
    assert results["mri"] is None
    assert results["curated_all"].iloc[0]["ct_phase"] == "PORTAL_VENOUS"
