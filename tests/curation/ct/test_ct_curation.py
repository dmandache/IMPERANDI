import pandas as pd

from imperandi.curation import curate_by_modality, split_by_modality
from imperandi.curation.ct.curate import curate_ct


def _base(desc, **extra):
    row = {
        "patient_id": "p1",
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


def test_ct_derived_and_localizer_are_not_selected():
    df = pd.DataFrame(
        [
            _base("Abdomen portal venous MIP", volume_id="mip"),
            _base("Topogram scout", volume_id="scout"),
            _base("Abdomen portal venous", volume_id="good"),
        ]
    )
    results = curate_ct(df)
    selected = results["selected_long"]

    assert len(selected) == 1
    assert selected.iloc[0]["volume_id"] == "good"


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
