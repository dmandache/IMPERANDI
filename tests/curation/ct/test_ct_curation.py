import pandas as pd

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
    df = pd.DataFrame([
        _base("Abdomen sans injection"),
        _base("Abdomen arterial"),
        _base("Abdomen portal venous"),
        _base("Abdomen tardif 5 min"),
    ])
    results = curate_ct(df)

    assert set(results["curated"]["ct_phase"]) == {
        "PRECONTRAST",
        "ARTERIAL",
        "PORTAL_VENOUS",
        "DELAYED",
    }
    assert set(results["selected_long"]["selection_slot"]) == {
        "CT_PRECONTRAST",
        "CT_ARTERIAL",
        "CT_PORTAL_VENOUS",
        "CT_DELAYED",
    }


def test_ct_derived_and_localizer_are_not_selected():
    df = pd.DataFrame([
        _base("Abdomen portal venous MIP", volume_id="mip"),
        _base("Topogram scout", volume_id="scout"),
        _base("Abdomen portal venous", volume_id="good"),
    ])
    results = curate_ct(df)
    selected = results["selected_long"]

    assert len(selected) == 1
    assert selected.iloc[0]["volume_id"] == "good"
