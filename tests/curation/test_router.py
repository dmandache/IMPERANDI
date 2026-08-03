import pandas as pd

from imperandi.curation import curate_by_modality, split_by_modality


def test_split_by_modality_and_curate_both_modalities():
    df = pd.DataFrame(
        [
            {
                "patient_key": "p1",
                "study_id": "ct-study",
                "series_id": "ct-series",
                "volume_id": "ct-vol",
                "date": "2020-01-01",
                "Modality": "CT",
                "SeriesDescription": "Abdomen portal venous",
                "ImageType": "ORIGINAL PRIMARY AXIAL",
                "Rows": 512,
                "Columns": 512,
                "SliceThickness": 2.0,
                "n_files": 120,
            },
            {
                "patient_key": "p1",
                "study_id": "mr-study",
                "series_id": "mr-series",
                "volume_id": "mr-vol",
                "date": "2020-01-01",
                "Modality": "MR",
                "SeriesDescription": "AX T1 DIXON VEIN_W",
            },
        ]
    )

    ct, mr, other = split_by_modality(df)
    assert len(ct) == 1
    assert len(mr) == 1
    assert other.empty

    results = curate_by_modality(df)
    assert results["ct"] is not None
    assert results["mri"] is not None
    assert set(results["selected_long_all"]["curation_modality"]) == {"CT", "MR"}


def test_router_can_disable_one_builtin_curator_without_dropping_rows():
    df = pd.DataFrame(
        [
            {
                "patient_key": "p1",
                "study_id": "mr-study",
                "series_id": "mr-series",
                "volume_id": "mr-vol",
                "date": "2020-01-01",
                "Modality": "MR",
                "SeriesDescription": "AX T2 FS",
            }
        ]
    )

    results = curate_by_modality(df, curators=["builtin:liver_ct"])

    assert len(results["curated_all"]) == 1
    assert results["curated_all"].loc[0, "curation_modality"] == "MR"
    assert "mri_sequence" not in results["curated_all"]


def test_missing_modality_is_never_silently_routed_to_ct():
    df = pd.DataFrame({"patient_key": ["p1"], "volume_id": ["v1"]})

    ct, mr, other = split_by_modality(df)

    assert ct.empty
    assert mr.empty
    assert len(other) == 1
