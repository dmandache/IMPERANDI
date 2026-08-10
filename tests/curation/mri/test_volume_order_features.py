import pandas as pd

from imperandi.curation.mri.curate import add_volume_order_features, curate_mri


def _volume_row(volume_id, **extra):
    row = {
        "patient_key": "p1",
        "study_id": "s1",
        "series_id": "series1",
        "volume_id": volume_id,
        "date": "2020-01-01",
        "time": "120000",
        "SeriesDescription": "Ax LAVA Gado",
    }
    row.update(extra)
    return row


def test_add_volume_order_features_handles_list_instance_numbers():
    df = pd.DataFrame(
        [
            _volume_row("vol1", InstanceNumber=[1, 2, 3], AcquisitionNumber=1),
            _volume_row("vol2", InstanceNumber=[1, 2, 3], AcquisitionNumber=2),
        ]
    )

    out = add_volume_order_features(df)

    assert out["volume_order_in_series"].tolist() == [1, 2]
    assert out["n_volumes_in_series"].tolist() == [2, 2]
    assert out["is_multivolume_series"].tolist() == [True, True]


def test_add_volume_order_features_handles_list_acquisition_numbers():
    df = pd.DataFrame(
        [
            _volume_row("vol1", InstanceNumber=[1, 2, 3], AcquisitionNumber=[1, 1]),
            _volume_row("vol2", InstanceNumber=[1, 2, 3], AcquisitionNumber=[2, 2]),
        ]
    )

    out = add_volume_order_features(df)

    assert out["volume_order_in_series"].tolist() == [1, 2]
    assert "volume_index_in_series" not in out.columns


def test_add_volume_order_features_preserves_existing_order_columns():
    df = pd.DataFrame(
        [
            _volume_row(
                "vol1",
                volume_order_in_series=7,
                n_volumes_in_series=9,
                volume_index_in_series=6,
                is_multivolume_series=True,
            )
        ]
    )

    out = add_volume_order_features(df)

    assert out["volume_order_in_series"].tolist() == [7]
    assert out["n_volumes_in_series"].tolist() == [9]
    assert "volume_index_in_series" not in out.columns
    assert out["is_multivolume_series"].tolist() == [True]


def test_add_volume_order_features_handles_minimal_dataframe():
    df = pd.DataFrame(
        [
            {"patient_key": "p1", "SeriesDescription": "T2 FS AX"},
            {"patient_key": "p1", "SeriesDescription": "T1 VIBE pre"},
        ]
    )

    out = add_volume_order_features(df)

    assert out["volume_order_in_series"].tolist() == [1, 1]
    assert "volume_index_in_series" not in out.columns
    assert out["n_volumes_in_series"].tolist() == [1, 1]
    assert out["is_multivolume_series"].tolist() == [False, False]


def test_curate_mri_handles_grouped_volume_rows_with_list_metadata():
    df = pd.DataFrame(
        [
            _volume_row(
                "vol1",
                Modality="MR",
                SeriesDescription="T2 FS AX",
                InstanceNumber=[1, 2, 3],
                AcquisitionNumber=[1, 1],
            ),
            _volume_row(
                "vol2",
                Modality="MR",
                SeriesDescription="T1 VIBE DIXON SANS IV CAIPI_W",
                InstanceNumber=[1, 2, 3],
                AcquisitionNumber=[2, 2],
            ),
        ]
    )

    results = curate_mri(df)

    assert {"curated", "selected_long", "selected_wide"}.issubset(results)
    curated = results["curated"]
    assert {"mri_sequence", "selection_slot", "selection_score"}.issubset(
        curated.columns
    )
