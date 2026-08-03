import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imperandi.qc.viewer_web_data import (
    FILTER_ALL_COLUMNS,
    filter_dataframe,
    get_image_path_columns,
    guess_ct_scan_col,
    guess_phase_col,
    guess_segmentation_cols,
    is_image_path_value,
    load_dataframe,
    validate_image_path_column,
)


def test_load_dataframe_from_csv_path(tmp_path):
    csv_path = tmp_path / "index.csv"
    expected = pd.DataFrame(
        [
            {"patient_key": "P1", "phase": "portal", "visit_order": 1},
            {"patient_key": "P2", "phase": "arterial", "visit_order": 2},
        ]
    )
    expected.to_csv(csv_path, index=False)

    loaded = load_dataframe(csv_path)

    pd.testing.assert_frame_equal(loaded, expected)


def test_load_dataframe_from_uploaded_tsv_bytes():
    payload = b"patient_key\tphase\tvisit_order\nP1\tportal\t1\nP2\tarterial\t2\n"

    loaded = load_dataframe(payload, source_name="index.tsv")

    assert loaded.to_dict("records") == [
        {"patient_key": "P1", "phase": "portal", "visit_order": 1},
        {"patient_key": "P2", "phase": "arterial", "visit_order": 2},
    ]


def test_guess_default_columns_and_segmentation_columns():
    df = pd.DataFrame(
        [
            {
                "patient_key": "P1",
                "nifti_path": "scan1.nii.gz",
                "phase": "portal",
                "mask_liver": "mask1.nii.gz",
                "mask_tumor": None,
            },
            {
                "patient_key": "P2",
                "nifti_path": "scan2.nii.gz",
                "phase": "arterial",
                "mask_liver": "",
                "mask_tumor": None,
            },
        ]
    )

    assert guess_ct_scan_col(df.columns) == "nifti_path"
    assert guess_phase_col(df.columns) == "phase"
    assert guess_segmentation_cols(df) == ["mask_liver"]


def test_image_path_detection_and_column_listing(tmp_path):
    ct_path = tmp_path / "scan1.nii.gz"
    seg_path = tmp_path / "mask1.nii.gz"
    ct_path.touch()
    seg_path.touch()

    df = pd.DataFrame(
        [
            {
                "nifti_path": str(ct_path),
                "mask_liver": str(seg_path),
                "phase": "portal",
            },
            {
                "nifti_path": "relative_scan2.nii.gz",
                "mask_liver": "",
                "phase": "arterial",
            },
        ]
    )

    assert is_image_path_value(str(ct_path)) is True
    assert is_image_path_value("portal") is False
    assert get_image_path_columns(df) == ["nifti_path", "mask_liver"]


def test_validate_image_path_column_requires_ct_values_to_all_be_paths(tmp_path):
    ct_path = tmp_path / "scan1.nii.gz"
    ct_path.touch()
    df = pd.DataFrame(
        [
            {"nifti_path": str(ct_path)},
            {"nifti_path": "not_a_path_value"},
        ]
    )

    with pytest.raises(ValueError, match="must contain file paths"):
        validate_image_path_column(
            df,
            "nifti_path",
            allow_empty=False,
            label="CT column 'nifti_path'",
        )


def test_validate_image_path_column_allows_empty_segmentation_values(tmp_path):
    seg_path = tmp_path / "mask1.nii.gz"
    seg_path.touch()
    df = pd.DataFrame(
        [
            {"mask_liver": str(seg_path)},
            {"mask_liver": ""},
            {"mask_liver": None},
        ]
    )

    validate_image_path_column(
        df,
        "mask_liver",
        allow_empty=True,
        label="Segmentation column 'mask_liver'",
    )


def test_filter_dataframe_supports_text_query_and_exact_modes():
    df = pd.DataFrame(
        [
            {"patient_key": "P1", "phase": "portal", "visit_order": 1},
            {"patient_key": "P2", "phase": "portal venous", "visit_order": 2},
            {"patient_key": "P3", "phase": "arterial", "visit_order": 3},
        ]
    )

    filtered = filter_dataframe(
        df,
        text="portal",
        column=FILTER_ALL_COLUMNS,
        mode="contains",
        query="visit_order >= 2",
    )
    exact = filter_dataframe(df, text="P3", column="patient_key", mode="exact")

    assert filtered["patient_key"].tolist() == ["P2"]
    assert exact["phase"].tolist() == ["arterial"]


def test_filter_dataframe_raises_for_unknown_column():
    df = pd.DataFrame([{"patient_key": "P1"}])

    with pytest.raises(KeyError, match="Column not found"):
        filter_dataframe(df, text="P1", column="missing", mode="contains")
