import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from imperandi.qc.viewer_web_data import (
    FILTER_ALL_COLUMNS,
    filter_dataframe,
    get_image_path_columns,
    guess_ct_scan_col,
    guess_phase_col,
    guess_segmentation_cols,
    is_empty_value,
    is_image_path_column,
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


def test_load_dataframe_validates_uploaded_sources():
    with pytest.raises(ValueError, match="No dataframe source"):
        load_dataframe(None)
    with pytest.raises(ValueError, match="source name is required"):
        load_dataframe(b"patient_key\nP1\n")
    with pytest.raises(ValueError, match="Unsupported dataframe format"):
        load_dataframe(b"data", source_name="index.xlsx")


@pytest.mark.parametrize(
    ("suffix", "writer"),
    [
        (".tab", lambda df, path: df.to_csv(path, sep="\t", index=False)),
        (".txt", lambda df, path: df.to_csv(path, sep=";", index=False)),
        (".json", lambda df, path: df.to_json(path, orient="records")),
        (".pkl", lambda df, path: df.to_pickle(path)),
        (".pickle", lambda df, path: df.to_pickle(path)),
    ],
)
def test_load_dataframe_supports_additional_formats(tmp_path, suffix, writer):
    expected = pd.DataFrame({"patient_key": ["P1"], "visit_order": [1]})
    path = tmp_path / f"index{suffix}"
    writer(expected, path)

    result = load_dataframe(path)

    pd.testing.assert_frame_equal(result, expected)


def test_load_dataframe_dispatches_parquet_reader(tmp_path, monkeypatch):
    expected = pd.DataFrame({"patient_key": ["P1"]})
    path = tmp_path / "index.parquet"
    path.touch()
    seen = []

    def fake_read_parquet(source):
        seen.append(source)
        return expected

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    assert load_dataframe(path) is expected
    assert seen == [path.resolve()]


def test_empty_and_image_path_detection_edge_cases(tmp_path, monkeypatch):
    image = tmp_path / "scan.nii.gz"
    image.touch()
    folder = tmp_path / "images"
    folder.mkdir()

    assert is_empty_value(None)
    assert is_empty_value(pd.NA)
    assert is_empty_value("  ")
    assert not is_empty_value([1, 2])
    assert not is_image_path_value(None)
    assert is_image_path_value(f"file://{image}")
    assert not is_image_path_value(folder)
    assert is_image_path_value("relative/scan.without_known_suffix")
    assert is_image_path_value("scan.MGZ")

    monkeypatch.setattr(Path, "exists", lambda _self: (_ for _ in ()).throw(OSError()))
    assert not is_image_path_value("plain_value")


def test_image_path_column_ratio_and_empty_controls():
    assert not is_image_path_column(pd.Series([None, ""]))
    assert not is_image_path_column(
        pd.Series(["scan.nii.gz", "not-a-path"]), min_valid_ratio=0.75
    )
    assert not is_image_path_column(pd.Series(["scan.nii.gz", None]), allow_empty=False)
    assert is_image_path_column(pd.Series(["scan.nii.gz", None]), allow_empty=True)


def test_validate_image_path_column_reports_selection_and_empty_errors():
    df = pd.DataFrame({"mask_liver": [None, ""]})

    with pytest.raises(KeyError, match="Column not found"):
        validate_image_path_column(df, "missing", allow_empty=True)
    with pytest.raises(ValueError, match="contains empty values"):
        validate_image_path_column(
            df,
            "mask_liver",
            allow_empty=False,
            label="Liver mask",
        )
    with pytest.raises(ValueError, match="does not contain any usable paths"):
        validate_image_path_column(df, "mask_liver", allow_empty=True)


def test_guess_column_fallbacks_and_preferences():
    assert guess_ct_scan_col(["custom", "ct_path"], preferred="custom") == "custom"
    assert guess_ct_scan_col(["patient", "derived_nifti_path"]) == "derived_nifti_path"
    assert guess_ct_scan_col(["patient", "volume_path"]) == "volume_path"
    assert guess_ct_scan_col(["patient", "description"]) == "patient"
    assert guess_ct_scan_col([]) is None

    assert guess_phase_col(["reviewed", "phase"], preferred="reviewed") == "reviewed"
    assert guess_phase_col(["totalseg_phase"]) == "totalseg_phase"
    assert guess_phase_col(["description"]) is None


def test_guess_segmentation_columns_supports_legacy_and_explicit_preferences():
    df = pd.DataFrame(
        {
            "liver_path": ["liver.nii.gz"],
            "liver_tumor_path": [None],
            "custom_mask": ["custom.nii.gz"],
        }
    )

    assert guess_segmentation_cols(df) == ["liver_path"]
    assert guess_segmentation_cols(df, preferred="custom_mask") == ["custom_mask"]
    assert guess_segmentation_cols(
        df, preferred=["missing", "liver_tumor_path", "custom_mask"]
    ) == ["custom_mask"]


def test_filter_dataframe_supports_empty_regex_case_and_validation_paths():
    df = pd.DataFrame(
        [
            {"patient_key": "P1", "phase": "Portal"},
            {"patient_key": "P2", "phase": "arterial"},
        ]
    )

    unchanged = filter_dataframe(df, text=" ", query="patient_key == 'P2'")
    regex = filter_dataframe(df, text="^P[12]$", column="patient_key", mode="regex")
    case_sensitive = filter_dataframe(
        df,
        text="portal",
        column="phase",
        mode="exact",
        case_sensitive=True,
    )

    assert unchanged["patient_key"].tolist() == ["P2"]
    assert regex["patient_key"].tolist() == ["P1", "P2"]
    assert case_sensitive.empty
    with pytest.raises(ValueError, match="Unsupported filter mode"):
        filter_dataframe(df, text="P1", column="patient_key", mode="fuzzy")
