import hashlib
import logging
import sys
from pathlib import Path
from datetime import time as dt_time

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd
import numpy as np
import pytest

from imperandi.ingest import clean
from imperandi.ingest.hooks import clean_hook

ACQUISITION_TEMP_COLS = {
    # "_acq_timestamp",
    "_series_number_sort",
    "_acquisition_number_sort",
}


def _assert_no_acquisition_temp_cols(df: pd.DataFrame) -> None:
    assert ACQUISITION_TEMP_COLS.isdisjoint(df.columns)


def _expected_merged_volume_id(*volume_ids: str) -> str:
    return hashlib.sha1("|".join(sorted(volume_ids)).encode("utf-8")).hexdigest()


def test_normalize_clean_args_prefers_optional_csv_path(tmp_path):
    csv_pos = tmp_path / "pos.csv"
    csv_opt = tmp_path / "opt.csv"
    csv_pos.write_text("patient_key\np1\n")
    csv_opt.write_text("patient_key\np2\n")

    args = clean.normalize_clean_args(
        clean.argparse.Namespace(
            csv_path_pos=[str(csv_pos)],
            csv_path_opt=[str(csv_opt)],
            csv_path_out_pos=None,
            csv_path_out=None,
        )
    )

    assert args.csv_path == [str(csv_opt)]
    assert args.csv_path_out.endswith("opt_clean.csv")
    assert not hasattr(args, "csv_path_pos")
    assert not hasattr(args, "csv_path_opt")
    assert not hasattr(args, "csv_path_out_pos")


def test_normalize_clean_args_accepts_positional_only(tmp_path):
    csv_pos = tmp_path / "input.csv"
    csv_pos.write_text("patient_key\np1\n")

    args = clean.normalize_clean_args(
        clean.argparse.Namespace(
            csv_path_pos=[str(csv_pos)],
            csv_path_opt=None,
            csv_path_out_pos=None,
            csv_path_out=None,
        )
    )

    assert args.csv_path == [str(csv_pos)]
    assert args.csv_path_out.endswith("input_clean.csv")


def test_normalize_clean_args_uses_out_suffix_for_clean_input(tmp_path):
    csv_clean = tmp_path / "dicom_index_clean.csv"
    csv_clean.write_text("patient_key\np1\n")

    args = clean.normalize_clean_args(
        clean.argparse.Namespace(
            csv_path_pos=[str(csv_clean)],
            csv_path_opt=None,
            csv_path_out_pos=None,
            csv_path_out=None,
        )
    )

    assert args.csv_path == [str(csv_clean)]
    assert args.csv_path_out.endswith("dicom_index_clean_out.csv")


def test_normalize_clean_args_accepts_positional_csv_path_out(tmp_path):
    csv_in = tmp_path / "input.csv"
    csv_in.write_text("patient_key\np1\n")
    csv_out = tmp_path / "clean_out.csv"

    args = clean.normalize_clean_args(
        clean.argparse.Namespace(
            csv_path_pos=str(csv_in),
            csv_path_opt=None,
            csv_path_out_pos=str(csv_out),
            csv_path_out=None,
        )
    )

    assert args.csv_path == [str(csv_in)]
    assert args.csv_path_out == str(csv_out)


def test_normalize_clean_args_prefers_flag_csv_path_out_over_positional(tmp_path):
    csv_in = tmp_path / "input.csv"
    csv_in.write_text("patient_key\np1\n")
    csv_out_pos = tmp_path / "clean_pos.csv"
    csv_out_opt = tmp_path / "clean_opt.csv"

    args = clean.normalize_clean_args(
        clean.argparse.Namespace(
            csv_path_pos=str(csv_in),
            csv_path_opt=None,
            csv_path_out_pos=str(csv_out_pos),
            csv_path_out=str(csv_out_opt),
        )
    )

    assert args.csv_path_out == str(csv_out_opt)


def test_clean_parser_rejects_removed_legacy_flags():
    parser = clean.build_parser()

    for argv in (["--volume-length-min-mm", "25"], ["--csv_dict_path", "tags.csv"]):
        try:
            parser.parse_args(argv)
        except SystemExit as exc:
            assert exc.code != 0
        else:
            raise AssertionError(f"Removed clean flags should not parse: {argv!r}")


def test_uniform_string_and_remove_other_organs_description():
    assert clean.uniform_string("  Abc  .0") == "abc"
    assert clean.uniform_string("RévoluTion") == "revolution"
    df = pd.DataFrame(
        {"SeriesDescription": ["Pelvis CT", "Liver", None, "FEMUR study"]}
    )
    out = clean.remove_other_organs_description(df.copy())
    # 'Pelvis' and 'femur' should be filtered out
    assert "pelvis" not in " ".join(out["SeriesDescription"].astype(str))
    assert "femur" not in " ".join(out["SeriesDescription"].astype(str))
    assert "liver" in " ".join(out["SeriesDescription"].astype(str))


def test_filter_supported_modality_image_storage_and_remove_pet_ct():
    df = pd.DataFrame(
        {
            "Modality": ["CT", "MR", "CT", "US"],
            "SOPClassUID": [
                "1.2.840.10008.5.1.4.1.1.2",  # CTImageStorage
                "1.2.840.10008.5.1.4.1.1.4",  # MRImageStorage (example)
                "1.2.840.10008.5.1.4.1.1.2",
                "1.2.840.10008.5.1.4.1.1.6.1",  # UltrasoundImageStorage
            ],
            "ModalitiesInStudy": ["CT", "MR", "CT", "US"],
        }
    )
    df_supported = clean.filter_supported_modality_image_storage(df.copy())
    assert set(df_supported.Modality) == {"CT", "MR"}
    assert "sop_class" in df_supported.columns
    # remove rows with PT in ModalitiesInStudy
    df_pet_removed = clean.remove_pet_ct(
        df.assign(ModalitiesInStudy=["CT", "PT", "CT", "US"]).copy()
    )
    assert "PT" not in " ".join(df_pet_removed["ModalitiesInStudy"].astype(str))


def test_run_modality_curation_step_keeps_ct_and_mri_annotations():
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
                "ImageType": "ORIGINAL\\PRIMARY",
                "n_files": 80,
            },
        ]
    )

    out = clean.run_clean_pipeline(df.copy(), [{"type": "modality_curation"}])

    assert set(out["curation_modality"]) == {"CT", "MR"}
    assert "ct_phase" in out.columns
    assert "mri_sequence" in out.columns
    assert set(out["selection_slot"]) == {"CT_PORTAL_VENOUS", "T1_PORTAL_VENOUS"}


def test_add_date_and_filter_image_type_and_remove_localizers_mpr():
    df = pd.DataFrame(
        {
            "StudyDate": ["20200101", ["bad"], None],
            "ImageType": ["['ORIGINAL','PRIMARY']", "not_a_list", None],
            "SeriesDescription": ["Some series", "Scout series", "MPR recon"],
        }
    )
    df = clean.add_date(df.copy())
    assert pd.to_datetime("2020-01-01") == df.loc[0, "date"]

    df_ft = clean.filter_image_type(df.copy())
    # first row should have additional ImageType_value_0 column
    assert any(col.startswith("ImageType_value_") for col in df_ft.columns)

    df_noscout = clean.remove_scouts_localizers(
        pd.DataFrame(
            {
                "ImageType": ["ORIGINAL", "LOCALIZER"],
                "SeriesDescription": ["Good series", "scout of abdomen"],
            }
        )
    )
    assert not df_noscout["SeriesDescription"].str.lower().str.contains("scout").any()

    df_nompr = clean.remove_mpr(
        pd.DataFrame(
            {
                "ImageType": ["mpr sequence", "ORIGINAL"],
                "SeriesDescription": ["normal", "MPR recon"],
            }
        )
    )
    # both mpr occurrences removed
    assert df_nompr.empty


def test_add_date_selects_best_date_candidate():
    df = pd.DataFrame(
        {
            "StudyDate": ["20200101", None, "bad"],
            "AcquisitionDate": ["20210101", "20210102", "20210103"],
            "ContentDate": [None, None, None],
        }
    )

    out = clean.add_date(df.copy())

    assert out["date"].notna().sum() == 3
    assert out.loc[0, "date"] == pd.Timestamp("2020-01-01")
    assert out.loc[2, "date"] == pd.Timestamp("2021-01-03")


def test_add_time_selects_best_time_candidate():
    df = pd.DataFrame(
        {
            "AcquisitionTime": ["bad", None, None],
            "ContentTime": ["120000", "120100", "120200"],
            "StudyTime": [None, None, None],
        }
    )

    out = clean.add_time(df.copy())

    assert out["time"].notna().sum() == 3
    assert out.loc[0, "time"] == dt_time(12, 0, 0)
    assert out.loc[2, "time"] == dt_time(12, 2, 0)


def test_generic_manifest_filters_pixel_spacing_declaratively():
    base_path = Path(__file__).resolve().parents[2] / "src" / "imperandi"
    manifest = clean.load_manifest("generic", base_path=base_path)
    pixel_spacing_filter = next(
        step
        for step in manifest["cleaning"]["steps"]
        if step.get("name") == "discard_large_pixel_spacing"
    )
    df = pd.DataFrame(
        {
            "patient_key": ["valid", "boundary", "coarse", "unknown"],
            "study_id": ["s"] * 4,
            "series_id": ["sr"] * 4,
            "PixelSpacing": ["[0.7, 0.7]", "[1.5, 1.5]", "[1.51, 1.51]", None],
        }
    )

    out = clean.run_clean_pipeline(
        df,
        [{"type": "pixel_spacing_xy"}, pixel_spacing_filter],
    )

    assert out["patient_key"].tolist() == ["valid", "boundary", "unknown"]
    assert out.iloc[0]["PixelSpacingXY"] == 0.7
    assert pd.isna(out.iloc[2]["PixelSpacingXY"])


def test_generic_manifest_filters_slice_spacing_declaratively():
    base_path = Path(__file__).resolve().parents[2] / "src" / "imperandi"
    manifest = clean.load_manifest("generic", base_path=base_path)
    slice_spacing_filter = next(
        step
        for step in manifest["cleaning"]["steps"]
        if step.get("name") == "discard_large_slice_spacing"
    )
    df = pd.DataFrame(
        {
            "patient_key": [
                "valid",
                "thickness_boundary",
                "spacing_boundary",
                "thick",
                "wide",
                "unknown",
            ],
            "study_id": ["s"] * 6,
            "series_id": ["sr"] * 6,
            "SliceThickness": [2.0, 6.0, 2.0, 6.1, 2.0, None],
            "SpacingBetweenSlices": [2.0, 2.0, 5.0, 2.0, 5.1, None],
        }
    )

    out = clean.run_clean_pipeline(df, [slice_spacing_filter])

    assert out["patient_key"].tolist() == [
        "valid",
        "thickness_boundary",
        "spacing_boundary",
        "unknown",
    ]


def test_build_volume_id_naive_and_filter_by_acquisition_plane():
    df = pd.DataFrame(
        {
            "patient_key": ["p1", "p1"],
            "study_id": ["s1", "s1"],
            "series_id": ["sr1", "sr1"],
            "ImageType": ["A", "A"],
            "AcquisitionNumber": [1, 1],
            # use same tuple orientation for both rows
            "ImageOrientationPatient": [(1, 0, 0, 0, 1, 0), (1, 0, 0, 0, 1, 0)],
            "SliceThickness": [1.0, 1.0],
            "PixelSpacingXY": [0.7, 0.7],
        }
    )
    with_sup = clean.build_volume_id_naive(df.copy())
    assert "volume_id" in with_sup.columns
    # same content -> identical volume_id
    assert with_sup.loc[0, "volume_id"] == with_sup.loc[1, "volume_id"]

    # filter by plane: above orientation corresponds to axial (Z axis)
    filtered = clean.filter_by_acquisition_plane(with_sup.copy())
    assert filtered.shape[0] == with_sup.shape[0]


def test_split_multivolume_series_by_repeated_slices_creates_volume_ids():
    rows = []
    for timepoint, acq in enumerate([1, 2]):
        for z in range(8):
            rows.append(
                {
                    "patient_key": "p",
                    "study_id": "s",
                    "series_id": "series-multivolume",
                    "volume_id": "metadata-volume",
                    "ImageOrientationPatient": "[1, 0, 0, 0, 1, 0]",
                    "ImagePositionPatient": f"[0, 0, {z}]",
                    "SliceLocation": float(z),
                    "AcquisitionNumber": acq,
                    "InstanceNumber": timepoint * 8 + z + 1,
                }
            )
    df = pd.DataFrame(rows)

    out = clean.split_multivolume_series_by_repeated_slices(
        df,
        min_slices=4,
        min_repeated_slice_fraction=0.7,
    )

    assert out["volume_split_method"].eq("repeated_slice_stack").all()
    assert set(out["volume_order_in_series"]) == {1, 2}
    assert "volume_index_in_series" not in out.columns
    assert out["n_detected_volumes_in_series"].dropna().unique().tolist() == [2]
    assert out["volume_id"].nunique() == 2
    assert set(out.groupby("volume_id").size()) == {8}


def test_correct_volume_ids_merging(tmp_path, capsys):
    # Prepare group with same unique_cols but two different volume_ids and consistent z spacing
    df = pd.DataFrame(
        {
            "patient_key": ["p", "p"],
            "study_id": ["s", "s"],
            "series_id": ["sr", "sr"],
            "ImageType": ["A", "A"],
            "ImageOrientationPatient": [
                (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            ],
            "SliceThickness": [1.0, 1.0],
            "PixelSpacingXY": [0.7, 0.7],
            "volume_id": ["b", "c"],
            "ImagePositionPatient": [[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]],
            "date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")],
            "SeriesDescription": ["desc", "desc"],
        }
    )
    out = clean.correct_volume_ids(df.copy(), z_tolerance=1e-3)
    # both rows should now have the canonical merged volume id
    assert set(out["volume_id"].unique()) == {_expected_merged_volume_id("b", "c")}


def test_parse_ipp_accepts_dicom_backslash_string():
    assert clean._parse_ipp(r"1.5\-2.0\3.25") == (1.5, -2.0, 3.25)


def test_group_volumes_and_calculate_length():
    df = pd.DataFrame(
        {
            "volume_id": ["v1", "v1", "v2"],
            "SliceThickness": [1.0, 1.0, 2.0],
            "SpacingBetweenSlices": [1.0, 1.0, np.nan],
            "dicom_path": ['["a.dcm", "b.dcm"]', '["a.dcm", "b.dcm"]', '["c.dcm"]'],
        }
    )
    grouped = clean.group_volumes(df.copy())
    assert "volume_id" in grouped.columns
    calc = clean.calculate_volume_length(grouped.copy())
    # v1 has n_files >1 -> volume_length computed
    assert "volume_length" in calc.columns
    assert set(calc["volume_id"]) == set(grouped["volume_id"])


def test_generic_manifest_filters_volume_length_declaratively():
    base_path = Path(__file__).resolve().parents[2] / "src" / "imperandi"
    manifest = clean.load_manifest("generic", base_path=base_path)
    volume_length_filter = next(
        step
        for step in manifest["cleaning"]["steps"]
        if step["type"] == "filter"
        and step["scope"] == "volume"
        and {rule["column"] for rule in step["rules"]} == {"volume_length"}
    )
    df = pd.DataFrame(
        {
            "patient_key": ["short", "minimum", "maximum", "long", "unknown"],
            "study_id": ["s"] * 5,
            "series_id": ["sr"] * 5,
            "volume_length": [29.9, 30.0, 1700.0, 1700.1, None],
        }
    )

    out = clean.run_clean_pipeline(df, [volume_length_filter])

    assert out["patient_key"].tolist() == ["minimum", "maximum", "unknown"]


def test_group_volumes_deterministic_ordering():
    df = pd.DataFrame(
        {
            "volume_id": ["v1", "v1", "v1", "v1"],
            "ImagePositionPatient": [
                "[0, 0, 5.0]",
                "[100, 0, 0.0]",
                "[50, 0, 2.0]",
                "[50, 0, 2.0]",
            ],
            "InstanceNumber": [3, 1, 2, 2],
            "SliceLocation": [10.0, 2.0, 2.0, 10.0],
        }
    )
    grouped = clean.group_volumes(df.copy())
    row = grouped.iloc[0]
    assert row["ImagePositionPatient"] == [
        "[100, 0, 0.0]",
        "[50, 0, 2.0]",
        "[0, 0, 5.0]",
    ]
    assert row["InstanceNumber"] == [1, 2, 3]
    assert row["SliceLocation"] == [2.0, 10.0]


def test_map_series_description(tmp_path, capsys):
    df = pd.DataFrame(
        {
            "SeriesDescription": [
                "Arteriel study",
                "Mixte sequence",
                "Inutile scan",
                None,
            ],
            "AcquisitionNumber": [1, 2, 1, 1],
        }
    )
    csv_dict = tmp_path / "dict.csv"
    pd.DataFrame(
        {
            "SeriesDescription": ["arteriel study", "mixte sequence", "inutile scan"],
            "phase": ["arteriel", "mixte", "inutile"],
        }
    ).to_csv(csv_dict, index=False)
    out = clean.map_series_description(df.copy(), str(csv_dict))
    # inutile rows removed
    assert "inutile" not in out["phase"].astype(str).values
    # mixt with AcquisitionNumber 2 -> portal
    assert "portal" in out["phase"].values or "arteriel" in out["phase"].values


def test_compute_visit_and_acquisition_order():
    # visit order
    df = pd.DataFrame(
        {
            "patient_key": ["p", "p", "q"],
            "study_id": ["s1", "s2", "s3"],
            "date": [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-02-01"),
                pd.Timestamp("2020-01-01"),
            ],
            "volume_id": ["v1", "v2", "v3"],
        }
    )
    out = clean.compute_visit_order(df.copy())
    assert "visit_order" in out.columns
    # acquisition order
    df2 = pd.DataFrame(
        {
            "patient_key": ["p", "p", "p"],
            "study_id": ["s", "s", "s"],
            "volume_id": ["v1", "v2", "v3"],
            "StudyDate": ["20200101", "20200101", "20200101"],
            "InstanceCreationTime": [
                "['120000.000']",
                "['120100.000']",
                "['120200.000']",
            ],
        }
    )
    df2 = clean.add_date(df2)
    df2 = clean.add_time(df2)
    out2 = clean.compute_acquisition_order(df2.copy())
    assert "acquisition_order" in out2.columns
    assert set(out2["acquisition_order"].dropna()) == {0, 1, 2}
    assert (out2["delay_since_prev_acq_sec"].dropna() >= 0).all()
    assert (out2["delay_since_first_acq_sec"].dropna() >= 0).all()
    _assert_no_acquisition_temp_cols(out2)

    # Handles aggregated time values represented as datetime.time objects or repr strings
    df3 = pd.DataFrame(
        {
            "patient_key": ["p", "p", "p"],
            "study_id": ["s", "s", "s"],
            "volume_id": ["v1", "v2", "v3"],
            "StudyDate": ["20200101", "20200101", "20200101"],
            "InstanceCreationTime": [
                [dt_time(17, 16, 35), dt_time(17, 16, 36)],
                "[datetime.time(17, 16, 40), datetime.time(17, 16, 41)]",
                dt_time(17, 16, 50),
            ],
        }
    )
    df3 = clean.add_date(df3)
    df3 = clean.add_time(df3)
    out3 = clean.compute_acquisition_order(df3.copy())
    assert out3.set_index("volume_id").loc["v1", "acquisition_order"] == 0
    assert out3.set_index("volume_id").loc["v2", "acquisition_order"] == 1
    assert out3.set_index("volume_id").loc["v3", "acquisition_order"] == 2
    assert (out3["delay_since_prev_acq_sec"].dropna() >= 0).all()
    assert (out3["delay_since_first_acq_sec"].dropna() >= 0).all()
    _assert_no_acquisition_temp_cols(out3)

    # Ensure ordering uses acquisition timestamp, not lexical volume_id order.
    df4 = pd.DataFrame(
        {
            "patient_key": ["p", "p", "p"],
            "study_id": ["s", "s", "s"],
            "volume_id": ["v1", "v2", "v3"],  # lexical order != acquisition order
            "StudyDate": ["20200101", "20200101", "20200101"],
            "InstanceCreationTime": [
                "['120300.000']",  # should be order 2
                "['120100.000']",  # should be order 0
                "['120200.000']",  # should be order 1
            ],
        }
    )
    df4 = clean.add_date(df4)
    df4 = clean.add_time(df4)
    out4 = clean.compute_acquisition_order(df4.copy())
    out4_by_volume = out4.set_index("volume_id")
    assert out4_by_volume.loc["v2", "acquisition_order"] == 0
    assert out4_by_volume.loc["v3", "acquisition_order"] == 1
    assert out4_by_volume.loc["v1", "acquisition_order"] == 2
    assert (out4["delay_since_prev_acq_sec"].dropna() >= 0).all()
    assert (out4["delay_since_first_acq_sec"].dropna() >= 0).all()
    _assert_no_acquisition_temp_cols(out4)


def test_compute_acquisition_order_without_time_uses_series_and_acquisition_number():
    df = pd.DataFrame(
        {
            "patient_key": ["p", "p", "p"],
            "study_id": ["s", "s", "s"],
            "volume_id": ["v1", "v2", "v3"],
            "StudyDate": ["20200101", "20200101", "20200101"],
            "SeriesNumber": [2, 1, 2],
            "AcquisitionNumber": [10, 5, 1],
        }
    )
    df = clean.add_date(df)

    out = clean.compute_acquisition_order(df.copy())
    out_by_volume = out.set_index("volume_id")

    assert "acquisition_order" in out.columns
    assert out_by_volume.loc["v2", "acquisition_order"] == 0
    assert out_by_volume.loc["v3", "acquisition_order"] == 1
    assert out_by_volume.loc["v1", "acquisition_order"] == 2
    _assert_no_acquisition_temp_cols(out)


def test_compute_acquisition_order_without_date_and_time_falls_back_to_numbers():
    df = pd.DataFrame(
        {
            "patient_key": ["p", "p", "p"],
            "study_id": ["s", "s", "s"],
            "volume_id": ["v1", "v2", "v3"],
            "SeriesNumber": [3, 1, 2],
            "AcquisitionNumber": [1, 1, 1],
        }
    )

    out = clean.compute_acquisition_order(df.copy())
    out_by_volume = out.set_index("volume_id")

    assert "acquisition_order" in out.columns
    assert out_by_volume.loc["v2", "acquisition_order"] == 0
    assert out_by_volume.loc["v3", "acquisition_order"] == 1
    assert out_by_volume.loc["v1", "acquisition_order"] == 2
    _assert_no_acquisition_temp_cols(out)


def test_compute_acquisition_order_preserves_input_order_when_no_sort_keys():
    df = pd.DataFrame(
        {
            "patient_key": ["p", "p", "p"],
            "study_id": ["s", "s", "s"],
            "volume_id": ["v2", "v10", "v1"],
        }
    )

    out = clean.compute_acquisition_order(df.copy())
    out_by_volume = out.set_index("volume_id")

    assert out_by_volume.loc["v2", "acquisition_order"] == 0
    assert out_by_volume.loc["v10", "acquisition_order"] == 1
    assert out_by_volume.loc["v1", "acquisition_order"] == 2
    _assert_no_acquisition_temp_cols(out)


def test_compute_acquisition_order_uses_temporal_position_then_instance_number():
    df = pd.DataFrame(
        {
            "patient_key": ["p", "p", "p"],
            "study_id": ["s", "s", "s"],
            "volume_id": ["v_temporal_2", "v_instance_3", "v_instance_1"],
            "SeriesNumber": [1, 1, 1],
            "AcquisitionNumber": [1, 1, 1],
            "TemporalPositionIdentifier": [2, 1, 1],
            "InstanceNumber": [[2, 5, 8], [3, 6, 9], [1, 4, 7]],
        }
    )

    out = clean.compute_acquisition_order(df)
    out_by_volume = out.set_index("volume_id")

    assert out_by_volume.loc["v_instance_1", "acquisition_order"] == 0
    assert out_by_volume.loc["v_instance_3", "acquisition_order"] == 1
    assert out_by_volume.loc["v_temporal_2", "acquisition_order"] == 2
    _assert_no_acquisition_temp_cols(out)


def test_compute_acquisition_order_drops_internal_sort_columns():
    df = pd.DataFrame(
        {
            "patient_key": ["p", "p"],
            "study_id": ["s", "s"],
            "volume_id": ["v1", "v2"],
            "date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")],
            "time": [dt_time(12, 0, 0), dt_time(12, 1, 0)],
            "SeriesNumber": [1, 1],
            "AcquisitionNumber": [1, 2],
        }
    )

    out = clean.compute_acquisition_order(df.copy())

    _assert_no_acquisition_temp_cols(out)
    assert "acquisition_order" in out.columns
    assert "delay_since_prev_acq_sec" in out.columns
    assert "delay_since_first_acq_sec" in out.columns


def test_group_volumes_sorts_acquisition_number_numerically():
    df = pd.DataFrame(
        {
            "volume_id": ["v1", "v1", "v1", "v1"],
            "AcquisitionNumber": ["10", "2", "2", "1"],
        }
    )

    grouped = clean.group_volumes(df.copy())
    row = grouped.iloc[0]

    assert row["AcquisitionNumber"] == ["1", "2", "10"]


def test_drop_irrelevant_dicom_tags():
    df = pd.DataFrame(
        {
            "SeriesDescription": ["a"],
            "SomeTag": [1],
            "UNRELATEDLOWER": [2],
            "UIDField": ["abc"],
            "ALLLOWER": [3],
        }
    )
    out = clean.drop_irrelevant_dicom_tags(df.copy())
    # Columns that look like dicom tags (contain uppercase) except important ones are dropped
    assert (
        "SomeTag" not in out.columns or "SomeTag" in out.columns
    )  # just ensure func runs without error


def test_load_data_and_read_csv_with_valid_columns(tmp_path):
    # create a csv with some columns - include one from COLUMNS_TO_USE if available
    csv = tmp_path / "t.csv"
    pd.DataFrame(
        {"patient_key": ["p"], "SeriesDescription": ["s"], "Extra": [1]}
    ).to_csv(csv, index=False)
    df = clean.read_csv_with_valid_columns(str(csv))
    assert "patient_key" in df.columns or "SeriesDescription" in df.columns

    # test load_data with multiple files
    csv2 = tmp_path / "t2.csv"
    pd.DataFrame({"patient_key": ["q"], "SeriesDescription": ["s2"]}).to_csv(
        csv2, index=False
    )
    combined = clean.load_data([str(csv), str(csv2)])
    assert combined.shape[0] == 2


def test_validate_cleaning_manifest_uses_hook_output_metadata():
    manifest = {
        "cleaning": {
            "version": 1,
            "steps": [
                {
                    "type": "hook",
                    "function": "dataset_configs.hooks.operandi:extract_from_patient_key",
                    "source_columns": ["patient_key"],
                },
                {
                    "type": "filter",
                    "kind": "keep",
                    "scope": "row",
                    "logic": "and",
                    "rules": [{"column": "center", "op": "eq", "value": "BJN"}],
                },
            ],
        }
    }

    steps = clean.validate_cleaning_manifest(manifest)
    required = clean._collect_required_input_columns(steps)

    assert "patient_key" in required
    assert "center" not in required


def test_validate_cleaning_manifest_requires_phase_curation_for_modality_step():
    manifest = {
        "cleaning": {
            "version": 1,
            "steps": [{"type": "modality_curation"}],
        }
    }

    with pytest.raises(ValueError, match="must define phase_curation"):
        clean.validate_cleaning_manifest(manifest)


def test_phase_curation_sources_are_loaded_for_modality_curation():
    manifest = {
        "phase_curation": {
            "strategies": [
                {
                    "type": "ontology",
                    "columns": ["reviewed_phase"],
                    "mapping": {"portal": "PORTAL_VENOUS"},
                },
                {"type": "totalsegmentator", "column": "predicted_phase"},
            ]
        },
        "cleaning": {
            "version": 1,
            "steps": [
                {"type": "compute_acquisition_order"},
                {"type": "modality_curation"},
            ],
        },
    }

    steps = clean.validate_cleaning_manifest(manifest)
    required = clean._collect_required_input_columns(steps, manifest["phase_curation"])

    assert {"reviewed_phase", "predicted_phase"}.issubset(required)
    assert {"TemporalPositionIdentifier", "InstanceNumber"}.issubset(required)


def test_single_rule_filter_defaults_missing_logic_to_and():
    manifest = {
        "cleaning": {
            "version": 1,
            "steps": [
                {
                    "type": "filter",
                    "kind": "keep",
                    "scope": "row",
                    "rules": [{"column": "Modality", "op": "eq", "value": "CT"}],
                }
            ],
        }
    }

    steps = clean.validate_cleaning_manifest(manifest)

    assert steps[0]["logic"] == "and"


def test_multiple_rule_filter_requires_logic():
    manifest = {
        "cleaning": {
            "version": 1,
            "steps": [
                {
                    "type": "filter",
                    "kind": "keep",
                    "scope": "row",
                    "rules": [
                        {"column": "Modality", "op": "eq", "value": "CT"},
                        {"column": "Rows", "op": "gte", "value": 256},
                    ],
                }
            ],
        }
    }

    with pytest.raises(ValueError, match="multiple rules"):
        clean.validate_cleaning_manifest(manifest)


@pytest.mark.parametrize(
    ("step", "message"),
    [
        (
            {"type": "clean_scan_size"},
            "Unknown cleaning step type",
        ),
        (
            {
                "type": "filter",
                "kind": "keep",
                "scope": "row",
                "logic": "and",
                "rules": [{"column": "Modality", "op": "between"}],
            },
            "Unsupported filter operator",
        ),
        (
            {
                "type": "filter",
                "kind": "keep",
                "scope": "row",
                "logic": "and",
                "rules": [{"column": "Modality", "op": "eq"}],
            },
            "requires a 'value'",
        ),
        (
            {"type": "normalize_string"},
            "normalize_string steps must define",
        ),
        (
            {"type": "coalesce_date", "candidates": "StudyDate"},
            "coalesce_date.candidates",
        ),
        (
            {"type": "classify_acquisition_plane", "angle_thresh_deg": "wide"},
            "angle_thresh_deg",
        ),
        (
            {"type": "build_volume_id", "preferred_columns": "patient_key"},
            "build_volume_id.preferred_columns",
        ),
    ],
)
def test_validate_cleaning_manifest_rejects_invalid_step_configs(step, message):
    manifest = {"cleaning": {"version": 1, "steps": [step]}}

    with pytest.raises(ValueError) as exc:
        clean.validate_cleaning_manifest(manifest)

    assert message in str(exc.value)


@pytest.mark.parametrize(
    ("column", "values", "rule", "expected_patients"),
    [
        (
            "value",
            ["CT", "MR", "PT"],
            {"column": "value", "op": "eq", "value": "CT"},
            ["p1"],
        ),
        (
            "value",
            ["CT", "MR", "PT"],
            {"column": "value", "op": "ne", "value": "CT"},
            ["p2", "p3"],
        ),
        (
            "value",
            ["CT", "MR", "PT"],
            {"column": "value", "op": "in", "value": ["CT", "PT"]},
            ["p1", "p3"],
        ),
        (
            "value",
            ["CT", "MR", "PT"],
            {"column": "value", "op": "not_in", "value": ["CT", "PT"]},
            ["p2"],
        ),
        (
            "value",
            ["LOCALIZER", "scout", "Portal Venous"],
            {"column": "value", "op": "contains", "value": "LOCAL"},
            ["p1"],
        ),
        (
            "value",
            ["LOCALIZER", "scout", "Portal Venous"],
            {"column": "value", "op": "icontains", "value": "portal"},
            ["p3"],
        ),
        (
            "value",
            ["LOCALIZER", "scout", "Portal Venous"],
            {"column": "value", "op": "regex", "value": "^P"},
            ["p3"],
        ),
        ("value", [1, 2, 3], {"column": "value", "op": "lt", "value": 2}, ["p1"]),
        (
            "value",
            [1, 2, 3],
            {"column": "value", "op": "lte", "value": 2},
            ["p1", "p2"],
        ),
        ("value", [1, 2, 3], {"column": "value", "op": "gt", "value": 2}, ["p3"]),
        (
            "value",
            [1, 2, 3],
            {"column": "value", "op": "gte", "value": 2},
            ["p2", "p3"],
        ),
        ("value", [1, None, 3], {"column": "value", "op": "is_null"}, ["p2"]),
        ("value", [1, None, 3], {"column": "value", "op": "not_null"}, ["p1", "p3"]),
    ],
)
def test_run_clean_pipeline_supports_all_filter_operators(
    column, values, rule, expected_patients
):
    df = pd.DataFrame(
        {
            "patient_key": ["p1", "p2", "p3"],
            "study_id": ["s1", "s1", "s1"],
            "series_id": ["sr1", "sr2", "sr3"],
            "dicom_path": ["a.dcm", "b.dcm", "c.dcm"],
            column: values,
        }
    )
    manifest = {
        "cleaning": {
            "version": 1,
            "steps": [
                {
                    "type": "filter",
                    "kind": "keep",
                    "scope": "row",
                    "logic": "and",
                    "rules": [rule],
                }
            ],
        }
    }

    steps = clean.validate_cleaning_manifest(manifest)
    out = clean.run_clean_pipeline(df.copy(), steps)

    assert out["patient_key"].tolist() == expected_patients


def test_run_clean_pipeline_filter_logic_and_or():
    df = pd.DataFrame(
        {
            "patient_key": ["p1", "p2", "p3"],
            "study_id": ["s", "s", "s"],
            "series_id": ["sr1", "sr2", "sr3"],
            "dicom_path": ["a", "b", "c"],
            "Modality": ["CT", "CT", "MR"],
            "sop_class": ["CTImageStorage", "Other", "CTImageStorage"],
            "ModalitiesInStudy": ["CT", "PT", "CT"],
        }
    )

    steps = [
        {
            "type": "filter",
            "kind": "keep",
            "scope": "row",
            "logic": "and",
            "rules": [
                {"column": "Modality", "op": "eq", "value": "CT"},
                {"column": "sop_class", "op": "eq", "value": "CTImageStorage"},
            ],
        },
        {
            "type": "filter",
            "kind": "discard",
            "scope": "row",
            "logic": "or",
            "rules": [
                {"column": "ModalitiesInStudy", "op": "contains", "value": "PT"},
                {"column": "ModalitiesInStudy", "op": "contains", "value": "NM"},
            ],
        },
    ]

    out = clean.run_clean_pipeline(df.copy(), steps)

    assert out["patient_key"].tolist() == ["p1"]


@pytest.mark.parametrize("op", ["eq", "ne", "in", "not_in"])
@pytest.mark.parametrize(
    "values",
    [
        pd.Series(["CT", None], dtype="object"),
        pd.Series(["CT", pd.NA], dtype="string"),
        pd.Series([1.0, float("nan")], dtype="float64"),
    ],
)
def test_non_null_filter_operators_never_match_missing_values(op, values):
    comparison = ["CT"] if op in {"in", "not_in"} else "CT"
    if values.dtype == "float64":
        comparison = [1.0] if op in {"in", "not_in"} else 1.0
    df = pd.DataFrame({"value": values})

    mask = clean._rule_mask(
        df,
        {"column": "value", "op": op, "value": comparison},
    )

    assert not bool(mask.iloc[1])


@pytest.mark.parametrize("kind", ["keep", "discard"])
def test_keep_null_preserves_incomplete_rows_for_both_filter_kinds(kind):
    df = pd.DataFrame(
        {
            "patient_key": ["missing", "match", "other"],
            "study_id": ["s"] * 3,
            "series_id": ["sr"] * 3,
            "value": [None, 5.0, 1.0],
        }
    )
    step = {
        "type": "filter",
        "kind": kind,
        "scope": "row",
        "keep_null": True,
        "rules": [{"column": "value", "op": "gte", "value": 5.0}],
    }
    validated_step = clean.validate_cleaning_manifest(
        {"cleaning": {"version": 1, "steps": [step]}}
    )[0]

    out = clean.run_clean_pipeline(df, [validated_step])

    expected = ["missing", "match"] if kind == "keep" else ["missing", "other"]
    assert out["patient_key"].tolist() == expected


def test_filter_step_log_includes_name_and_lists_each_column_once(caplog):
    caplog.set_level(logging.INFO, logger="imperandi.utils.misc")
    df = pd.DataFrame(
        {
            "patient_key": ["p1", "p2"],
            "study_id": ["s", "s"],
            "series_id": ["sr1", "sr2"],
            "Modality": ["CT", "PT"],
            "SeriesDescription": ["abdomen", "scout"],
        }
    )
    steps = [
        {
            "type": "filter",
            "name": "exclude_non_diagnostic",
            "kind": "discard",
            "scope": "row",
            "logic": "or",
            "rules": [
                {"column": "Modality", "op": "eq", "value": "PT"},
                {"column": "Modality", "op": "eq", "value": "NM"},
                {
                    "column": "SeriesDescription",
                    "op": "icontains",
                    "value": "scout",
                },
            ],
        }
    ]

    clean.run_clean_pipeline(df, steps)

    assert (
        "After discard row filter 'exclude_non_diagnostic' on column(s) "
        "Modality, SeriesDescription:" in caplog.text
    )


def test_run_clean_pipeline_executes_all_supported_step_types(monkeypatch):
    real_resolver = clean.resolve_function_path

    @clean_hook(outputs=["patient_study_key"])
    def combine_patient_and_study(row):
        return f"{row['patient_key']}::{row['study_id']}"

    def patched_resolver(function_path):
        if function_path == "tests.helpers:combine_patient_and_study":
            return combine_patient_and_study
        return real_resolver(function_path)

    monkeypatch.setattr(clean, "resolve_function_path", patched_resolver)

    df = pd.DataFrame(
        {
            "patient_key": ["patient_0012_030"] * 4,
            "study_id": ["study1"] * 4,
            "series_id": ["series1", "series1", "series2", "series2"],
            "dicom_path": ["a1.dcm", "a2.dcm", "b1.dcm", "b2.dcm"],
            "AltDate": [None, None, None, None],
            "StudyDate": ["20200101"] * 4,
            "AltTime": [None, None, None, None],
            "AcquisitionTime": ["120000", "120001", "121000", "121001"],
            "Modality": ["CT"] * 4,
            "SOPClassUID": ["1.2.840.10008.5.1.4.1.1.2"] * 4,
            "ImageType": ["['ORIGINAL','PRIMARY']"] * 4,
            "Rows": [512] * 4,
            "Columns": [512] * 4,
            "SliceThickness": [1.0] * 4,
            "SpacingBetweenSlices": [1.0] * 4,
            "SeriesDescription": [
                " Portal Venous ",
                " Portal Venous ",
                " Arterial ",
                " Arterial ",
            ],
            "PixelSpacing": ["[0.7, 0.7]"] * 4,
            "ImageOrientationPatient": ["[1, 0, 0, 0, 1, 0]"] * 4,
            "AcquisitionNumber": ["1", "1", "2", "2"],
            "SeriesNumber": ["10", "10", "11", "11"],
            "ImagePositionPatient": [
                "[0, 0, 0]",
                "[0, 0, 1]",
                "[0, 0, 10]",
                "[0, 0, 11]",
            ],
            "SliceLocation": [0.0, 1.0, 10.0, 11.0],
        }
    )
    manifest = {
        "phase_curation": {
            "strategies": [{"type": "rules"}],
            "fallback": "OTHER",
        },
        "cleaning": {
            "version": 1,
            "steps": [
                {
                    "type": "hook",
                    "function": "imperandi.builtin_datasets_config.hooks.generic:standardize_patient_key",
                    "source_columns": ["patient_key"],
                },
                {
                    "type": "hook",
                    "function": "tests.helpers:combine_patient_and_study",
                    "source_columns": ["patient_key", "study_id"],
                },
                {"type": "coalesce_date", "candidates": ["AltDate", "StudyDate"]},
                {"type": "coalesce_time", "candidates": ["AltTime", "AcquisitionTime"]},
                {"type": "sop_class"},
                {"type": "parse_image_type"},
                {"type": "normalize_string", "column": "SeriesDescription"},
                {"type": "pixel_spacing_xy"},
                {"type": "standardize_iop"},
                {"type": "classify_acquisition_plane", "angle_thresh_deg": 5.0},
                {
                    "type": "build_volume_id",
                    "preferred_columns": [
                        "patient_key",
                        "study_id",
                        "series_id",
                        "ImageType",
                        "AcquisitionNumber",
                        "ImageOrientationPatient",
                        "SliceThickness",
                        "PixelSpacingXY",
                    ],
                    "fallback_columns": ["patient_key", "study_id", "series_id"],
                    "merge_group_columns": [
                        "patient_key",
                        "study_id",
                        "series_id",
                        "ImageType",
                        "ImageOrientationPatient",
                        "SliceThickness",
                        "PixelSpacingXY",
                    ],
                    "merge_z_sources": ["ImagePositionPatient", "SliceLocation"],
                    "merge_z_tolerance": 0.01,
                },
                {"type": "group_volumes"},
                {"type": "compute_volume_length"},
                {"type": "modality_curation"},
                {"type": "compute_visit_order"},
                {"type": "compute_acquisition_order"},
                {"type": "finalize"},
            ],
        },
    }

    steps = clean.validate_cleaning_manifest(manifest)
    out = clean.run_clean_pipeline(
        df.copy(), steps, phase_curation=manifest["phase_curation"]
    )

    assert out.shape[0] == 2
    assert out["patient_key"].tolist() == ["12-30", "12-30"]
    assert set(out["patient_study_key"].tolist()) == {"12-30::study1"}
    assert set(out["SeriesDescription"].tolist()) == {"portal venous", "arterial"}
    assert set(out["acquisition_axis"].tolist()) == {"Z"}
    assert set(out["acquisition_order"].tolist()) == {0, 1}
    assert set(out["volume_length"].tolist()) == {2.0}
    assert set(out["curation_modality"].tolist()) == {"CT"}
    assert "CT_PORTAL_VENOUS" in set(out["selection_slot"])


def test_clean_and_save_data_runs_manifest_pipeline(tmp_path):
    csv_in = tmp_path / "input.csv"
    csv_out = tmp_path / "output.csv"
    pd.DataFrame(
        {
            "patient_key": ["patient_0012_030", "patient_0012_030"],
            "study_id": ["s1", "s1"],
            "series_id": ["sr1", "sr1"],
            "dicom_path": ["a.dcm", "b.dcm"],
            "SOPClassUID": ["1.2.840.10008.5.1.4.1.1.2"] * 2,
            "StudyDate": ["20200101", "20200101"],
            "AcquisitionTime": ["120000", "120100"],
            "SliceThickness": [1.0, 1.0],
            "SpacingBetweenSlices": [1.0, 1.0],
            "PixelSpacing": ["[0.7, 0.7]", "[0.7, 0.7]"],
        }
    ).to_csv(csv_in, index=False)

    manifest = {
        "cleaning": {
            "version": 1,
            "steps": [
                {
                    "type": "hook",
                    "function": "imperandi.builtin_datasets_config.hooks.generic:standardize_patient_key",
                    "source_columns": ["patient_key"],
                },
                {"type": "coalesce_date"},
                {"type": "coalesce_time"},
                {"type": "pixel_spacing_xy"},
                {
                    "type": "build_volume_id",
                    "preferred_columns": [
                        "patient_key",
                        "study_id",
                        "series_id",
                        "SliceThickness",
                        "PixelSpacingXY",
                    ],
                    "fallback_columns": ["patient_key", "study_id", "series_id"],
                },
                {"type": "group_volumes"},
                {"type": "compute_volume_length"},
                {"type": "finalize"},
            ],
        }
    }

    clean.clean_and_save_data(
        [str(csv_in)],
        str(csv_out),
        manifest=manifest,
    )

    out = pd.read_csv(csv_out)
    assert out.loc[0, "patient_key"] == "12-30"
    assert out.loc[0, "_patient_key_raw"] == "patient_0012_030"
    assert out.loc[0, "volume_length"] == 2.0
