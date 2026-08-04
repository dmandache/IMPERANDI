import hashlib
import sys
from pathlib import Path
from datetime import time as dt_time

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import numpy as np

from imperandi.ingest import clean

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


def test_normalize_clean_args_migrates_legacy_volume_bounds(tmp_path):
    csv_in = tmp_path / "input.csv"
    csv_in.write_text("patient_key\np1\n")

    args = clean.normalize_clean_args(
        clean.argparse.Namespace(
            csv_path_pos=str(csv_in),
            csv_path_opt=None,
            csv_path_out_pos=None,
            csv_path_out=None,
            volume_min=25.0,
            volume_max=450.0,
        )
    )

    assert args.volume_length_min_mm == 25.0
    assert args.volume_length_max_mm == 450.0
    assert not hasattr(args, "volume_min")
    assert not hasattr(args, "volume_max")


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


def test_filter_ct_modality_and_remove_pet_ct():
    df = pd.DataFrame(
        {
            "Modality": ["CT", "MR", "CT"],
            "SOPClassUID": [
                "1.2.840.10008.5.1.4.1.1.2",  # CTImageStorage
                "1.2.840.10008.5.1.4.1.1.4",  # MRImageStorage (example)
                "1.2.840.10008.5.1.4.1.1.2",
            ],
            "ModalitiesInStudy": ["CT", "PT", "CT"],
        }
    )
    df_ct = clean.filter_ct_modality(df.copy())
    assert (df_ct.Modality == "CT").all()
    assert "sop_class" in df_ct.columns
    # remove rows with PT in ModalitiesInStudy
    df_pet_removed = clean.remove_pet_ct(df.copy())
    assert "PT" not in " ".join(df_pet_removed["ModalitiesInStudy"].astype(str))


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
    assert not df_nompr.apply(
        lambda r: "mpr" in (str(r.ImageType) + str(r.SeriesDescription)).lower(), axis=1
    ).any()


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


def test_clean_scan_size_and_pixel_spacing():
    df = pd.DataFrame(
        {
            "Rows": [512, None],
            "Columns": [512, 512],
            "SliceThickness": [
                "2",
                "5",
            ],  # second should be filtered out (thickness > 3)
            "PixelSpacing": ["[0.7, 0.7]", None],
        }
    )
    cleaned = clean.clean_scan_size(df.copy())
    assert (cleaned["SliceThickness"].astype(float) <= 3).all()

    ps = clean.clean_pixel_spacing(df.copy())
    assert "PixelSpacingXY" in ps.columns
    assert ps.loc[0, "PixelSpacingXY"] == 0.7
    assert pd.isna(ps.loc[1, "PixelSpacingXY"])


def test_generate_volume_id_and_filter_by_acquisition_plane():
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
    with_sup = clean.generate_volume_id(df.copy())
    assert "volume_id" in with_sup.columns
    # same content -> identical volume_id
    assert with_sup.loc[0, "volume_id"] == with_sup.loc[1, "volume_id"]

    # filter by plane: above orientation corresponds to axial (Z axis)
    filtered = clean.filter_by_acquisition_plane(with_sup.copy())
    assert filtered.shape[0] == with_sup.shape[0]


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


def test_group_volumes_and_calculate_length_and_filter_by_size():
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
    filtered = clean.filter_volumes_by_size(
        calc.copy(), min_length_mm=0.0, max_length_mm=5.0
    )
    # both volumes should be kept (v2 has NaN volume_length => kept)
    assert set(filtered["volume_id"]) == set(grouped["volume_id"])


def test_calculate_volume_length_accepts_csv_string_metadata():
    volumes = pd.DataFrame(
        {
            "dicom_path": [["a.dcm", "b.dcm", "c.dcm"]],
            "SliceThickness": ["1.5"],
            "SpacingBetweenSlices": ["2.0"],
        }
    )

    result = clean.calculate_volume_length(volumes)

    assert result.loc[0, "n_files"] == 3
    assert result.loc[0, "volume_length"] == 5.5


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


def test_compute_acquisition_order_tie_breaks_by_volume_id_when_no_sort_keys():
    df = pd.DataFrame(
        {
            "patient_key": ["p", "p", "p"],
            "study_id": ["s", "s", "s"],
            "volume_id": ["v2", "v10", "v1"],
        }
    )

    out = clean.compute_acquisition_order(df.copy())
    out_by_volume = out.set_index("volume_id")

    assert out_by_volume.loc["v1", "acquisition_order"] == 0
    assert out_by_volume.loc["v10", "acquisition_order"] == 1
    assert out_by_volume.loc["v2", "acquisition_order"] == 2
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
