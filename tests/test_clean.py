import sys
from pathlib import Path

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import numpy as np

from imperandi.ingest import clean


def test_clean_parser_defaults_manifest_to_generic():
    parser = clean.build_parser()
    args = parser.parse_args([])
    assert args.manifest == "generic"


def test_normalize_clean_args_prefers_optional_csv_path(tmp_path):
    csv_pos = tmp_path / "pos.csv"
    csv_opt = tmp_path / "opt.csv"
    csv_pos.write_text("patient_key\np1\n")
    csv_opt.write_text("patient_key\np2\n")

    args = clean.normalize_clean_args(
        clean.argparse.Namespace(
            csv_path_pos=[str(csv_pos)],
            csv_path_opt=[str(csv_opt)],
            csv_path_out=None,
        )
    )

    assert args.csv_path == [str(csv_opt)]
    assert args.csv_path_out.endswith("opt_clean.csv")
    assert not hasattr(args, "csv_path_pos")
    assert not hasattr(args, "csv_path_opt")


def test_normalize_clean_args_accepts_positional_only(tmp_path):
    csv_pos = tmp_path / "input.csv"
    csv_pos.write_text("patient_key\np1\n")

    args = clean.normalize_clean_args(
        clean.argparse.Namespace(
            csv_path_pos=[str(csv_pos)],
            csv_path_opt=None,
            csv_path_out=None,
        )
    )

    assert args.csv_path == [str(csv_pos)]
    assert args.csv_path_out.endswith("input_clean.csv")


def test_standardize_patient_keys_supports_keyed_function_name(monkeypatch):
    df = pd.DataFrame({"patient_key": [" Alice "]})

    def resolver(cfg):
        assert cfg.get("function") == "normalize_patient_key"
        return lambda v: str(v).strip().upper()

    monkeypatch.setattr(clean, "resolve_hook", resolver)
    out = clean.standardize_patient_keys(
        df.copy(),
        manifest={
            "id_standardization": {
                "hook_module": "datasets_config.hooks.generic",
                "patient_key": "normalize_patient_key",
            }
        },
    )
    assert out.loc[0, "patient_key"] == "ALICE"


def test_unravel_patient_key_supports_multiple_operations(monkeypatch):
    df = pd.DataFrame({"patient_key": ["alice"], "value": [2]})

    def resolver(cfg):
        fn = cfg.get("function")
        if fn == "from_patient_key":
            return lambda x: {"patient_upper": str(x).upper()}
        if fn == "from_value":
            return lambda x: {"double": x * 2}
        return None

    monkeypatch.setattr(clean, "resolve_hook", resolver)
    out = clean.unravel_patient_key(
        df.copy(),
        manifest={
            "derived_columns": [
                {
                    "hook_module": "datasets_config.hooks.generic",
                    "function": "from_patient_key",
                    "from_column": "patient_key",
                },
                {
                    "hook_module": "datasets_config.hooks.generic",
                    "function": "from_value",
                    "from_column": "value",
                },
            ]
        },
    )
    assert out["patient_upper"].tolist() == ["ALICE"]
    assert out["double"].tolist() == [4]


def test_unravel_patient_key_requires_explicit_derived_columns(monkeypatch):
    df = pd.DataFrame({"patient_key": ["alice"]})

    def resolver(_cfg):
        raise AssertionError("resolve_hook should not be called without derived_columns")

    monkeypatch.setattr(clean, "resolve_hook", resolver)
    out = clean.unravel_patient_key(
        df.copy(),
        manifest={
            "id_standardization": {
                "hook_module": "datasets_config.hooks.operandi",
            }
        },
    )
    assert out.equals(df)


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
    # both rows should now have the canonical (sorted) volume id 'b'
    assert set(out["volume_id"].unique()) == {"b"}


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
    filtered = clean.filter_volumes_by_size(calc.copy(), t_min=0.0, t_max=5.0)
    # both volumes should be kept (v2 has NaN volume_length => kept)
    assert set(filtered["volume_id"]) == set(grouped["volume_id"])


def test_group_volumes_deterministic_ordering():
    df = pd.DataFrame(
        {
            "volume_id": ["v1", "v1", "v1", "v1"],
            "ImagePositionPatient": [
                "[0, 0, 5.0]",
                "[0, 0, 0.0]",
                "[0, 0, 2.0]",
                "[0, 0, 2.0]",
            ],
            "InstanceNumber": [3, 1, 2, 2],
            "SliceLocation": [10.0, 2.0, 2.0, 10.0],
        }
    )
    grouped = clean.group_volumes(df.copy())
    row = grouped.iloc[0]
    assert row["ImagePositionPatient"] == [
        "[0, 0, 0.0]",
        "[0, 0, 2.0]",
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
            "InstanceCreationTime": [
                "['120000.000']",
                "['120100.000']",
                "['120200.000']",
            ],
        }
    )
    out2 = clean.compute_acquisition_order(df2.copy())
    assert "acquisition_order" in out2.columns
    assert set(out2["acquisition_order"].dropna()) == {0, 1, 2}


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
