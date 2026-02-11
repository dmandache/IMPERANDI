import sys
from pathlib import Path

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import pandas as pd

from imperandi.extract import radiomics as radiomics_module


def test_radiomics_parser_defaults_manifest_to_generic():
    parser = radiomics_module.build_parser()
    args = parser.parse_args([])
    assert args.manifest == "generic"


def test_normalize_radiomics_args_defaults(tmp_path):
    csv_path = tmp_path / "nifti_index.csv"
    csv_path.write_text("nifti_path\n")

    args = argparse.Namespace(
        csv_path_pos=str(csv_path),
        csv_path_opt=None,
        csv_path_out=None,
        error_csv_path=None,
        skip_filter=False,
        verbose=False,
        dry_run=False,
    )

    out = radiomics_module.normalize_radiomics_args(args)

    assert out.csv_path == str(csv_path.resolve())
    assert out.csv_path_out == str(csv_path.parent / "nifti_index_radiomics.csv")
    assert out.error_csv_path == str(csv_path.parent / "radiomics_errors.csv")
    assert not hasattr(out, "csv_path_pos")
    assert not hasattr(out, "csv_path_opt")


def test_extract_radiomics_from_dataframe_records_missing_image():
    df = pd.DataFrame([{"nifti_path": "missing_file.nii.gz"}])
    df_out, df_err = radiomics_module.extract_radiomics_from_dataframe(
        df,
        extractor=object(),
        sitk_module=object(),
        verbose=False,
    )

    assert len(df_out) == 1
    assert len(df_err) == 1
    assert "missing or invalid" in df_err.loc[0, "error_message"]


def test_main_writes_output_and_error_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": "x.nii.gz"}]).to_csv(csv_path, index=False)

    monkeypatch.setattr(
        radiomics_module,
        "_load_radiomics_dependencies",
        lambda: (object(), object()),
    )
    monkeypatch.setattr(
        radiomics_module,
        "_create_radiomics_extractor",
        lambda featureextractor_module, settings: object(),
    )

    def fake_extract(df, *, extractor, sitk_module, verbose):
        out = df.copy()
        out["liver_original_shape_VoxelVolume"] = 1.0
        err = pd.DataFrame([{"nifti_path": "x.nii.gz", "error_message": "mock error"}])
        return out, err

    monkeypatch.setattr(
        radiomics_module, "extract_radiomics_from_dataframe", fake_extract
    )

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        skip_filter=True,
        verbose=False,
    )

    radiomics_module.main(args)

    out_df = pd.read_csv(args.csv_path_out)
    assert "liver_original_shape_VoxelVolume" in out_df.columns

    err_df = pd.read_csv(args.error_csv_path)
    assert len(err_df) == 1
    assert err_df.loc[0, "error_message"] == "mock error"
