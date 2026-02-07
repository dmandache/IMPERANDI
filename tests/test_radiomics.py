import sys
from pathlib import Path

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import numpy as np
import pandas as pd

from imperandi.extract import radiomics as radiomics_module


def test_normalize_radiomics_args_defaults(tmp_path):
    csv_path = tmp_path / "nifti_index.csv"
    csv_path.write_text("nifti_path\n")

    args = argparse.Namespace(
        csv_path_pos=str(csv_path),
        csv_path_opt=None,
        csv_path_out=None,
        error_csv_path=None,
        organ_mask_col="mask_liver",
        tumor_mask_col="mask_liver_tumor",
        skip_filter=False,
        verbose=False,
        dry_run=False,
    )

    out = radiomics_module.normalize_radiomics_args(args)

    assert out.csv_path == str(csv_path.resolve())
    assert out.csv_path_out == str(csv_path.parent / "nifti_index_radiomics.csv")
    assert out.error_csv_path == str(csv_path.parent / "radiomics_errors.csv")
    assert out.organ_mask_col == "mask_liver"
    assert out.tumor_mask_col == "mask_liver_tumor"
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
    pd.DataFrame(
        [
            {
                "nifti_path": "x.nii.gz",
                "mask_liver": "liver.nii.gz",
                "mask_liver_tumor": "tumor.nii.gz",
            }
        ]
    ).to_csv(csv_path, index=False)

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

    def fake_extract(
        df,
        *,
        extractor,
        sitk_module,
        organ_mask_col,
        tumor_mask_col,
        organ_prefix,
        verbose,
    ):
        assert organ_mask_col == "mask_liver"
        assert tumor_mask_col == "mask_liver_tumor"
        assert organ_prefix == "liver"
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
        organ_mask_col="mask_liver",
        tumor_mask_col="mask_liver_tumor",
        skip_filter=True,
        verbose=False,
    )

    radiomics_module.main(args)

    out_df = pd.read_csv(args.csv_path_out)
    assert "liver_original_shape_VoxelVolume" in out_df.columns

    err_df = pd.read_csv(args.error_csv_path)
    assert len(err_df) == 1
    assert err_df.loc[0, "error_message"] == "mock error"


def test_extract_radiomics_organ_minus_tumor_shape_on_full_mask(tmp_path):
    class FakeImage:
        def __init__(self, voxels):
            self.voxels = set(voxels)
            self._size = (1, 1, 1)
            self._spacing = (1.0, 1.0, 1.0)
            self._origin = (0.0, 0.0, 0.0)
            self._direction = (1.0,) * 9

        def GetSize(self):
            return self._size

        def GetSpacing(self):
            return self._spacing

        def GetOrigin(self):
            return self._origin

        def GetDirection(self):
            return self._direction

    class FakeNotMask:
        def __init__(self, mask):
            self.mask = mask

    class FakeSITK:
        sitkUInt8 = 1
        sitkNearestNeighbor = 0

        def __init__(self, image_map):
            self._image_map = image_map

        def ReadImage(self, path):
            return self._image_map[path]

        def GetArrayViewFromImage(self, image):
            return np.array([len(image.voxels)], dtype=np.uint8)

        def Cast(self, image, _dtype):
            return image

        def NotEqual(self, image, _value):
            return image

        def Not(self, image):
            return FakeNotMask(image)

        def And(self, lhs, rhs):
            return FakeImage(lhs.voxels - rhs.mask.voxels)

        class ResampleImageFilter:
            def SetReferenceImage(self, _image):
                return None

            def SetInterpolator(self, _interp):
                return None

            def SetDefaultPixelValue(self, _value):
                return None

            def Execute(self, image):
                return image

    class FakeExtractor:
        def execute(self, _image, mask):
            n = len(mask.voxels)
            return {
                "original_shape_VoxelVolume": n,
                "original_firstorder_Mean": n * 10,
            }

    image_path = tmp_path / "ct.nii.gz"
    organ_path = tmp_path / "organ.nii.gz"
    tumor_path = tmp_path / "tumor.nii.gz"
    image_path.write_text("ct")
    organ_path.write_text("organ")
    tumor_path.write_text("tumor")

    sitk_module = FakeSITK(
        {
            str(image_path): FakeImage({0, 1, 2, 3, 4}),
            str(organ_path): FakeImage({1, 2, 3, 4}),
            str(tumor_path): FakeImage({4}),
        }
    )

    features, message = radiomics_module.extract_radiomics_organ_minus_tumor(
        image_path=str(image_path),
        organ_mask_path=str(organ_path),
        tumor_mask_path=str(tumor_path),
        extractor=FakeExtractor(),
        sitk_module=sitk_module,
        prefix="organ",
    )

    assert message is None
    assert features["organ_original_shape_VoxelVolume"] == 4
    assert features["organ_original_firstorder_Mean"] == 30


def test_main_allows_missing_tumor_mask_column(tmp_path, monkeypatch):
    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": "x.nii.gz", "mask_liver": "liver.nii.gz"}]).to_csv(
        csv_path, index=False
    )

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

    observed = {}

    def fake_extract(
        df,
        *,
        extractor,
        sitk_module,
        organ_mask_col,
        tumor_mask_col,
        organ_prefix,
        verbose,
    ):
        observed["organ_mask_col"] = organ_mask_col
        observed["tumor_mask_col"] = tumor_mask_col
        observed["organ_prefix"] = organ_prefix
        return df.copy(), pd.DataFrame([])

    monkeypatch.setattr(
        radiomics_module, "extract_radiomics_from_dataframe", fake_extract
    )

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        organ_mask_col="mask_liver",
        tumor_mask_col="mask_tumor_missing",
        skip_filter=True,
        verbose=False,
    )

    radiomics_module.main(args)

    assert observed["organ_mask_col"] == "mask_liver"
    assert observed["tumor_mask_col"] is None
    assert observed["organ_prefix"] == "liver"
