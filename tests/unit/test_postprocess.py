import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import yaml

from imperandi import cli
from imperandi.process import postprocess as pp


def profile(method="zscore", kind="normalization", **parameters):
    return {"mask": "all", "steps": [{"type": kind, "method": method, **parameters}]}


def save_image(path, data=None, affine=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if data is None:
        data = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    image = nib.Nifti1Image(
        np.asarray(data, dtype=np.float32),
        affine if affine is not None else np.diag([1.5, 2, 3, 1]),
    )
    image.set_qform(image.affine, 1)
    image.set_sform(image.affine, 2)
    nib.save(image, path)
    return path


def cohort(tmp_path, rows=None):
    image = save_image(tmp_path / "input.nii.gz")
    csv = tmp_path / "nifti_index.csv"
    pd.DataFrame(
        rows
        if rows is not None
        else [{"volume_id": "001", "nifti_path": image.name, "Modality": "MR"}]
    ).to_csv(csv, index=False)
    return csv, image


def test_masked_zscore_keeps_fixed_selection_and_background():
    data = np.arange(27, dtype=float).reshape(3, 3, 3)
    mask = data >= 10
    output, report = pp.process_array(data, np.eye(4), profile(), mask)
    assert output.dtype == np.float32
    np.testing.assert_array_equal(output[~mask], data[~mask])
    assert output[mask].mean() == pytest.approx(0, abs=1e-7)
    assert output[mask].std() == pytest.approx(1)
    assert report["before"]["count"] == 17
    assert report["steps"][0]["scale"] == pytest.approx(data[mask].std())


@pytest.mark.parametrize("method", ["zscore", "robust_zscore", "minmax", "percentile"])
def test_normalization_numeric_results(method):
    data = np.arange(1, 28, dtype=float).reshape(3, 3, 3)
    data[-1, -1, -1] = 1000
    output, _ = pp.process_array(data, np.eye(4), profile(method))
    if method == "zscore":
        expected = (data - data.mean()) / data.std()
    elif method == "robust_zscore":
        expected = (data - np.median(data)) / (
            1.4826 * np.median(np.abs(data - np.median(data)))
        )
    else:
        low, high = (
            np.percentile(data, [1, 99])
            if method == "percentile"
            else (data.min(), data.max())
        )
        expected = (np.clip(data, low, high) - low) / (high - low)
    np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("method", ["zscore", "robust_zscore", "minmax", "percentile"])
def test_constant_normalization_is_finite(method):
    config = profile(method)
    if method in ("minmax", "percentile"):
        config["steps"][0]["output_range"] = [-1, 1]
    output, report = pp.process_array(np.ones((3, 3, 3)), np.eye(4), config)
    np.testing.assert_array_equal(
        output, -1 if method in ("minmax", "percentile") else 0
    )
    assert report["steps"][0]["degenerate"]


@pytest.mark.parametrize(
    "method,parameters",
    [
        ("window", {"lower": 5, "upper": 20}),
        ("percentile", {"percentiles": [10, 90]}),
        ("gamma", {"gamma": 2}),
        ("histogram", {"bins": 8}),
    ],
)
def test_contrast_methods_are_monotone_and_match_bounds(method, parameters):
    data = np.arange(27, dtype=float).reshape(3, 3, 3)
    output, _ = pp.process_array(
        data, np.eye(4), profile(method, "contrast", **parameters)
    )
    assert np.all(np.diff(output.ravel()) >= 0)
    if method == "window":
        np.testing.assert_array_equal(output, np.clip(data, 5, 20))
    elif method == "percentile":
        np.testing.assert_allclose(
            output, np.clip(data, *np.percentile(data, [10, 90]))
        )
    elif method == "gamma":
        np.testing.assert_allclose(output, (data / 26) ** 2 * 26, atol=1e-6)
    else:
        assert 0 <= output.min() <= output.max() <= 1
        assert not np.allclose(output, data / 26)


def test_order_and_zero_background():
    data = np.arange(27, dtype=float).reshape(3, 3, 3)
    config = {
        "mask": "positive",
        "outside_mask": "zero",
        "steps": [
            {"type": "contrast", "method": "window", "lower": 5, "upper": 20},
            {"type": "normalization", "method": "zscore"},
        ],
    }
    output, _ = pp.process_array(data, np.eye(4), config)
    clipped = np.clip(data[data > 0], 5, 20)
    np.testing.assert_allclose(
        output[data > 0], (clipped - clipped.mean()) / clipped.std(), atol=1e-7
    )
    assert output[0, 0, 0] == 0


@pytest.mark.parametrize(
    "bad,match",
    [
        ({"steps": []}, "non-empty"),
        ({"version": True, **profile()}, "version"),
        ({"version": 2, **profile()}, "version"),
        (profile("unknown"), "Unsupported"),
        (profile("zscore", typo=1), "Unknown"),
        (profile("percentile", percentiles=[99, 1]), "bounds"),
        (profile("minmax", output_range=[0, float("inf")]), "finite"),
        (profile("gamma", "contrast", gamma=0), "positive"),
        (profile("window", "contrast"), "finite"),
        (profile("histogram", "contrast", bins=True), "integer"),
        (profile("n4", "bias_correction", iterations=[]), "non-empty"),
        (profile("n4", "bias_correction", shrink_factor=1.2), "integer"),
        ({"modalities": {"MR": profile(), "MRI": profile()}}, "duplicate"),
        ({"modalities": {"CT": profile()}, **profile()}, "either"),
        (
            {
                "steps": [
                    {"type": "normalization", "method": "zscore"},
                    {"type": "bias_correction"},
                ]
            },
            "before",
        ),
    ],
)
def test_config_rejects_invalid_settings(bad, match):
    with pytest.raises(ValueError, match=match):
        pp.validate_config(bad)


@pytest.mark.parametrize(
    "data,mask,match",
    [
        (np.ones((2, 2, 2, 2)), None, "3-D"),
        (np.full((3, 3, 3), np.nan), None, "NaN"),
        (np.full((3, 3, 3), np.inf), None, "infinite"),
        (np.ones((3, 3, 3)), np.zeros((3, 3, 3)), "empty"),
        (np.ones((3, 3, 3)), np.ones((2, 2, 2)), "shape"),
    ],
)
def test_invalid_data_fails_explicitly(data, mask, match):
    with pytest.raises(ValueError, match=match):
        pp.process_array(data, np.eye(4), profile(), mask)


def test_file_geometry_dtype_scaling_and_source_preserved(tmp_path):
    affine = np.array([[0, -2, 0, 11], [1.5, 0, 0, -7], [0, 0, 3, 20], [0, 0, 0, 1]])
    source = save_image(tmp_path / "source.nii.gz", affine=affine)
    original = source.read_bytes()
    output, sidecar, resumed = pp.process_volume(
        source, tmp_path / "out", pp._validate_profile(profile())
    )
    assert not resumed
    result = nib.load(output)
    np.testing.assert_allclose(result.affine, affine)
    np.testing.assert_array_equal(result.get_qform(), nib.load(source).get_qform())
    assert int(result.header["qform_code"]) == 1
    assert int(result.header["sform_code"]) == 2
    assert result.header.get_zooms() == (1.5, 2, 3)
    assert result.get_data_dtype() == np.float32
    assert result.get_fdata().std() == pytest.approx(1)
    assert source.read_bytes() == original
    provenance = json.loads(sidecar.read_text())
    assert provenance["signature"]["input"]["path"] == str(source)
    assert provenance["statistics"]["after"]["mean"] == pytest.approx(0, abs=1e-7)


def test_scaled_integer_input_and_uncoded_qform(tmp_path):
    source = tmp_path / "scaled.nii.gz"
    image = nib.Nifti1Image(np.arange(64, dtype=np.int16).reshape(4, 4, 4), np.eye(4))
    image.header.set_slope_inter(2, 10)
    image.set_qform(None, 0)
    nib.save(image, source)
    config = profile("window", "contrast", lower=20, upper=80)
    output, _, _ = pp.process_volume(source, tmp_path / "out", config)
    result = nib.load(output)
    np.testing.assert_array_equal(
        result.get_fdata(), np.clip(nib.load(source).get_fdata(), 20, 80)
    )
    assert int(result.header["qform_code"]) == 0


def test_file_mask_geometry_must_match(tmp_path):
    source = save_image(tmp_path / "source.nii.gz")
    mask = save_image(tmp_path / "mask.nii.gz", np.ones((4, 4, 4)), affine=np.eye(4))
    with pytest.raises(ValueError, match="geometry"):
        pp.process_volume(source, tmp_path / "out", profile(), mask_path=mask)
    assert not (tmp_path / "out").exists()


def test_resume_invalidation_and_corrupt_output_recovery(tmp_path):
    source = save_image(tmp_path / "source.nii.gz")
    output, sidecar, _ = pp.process_volume(source, tmp_path / "out", profile())
    assert pp.process_volume(source, tmp_path / "out", profile())[2]
    output.write_bytes(b"corrupt")
    assert not pp.process_volume(source, tmp_path / "out", profile())[2]
    assert nib.load(output).shape == (4, 4, 4)
    assert not pp.process_volume(source, tmp_path / "out", profile(), force=True)[2]
    assert not pp.process_volume(source, tmp_path / "out", profile(), resume=False)[2]
    changed, _, reused = pp.process_volume(source, tmp_path / "out", profile("minmax"))
    assert not reused and changed != output
    source = save_image(source, np.arange(64).reshape(4, 4, 4) + 100)
    changed, _, reused = pp.process_volume(source, tmp_path / "out", profile())
    assert not reused and changed != output
    assert sidecar.is_file()


def test_mask_fingerprint_and_same_named_sources_do_not_collide(tmp_path):
    source = save_image(tmp_path / "one" / "image.nii.gz")
    second = save_image(tmp_path / "two" / "image.nii.gz")
    mask = save_image(tmp_path / "mask.nii.gz", np.ones((4, 4, 4)))
    first_out, _, _ = pp.process_volume(
        source, tmp_path / "out", profile(), mask_path=mask, strict=True
    )
    second_out, _, _ = pp.process_volume(
        second, tmp_path / "out", profile(), mask_path=mask, strict=True
    )
    assert first_out != second_out
    values = np.ones((4, 4, 4))
    values[0] = 0
    save_image(mask, values)
    changed, _, reused = pp.process_volume(
        source, tmp_path / "out", profile(), mask_path=mask, strict=True
    )
    assert not reused and changed != first_out


def test_cli_end_to_end_resume_and_failure_retry(tmp_path):
    csv, source = cohort(tmp_path)
    args = ["postprocess", str(csv), "--normalization", "z-score", "--mask", "all"]
    original = csv.read_bytes()
    assert cli.main(args) == 0
    result_csv = tmp_path / "nifti_index_postprocessed.csv"
    result = pd.read_csv(result_csv)
    assert result.loc[0, "postprocess_status"] == "processed"
    assert result.loc[0, "source_nifti_path"] == str(source)
    assert pd.read_csv(result_csv, dtype=str).loc[0, "volume_id"] == "001"
    assert Path(result.loc[0, "nifti_path"]).is_file()
    assert csv.read_bytes() == original
    assert cli.main(args) == 0
    assert pd.read_csv(result_csv).loc[0, "postprocess_status"] == "resumed"
    source.write_bytes(b"bad nifti")
    assert cli.main(args) == 1
    failed = pd.read_csv(result_csv)
    assert failed.loc[0, "postprocess_status"] == "failed"
    assert pd.isna(failed.loc[0, "nifti_path"])
    assert len(pd.read_csv(tmp_path / "postprocess_errors.csv")) == 1
    save_image(source)
    assert cli.main(args) == 0
    assert pd.read_csv(tmp_path / "postprocess_errors.csv").empty


def test_manifest_modality_dispatch_retains_rows_and_masks(tmp_path):
    csv, source = cohort(tmp_path)
    mask = save_image(tmp_path / "mask.nii.gz", np.ones((4, 4, 4)))
    pd.DataFrame(
        [
            {"nifti_path": source.name, "Modality": "MRI", "mask_liver": mask.name},
            {"nifti_path": source.name, "Modality": "CT", "mask_liver": mask.name},
            {"nifti_path": "missing.nii.gz", "Modality": "MR", "mask_liver": mask.name},
        ]
    ).to_csv(csv, index=False)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "image_postprocessing": {
                    "modalities": {"MR": {**profile(), "mask_column": "mask_liver"}}
                }
            }
        )
    )
    out_csv = tmp_path / "elsewhere" / "out.csv"
    assert (
        cli.main(
            [
                "postprocess",
                str(csv),
                "--manifest",
                str(manifest),
                "--csv-path-out",
                str(out_csv),
            ]
        )
        == 1
    )
    result = pd.read_csv(out_csv)
    assert list(result.postprocess_status) == ["processed", "skipped", "failed"]
    assert result.loc[1, "nifti_path"] == str(source)
    assert all(result.mask_liver == str(mask))
    assert pd.isna(result.loc[2, "nifti_path"])


def test_cli_dry_run_validates_manifest_without_n4_dependency(
    tmp_path, monkeypatch, capsys
):
    csv, _ = cohort(tmp_path)
    monkeypatch.setattr(
        pp, "_load_sitk", lambda: pytest.fail("dry run imported SimpleITK")
    )
    assert (
        cli.main(
            [
                "postprocess",
                str(csv),
                "--bias-correction",
                "n4",
                "--normalization",
                "zscore",
                "--dry-run",
            ]
        )
        == 0
    )
    assert "postprocessing_config" in capsys.readouterr().out
    assert not (tmp_path / "POSTPROCESSED").exists()
    assert not (tmp_path / "nifti_index_postprocessed.csv").exists()


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--normalization", "none"],
        ["--contrast", "window"],
        ["--normalization", "minmax", "--csv-path-out", "SAME"],
    ],
)
def test_cli_rejects_noop_and_invalid_paths(tmp_path, extra):
    csv, _ = cohort(tmp_path)
    extra = [str(csv) if arg == "SAME" else arg for arg in extra]
    assert cli.main(["postprocess", str(csv), *extra, "--dry-run"]) == 2


def test_cli_methods_replace_manifest_and_named_paths_win(tmp_path, capsys):
    csv, _ = cohort(tmp_path)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "image_postprocessing": {
                    "modalities": {"MR": profile("n4", "bias_correction")}
                }
            }
        )
    )
    parser = pp.build_parser()
    args = pp.normalize_postprocess_args(
        parser.parse_args(
            [
                "missing.csv",
                "unused",
                "--csv-path",
                str(csv),
                "--output-dir",
                str(tmp_path / "actual"),
                "--manifest",
                str(manifest),
                "--normalization",
                "zscore",
                "--mask",
                "positive",
            ]
        )
    )
    assert args.csv_path == str(csv)
    assert args.output_dir == str(tmp_path / "actual")
    assert args.postprocessing_config["mask"] == "positive"
    assert args.postprocessing_config["steps"] == [
        {"type": "normalization", "method": "zscore"}
    ]


def test_missing_optional_n4_dependency_has_actionable_error(tmp_path, monkeypatch):
    csv, _ = cohort(tmp_path)

    def unavailable():
        raise RuntimeError(
            "N4 requires optional dependencies. Install imperandi[postprocess]"
        )

    monkeypatch.setattr(pp, "_load_sitk", unavailable)
    assert cli.main(["postprocess", str(csv), "--bias-correction", "n4"]) == 2
    assert not (tmp_path / "POSTPROCESSED").exists()


def test_n4_reduces_synthetic_bias_without_changing_grid(tmp_path):
    pytest.importorskip("SimpleITK")
    x, y, z = np.meshgrid(*[np.linspace(-1, 1, 32)] * 3, indexing="ij")
    mask = (x * x + y * y + z * z) < 0.8**2
    data = np.zeros(x.shape)
    data[mask] = 100 * np.exp(0.5 * x[mask])
    source = save_image(tmp_path / "biased.nii.gz", data)
    config = {
        "mask": "positive",
        "steps": [
            {
                "type": "bias_correction",
                "method": "n4",
                "iterations": [30, 20],
                "shrink_factor": 2,
            }
        ],
    }
    output, _, _ = pp.process_volume(source, tmp_path / "out", config)
    result = nib.load(output)
    corrected = result.get_fdata()
    assert (
        corrected[mask].std() / corrected[mask].mean()
        < 0.75 * data[mask].std() / data[mask].mean()
    )
    np.testing.assert_array_equal(corrected[~mask], 0)
    np.testing.assert_array_equal(result.affine, nib.load(source).affine)


def test_n4_rejects_nonpositive_foreground():
    pytest.importorskip("SimpleITK")
    with pytest.raises(ValueError, match="positive intensities"):
        pp.process_array(
            -np.ones((8, 8, 8)), np.eye(4), profile("n4", "bias_correction")
        )


def test_array_api_rejects_complex_data_and_missing_explicit_mask():
    with pytest.raises(ValueError, match="real scalar"):
        pp.process_array(np.ones((3, 3, 3), dtype=complex) * 1j, np.eye(4), profile())
    with pytest.raises(ValueError, match="mask array is required"):
        pp.process_array(
            np.ones((3, 3, 3)), np.eye(4), {**profile(), "mask_column": "roi"}
        )


@pytest.mark.parametrize("name", ["generic", "blueprint_manifest_example"])
def test_bundled_postprocessing_profiles_are_valid(name):
    config = pp.load_manifest(name, base_path=Path(__file__).parent)[
        "image_postprocessing"
    ]
    validated = pp.validate_config(config)
    assert "MR" in validated["modalities"]
    assert validated["modalities"]["MR"]["steps"][-1]["method"] == "zscore"
