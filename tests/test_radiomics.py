import sys
from pathlib import Path

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import logging
import pandas as pd
import pytest

from imperandi.extract import radiomics as radiomics_module


def _fake_extractors_factory(*args, **kwargs):
    return {"all": object(), "shape": object(), "non_shape": object()}


def test_normalize_radiomics_args_defaults(tmp_path):
    csv_path = tmp_path / "nifti_index.csv"
    csv_path.write_text("nifti_path\n")

    args = argparse.Namespace(
        csv_path_pos=str(csv_path),
        csv_path_opt=None,
        csv_path_out_pos=None,
        csv_path_out=None,
        error_csv_path=None,
        skip_filter=False,
        filter_args=None,
        manifest=None,
        pyradiomics_settings=None,
        verbose=False,
        dry_run=False,
    )

    out = radiomics_module.normalize_radiomics_args(args)

    assert out.csv_path == str(csv_path.resolve())
    assert out.csv_path_out == str(csv_path.parent / "nifti_index_radiomics.csv")
    assert out.error_csv_path == str(csv_path.parent / "radiomics_errors.csv")
    assert out.pyradiomics_settings is None
    assert out.filters == {}
    assert not hasattr(out, "csv_path_pos")
    assert not hasattr(out, "csv_path_opt")
    assert not hasattr(out, "csv_path_out_pos")


def test_normalize_radiomics_args_accepts_positional_csv_path_out(tmp_path):
    csv_path = tmp_path / "nifti_index.csv"
    csv_path.write_text("nifti_path\n")
    csv_out = tmp_path / "radiomics_out.csv"

    args = argparse.Namespace(
        csv_path_pos=str(csv_path),
        csv_path_opt=None,
        csv_path_out_pos=str(csv_out),
        csv_path_out=None,
        error_csv_path=None,
        skip_filter=False,
        filter_args=None,
        manifest=None,
        pyradiomics_settings=None,
        verbose=False,
        dry_run=False,
    )

    out = radiomics_module.normalize_radiomics_args(args)

    assert out.csv_path_out == str(csv_out)
    assert not hasattr(out, "csv_path_out_pos")


def test_normalize_radiomics_args_validates_pyradiomics_settings_path(tmp_path, monkeypatch):
    csv_path = tmp_path / "nifti_index.csv"
    csv_path.write_text("nifti_path\n")
    settings_path = tmp_path / "params.yaml"
    settings_path.write_text("setting:\n  binWidth: 5\n")
    monkeypatch.chdir(tmp_path)

    args = argparse.Namespace(
        csv_path_pos=str(csv_path),
        csv_path_opt=None,
        csv_path_out_pos=None,
        csv_path_out=None,
        error_csv_path=None,
        skip_filter=False,
        filter_args=None,
        manifest=None,
        pyradiomics_settings="params.yaml",
        verbose=False,
        dry_run=False,
    )

    out = radiomics_module.normalize_radiomics_args(args)
    assert out.pyradiomics_settings == str(settings_path.resolve())


def test_normalize_radiomics_args_rejects_missing_or_non_yaml_settings_path(tmp_path):
    csv_path = tmp_path / "nifti_index.csv"
    csv_path.write_text("nifti_path\n")

    args_missing = argparse.Namespace(
        csv_path_pos=str(csv_path),
        csv_path_opt=None,
        csv_path_out_pos=None,
        csv_path_out=None,
        error_csv_path=None,
        skip_filter=False,
        filter_args=None,
        manifest=None,
        pyradiomics_settings=str(tmp_path / "missing.yaml"),
        verbose=False,
        dry_run=False,
    )
    with pytest.raises(FileNotFoundError):
        radiomics_module.normalize_radiomics_args(args_missing)

    bad_path = tmp_path / "params.txt"
    bad_path.write_text("setting:\n  binWidth: 5\n")
    args_bad_suffix = argparse.Namespace(
        csv_path_pos=str(csv_path),
        csv_path_opt=None,
        csv_path_out_pos=None,
        csv_path_out=None,
        error_csv_path=None,
        skip_filter=False,
        filter_args=None,
        manifest=None,
        pyradiomics_settings=str(bad_path),
        verbose=False,
        dry_run=False,
    )
    with pytest.raises(ValueError):
        radiomics_module.normalize_radiomics_args(args_bad_suffix)


def test_resolve_pyradiomics_settings_source_prefers_manifest_and_warns_twice(
    tmp_path, monkeypatch, caplog
):
    settings_path = tmp_path / "params.yaml"
    settings_path.write_text("setting:\n  binWidth: 5\n")
    monkeypatch.setattr(
        radiomics_module,
        "load_manifest",
        lambda *args, **kwargs: {
            "radiomics": {"pyradiomics": {"setting": {"binWidth": 9}}}
        },
    )

    args = argparse.Namespace(
        manifest="generic",
        pyradiomics_settings=str(settings_path),
    )
    caplog.set_level(logging.WARNING, logger=radiomics_module.__name__)

    source_kind, source_path, source_dict = (
        radiomics_module._resolve_pyradiomics_settings_source(args)
    )

    assert source_kind == "manifest"
    assert source_path is None
    assert source_dict == {"setting": {"binWidth": 9}}
    warnings = [
        record.message
        for record in caplog.records
        if record.levelno >= logging.WARNING
    ]
    assert len(warnings) == 2
    assert "Both --manifest and --pyradiomics_settings were provided" in warnings[0]
    assert "preferring manifest settings" in warnings[1]


def test_resolve_pyradiomics_settings_source_uses_cli_file_when_manifest_has_no_radiomics(
    tmp_path, monkeypatch, caplog
):
    settings_path = tmp_path / "params.yaml"
    settings_path.write_text("setting:\n  binWidth: 5\n")
    monkeypatch.setattr(radiomics_module, "load_manifest", lambda *args, **kwargs: {})

    args = argparse.Namespace(
        manifest="generic",
        pyradiomics_settings=str(settings_path),
    )
    caplog.set_level(logging.WARNING, logger=radiomics_module.__name__)

    source_kind, source_path, source_dict = (
        radiomics_module._resolve_pyradiomics_settings_source(args)
    )

    assert source_kind == "cli_file"
    assert source_path == str(settings_path.resolve())
    assert source_dict is None
    warnings = [
        record.message
        for record in caplog.records
        if record.levelno >= logging.WARNING
    ]
    assert len(warnings) == 1
    assert "Both --manifest and --pyradiomics_settings were provided" in warnings[0]


def test_normalize_radiomics_args_parses_filters(tmp_path):
    csv_path = tmp_path / "nifti_index.csv"
    csv_path.write_text("nifti_path\n")

    args = argparse.Namespace(
        csv_path_pos=str(csv_path),
        csv_path_opt=None,
        csv_path_out_pos=None,
        csv_path_out=None,
        error_csv_path=None,
        skip_filter=False,
        filter_args=["phase=portal,arteriel", "followup_months=0,3"],
        manifest=None,
        pyradiomics_settings=None,
        verbose=False,
        dry_run=False,
    )

    out = radiomics_module.normalize_radiomics_args(args)

    assert out.filters == {
        "phase": ["portal", "arteriel"],
        "followup_months": ["0", "3"],
    }
    assert not hasattr(out, "filter_args")


@pytest.mark.parametrize(
    "raw_filter, expected_message",
    [
        ("phase", "exactly one '='"),
        (" =portal", "column name is empty"),
        ("phase=,", "at least one value is required"),
    ],
)
def test_normalize_radiomics_args_rejects_invalid_filters(
    tmp_path, raw_filter, expected_message
):
    csv_path = tmp_path / "nifti_index.csv"
    csv_path.write_text("nifti_path\n")

    args = argparse.Namespace(
        csv_path_pos=str(csv_path),
        csv_path_opt=None,
        csv_path_out_pos=None,
        csv_path_out=None,
        error_csv_path=None,
        skip_filter=False,
        filter_args=[raw_filter],
        manifest=None,
        pyradiomics_settings=None,
        verbose=False,
        dry_run=False,
    )

    with pytest.raises(ValueError, match=expected_message):
        radiomics_module.normalize_radiomics_args(args)


def test_resolve_radiomics_filters_prefers_manifest_for_same_column(
    tmp_path, monkeypatch
):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"radiomics": {"filters": {"phase": ["portal"], "site": ["A"]}}}'
    )
    args = argparse.Namespace(
        skip_filter=False,
        filters={"phase": ["arteriel"], "followup_months": ["0", "3"]},
        manifest=str(manifest_path),
    )

    filters = radiomics_module._resolve_radiomics_filters(args)

    assert filters == {
        "phase": ["portal"],
        "followup_months": ["0", "3"],
        "site": ["A"],
    }


def test_resolve_radiomics_filters_skip_filter_bypasses_manifest_and_cli(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"radiomics": {"filters": {"phase": ["portal"]}}}')
    args = argparse.Namespace(
        skip_filter=True,
        filters={"phase": ["arteriel"]},
        manifest=str(manifest_path),
    )

    assert radiomics_module._resolve_radiomics_filters(args) == {}


def test_apply_explicit_filters_requires_existing_columns():
    df = pd.DataFrame([{"phase": "portal"}])

    with pytest.raises(ValueError, match="missing from input CSV"):
        radiomics_module._apply_explicit_filters(df, {"site": ["A"]})


def test_apply_explicit_filters_keeps_rows_when_no_filters():
    df = pd.DataFrame([{"phase": "mixte"}, {"phase": "portal"}])

    out = radiomics_module._apply_explicit_filters(df, {})

    pd.testing.assert_frame_equal(out, df)


def test_apply_explicit_filters_uses_and_semantics_and_preserves_order():
    df = pd.DataFrame(
        [
            {"phase": "portal", "followup_months": 3, "row_id": "keep-1"},
            {"phase": "arteriel", "followup_months": 0, "row_id": "drop"},
            {"phase": "portal", "followup_months": 0, "row_id": "keep-2"},
        ]
    )

    out = radiomics_module._apply_explicit_filters(
        df,
        {"phase": ["portal"], "followup_months": [0, 3]},
    )

    assert out["row_id"].tolist() == ["keep-1", "keep-2"]


def test_configure_pyradiomics_output_toggles_logger_state():
    pyradiomics_logger = logging.getLogger("radiomics")
    old_disabled = pyradiomics_logger.disabled
    old_propagate = pyradiomics_logger.propagate
    old_level = pyradiomics_logger.level
    try:
        radiomics_module._configure_pyradiomics_output(enabled=False, verbose=False)
        assert pyradiomics_logger.disabled is True
        assert pyradiomics_logger.propagate is False

        radiomics_module._configure_pyradiomics_output(enabled=True, verbose=False)
        assert pyradiomics_logger.disabled is False
        assert pyradiomics_logger.propagate is True
    finally:
        pyradiomics_logger.disabled = old_disabled
        pyradiomics_logger.propagate = old_propagate
        pyradiomics_logger.setLevel(old_level)


def test_execute_extractor_logs_organ_and_extractor(caplog):
    class FakeExtractor:
        def execute(self, image, mask):
            return {"original_firstorder_Mean": 1.0}

    caplog.set_level(logging.DEBUG, logger=radiomics_module.__name__)
    out = radiomics_module._execute_extractor(
        FakeExtractor(),
        image="image",
        mask="mask",
        organ="liver",
        extractor_name="shape",
        row_idx=7,
    )

    assert out["original_firstorder_Mean"] == 1.0
    assert (
        "Executing PyRadiomics extractor | row=7 | organ=liver | extractor=shape"
        in caplog.text
    )


def test_main_records_missing_image_path_error(tmp_path, monkeypatch):
    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": "missing_file.nii.gz"}]).to_csv(csv_path, index=False)

    monkeypatch.setattr(
        radiomics_module,
        "_load_radiomics_dependencies",
        lambda: (object(), object()),
    )
    monkeypatch.setattr(
        radiomics_module,
        "_create_radiomics_extractors",
        _fake_extractors_factory,
    )

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        skip_filter=True,
        filters={},
        manifest=None,
        verbose=False,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )

    radiomics_module.main(args)

    out_df = pd.read_csv(args.csv_path_out)
    err_df = pd.read_csv(args.error_csv_path)
    assert len(out_df) == 1
    assert len(err_df) == 1
    assert "missing or invalid" in err_df.loc[0, "error_message"]


def test_main_uses_tqdm_even_when_not_verbose(tmp_path, monkeypatch):
    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": "missing_file.nii.gz"}]).to_csv(csv_path, index=False)

    monkeypatch.setattr(
        radiomics_module,
        "_load_radiomics_dependencies",
        lambda: (object(), object()),
    )
    monkeypatch.setattr(
        radiomics_module,
        "_create_radiomics_extractors",
        _fake_extractors_factory,
    )

    tqdm_calls = {"count": 0, "desc": None, "unit": None}

    def fake_tqdm(it, **kwargs):
        tqdm_calls["count"] += 1
        tqdm_calls["desc"] = kwargs.get("desc")
        tqdm_calls["unit"] = kwargs.get("unit")
        return it

    monkeypatch.setattr(radiomics_module, "tqdm", fake_tqdm)

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        skip_filter=True,
        filters={},
        manifest=None,
        verbose=False,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )

    radiomics_module.main(args)

    assert tqdm_calls["count"] == 1
    assert tqdm_calls["desc"] == "Radiomics"
    assert tqdm_calls["unit"] == "row"


def test_main_writes_output_and_error_csv(tmp_path, monkeypatch):
    good_nifti = tmp_path / "good.nii.gz"
    bad_nifti = tmp_path / "bad.nii.gz"
    good_mask = tmp_path / "good_mask.nii.gz"
    bad_mask = tmp_path / "bad_mask.nii.gz"
    good_nifti.write_text("nifti")
    bad_nifti.write_text("nifti")
    good_mask.write_text("mask")
    bad_mask.write_text("mask")

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame(
        [
            {"nifti_path": str(good_nifti), "mask_liver": str(good_mask)},
            {"nifti_path": str(bad_nifti), "mask_liver": str(bad_mask)},
        ]
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(
        radiomics_module,
        "_load_radiomics_dependencies",
        lambda: (object(), object()),
    )
    monkeypatch.setattr(
        radiomics_module,
        "_create_radiomics_extractors",
        _fake_extractors_factory,
    )

    def fake_liver_minus_tumor(
        image_path,
        liver_mask_path,
        tumor_mask_path,
        *,
        extractors,
        sitk_module,
        prefix,
        row_idx=None,
    ):
        if Path(image_path).name == "good.nii.gz":
            return {"liver_original_shape_VoxelVolume": 1.0}, None
        return {}, "mock error"

    monkeypatch.setattr(
        radiomics_module, "extract_radiomics_organ_minus_tumor", fake_liver_minus_tumor
    )
    monkeypatch.setattr(
        radiomics_module, "extract_radiomics_safe", lambda *a, **k: ({}, None)
    )

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        skip_filter=True,
        filters={},
        manifest=None,
        verbose=False,
    )

    radiomics_module.main(args)

    out_df = pd.read_csv(args.csv_path_out)
    assert "liver_original_shape_VoxelVolume" in out_df.columns

    err_df = pd.read_csv(args.error_csv_path)
    assert len(err_df) == 1
    assert "mock error" in err_df.loc[0, "error_message"]


def test_main_resume_skips_completed_rows(tmp_path, monkeypatch):
    good_nifti = tmp_path / "good.nii.gz"
    good_mask = tmp_path / "good_mask.nii.gz"
    good_nifti.write_text("nifti")
    good_mask.write_text("mask")
    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame(
        [{"nifti_path": str(good_nifti), "mask_liver": str(good_mask)}]
    ).to_csv(csv_path, index=False)

    calls = {"count": 0}
    dep_calls = {"count": 0}

    def fake_load_deps():
        dep_calls["count"] += 1
        return object(), object()

    monkeypatch.setattr(radiomics_module, "_load_radiomics_dependencies", fake_load_deps)
    monkeypatch.setattr(
        radiomics_module,
        "_create_radiomics_extractors",
        _fake_extractors_factory,
    )

    def fake_liver(*args, **kwargs):
        calls["count"] += 1
        return {"f": 1.0}, None

    monkeypatch.setattr(radiomics_module, "extract_radiomics_organ_minus_tumor", fake_liver)
    monkeypatch.setattr(radiomics_module, "extract_radiomics_safe", lambda *a, **k: ({}, None))

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        skip_filter=True,
        filters={},
        manifest=None,
        verbose=False,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )
    radiomics_module.main(args)
    assert calls["count"] == 1

    calls["count"] = 0
    args.resume = True
    radiomics_module.main(args)
    assert calls["count"] == 0
    assert dep_calls["count"] == 1


def test_main_resume_reprocesses_when_manifest_filters_change(tmp_path, monkeypatch):
    portal_nifti = tmp_path / "portal.nii.gz"
    arterial_nifti = tmp_path / "arterial.nii.gz"
    portal_mask = tmp_path / "portal_mask.nii.gz"
    arterial_mask = tmp_path / "arterial_mask.nii.gz"
    for path in (portal_nifti, arterial_nifti, portal_mask, arterial_mask):
        path.write_text("x")

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame(
        [
            {
                "nifti_path": str(portal_nifti),
                "mask_liver": str(portal_mask),
                "phase": "portal",
            },
            {
                "nifti_path": str(arterial_nifti),
                "mask_liver": str(arterial_mask),
                "phase": "arterial",
            },
        ]
    ).to_csv(csv_path, index=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"radiomics": {"filters": {"phase": ["portal"]}}}')

    monkeypatch.setattr(
        radiomics_module,
        "_load_radiomics_dependencies",
        lambda: (object(), object()),
    )
    monkeypatch.setattr(
        radiomics_module,
        "_create_radiomics_extractors",
        _fake_extractors_factory,
    )

    processed = []

    def fake_liver(
        image_path,
        liver_mask_path,
        tumor_mask_path,
        *,
        extractors,
        sitk_module,
        prefix,
        row_idx=None,
    ):
        processed.append(Path(image_path).name)
        return {"liver_original_shape_VoxelVolume": 1.0}, None

    monkeypatch.setattr(
        radiomics_module, "extract_radiomics_organ_minus_tumor", fake_liver
    )
    monkeypatch.setattr(
        radiomics_module, "extract_radiomics_safe", lambda *a, **k: ({}, None)
    )

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        skip_filter=False,
        filters={},
        manifest=str(manifest_path),
        verbose=False,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )
    radiomics_module.main(args)

    manifest_path.write_text('{"radiomics": {"filters": {"phase": ["arterial"]}}}')
    args.resume = True
    radiomics_module.main(args)

    assert processed == ["portal.nii.gz", "arterial.nii.gz"]


def test_main_preserves_foreign_columns_from_existing_output(tmp_path, monkeypatch):
    nifti = tmp_path / "good.nii.gz"
    mask = tmp_path / "good_mask.nii.gz"
    nifti.write_text("nifti")
    mask.write_text("mask")
    csv_path = tmp_path / "nifti_index.csv"
    out_path = tmp_path / "out.csv"
    pd.DataFrame([{"nifti_path": str(nifti), "mask_liver": str(mask)}]).to_csv(
        csv_path, index=False
    )
    pd.DataFrame([{"nifti_path": str(nifti), "foreign_col": "keep"}]).to_csv(
        out_path, index=False
    )

    monkeypatch.setattr(
        radiomics_module,
        "_load_radiomics_dependencies",
        lambda: (object(), object()),
    )
    monkeypatch.setattr(
        radiomics_module,
        "_create_radiomics_extractors",
        _fake_extractors_factory,
    )
    monkeypatch.setattr(
        radiomics_module,
        "extract_radiomics_organ_minus_tumor",
        lambda *a, **k: ({"liver_original_shape_VoxelVolume": 1.0}, None),
    )
    monkeypatch.setattr(
        radiomics_module, "extract_radiomics_safe", lambda *a, **k: ({}, None)
    )

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(out_path),
        error_csv_path=str(tmp_path / "errors.csv"),
        skip_filter=True,
        filters={},
        manifest=None,
        verbose=False,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )
    radiomics_module.main(args)

    out_df = pd.read_csv(out_path)
    assert "foreign_col" in out_df.columns
    assert out_df.loc[0, "foreign_col"] == "keep"


def test_main_does_not_filter_phase_by_default(tmp_path, monkeypatch):
    nifti_a = tmp_path / "mixte.nii.gz"
    nifti_b = tmp_path / "portal.nii.gz"
    mask_a = tmp_path / "mixte_mask.nii.gz"
    mask_b = tmp_path / "portal_mask.nii.gz"
    for path in (nifti_a, nifti_b, mask_a, mask_b):
        path.write_text("x")

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame(
        [
            {"nifti_path": str(nifti_a), "mask_liver": str(mask_a), "phase": "mixte"},
            {"nifti_path": str(nifti_b), "mask_liver": str(mask_b), "phase": "portal"},
        ]
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(
        radiomics_module,
        "_load_radiomics_dependencies",
        lambda: (object(), object()),
    )
    monkeypatch.setattr(
        radiomics_module,
        "_create_radiomics_extractors",
        _fake_extractors_factory,
    )
    monkeypatch.setattr(
        radiomics_module,
        "extract_radiomics_organ_minus_tumor",
        lambda *a, **k: ({"liver_original_shape_VoxelVolume": 1.0}, None),
    )
    monkeypatch.setattr(
        radiomics_module, "extract_radiomics_safe", lambda *a, **k: ({}, None)
    )

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        skip_filter=False,
        filters={},
        manifest=None,
        verbose=False,
    )

    radiomics_module.main(args)

    out_df = pd.read_csv(args.csv_path_out)
    assert sorted(out_df["phase"].tolist()) == ["mixte", "portal"]


def test_main_applies_explicit_filters_from_manifest_and_cli(tmp_path, monkeypatch):
    nifti_a = tmp_path / "a.nii.gz"
    nifti_b = tmp_path / "b.nii.gz"
    nifti_c = tmp_path / "c.nii.gz"
    mask_a = tmp_path / "a_mask.nii.gz"
    mask_b = tmp_path / "b_mask.nii.gz"
    mask_c = tmp_path / "c_mask.nii.gz"
    for path in (nifti_a, nifti_b, nifti_c, mask_a, mask_b, mask_c):
        path.write_text("x")

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame(
        [
            {
                "nifti_path": str(nifti_a),
                "mask_liver": str(mask_a),
                "phase": "portal",
                "followup_months": 0,
            },
            {
                "nifti_path": str(nifti_b),
                "mask_liver": str(mask_b),
                "phase": "arteriel",
                "followup_months": 0,
            },
            {
                "nifti_path": str(nifti_c),
                "mask_liver": str(mask_c),
                "phase": "portal",
                "followup_months": 3,
            },
        ]
    ).to_csv(csv_path, index=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"radiomics": {"filters": {"phase": ["portal"]}}}'
    )

    monkeypatch.setattr(
        radiomics_module,
        "_load_radiomics_dependencies",
        lambda: (object(), object()),
    )
    monkeypatch.setattr(
        radiomics_module,
        "_create_radiomics_extractors",
        _fake_extractors_factory,
    )
    monkeypatch.setattr(
        radiomics_module,
        "extract_radiomics_organ_minus_tumor",
        lambda *a, **k: ({"liver_original_shape_VoxelVolume": 1.0}, None),
    )
    monkeypatch.setattr(
        radiomics_module, "extract_radiomics_safe", lambda *a, **k: ({}, None)
    )

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        skip_filter=False,
        filters={"phase": ["arteriel"], "followup_months": [0]},
        manifest=str(manifest_path),
        verbose=False,
    )

    radiomics_module.main(args)

    out_df = pd.read_csv(args.csv_path_out)
    assert out_df["nifti_path"].tolist() == [str(nifti_a)]


def test_project_radiomics_features_keeps_non_original_and_drops_diagnostics():
    result = {
        "original_firstorder_Mean": 1.0,
        "wavelet-HLL_glcm_Contrast": 2.0,
        "log-sigma-1-0-mm-3D_glrlm_RunLengthNonUniformity": 3.0,
        "diagnostics_Image-original_Spacing": (1.0, 1.0, 1.0),
    }
    out = radiomics_module._project_radiomics_features(result, prefix="liver")
    assert out == {
        "liver_original_firstorder_Mean": 1.0,
        "liver_wavelet-HLL_glcm_Contrast": 2.0,
        "liver_log-sigma-1-0-mm-3D_glrlm_RunLengthNonUniformity": 3.0,
    }


def test_create_radiomics_extractors_configures_shape_and_non_shape():
    class FakeExtractor:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.calls = []
            self.featureClassNames = ["firstorder", "shape", "glcm"]

        def disableAllFeatures(self):
            self.calls.append(("disableAllFeatures",))

        def enableFeatureClassByName(self, name):
            self.calls.append(("enableFeatureClassByName", name))

    class FakeFeatureExtractorModule:
        RadiomicsFeatureExtractor = FakeExtractor

    extractors = radiomics_module._create_radiomics_extractors(
        FakeFeatureExtractorModule,
        {"binWidth": 25},
    )
    assert set(extractors.keys()) == {"all", "shape", "non_shape"}

    shape_calls = extractors["shape"].calls
    assert ("disableAllFeatures",) in shape_calls
    assert ("enableFeatureClassByName", "shape") in shape_calls

    non_shape_calls = extractors["non_shape"].calls
    assert ("disableAllFeatures",) in non_shape_calls
    assert ("enableFeatureClassByName", "firstorder") in non_shape_calls
    assert ("enableFeatureClassByName", "glcm") in non_shape_calls
    assert ("enableFeatureClassByName", "shape") not in non_shape_calls


def test_create_radiomics_extractors_supports_settings_path_and_manifest_dict():
    class FakeExtractor:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.calls = []
            self.featureClassNames = ["firstorder", "shape"]

        def disableAllFeatures(self):
            self.calls.append(("disableAllFeatures",))

        def enableFeatureClassByName(self, name):
            self.calls.append(("enableFeatureClassByName", name))

    class FakeFeatureExtractorModule:
        RadiomicsFeatureExtractor = FakeExtractor

    path_extractors = radiomics_module._create_radiomics_extractors(
        FakeFeatureExtractorModule,
        settings_path="params.yaml",
    )
    assert path_extractors["all"].args == ("params.yaml",)
    assert path_extractors["all"].kwargs == {}

    dict_extractors = radiomics_module._create_radiomics_extractors(
        FakeFeatureExtractorModule,
        settings_dict={"setting": {"binWidth": 7}},
    )
    assert dict_extractors["all"].args == ({"setting": {"binWidth": 7}},)
    assert dict_extractors["all"].kwargs == {}


def test_create_radiomics_extractors_preserves_multiple_image_types_from_manifest_dict():
    class FakeExtractor:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.calls = []
            self.featureClassNames = ["firstorder", "shape"]

        def disableAllFeatures(self):
            self.calls.append(("disableAllFeatures",))

        def enableFeatureClassByName(self, name):
            self.calls.append(("enableFeatureClassByName", name))

    class FakeFeatureExtractorModule:
        RadiomicsFeatureExtractor = FakeExtractor

    manifest_like_settings = {
        "setting": {
            "binWidth": 25,
            "resampledPixelSpacing": [1, 1, 1],
        },
        "imageType": {
            "Original": {},
            "Wavelet": {},
            "LoG": {"sigma": [1.0, 2.0, 3.0]},
        },
    }
    extractors = radiomics_module._create_radiomics_extractors(
        FakeFeatureExtractorModule,
        settings_dict=manifest_like_settings,
    )

    assert extractors["all"].args == (manifest_like_settings,)
    assert extractors["shape"].args == (manifest_like_settings,)
    assert extractors["non_shape"].args == (manifest_like_settings,)


def test_create_radiomics_extractors_rejects_multiple_sources():
    class FakeFeatureExtractorModule:
        class RadiomicsFeatureExtractor:
            def __init__(self, *args, **kwargs):
                pass

    with pytest.raises(ValueError):
        radiomics_module._create_radiomics_extractors(
            FakeFeatureExtractorModule,
            {"binWidth": 25},
            settings_path="params.yaml",
        )


def test_build_dataset_strategy_describes_extractor_plan():
    strategy = radiomics_module._build_dataset_strategy(
        ["mask_liver", "mask_liver_tumor", "mask_kidney"]
    )

    assert "kidney: all on mask_kidney (no paired tumor mask column)" in strategy
    assert (
        "liver: shape on mask_liver; non_shape on liver_minus_tumor; "
        "fallback all on mask_liver if mask_liver_tumor missing/empty"
    ) in strategy
    assert "liver_tumor: all on mask_liver_tumor" in strategy


def test_extract_radiomics_organ_minus_tumor_uses_shape_and_non_shape_extractors(tmp_path):
    image_path = tmp_path / "img.nii.gz"
    organ_path = tmp_path / "organ.nii.gz"
    tumor_path = tmp_path / "tumor.nii.gz"
    image_path.write_text("x")
    organ_path.write_text("x")
    tumor_path.write_text("x")

    class FakeExtractor:
        def __init__(self, result):
            self.result = result
            self.calls = []

        def execute(self, image, mask):
            self.calls.append((image, mask))
            return dict(self.result)

    extractors = {
        "all": FakeExtractor({"original_firstorder_Mean": 99.0}),
        "shape": FakeExtractor({"original_shape_VoxelVolume": 1.0}),
        "non_shape": FakeExtractor({"wavelet-HLL_glcm_Contrast": 2.0}),
    }

    class FakeSitk:
        sitkUInt8 = "uint8"

        @staticmethod
        def ReadImage(path):
            return path

        @staticmethod
        def GetArrayViewFromImage(img):
            if str(img) == str(tumor_path):
                return [1]
            return [1]

        @staticmethod
        def Cast(img, _dtype):
            return img

        @staticmethod
        def NotEqual(img, _value):
            return f"bin({img})"

        @staticmethod
        def Not(img):
            return f"not({img})"

        @staticmethod
        def And(a, b):
            return f"{a}&{b}"

    old_resample = radiomics_module._resample_to_reference_if_needed
    old_is_existing = radiomics_module._is_existing_path
    try:
        radiomics_module._resample_to_reference_if_needed = lambda mask, ref, sitk_module: mask
        radiomics_module._is_existing_path = lambda value: bool(value)
        features, msg = radiomics_module.extract_radiomics_organ_minus_tumor(
            str(image_path),
            str(organ_path),
            str(tumor_path),
            extractors=extractors,
            sitk_module=FakeSitk,
            prefix="liver",
        )
    finally:
        radiomics_module._resample_to_reference_if_needed = old_resample
        radiomics_module._is_existing_path = old_is_existing

    assert msg is None
    assert features["liver_original_shape_VoxelVolume"] == 1.0
    assert features["liver_wavelet-HLL_glcm_Contrast"] == 2.0
    assert len(extractors["shape"].calls) == 1
    assert len(extractors["non_shape"].calls) == 1
    assert len(extractors["all"].calls) == 0


def test_extract_radiomics_organ_minus_tumor_missing_tumor_uses_all_extractor(tmp_path):
    image_path = tmp_path / "img.nii.gz"
    organ_path = tmp_path / "organ.nii.gz"
    image_path.write_text("x")
    organ_path.write_text("x")

    class FakeExtractor:
        def __init__(self, result):
            self.result = result
            self.calls = []

        def execute(self, image, mask):
            self.calls.append((image, mask))
            return dict(self.result)

    extractors = {
        "all": FakeExtractor({"original_firstorder_Mean": 99.0}),
        "shape": FakeExtractor({"original_shape_VoxelVolume": 1.0}),
        "non_shape": FakeExtractor({"wavelet-HLL_glcm_Contrast": 2.0}),
    }

    class FakeSitk:
        sitkUInt8 = "uint8"

        @staticmethod
        def ReadImage(path):
            return path

        @staticmethod
        def GetArrayViewFromImage(_img):
            return [1]

        @staticmethod
        def Cast(img, _dtype):
            return img

        @staticmethod
        def NotEqual(img, _value):
            return f"bin({img})"

    old_is_existing = radiomics_module._is_existing_path
    try:
        radiomics_module._is_existing_path = (
            lambda value: bool(value) and str(value) == str(organ_path)
        )
        features, msg = radiomics_module.extract_radiomics_organ_minus_tumor(
            str(image_path),
            str(organ_path),
            None,
            extractors=extractors,
            sitk_module=FakeSitk,
            prefix="liver",
        )
    finally:
        radiomics_module._is_existing_path = old_is_existing

    assert msg is None
    assert features["liver_original_firstorder_Mean"] == 99.0
    assert len(extractors["all"].calls) == 1
    assert len(extractors["shape"].calls) == 0
    assert len(extractors["non_shape"].calls) == 0


def test_extract_radiomics_organ_minus_tumor_empty_tumor_uses_all_extractor(tmp_path):
    image_path = tmp_path / "img.nii.gz"
    organ_path = tmp_path / "organ.nii.gz"
    tumor_path = tmp_path / "tumor.nii.gz"
    image_path.write_text("x")
    organ_path.write_text("x")
    tumor_path.write_text("x")

    class FakeExtractor:
        def __init__(self, result):
            self.result = result
            self.calls = []

        def execute(self, image, mask):
            self.calls.append((image, mask))
            return dict(self.result)

    extractors = {
        "all": FakeExtractor({"original_firstorder_Mean": 99.0}),
        "shape": FakeExtractor({"original_shape_VoxelVolume": 1.0}),
        "non_shape": FakeExtractor({"wavelet-HLL_glcm_Contrast": 2.0}),
    }

    class FakeSitk:
        sitkUInt8 = "uint8"

        @staticmethod
        def ReadImage(path):
            return path

        @staticmethod
        def GetArrayViewFromImage(img):
            if str(img) == str(tumor_path):
                return [0]
            return [1]

        @staticmethod
        def Cast(img, _dtype):
            return img

        @staticmethod
        def NotEqual(img, _value):
            return f"bin({img})"

    old_is_existing = radiomics_module._is_existing_path
    try:
        radiomics_module._is_existing_path = lambda value: bool(value)
        features, msg = radiomics_module.extract_radiomics_organ_minus_tumor(
            str(image_path),
            str(organ_path),
            str(tumor_path),
            extractors=extractors,
            sitk_module=FakeSitk,
            prefix="liver",
        )
    finally:
        radiomics_module._is_existing_path = old_is_existing

    assert msg is None
    assert features["liver_original_firstorder_Mean"] == 99.0
    assert len(extractors["all"].calls) == 1
    assert len(extractors["shape"].calls) == 0
    assert len(extractors["non_shape"].calls) == 0
