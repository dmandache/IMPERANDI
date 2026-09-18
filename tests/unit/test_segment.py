import argparse
import copy
import sys
import logging
from pathlib import Path
import types

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import yaml

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from imperandi.process import segment as segment_module
from imperandi.utils.multiprocessing import MPStrategy
import imperandi.utils.multiprocessing as mp_utils


def save_nifti(data, affine, path):
    nib.save(nib.Nifti1Image(data.astype(np.uint8), affine), path)


class DummyBackend:
    def __init__(self, outputs):
        self.outputs = outputs

    def run(self, *, input_path, output_dir, task, **kwargs):
        out_name = self.outputs[task]
        (Path(output_dir) / out_name).write_text("mask")


class DummyTqdmBar:
    def __init__(self, total=None, **kwargs):
        self.total = total

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def update(self, n=1):
        return None

    def refresh(self):
        return None


def passthrough_tqdm(it=None, **kwargs):
    if it is None:
        return DummyTqdmBar(**kwargs)
    return it


def make_strategy(**overrides):
    data = {
        "mode": "process_pool",
        "start_method": "spawn",
        "max_workers": 2,
        "max_in_flight": 2,
        "recycle_every": 0,
        "env": {},
        "use_gpu": False,
        "gpu_count": 0,
        "hard_timeout_supported": False,
        "reasons": {"test": True},
    }
    data.update(overrides)
    return MPStrategy(**data)


def patch_strategy(monkeypatch, **overrides):
    monkeypatch.setattr(
        mp_utils,
        "decide_multiprocessing_strategy",
        lambda **kwargs: make_strategy(**overrides),
    )
    monkeypatch.setattr(mp_utils, "apply_strategy_env", lambda *a, **k: None)


def write_segmentation_manifest(path, config, *, modality="CT"):
    resolved = copy.deepcopy(config)
    backend = resolved.pop("backend", "totalsegmentator")
    crop = resolved.pop("crop", None)
    crop_margin_mm = resolved.pop("crop_margin_mm", None)
    segmentation = {
        "backend": backend,
        "modalities": {modality: resolved},
    }
    if crop is not None:
        segmentation["crop"] = crop
    if crop_margin_mm is not None:
        segmentation["crop_margin_mm"] = crop_margin_mm
    manifest = {
        "segmentation": segmentation
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def test_normalize_segment_args_accepts_positional_csv_path_out(tmp_path):
    csv_path = tmp_path / "nifti_index.csv"
    csv_path.write_text("nifti_path\n")
    csv_out = tmp_path / "seg_custom.csv"

    args = argparse.Namespace(
        csv_path_pos=str(csv_path),
        csv_path_opt=None,
        csv_path_out_pos=str(csv_out),
        csv_path_out=None,
        error_csv_path=None,
    )

    out = segment_module.normalize_segment_args(args)

    assert out.csv_path == str(csv_path.resolve())
    assert out.csv_path_out == str(csv_out)
    assert not hasattr(out, "csv_path_out_pos")


def test_normalize_segment_args_prefers_flag_csv_path_out_over_positional(tmp_path):
    csv_path = tmp_path / "nifti_index.csv"
    csv_path.write_text("nifti_path\n")
    csv_out_pos = tmp_path / "seg_pos.csv"
    csv_out_opt = tmp_path / "seg_opt.csv"

    args = argparse.Namespace(
        csv_path_pos=str(csv_path),
        csv_path_opt=None,
        csv_path_out_pos=str(csv_out_pos),
        csv_path_out=str(csv_out_opt),
        error_csv_path=None,
    )

    out = segment_module.normalize_segment_args(args)

    assert out.csv_path_out == str(csv_out_opt)


def test_crop_cli_uses_default_or_explicit_margin():
    parser = segment_module.build_parser()

    defaults = parser.parse_args(["--crop"])
    explicit = parser.parse_args(["--crop", "--crop-margin-mm", "7.5"])
    inherited = parser.parse_args([])
    disabled = parser.parse_args(["--no-crop"])

    assert defaults.crop and defaults.crop_margin_mm is None
    assert explicit.crop and explicit.crop_margin_mm == 7.5
    assert inherited.crop is None and inherited.crop_margin_mm is None
    assert disabled.crop is False


def test_crop_cli_and_manifest_precedence():
    config = {"crop": True, "crop_margin_mm": 35.0}

    assert (
        segment_module._requested_crop_margin(argparse.Namespace(), config) == 35.0
    )
    assert (
        segment_module._requested_crop_margin(
            argparse.Namespace(crop=True, crop_margin_mm=7.5), config
        )
        == 7.5
    )
    assert (
        segment_module._requested_crop_margin(
            argparse.Namespace(crop=False, crop_margin_mm=None), config
        )
        is None
    )


@pytest.mark.parametrize("margin", [-1, float("nan"), float("inf")])
def test_normalize_segment_args_rejects_invalid_crop_margin(tmp_path, margin):
    csv_path = tmp_path / "nifti_index.csv"
    csv_path.write_text("nifti_path\n")
    args = argparse.Namespace(
        csv_path_pos=str(csv_path),
        csv_path_opt=None,
        csv_path_out_pos=None,
        csv_path_out=None,
        error_csv_path=None,
        crop=True,
        crop_margin_mm=margin,
    )

    with pytest.raises(ValueError, match="--crop-margin-mm"):
        segment_module.normalize_segment_args(args)


def test_crop_to_segmented_organs_uses_union_margin_and_preserves_world_space(
    tmp_path,
):
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, -5.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    image_path = tmp_path / "scan.nii.gz"
    liver_path = tmp_path / "liver.nii.gz"
    spleen_path = tmp_path / "spleen.nii.gz"
    image_data = np.arange(10 * 12 * 8, dtype=np.int16).reshape(10, 12, 8)
    liver = np.zeros(image_data.shape, dtype=np.uint8)
    spleen = np.zeros_like(liver)
    liver[3:5, 4:7, 2:4] = 1
    spleen[6:8, 8:10, 4:6] = 1
    nib.save(nib.Nifti1Image(image_data, affine), image_path)
    save_nifti(liver, affine, liver_path)
    save_nifti(spleen, affine, spleen_path)

    assert segment_module.crop_to_segmented_organs(
        image_path, [liver_path, spleen_path], margin_mm=2.0
    )

    # Margin in voxels is ceil(2 mm / [2, 1, 4]) == [1, 2, 1].
    expected_slice = np.s_[2:9, 2:12, 1:7]
    cropped_image = nib.load(image_path)
    cropped_liver = nib.load(liver_path)
    cropped_spleen = nib.load(spleen_path)
    assert cropped_image.shape == (7, 10, 6)
    assert cropped_liver.shape == cropped_image.shape == cropped_spleen.shape
    np.testing.assert_array_equal(
        np.asanyarray(cropped_image.dataobj), image_data[expected_slice]
    )
    np.testing.assert_array_equal(
        np.asanyarray(cropped_liver.dataobj), liver[expected_slice]
    )
    np.testing.assert_array_equal(
        np.asanyarray(cropped_spleen.dataobj), spleen[expected_slice]
    )
    np.testing.assert_allclose(
        cropped_image.affine,
        affine
        @ np.array(
            [
                [1, 0, 0, 2],
                [0, 1, 0, 2],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
            ]
        ),
    )
    # The first retained voxel keeps its pre-crop world coordinate.
    np.testing.assert_allclose(
        nib.affines.apply_affine(cropped_image.affine, [0, 0, 0]),
        nib.affines.apply_affine(affine, [2, 2, 1]),
    )


def test_crop_to_segmented_organs_skips_all_empty_masks(tmp_path):
    affine = np.eye(4)
    image_path = tmp_path / "scan.nii.gz"
    mask_path = tmp_path / "liver.nii.gz"
    image = np.arange(64, dtype=np.int16).reshape(4, 4, 4)
    nib.save(nib.Nifti1Image(image, affine), image_path)
    save_nifti(np.zeros_like(image), affine, mask_path)

    assert not segment_module.crop_to_segmented_organs(
        image_path, [mask_path], margin_mm=0
    )
    np.testing.assert_array_equal(np.asanyarray(nib.load(image_path).dataobj), image)


def test_crop_margin_is_clipped_to_true_image_size_without_padding(tmp_path):
    affine = np.eye(4)
    image_path = tmp_path / "scan.nii.gz"
    mask_path = tmp_path / "liver.nii.gz"
    image = np.arange(4**3, dtype=np.int16).reshape(4, 4, 4)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[2, 2, 2] = 1
    nib.save(nib.Nifti1Image(image, affine), image_path)
    save_nifti(mask, affine, mask_path)

    assert segment_module.crop_to_segmented_organs(
        image_path, [mask_path], margin_mm=50
    )

    # The requested margin exceeds the available extent, so the true image
    # size wins; no synthetic voxels are added.
    assert nib.load(image_path).shape == image.shape
    assert nib.load(mask_path).shape == mask.shape
    np.testing.assert_array_equal(np.asanyarray(nib.load(image_path).dataobj), image)


def test_process_single_volume_crops_image_and_resolved_masks(tmp_path):
    affine = np.eye(4)
    image_path = tmp_path / "scan.nii.gz"
    image = np.arange(6**3, dtype=np.int16).reshape(6, 6, 6)
    nib.save(nib.Nifti1Image(image, affine), image_path)
    config = {
        "backend": "totalsegmentator",
        "tasks": [{"task": "total", "output": "liver", "extra": {}}],
    }

    class NiftiBackend:
        def run(self, *, input_path, output_dir, task, **kwargs):
            liver = np.zeros((6, 6, 6), dtype=np.uint8)
            liver[2:4, 1:5, 3:5] = 1
            save_nifti(liver, affine, output_dir / "liver.nii.gz")

    idx, out_dir, error, warning, outputs = segment_module.process_single_volume(
        4,
        {"nifti_path": str(image_path)},
        config,
        verbose=False,
        force=True,
        crop_margin_mm=0,
        backend=NiftiBackend(),
    )

    assert (idx, out_dir, error, warning) == (4, str(tmp_path), None, None)
    assert outputs == {"liver": "liver"}
    assert nib.load(image_path).shape == (2, 4, 2)
    assert nib.load(tmp_path / "liver.nii.gz").shape == (2, 4, 2)


def test_segment_volume_crops_before_morphological_postprocessing(tmp_path):
    affine = np.eye(4)
    image_path = tmp_path / "scan.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((7, 7, 7)), affine), image_path)
    config = {
        "backend": "totalsegmentator",
        "tasks": [{"task": "total", "output": "liver", "extra": {}}],
        "postprocess": {
            "operations": [
                {
                    "op": "dilate",
                    "input": "liver",
                    "output": "dilated",
                    "radius_vox": 1,
                }
            ]
        },
    }

    class SingleVoxelBackend:
        def run(self, *, input_path, output_dir, task, **kwargs):
            liver = np.zeros((7, 7, 7), dtype=np.uint8)
            liver[3, 3, 3] = 1
            save_nifti(liver, affine, output_dir / "liver.nii.gz")

    segment_module.segment_volume(
        image_path,
        tmp_path,
        config,
        force=True,
        crop_margin_mm=0,
        backend=SingleVoxelBackend(),
    )

    # With pre-postprocessing cropping the dilation runs on a 1x1x1 grid and
    # cannot expand beyond it. Cropping after dilation would produce 3x3x3.
    assert nib.load(image_path).shape == (1, 1, 1)
    dilated = nib.load(tmp_path / "dilated.nii.gz")
    assert dilated.shape == (1, 1, 1)
    assert np.asanyarray(dilated.dataobj).sum() == 1


def test_load_segmentation_config_default():
    cfg = segment_module.load_segmentation_config(
        None, base_path=Path(__file__).resolve().parents[2] / "src" / "imperandi"
    )
    assert set(cfg["modalities"]) == {"CT", "MR"}
    assert cfg["modalities"]["CT"]["tasks"][0]["task"] == "total"
    assert cfg["modalities"]["MR"]["tasks"][0]["task"] == "total_mr"
    assert cfg["backend"] == "totalsegmentator"
    assert cfg["crop"] is False
    assert cfg["crop_margin_mm"] == 50.0


def test_load_segmentation_config_missing(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError):
        segment_module.load_segmentation_config(
            str(missing),
            base_path=Path(__file__).resolve().parents[2] / "src" / "imperandi",
        )


def test_validate_segmentation_config_requires_modality_map():
    with pytest.raises(ValueError, match="segmentation.modalities"):
        segment_module.validate_segmentation_config(
            {
                "backend": "totalsegmentator",
                "tasks": [{"task": "total"}],
            }
        )


def test_validate_segmentation_config_accepts_crop_settings():
    config = segment_module.validate_segmentation_config(
        {
            "crop": True,
            "crop_margin_mm": 12,
            "modalities": {"CT": {"tasks": [{"task": "total"}]}},
        }
    )

    assert config["crop"] is True
    assert config["crop_margin_mm"] == 12.0


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("crop", "yes", "segmentation.crop"),
        ("crop_margin_mm", -1, "segmentation.crop_margin_mm"),
        ("crop_margin_mm", float("inf"), "segmentation.crop_margin_mm"),
    ],
)
def test_validate_segmentation_config_rejects_invalid_crop_settings(
    key, value, message
):
    with pytest.raises(ValueError, match=message):
        segment_module.validate_segmentation_config(
            {
                key: value,
                "modalities": {"CT": {"tasks": [{"task": "total"}]}},
            }
        )


def test_validate_segmentation_config_rejects_model_modality_mismatch():
    with pytest.raises(ValueError, match="MR model"):
        segment_module.validate_segmentation_config(
            {
                "backend": "totalsegmentator",
                "modalities": {
                    "CT": {"tasks": [{"task": "total_mr"}]},
                },
            }
        )


def test_resolve_segmentation_config_uses_ct_and_mr_models():
    config = segment_module.validate_segmentation_config(
        {
            "backend": "totalsegmentator",
            "modalities": {
                "CT": {"tasks": [{"task": "total"}]},
                "MRI": {"tasks": [{"task": "total_mr"}]},
            },
        }
    )

    ct = segment_module.resolve_segmentation_config_for_modality(config, "CT")
    mr = segment_module.resolve_segmentation_config_for_modality(config, "MR")

    assert ct["tasks"][0]["task"] == "total"
    assert mr["tasks"][0]["task"] == "total_mr"
    assert segment_module.resolve_segmentation_config_for_modality(config, "US") is None


def test_resolve_prefetch_task_name_prefers_fast_variant_from_extra():
    task = {"task": "total", "extra": {"fastest": True}}
    assert segment_module._resolve_prefetch_task_name(task) == "total_fast"


def test_resolve_prefetch_task_name_keeps_original_without_fast_flags():
    task = {"task": "total", "extra": {"roi_subset_robust": ["liver"]}}
    assert segment_module._resolve_prefetch_task_name(task) == "total"


def test_resolve_prefetch_task_name_handles_body_mr_fast_alias():
    task = {"task": "body_mr", "extra": {"fast": True}}
    assert segment_module._resolve_prefetch_task_name(task) == "body_mr_fast"


def test_resolve_runtime_task_strips_fast_suffix_for_execution():
    task_name, extra = segment_module._resolve_runtime_task("total_fast", {})
    assert task_name == "total"
    assert extra["fast"] is True


def test_prefetch_totalsegmentator_models_requires_supported_liver_lesions_version(
    monkeypatch,
):
    monkeypatch.setattr(
        segment_module, "_get_totalsegmentator_version", lambda: "2.12.0"
    )

    with pytest.raises(
        RuntimeError,
        match=r"task needs totalsegmentator version >= 2\.13\.0, current version==2\.12\.0",
    ):
        segment_module.prefetch_totalsegmentator_models(
            {
                "backend": "totalsegmentator",
                "tasks": [{"task": "liver_lesions"}],
            }
        )


def test_prefetch_totalsegmentator_models_downloads_liver_lesions_when_version_supported(
    monkeypatch,
):
    calls = []

    fake_root = types.ModuleType("totalsegmentator")
    fake_python_api = types.ModuleType("totalsegmentator.python_api")

    def fake_download_pretrained_weights(task_id):
        calls.append(task_id)

    fake_python_api.download_pretrained_weights = fake_download_pretrained_weights
    fake_root.python_api = fake_python_api

    monkeypatch.setitem(sys.modules, "totalsegmentator", fake_root)
    monkeypatch.setitem(sys.modules, "totalsegmentator.python_api", fake_python_api)
    monkeypatch.setattr(
        segment_module, "_get_totalsegmentator_version", lambda: "2.13.0"
    )

    segment_module.prefetch_totalsegmentator_models(
        {
            "backend": "totalsegmentator",
            "tasks": [{"task": "liver_lesions"}],
        }
    )

    assert calls == [591]


def test_prefetch_downloads_only_models_for_active_modalities(monkeypatch):
    calls = []
    fake_root = types.ModuleType("totalsegmentator")
    fake_python_api = types.ModuleType("totalsegmentator.python_api")
    fake_python_api.download_pretrained_weights = calls.append
    fake_root.python_api = fake_python_api
    monkeypatch.setitem(sys.modules, "totalsegmentator", fake_root)
    monkeypatch.setitem(sys.modules, "totalsegmentator.python_api", fake_python_api)

    config = segment_module.validate_segmentation_config(
        {
            "backend": "totalsegmentator",
            "modalities": {
                "CT": {"tasks": [{"task": "total"}]},
                "MR": {"tasks": [{"task": "total_mr"}]},
            },
        }
    )

    segment_module.prefetch_totalsegmentator_models(config, modalities={"MR"})

    assert calls == [850, 851]


def test_infer_task_fetch_outputs_supports_aliasing_backend_filename():
    task = {
        "task": "liver_lesions",
        "output": "liver_tumor.nii.gz",
        "fetch_output": "liver_lesion.nii.gz",
    }

    assert segment_module.infer_task_fetch_outputs(task) == {
        "liver_tumor": "liver_lesion"
    }


def test_segment_volume_infers_outputs_from_created_segmentations(tmp_path):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    tasks_config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"task": "task_a", "extra": {}},
        ],
    }

    class DynamicBackend:
        def run(self, *, input_path, output_dir, task, **kwargs):
            (Path(output_dir) / "inferred_mask.nii.gz").write_text("mask")

    resolved = {}
    warnings = segment_module.segment_volume(
        nifti,
        tmp_path,
        tasks_config,
        verbose=False,
        force=True,
        backend=DynamicBackend(),
        resolved_output_to_fetch=resolved,
    )

    assert warnings == []
    assert resolved == {"inferred_mask": "inferred_mask"}


@pytest.mark.parametrize("declare_total_output", [False, True])
@pytest.mark.parametrize("reuse_total_masks", [False, True])
def test_merge_accepts_masks_not_enumerated_by_total(
    tmp_path, declare_total_output, reuse_total_masks
):
    nifti = tmp_path / "vol.nii.gz"
    save_nifti(np.zeros((4, 4, 4)), np.eye(4), nifti)
    total = {"task": "total"}
    if declare_total_output:
        total["output"] = "spleen"
    config = segment_module.validate_segmentation_config(
        {
            "modalities": {
                "CT": {
                    "tasks": [
                        total,
                        {
                            "task": "liver_lesions",
                            "output": "liver_tumor",
                            "fetch_output": "liver_lesions",
                        },
                    ],
                    "postprocess": {
                        "operations": [
                            {
                                "op": "union",
                                "inputs": ["liver", "liver_tumor"],
                                "output": "merged",
                            }
                        ]
                    },
                }
            },
        }
    )
    calls = []
    liver = np.zeros((4, 4, 4), dtype=np.uint8)
    liver[1, 1, 1] = 1
    tumor = np.zeros_like(liver)
    tumor[2, 2, 2] = 1
    if reuse_total_masks:
        for name in ("liver", "spleen"):
            save_nifti(liver, np.eye(4), tmp_path / f"{name}.nii.gz")

    class MultiMaskBackend:
        def run(self, *, input_path, output_dir, task, **kwargs):
            calls.append(task)
            if task == "total" and reuse_total_masks:
                return
            outputs = (
                {"liver": liver, "spleen": liver}
                if task == "total"
                else {"liver_lesions": tumor}
            )
            for name, data in outputs.items():
                save_nifti(data, np.eye(4), output_dir / f"{name}.nii.gz")

    idx, out_dir, error, warning, outputs = segment_module.process_single_volume(
        0,
        {"nifti_path": str(nifti), "Modality": "CT"},
        config,
        verbose=False,
        force=False,
        backend=MultiMaskBackend(),
    )
    assert (idx, out_dir, error, warning) == (0, str(tmp_path), None, None)
    assert calls == (
        ["liver_lesions"]
        if declare_total_output and reuse_total_masks
        else ["total", "liver_lesions"]
    )
    expected_outputs = {
        "liver": "liver",
        "liver_tumor": "liver_lesions",
        "merged": "merged",
    }
    if declare_total_output or not reuse_total_masks:
        expected_outputs["spleen"] = "spleen"
    assert outputs == expected_outputs
    np.testing.assert_array_equal(
        nib.load(tmp_path / "merged.nii.gz").get_fdata(), liver | tumor
    )


@pytest.mark.parametrize("missing_first", [True, False])
def test_operations_fail_at_fetch_without_writing_partial_merge(
    tmp_path, missing_first
):
    present = tmp_path / "present.nii.gz"
    save_nifti(np.ones((4, 4, 4)), np.eye(4), present)
    original = present.read_bytes()
    masks = (
        ["missing.nii.gz", present.name]
        if missing_first
        else [present.name, "missing.nii.gz"]
    )
    with pytest.raises(FileNotFoundError, match="missing.nii.gz"):
        operations = segment_module.validate_operations(
            {"operations": [{"op": "union", "inputs": masks, "output": "merged"}]},
            task_outputs={"present"},
        )
        segment_module.apply_operations(tmp_path, operations, {})
    assert not (tmp_path / "merged.nii.gz").exists()
    assert present.read_bytes() == original


def test_undeclared_missing_mask_fails_after_all_tasks_run(tmp_path):
    nifti = tmp_path / "vol.nii.gz"
    save_nifti(np.zeros((4, 4, 4)), np.eye(4), nifti)
    config = segment_module.validate_segmentation_config(
        {
            "modalities": {
                "CT": {
                    "tasks": [
                        {"task": "total"},
                        {
                            "task": "liver_lesions",
                            "output": "liver_tumor",
                            "fetch_output": "liver_lesions",
                        },
                    ],
                    "postprocess": {
                        "operations": [
                            {
                                "op": "union",
                                "inputs": ["liver", "liver_tumor"],
                                "output": "merged",
                            }
                        ],
                        "on_failure": "warn_only",
                    },
                }
            },
        }
    )
    calls = []

    class MissingLiverBackend:
        def run(self, *, input_path, output_dir, task, **kwargs):
            calls.append(task)
            name = "spleen" if task == "total" else "liver_lesions"
            save_nifti(np.ones((4, 4, 4)), np.eye(4), output_dir / f"{name}.nii.gz")

    _, out_dir, error, _, _ = segment_module.process_single_volume(
        0,
        {"nifti_path": str(nifti), "Modality": "CT"},
        config,
        verbose=False,
        force=False,
        backend=MissingLiverBackend(),
    )
    assert calls == ["total", "liver_lesions"]
    assert out_dir is None
    assert "liver.nii.gz" in error
    assert "No such file" in error
    assert not (tmp_path / "merged.nii.gz").exists()


@pytest.mark.parametrize("has_liver", [False, True])
def test_existing_merged_output_does_not_hide_missing_merge_input(tmp_path, has_liver):
    nifti = tmp_path / "vol.nii.gz"
    save_nifti(np.zeros((4, 4, 4)), np.eye(4), nifti)
    config = {
        "tasks": [{"task": "total", "output": "spleen"}],
        "postprocess": {
            "operations": [{"op": "union", "inputs": ["liver"], "output": "merged"}]
        },
    }
    for name in ["spleen", "merged", *(["liver"] if has_liver else [])]:
        save_nifti(np.ones((4, 4, 4)), np.eye(4), tmp_path / f"{name}.nii.gz")
    merged_before = (tmp_path / "merged.nii.gz").read_bytes()
    assert not segment_module._has_existing_task_outputs(tmp_path, config)

    class UnexpectedBackend:
        def run(self, **kwargs):
            pytest.fail("The declared task output already exists")

    resolved = {}
    if has_liver:
        assert (
            segment_module.segment_volume(
                nifti,
                tmp_path,
                config,
                backend=UnexpectedBackend(),
                resolved_output_to_fetch=resolved,
            )
            == []
        )
        assert resolved["liver"] == "liver"
        assert segment_module._has_existing_task_outputs(tmp_path, config)
    else:
        with pytest.raises(FileNotFoundError, match="liver.nii.gz"):
            segment_module.segment_volume(
                nifti, tmp_path, config, backend=UnexpectedBackend()
            )
    assert (tmp_path / "merged.nii.gz").read_bytes() == merged_before


def test_process_single_volume_success(tmp_path):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    tasks_config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "a", "task": "task_a", "output": "a.nii.gz", "extra": {}},
        ],
    }

    backend = DummyBackend({"task_a": "a.nii.gz"})

    idx, out_dir, err, warning, outputs = segment_module.process_single_volume(
        0,
        {"nifti_path": str(nifti)},
        tasks_config,
        verbose=False,
        force=True,
        backend=backend,
    )

    assert err is None
    assert warning is None
    assert out_dir == str(tmp_path)
    assert outputs == {"a": "a"}


def test_process_single_volume_infers_outputs_when_not_declared(tmp_path):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    tasks_config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"task": "task_a", "extra": {}},
        ],
    }

    class DynamicBackend:
        def run(self, *, input_path, output_dir, task, **kwargs):
            (Path(output_dir) / "created_here.nii.gz").write_text("mask")

    idx, out_dir, err, warning, outputs = segment_module.process_single_volume(
        0,
        {"nifti_path": str(nifti)},
        tasks_config,
        verbose=False,
        force=True,
        backend=DynamicBackend(),
    )

    assert idx == 0
    assert err is None
    assert warning is None
    assert out_dir == str(tmp_path)
    assert outputs == {"created_here": "created_here"}


def test_process_single_volume_missing_output(tmp_path):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    tasks_config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "a", "task": "task_a", "output": "a.nii.gz", "extra": {}},
        ],
    }

    class NoWriteBackend:
        def run(self, *, input_path, output_dir, task, **kwargs):
            return None

    idx, out_dir, err, warning, outputs = segment_module.process_single_volume(
        0,
        {"nifti_path": str(nifti)},
        tasks_config,
        verbose=False,
        force=True,
        backend=NoWriteBackend(),
    )

    assert out_dir is None
    assert "Expected mask not produced" in err
    assert warning is None
    assert outputs is None


def test_process_single_volume_skips_unconfigured_modality_before_file_access():
    config = segment_module.validate_segmentation_config(
        {
            "backend": "totalsegmentator",
            "modalities": {"CT": {"tasks": [{"task": "total"}]}},
        }
    )

    result = segment_module.process_single_volume(
        3,
        {"nifti_path": "missing.nii.gz", "Modality": "MR"},
        config,
        verbose=False,
        force=False,
    )

    assert result == (3, None, None, None, {})


def test_main_dispatches_totalsegmentator_models_by_modality(tmp_path, monkeypatch):
    ct_dir = tmp_path / "ct"
    mr_dir = tmp_path / "mr"
    ct_dir.mkdir()
    mr_dir.mkdir()
    ct_nifti = ct_dir / "scan.nii.gz"
    mr_nifti = mr_dir / "scan.nii.gz"
    ct_nifti.write_text("nifti")
    mr_nifti.write_text("nifti")

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame(
        [
            {"nifti_path": str(ct_nifti), "Modality": "CT"},
            {"nifti_path": str(mr_nifti), "Modality": "MRI"},
        ]
    ).to_csv(csv_path, index=False)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "segmentation": {
                    "backend": "totalsegmentator",
                    "modalities": {
                        "CT": {"tasks": [{"task": "total", "output": "liver"}]},
                        "MR": {"tasks": [{"task": "total_mr", "output": "liver"}]},
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    calls = []

    def fake_process_single_volume(
        idx, row, tasks_config, *, verbose, force, backend=None, **kwargs
    ):
        task_name = tasks_config["tasks"][0]["task"]
        calls.append((row["Modality"], task_name))
        out_dir = Path(row["nifti_path"]).parent
        (out_dir / "liver.nii.gz").write_text("mask")
        return idx, str(out_dir), None, None, {"liver": "liver"}

    monkeypatch.setattr(
        segment_module, "process_single_volume", fake_process_single_volume
    )
    monkeypatch.setattr(
        segment_module, "prefetch_totalsegmentator_models", lambda *a, **k: None
    )
    monkeypatch.setattr(segment_module, "tqdm", passthrough_tqdm)
    patch_strategy(monkeypatch, mode="serial", max_workers=1, max_in_flight=1)

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "segmented.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        manifest=str(manifest_path),
        num_workers=1,
        verbose=False,
        force=False,
        start_method="spawn",
        timeout_sec=10,
    )

    segment_module.main(args)

    assert calls == [("CT", "total"), ("MRI", "total_mr")]
    out = pd.read_csv(args.csv_path_out)
    assert out["mask_liver"].notna().all()


def test_main_writes_mask_columns(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    csv_path = tmp_path / "nifti_index.csv"
    df = pd.DataFrame([{"nifti_path": str(nifti), "Modality": "CT"}])
    df.to_csv(csv_path, index=False)

    config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "liver", "task": "total", "output": "liver.nii.gz", "extra": {}},
            {
                "key": "vessels",
                "task": "liver_vessels",
                "output": "vessels.nii.gz",
                "extra": {},
            },
        ],
        "postprocess": {
            "operations": [
                {
                    "op": "union",
                    "inputs": ["liver", "vessels"],
                    "output": "merged.nii.gz",
                }
            ]
        },
    }

    config_path = tmp_path / "manifest.yaml"
    write_segmentation_manifest(config_path, config)

    class DummyFuture:
        def __init__(self, result):
            self._result = result

        def result(self, timeout=None):
            return self._result

        def cancel(self):
            return None

    class DummyPool:
        def __init__(self, max_workers=None, mp_context=None):
            self._processes = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def submit(self, fn, *args, **kwargs):
            idx = args[0]
            row = args[1]
            out_dir = str(Path(row["nifti_path"]).parent)
            out_path = Path(out_dir)
            (out_path / "liver.nii.gz").write_text("mask")
            (out_path / "vessels.nii.gz").write_text("mask")
            (out_path / "merged.nii.gz").write_text("mask")
            return DummyFuture((idx, out_dir, None, None))

        def shutdown(self, wait=False, cancel_futures=True):
            return None

    monkeypatch.setattr(segment_module, "ProcessPoolExecutor", DummyPool)
    monkeypatch.setattr(segment_module, "tqdm", passthrough_tqdm)
    monkeypatch.setattr(
        segment_module, "prefetch_totalsegmentator_models", lambda *a, **k: None
    )
    patch_strategy(monkeypatch, mode="process_pool", max_workers=2, max_in_flight=2)

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "segmented.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        manifest=str(config_path),
        num_workers=2,
        verbose=False,
        force=False,
        start_method="spawn",
        timeout_sec=10,
    )

    segment_module.main(args)

    out_df = pd.read_csv(args.csv_path_out)
    assert "mask_liver" in out_df.columns
    assert "mask_vessels" in out_df.columns
    assert "mask_merged" in out_df.columns
    assert out_df.loc[0, "mask_liver"].endswith("liver.nii.gz")
    assert out_df.loc[0, "mask_vessels"].endswith("vessels.nii.gz")
    assert out_df.loc[0, "mask_merged"].endswith("merged.nii.gz")


def test_main_maps_fetch_output_path_into_logical_mask_column(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    csv_path = tmp_path / "nifti_index.csv"
    df = pd.DataFrame([{"nifti_path": str(nifti), "Modality": "CT"}])
    df.to_csv(csv_path, index=False)

    config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"task": "total", "output": "liver.nii.gz", "extra": {}},
            {
                "task": "liver_lesions",
                "output": "liver_tumor.nii.gz",
                "fetch_output": "liver_lesion.nii.gz",
                "extra": {},
            },
        ],
        "postprocess": {
            "operations": [
                {
                    "op": "union",
                    "inputs": ["liver", "liver_tumor"],
                    "output": "merged.nii.gz",
                }
            ]
        },
    }

    config_path = tmp_path / "manifest.yaml"
    write_segmentation_manifest(config_path, config)

    class DummyFuture:
        def __init__(self, result):
            self._result = result

        def result(self, timeout=None):
            return self._result

        def cancel(self):
            return None

    class DummyPool:
        def __init__(self, max_workers=None, mp_context=None):
            self._processes = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def submit(self, fn, *args, **kwargs):
            idx = args[0]
            row = args[1]
            out_dir = str(Path(row["nifti_path"]).parent)
            out_path = Path(out_dir)
            (out_path / "liver.nii.gz").write_text("mask")
            (out_path / "liver_lesion.nii.gz").write_text("mask")
            (out_path / "merged.nii.gz").write_text("mask")
            return DummyFuture((idx, out_dir, None, None))

        def shutdown(self, wait=False, cancel_futures=True):
            return None

    monkeypatch.setattr(segment_module, "ProcessPoolExecutor", DummyPool)
    monkeypatch.setattr(segment_module, "tqdm", passthrough_tqdm)
    monkeypatch.setattr(
        segment_module, "prefetch_totalsegmentator_models", lambda *a, **k: None
    )
    patch_strategy(monkeypatch, mode="process_pool", max_workers=2, max_in_flight=2)

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "segmented.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        manifest=str(config_path),
        num_workers=2,
        verbose=False,
        force=False,
        start_method="spawn",
        timeout_sec=10,
    )

    segment_module.main(args)

    out_df = pd.read_csv(args.csv_path_out)
    assert "mask_liver_tumor" in out_df.columns
    assert out_df.loc[0, "mask_liver"].endswith("liver.nii.gz")
    assert out_df.loc[0, "mask_liver_tumor"].endswith("liver_lesion.nii.gz")
    assert out_df.loc[0, "mask_merged"].endswith("merged.nii.gz")


def test_main_adds_mask_columns_for_runtime_inferred_outputs(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")
    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": str(nifti), "Modality": "CT"}]).to_csv(
        csv_path, index=False
    )

    config_path = tmp_path / "manifest.yaml"
    write_segmentation_manifest(
        config_path,
        {
            "backend": "totalsegmentator",
            "tasks": [{"task": "task_a"}],
        },
    )

    monkeypatch.setattr(
        segment_module, "prefetch_totalsegmentator_models", lambda *a, **k: None
    )
    monkeypatch.setattr(segment_module, "tqdm", passthrough_tqdm)
    patch_strategy(monkeypatch, mode="serial", max_workers=1, max_in_flight=1)

    def fake_process_single_volume(
        idx, row, tasks_config, *, verbose, force, backend=None, **kwargs
    ):
        out_dir = Path(row["nifti_path"]).parent
        (out_dir / "runtime_inferred.nii.gz").write_text("mask")
        return idx, str(out_dir), None, None, {"runtime_inferred": "runtime_inferred"}

    monkeypatch.setattr(
        segment_module, "process_single_volume", fake_process_single_volume
    )

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "segmented.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        manifest=str(config_path),
        num_workers=1,
        verbose=False,
        force=False,
        start_method="spawn",
        timeout_sec=10,
    )

    segment_module.main(args)

    out_df = pd.read_csv(args.csv_path_out)
    assert "mask_runtime_inferred" in out_df.columns
    assert out_df.loc[0, "mask_runtime_inferred"].endswith("runtime_inferred.nii.gz")


def test_main_records_warning_when_merged_mask_missing(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": str(nifti), "Modality": "CT"}]).to_csv(
        csv_path, index=False
    )

    config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "liver", "task": "total", "output": "liver.nii.gz", "extra": {}},
        ],
        "postprocess": {
            "operations": [
                {"op": "union", "inputs": ["liver"], "output": "merged.nii.gz"}
            ]
        },
    }

    config_path = tmp_path / "manifest.yaml"
    write_segmentation_manifest(config_path, config)

    class DummyFuture:
        def __init__(self, result):
            self._result = result

        def result(self, timeout=None):
            return self._result

        def cancel(self):
            return None

    class DummyPool:
        def __init__(self, max_workers=None, mp_context=None):
            self._processes = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def submit(self, fn, *args, **kwargs):
            idx = args[0]
            row = args[1]
            out_dir = Path(row["nifti_path"]).parent
            (out_dir / "liver.nii.gz").write_text("mask")
            return DummyFuture((idx, str(out_dir), None, None))

        def shutdown(self, wait=False, cancel_futures=True):
            return None

    monkeypatch.setattr(segment_module, "ProcessPoolExecutor", DummyPool)
    monkeypatch.setattr(segment_module, "tqdm", passthrough_tqdm)
    monkeypatch.setattr(
        segment_module, "prefetch_totalsegmentator_models", lambda *a, **k: None
    )
    patch_strategy(monkeypatch, mode="process_pool", max_workers=2, max_in_flight=2)

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "segmented.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        manifest=str(config_path),
        num_workers=2,
        verbose=False,
        force=False,
        start_method="spawn",
        timeout_sec=10,
    )

    segment_module.main(args)

    out_df = pd.read_csv(args.csv_path_out)
    assert "warning_message" in out_df.columns
    assert "missing postprocess mask" in out_df.loc[0, "warning_message"]
    assert pd.isna(out_df.loc[0, "mask_merged"])


def test_main_single_worker_avoids_process_pool(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": str(nifti), "Modality": "CT"}]).to_csv(
        csv_path, index=False
    )

    config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "liver", "task": "total", "output": "liver.nii.gz", "extra": {}},
        ],
    }
    config_path = tmp_path / "manifest.yaml"
    write_segmentation_manifest(config_path, config)

    monkeypatch.setattr(
        segment_module, "prefetch_totalsegmentator_models", lambda *a, **k: None
    )
    monkeypatch.setattr(segment_module, "tqdm", passthrough_tqdm)
    patch_strategy(monkeypatch, mode="serial", max_workers=1, max_in_flight=1)

    def fail_if_pool_used(*args, **kwargs):
        raise AssertionError(
            "ProcessPoolExecutor should not be used in single-worker mode"
        )

    monkeypatch.setattr(segment_module, "ProcessPoolExecutor", fail_if_pool_used)

    def fake_process_single_volume(
        idx, row, tasks_config, *, verbose, force, backend=None, **kwargs
    ):
        out_dir = Path(row["nifti_path"]).parent
        (out_dir / "liver.nii.gz").write_text("mask")
        return idx, str(out_dir), None, None

    monkeypatch.setattr(
        segment_module, "process_single_volume", fake_process_single_volume
    )

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "segmented.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        manifest=str(config_path),
        num_workers=1,
        verbose=False,
        force=False,
        start_method="spawn",
        timeout_sec=10,
    )

    segment_module.main(args)

    out_df = pd.read_csv(args.csv_path_out)
    assert out_df.loc[0, "mask_liver"].endswith("liver.nii.gz")


def test_main_uses_strategy_effective_worker_count(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")
    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": str(nifti), "Modality": "CT"}]).to_csv(
        csv_path, index=False
    )
    config_path = tmp_path / "manifest.yaml"
    write_segmentation_manifest(
        config_path,
        {
            "backend": "totalsegmentator",
            "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
        },
    )

    monkeypatch.setattr(
        segment_module, "prefetch_totalsegmentator_models", lambda *a, **k: None
    )
    monkeypatch.setattr(segment_module, "tqdm", passthrough_tqdm)
    patch_strategy(monkeypatch, mode="process_pool", max_workers=1, max_in_flight=1)

    def fail_if_pool_used(*args, **kwargs):
        raise AssertionError(
            "ProcessPoolExecutor should not be used when strategy max_workers=1"
        )

    monkeypatch.setattr(segment_module, "ProcessPoolExecutor", fail_if_pool_used)

    def fake_process_single_volume(
        idx, row, tasks_config, *, verbose, force, backend=None, **kwargs
    ):
        out_dir = Path(row["nifti_path"]).parent
        (out_dir / "liver.nii.gz").write_text("mask")
        return idx, str(out_dir), None, None

    monkeypatch.setattr(
        segment_module, "process_single_volume", fake_process_single_volume
    )

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "segmented.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        manifest=str(config_path),
        num_workers=4,
        verbose=False,
        force=False,
        start_method="fork",
        timeout_sec=10,
    )

    segment_module.main(args)
    out_df = pd.read_csv(args.csv_path_out)
    assert out_df.loc[0, "mask_liver"].endswith("liver.nii.gz")


def test_main_uses_strategy_effective_start_method(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")
    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": str(nifti), "Modality": "CT"}]).to_csv(
        csv_path, index=False
    )
    config_path = tmp_path / "manifest.yaml"
    write_segmentation_manifest(
        config_path,
        {
            "backend": "totalsegmentator",
            "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
        },
    )

    monkeypatch.setattr(
        segment_module, "prefetch_totalsegmentator_models", lambda *a, **k: None
    )
    monkeypatch.setattr(segment_module, "tqdm", passthrough_tqdm)
    patch_strategy(
        monkeypatch,
        mode="process_pool",
        max_workers=2,
        max_in_flight=1,
        start_method="forkserver",
    )

    start_methods = []

    def fake_get_context(method):
        start_methods.append(method)
        return object()

    monkeypatch.setattr(segment_module.mp, "get_context", fake_get_context)

    class DummyFuture:
        def __init__(self, result):
            self._result = result

        def result(self, timeout=None):
            return self._result

        def cancel(self):
            return None

    class DummyPool:
        def __init__(self, max_workers=None, mp_context=None):
            self._processes = None

        def submit(self, fn, *args, **kwargs):
            idx = args[0]
            row = args[1]
            out_dir = Path(row["nifti_path"]).parent
            (out_dir / "liver.nii.gz").write_text("mask")
            return DummyFuture((idx, str(out_dir), None, None))

        def shutdown(self, wait=False, cancel_futures=True):
            return None

    monkeypatch.setattr(segment_module, "ProcessPoolExecutor", DummyPool)

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "segmented.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        manifest=str(config_path),
        num_workers=4,
        verbose=False,
        force=False,
        start_method="fork",
        timeout_sec=10,
    )

    segment_module.main(args)
    assert start_methods == ["forkserver"]


def test_main_enables_gpu_worker_pinning_for_multi_gpu(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")
    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": str(nifti), "Modality": "CT"}]).to_csv(
        csv_path, index=False
    )
    config_path = tmp_path / "manifest.yaml"
    write_segmentation_manifest(
        config_path,
        {
            "backend": "totalsegmentator",
            "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
        },
    )

    monkeypatch.setattr(
        segment_module, "prefetch_totalsegmentator_models", lambda *a, **k: None
    )
    monkeypatch.setattr(segment_module, "tqdm", passthrough_tqdm)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    patch_strategy(
        monkeypatch,
        mode="process_pool",
        max_workers=2,
        max_in_flight=1,
        use_gpu=True,
        gpu_count=2,
    )

    class DummyFuture:
        def __init__(self, result):
            self._result = result

        def result(self, timeout=None):
            return self._result

        def cancel(self):
            return None

    class DummyPool:
        last_initializer = None
        last_initargs = None

        def __init__(
            self,
            max_workers=None,
            mp_context=None,
            initializer=None,
            initargs=(),
        ):
            self._processes = None
            DummyPool.last_initializer = initializer
            DummyPool.last_initargs = initargs

        def submit(self, fn, *args, **kwargs):
            idx = args[0]
            row = args[1]
            out_dir = Path(row["nifti_path"]).parent
            (out_dir / "liver.nii.gz").write_text("mask")
            return DummyFuture((idx, str(out_dir), None, None))

        def shutdown(self, wait=False, cancel_futures=True):
            return None

    monkeypatch.setattr(segment_module, "ProcessPoolExecutor", DummyPool)

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "segmented.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        manifest=str(config_path),
        num_workers=2,
        verbose=False,
        force=False,
        start_method="spawn",
        timeout_sec=10,
    )

    segment_module.main(args)

    assert DummyPool.last_initializer is segment_module._worker_gpu_initializer
    assert DummyPool.last_initargs == (["4", "5"],)


def test_main_bounds_in_flight_submissions(tmp_path, monkeypatch):
    paths = []
    for i in range(5):
        nifti = tmp_path / f"vol_{i}.nii.gz"
        nifti.write_text("nifti")
        paths.append(str(nifti))

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": p, "Modality": "CT"} for p in paths]).to_csv(
        csv_path, index=False
    )
    config_path = tmp_path / "manifest.yaml"
    write_segmentation_manifest(
        config_path,
        {
            "backend": "totalsegmentator",
            "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
        },
    )

    monkeypatch.setattr(
        segment_module, "prefetch_totalsegmentator_models", lambda *a, **k: None
    )
    monkeypatch.setattr(segment_module, "tqdm", passthrough_tqdm)
    patch_strategy(monkeypatch, mode="process_pool", max_workers=4, max_in_flight=2)

    class DummyFuture:
        def __init__(self, result, pool):
            self._result = result
            self._pool = pool

        def result(self, timeout=None):
            self._pool.outstanding -= 1
            return self._result

        def cancel(self):
            return None

    class DummyPool:
        last = None

        def __init__(self, max_workers=None, mp_context=None):
            self._processes = None
            self.outstanding = 0
            self.max_outstanding = 0
            DummyPool.last = self

        def submit(self, fn, *args, **kwargs):
            idx = args[0]
            row = args[1]
            out_dir = Path(row["nifti_path"]).parent
            (out_dir / "liver.nii.gz").write_text("mask")
            self.outstanding += 1
            self.max_outstanding = max(self.max_outstanding, self.outstanding)
            return DummyFuture((idx, str(out_dir), None, None), self)

        def shutdown(self, wait=False, cancel_futures=True):
            return None

    monkeypatch.setattr(segment_module, "ProcessPoolExecutor", DummyPool)

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "segmented.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        manifest=str(config_path),
        num_workers=4,
        verbose=False,
        force=False,
        start_method="spawn",
        timeout_sec=10,
    )

    segment_module.main(args)
    assert DummyPool.last is not None
    assert DummyPool.last.max_outstanding <= 2


def test_main_enforces_wall_timeout_per_row(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")
    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": str(nifti), "Modality": "CT"}]).to_csv(
        csv_path, index=False
    )
    config_path = tmp_path / "manifest.yaml"
    write_segmentation_manifest(
        config_path,
        {
            "backend": "totalsegmentator",
            "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
        },
    )

    monkeypatch.setattr(
        segment_module, "prefetch_totalsegmentator_models", lambda *a, **k: None
    )
    monkeypatch.setattr(segment_module, "tqdm", passthrough_tqdm)
    patch_strategy(monkeypatch, mode="process_pool", max_workers=2, max_in_flight=1)

    class HangingFuture:
        def result(self, timeout=None):
            raise segment_module.TimeoutError()

        def cancel(self):
            return None

    class DummyPool:
        shutdown_calls = []

        def __init__(self, max_workers=None, mp_context=None):
            self._processes = None

        def submit(self, fn, *args, **kwargs):
            return HangingFuture()

        def shutdown(self, wait=False, cancel_futures=True):
            DummyPool.shutdown_calls.append((wait, cancel_futures))
            return None

    current = {"t": 0.0}

    def fake_monotonic():
        current["t"] += 0.6
        return current["t"]

    monkeypatch.setattr(segment_module, "ProcessPoolExecutor", DummyPool)
    monkeypatch.setattr(segment_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(segment_module.time, "sleep", lambda *_a, **_k: None)

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "segmented.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        manifest=str(config_path),
        num_workers=2,
        verbose=False,
        force=False,
        start_method="spawn",
        timeout_sec=1,
    )

    segment_module.main(args)

    err_df = pd.read_csv(args.error_csv_path)
    assert "timeout after 1s" in err_df.loc[0, "error_message"]
    assert (False, True) in DummyPool.shutdown_calls


def test_main_force_shutdown_terminates_and_joins_workers(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")
    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": str(nifti), "Modality": "CT"}]).to_csv(
        csv_path, index=False
    )
    config_path = tmp_path / "manifest.yaml"
    write_segmentation_manifest(
        config_path,
        {
            "backend": "totalsegmentator",
            "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
        },
    )

    monkeypatch.setattr(
        segment_module, "prefetch_totalsegmentator_models", lambda *a, **k: None
    )
    monkeypatch.setattr(segment_module, "tqdm", passthrough_tqdm)
    patch_strategy(monkeypatch, mode="process_pool", max_workers=2, max_in_flight=1)

    class HangingFuture:
        def result(self, timeout=None):
            raise segment_module.TimeoutError()

        def cancel(self):
            return None

    class DummyProcess:
        def __init__(self):
            self.alive = True
            self.terminate_calls = 0
            self.kill_calls = 0
            self.join_calls = []

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminate_calls += 1
            self.alive = False

        def kill(self):
            self.kill_calls += 1
            self.alive = False

        def join(self, timeout=None):
            self.join_calls.append(timeout)
            return None

    class DummyManagerThread:
        def __init__(self):
            self.join_calls = []

        def join(self, timeout=None):
            self.join_calls.append(timeout)
            return None

    class DummyPool:
        last_process = None
        last_manager = None
        shutdown_calls = []

        def __init__(self, max_workers=None, mp_context=None):
            proc = DummyProcess()
            manager = DummyManagerThread()
            self._processes = {1: proc}
            self._executor_manager_thread = manager
            DummyPool.last_process = proc
            DummyPool.last_manager = manager

        def submit(self, fn, *args, **kwargs):
            return HangingFuture()

        def shutdown(self, wait=False, cancel_futures=True):
            DummyPool.shutdown_calls.append((wait, cancel_futures))
            return None

    current = {"t": 0.0}

    def fake_monotonic():
        current["t"] += 0.6
        return current["t"]

    monkeypatch.setattr(segment_module, "ProcessPoolExecutor", DummyPool)
    monkeypatch.setattr(segment_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(segment_module.time, "sleep", lambda *_a, **_k: None)

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "segmented.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        manifest=str(config_path),
        num_workers=2,
        verbose=False,
        force=False,
        start_method="spawn",
        timeout_sec=1,
    )

    segment_module.main(args)

    assert DummyPool.last_process is not None
    assert DummyPool.last_process.terminate_calls >= 1
    assert len(DummyPool.last_process.join_calls) >= 1
    assert DummyPool.last_manager is not None
    assert DummyPool.last_manager.join_calls
    assert (False, True) in DummyPool.shutdown_calls
    assert (True, True) in DummyPool.shutdown_calls


def test_main_recycles_executor_by_recycle_every(tmp_path, monkeypatch):
    paths = []
    for i in range(3):
        nifti = tmp_path / f"vol_{i}.nii.gz"
        nifti.write_text("nifti")
        paths.append(str(nifti))

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": p, "Modality": "CT"} for p in paths]).to_csv(
        csv_path, index=False
    )
    config_path = tmp_path / "manifest.yaml"
    write_segmentation_manifest(
        config_path,
        {
            "backend": "totalsegmentator",
            "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
        },
    )

    monkeypatch.setattr(
        segment_module, "prefetch_totalsegmentator_models", lambda *a, **k: None
    )
    monkeypatch.setattr(segment_module, "tqdm", passthrough_tqdm)
    patch_strategy(
        monkeypatch,
        mode="process_pool",
        max_workers=2,
        max_in_flight=2,
        recycle_every=1,
    )

    class DummyFuture:
        def __init__(self, result):
            self._result = result

        def result(self, timeout=None):
            return self._result

        def cancel(self):
            return None

    class DummyPool:
        init_count = 0

        def __init__(self, max_workers=None, mp_context=None):
            self._processes = None
            DummyPool.init_count += 1

        def submit(self, fn, *args, **kwargs):
            idx = args[0]
            row = args[1]
            out_dir = Path(row["nifti_path"]).parent
            (out_dir / "liver.nii.gz").write_text("mask")
            return DummyFuture((idx, str(out_dir), None, None))

        def shutdown(self, wait=False, cancel_futures=True):
            return None

    monkeypatch.setattr(segment_module, "ProcessPoolExecutor", DummyPool)

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "segmented.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        manifest=str(config_path),
        num_workers=3,
        verbose=False,
        force=False,
        start_method="spawn",
        timeout_sec=10,
    )

    segment_module.main(args)
    assert DummyPool.init_count == 3


def test_main_subprocess_mode_currently_degrades_to_serial(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")
    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": str(nifti), "Modality": "CT"}]).to_csv(
        csv_path, index=False
    )
    config_path = tmp_path / "manifest.yaml"
    write_segmentation_manifest(
        config_path,
        {
            "backend": "totalsegmentator",
            "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
        },
    )

    monkeypatch.setattr(
        segment_module, "prefetch_totalsegmentator_models", lambda *a, **k: None
    )
    monkeypatch.setattr(segment_module, "tqdm", passthrough_tqdm)
    patch_strategy(
        monkeypatch,
        mode="subprocess_per_case",
        max_workers=1,
        max_in_flight=1,
    )

    def fail_if_pool_used(*args, **kwargs):
        raise AssertionError(
            "ProcessPoolExecutor should not be used in subprocess fallback mode"
        )

    monkeypatch.setattr(segment_module, "ProcessPoolExecutor", fail_if_pool_used)

    def fake_process_single_volume(
        idx, row, tasks_config, *, verbose, force, backend=None, **kwargs
    ):
        out_dir = Path(row["nifti_path"]).parent
        (out_dir / "liver.nii.gz").write_text("mask")
        return idx, str(out_dir), None, None

    monkeypatch.setattr(
        segment_module, "process_single_volume", fake_process_single_volume
    )
    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "segmented.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        manifest=str(config_path),
        num_workers=4,
        verbose=False,
        force=False,
        start_method="spawn",
        timeout_sec=10,
    )

    segment_module.main(args)
    out_df = pd.read_csv(args.csv_path_out)
    assert out_df.loc[0, "mask_liver"].endswith("liver.nii.gz")


@pytest.mark.parametrize(
    "existing_masks, force", [(False, False), (True, False), (True, True)]
)
def test_main_resume_skips_completed_rows(
    tmp_path, monkeypatch, caplog, existing_masks, force
):
    caplog.set_level(logging.INFO, logger=segment_module.__name__)
    if existing_masks:
        (tmp_path / "liver.nii.gz").write_text("mask")
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")
    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": str(nifti), "Modality": "CT"}]).to_csv(
        csv_path, index=False
    )
    config_path = tmp_path / "manifest.yaml"
    write_segmentation_manifest(
        config_path,
        {
            "backend": "totalsegmentator",
            "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
        },
    )

    prefetch_calls = {"count": 0}

    def fake_prefetch(*args, **kwargs):
        prefetch_calls["count"] += 1

    monkeypatch.setattr(
        segment_module, "prefetch_totalsegmentator_models", fake_prefetch
    )
    monkeypatch.setattr(segment_module, "tqdm", passthrough_tqdm)
    patch_strategy(monkeypatch, mode="serial", max_workers=1, max_in_flight=1)

    calls = {"count": 0}

    def fake_process_single_volume(
        idx, row, tasks_config, *, verbose, force, backend=None, **kwargs
    ):
        calls["count"] += 1
        out_dir = Path(row["nifti_path"]).parent
        (out_dir / "liver.nii.gz").write_text("mask")
        return idx, str(out_dir), None, None

    monkeypatch.setattr(
        segment_module, "process_single_volume", fake_process_single_volume
    )

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "segmented.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        manifest=str(config_path),
        num_workers=1,
        verbose=False,
        force=force,
        start_method="spawn",
        timeout_sec=10,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )
    segment_module.main(args)
    expected_resumed = int(existing_masks and not force)
    assert f"{expected_resumed} resumed" in caplog.text
    assert calls["count"] == 1

    calls["count"] = 0
    args.resume = True
    segment_module.main(args)
    assert calls["count"] == 0
    assert prefetch_calls["count"] == 1


def test_main_resume_uses_source_idx_after_deduplicating_inputs(tmp_path, monkeypatch):
    nifti_a = tmp_path / "a.nii.gz"
    nifti_b = tmp_path / "b.nii.gz"
    nifti_a.write_text("nifti")
    nifti_b.write_text("nifti")
    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame(
        [
            {"nifti_path": str(nifti_a), "Modality": "CT"},
            {"nifti_path": str(nifti_a), "Modality": "CT"},
            {"nifti_path": str(nifti_b), "Modality": "CT"},
        ]
    ).to_csv(csv_path, index=False)
    config_path = tmp_path / "manifest.yaml"
    write_segmentation_manifest(
        config_path,
        {
            "backend": "totalsegmentator",
            "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
        },
    )

    monkeypatch.setattr(
        segment_module, "prefetch_totalsegmentator_models", lambda *a, **k: None
    )
    monkeypatch.setattr(segment_module, "tqdm", passthrough_tqdm)
    patch_strategy(monkeypatch, mode="serial", max_workers=1, max_in_flight=1)

    processed_indices = []

    def fake_process_single_volume(
        idx, row, tasks_config, *, verbose, force, backend=None, **kwargs
    ):
        processed_indices.append(idx)
        out_dir = Path(row["nifti_path"]).parent
        (out_dir / "liver.nii.gz").write_text("mask")
        return idx, str(out_dir), None, None

    monkeypatch.setattr(
        segment_module, "process_single_volume", fake_process_single_volume
    )

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "segmented.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        manifest=str(config_path),
        num_workers=1,
        verbose=False,
        force=False,
        start_method="spawn",
        timeout_sec=10,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )
    segment_module.main(args)
    assert processed_indices == [0, 2]

    args.resume = True
    segment_module.main(args)
    assert processed_indices == [0, 2]


def test_main_resume_reprocesses_when_segmentation_config_changes(
    tmp_path, monkeypatch
):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")
    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": str(nifti), "Modality": "CT"}]).to_csv(
        csv_path, index=False
    )
    config_path = tmp_path / "manifest.yaml"
    write_segmentation_manifest(
        config_path,
        {
            "backend": "totalsegmentator",
            "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
        },
    )

    monkeypatch.setattr(
        segment_module, "prefetch_totalsegmentator_models", lambda *a, **k: None
    )
    monkeypatch.setattr(segment_module, "tqdm", passthrough_tqdm)
    patch_strategy(monkeypatch, mode="serial", max_workers=1, max_in_flight=1)

    calls = []

    def fake_process_single_volume(
        idx, row, tasks_config, *, verbose, force, backend=None, **kwargs
    ):
        output_name = tasks_config["tasks"][0]["output"]
        calls.append(output_name)
        out_dir = Path(row["nifti_path"]).parent
        (out_dir / output_name).write_text("mask")
        return idx, str(out_dir), None, None

    monkeypatch.setattr(
        segment_module, "process_single_volume", fake_process_single_volume
    )

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "segmented.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        manifest=str(config_path),
        num_workers=1,
        verbose=False,
        force=False,
        start_method="spawn",
        timeout_sec=10,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )
    segment_module.main(args)

    write_segmentation_manifest(
        config_path,
        {
            "backend": "totalsegmentator",
            "tasks": [{"key": "spleen", "task": "total", "output": "spleen.nii.gz"}],
        },
    )
    args.resume = True
    segment_module.main(args)

    assert calls == ["liver.nii.gz", "spleen.nii.gz"]

    manifest = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest["dataset_name"] = "unrelated change"
    config_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    segment_module.main(args)

    assert calls == ["liver.nii.gz", "spleen.nii.gz"]


def test_main_does_not_blank_existing_mask_or_warning_columns(tmp_path, monkeypatch):
    nifti_a = tmp_path / "a.nii.gz"
    nifti_b = tmp_path / "b.nii.gz"
    nifti_a.write_text("nifti")
    nifti_b.write_text("nifti")
    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame(
        [
            {
                "nifti_path": str(nifti_a),
                "Modality": "CT",
                "mask_liver": "preexisting-a",
                "warning_message": "warn-a",
            },
            {
                "nifti_path": str(nifti_b),
                "Modality": "CT",
                "mask_liver": "preexisting-b",
                "warning_message": "warn-b",
            },
        ]
    ).to_csv(csv_path, index=False)
    config_path = tmp_path / "manifest.yaml"
    write_segmentation_manifest(
        config_path,
        {
            "backend": "totalsegmentator",
            "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
        },
    )

    monkeypatch.setattr(
        segment_module, "prefetch_totalsegmentator_models", lambda *a, **k: None
    )
    monkeypatch.setattr(segment_module, "tqdm", passthrough_tqdm)
    patch_strategy(monkeypatch, mode="serial", max_workers=1, max_in_flight=1)

    def fake_process_single_volume(
        idx, row, tasks_config, *, verbose, force, backend=None, **kwargs
    ):
        if idx == 0:
            out_dir = Path(row["nifti_path"]).parent
            (out_dir / "liver.nii.gz").write_text("mask")
            return idx, str(out_dir), None, None
        return idx, None, "mock failure", None

    monkeypatch.setattr(
        segment_module, "process_single_volume", fake_process_single_volume
    )
    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "segmented.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        manifest=str(config_path),
        num_workers=1,
        verbose=False,
        force=False,
        start_method="spawn",
        timeout_sec=10,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )
    segment_module.main(args)

    out_df = pd.read_csv(args.csv_path_out)
    assert str(nifti_a.parent / "liver.nii.gz") == out_df.loc[0, "mask_liver"]
    assert out_df.loc[1, "mask_liver"] == "preexisting-b"
    assert out_df.loc[1, "warning_message"] == "warn-b"
