"""Voxel-level checks for sequential segmentation postprocessing."""

import copy
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import yaml

from imperandi.process import mask_operations as ops
from imperandi.process import segment


def save_mask(directory, name, data, affine=None):
    affine = np.eye(4) if affine is None else affine
    nib.save(
        nib.Nifti1Image(np.asarray(data, dtype=np.uint8), affine),
        directory / f"{name}.nii.gz",
    )


def read_mask(directory, name):
    return np.asanyarray(nib.load(directory / f"{name}.nii.gz").dataobj)


def run_operations(directory, operations, aliases=None):
    normalized = ops.validate_operations(
        {"operations": operations}, task_outputs={"a", "b", "c"}
    )
    ops.apply_operations(directory, normalized, aliases or {})
    return normalized


@pytest.mark.parametrize(
    "op,expected",
    [
        ("union", [0, 1, 1, 1]),
        ("or", [0, 1, 1, 1]),
        ("intersection", [0, 0, 0, 1]),
        ("and", [0, 0, 0, 1]),
        ("difference", [0, 1, 0, 0]),
        ("subtract", [0, 1, 0, 0]),
        ("xor", [0, 1, 1, 0]),
    ],
)
def test_logical_truth_tables(tmp_path, op, expected):
    save_mask(tmp_path, "a", np.array([0, 2, 0, 2]).reshape(4, 1, 1))
    save_mask(tmp_path, "b", np.array([0, 0, 3, 3]).reshape(4, 1, 1))
    run_operations(tmp_path, [{"op": op, "inputs": ["a", "b"], "output": "result"}])
    result = read_mask(tmp_path, "result")
    np.testing.assert_array_equal(result.ravel(), expected)
    assert result.dtype == np.uint8
    assert read_mask(tmp_path, "a").max() == 2  # sources are not rewritten


def test_difference_subtracts_all_remaining_masks_and_not_uses_image_domain(tmp_path):
    for name, values in {"a": [1, 1, 1], "b": [1, 0, 0], "c": [0, 0, 1]}.items():
        save_mask(tmp_path, name, np.array(values).reshape(3, 1, 1))
    run_operations(
        tmp_path,
        [
            {"op": "difference", "inputs": ["a", "b", "c"], "output": "remaining"},
            {"op": "not", "input": "remaining", "output": "inverse"},
        ],
    )
    np.testing.assert_array_equal(read_mask(tmp_path, "remaining").ravel(), [0, 1, 0])
    np.testing.assert_array_equal(read_mask(tmp_path, "inverse").ravel(), [1, 0, 1])


@pytest.mark.parametrize("op", ["dilate", "dilation"])
def test_dilation_respects_anisotropic_spacing_and_affine(tmp_path, op):
    data = np.zeros((9, 9, 9))
    data[4, 4, 4] = 1
    affine = np.diag([1.0, 2.0, 4.0, 1.0])
    affine[:3, 3] = [12.0, -7.0, 5.0]
    save_mask(tmp_path, "a", data, affine)
    run_operations(
        tmp_path, [{"op": op, "input": "a", "output": "result", "radius_mm": 2}]
    )
    expected = np.zeros_like(data)
    expected[2:7, 4, 4] = 1
    expected[4, [3, 5], 4] = 1
    np.testing.assert_array_equal(read_mask(tmp_path, "result"), expected)
    np.testing.assert_allclose(nib.load(tmp_path / "result.nii.gz").affine, affine)


@pytest.mark.parametrize("radius", [0, 0.25])
def test_subvoxel_radius_is_identity(tmp_path, radius):
    data = np.ones((3, 3, 3))
    save_mask(tmp_path, "a", data)
    run_operations(
        tmp_path,
        [{"op": "erode", "input": "a", "output": "result", "radius_mm": radius}],
    )
    np.testing.assert_array_equal(read_mask(tmp_path, "result"), data)


@pytest.mark.parametrize("op", ["erode", "erosion"])
def test_erosion_and_iterations(tmp_path, op):
    data = np.zeros((11, 11, 11))
    data[2:9, 2:9, 2:9] = 1
    save_mask(tmp_path, "a", data)
    run_operations(
        tmp_path, [{"op": op, "input": "a", "output": "result", "iterations": 2}]
    )
    expected = np.zeros_like(data)
    expected[4:7, 4:7, 4:7] = 1
    np.testing.assert_array_equal(read_mask(tmp_path, "result"), expected)


@pytest.mark.parametrize("op", ["open", "opening"])
def test_opening_removes_isolated_voxel(tmp_path, op):
    data = np.zeros((9, 9, 9))
    data[1, 1, 1] = 1
    data[4:7, 4:7, 4:7] = 1
    save_mask(tmp_path, "a", data)
    run_operations(tmp_path, [{"op": op, "input": "a", "output": "result"}])
    result = read_mask(tmp_path, "result")
    assert result[1, 1, 1] == 0
    assert result[5, 5, 5] == 1


@pytest.mark.parametrize("op", ["close", "closing", "fill_holes"])
def test_closing_and_filling_repair_internal_hole(tmp_path, op):
    data = np.zeros((9, 9, 9))
    data[2:7, 2:7, 2:7] = 1
    data[4, 4, 4] = 0
    save_mask(tmp_path, "a", data)
    run_operations(tmp_path, [{"op": op, "input": "a", "output": "result"}])
    data[4, 4, 4] = 1
    np.testing.assert_array_equal(read_mask(tmp_path, "result"), data)


@pytest.mark.parametrize("op", ["largest_cc", "largest_component"])
def test_largest_component_handles_empty_and_disconnected_masks(tmp_path, op):
    data = np.zeros((9, 9, 9))
    save_mask(tmp_path, "b", data)
    data[2:5, 2:5, 2:5] = 1
    data[7, 7, 7] = 1
    save_mask(tmp_path, "a", data)
    run_operations(
        tmp_path,
        [
            {"op": op, "input": "a", "output": "result"},
            {"op": op, "input": "b", "output": "empty"},
        ],
    )
    data[7, 7, 7] = 0
    np.testing.assert_array_equal(read_mask(tmp_path, "result"), data)
    assert not read_mask(tmp_path, "empty").any()


def test_sequence_cleans_individual_mask_then_merges_then_cleans_result(tmp_path):
    a = np.zeros((9, 9, 9))
    a[2:5, 2:5, 2:5] = 1
    a[7, 7, 7] = 1
    b = np.zeros_like(a)
    b[4:7, 2:5, 2:5] = 1
    save_mask(tmp_path, "a", a)
    save_mask(tmp_path, "backend_b", b)
    run_operations(
        tmp_path,
        [
            {"op": "largest_cc", "input": "mask_a.nii.gz", "output": "clean"},
            {"op": "union", "inputs": ["clean", "mask_b"], "output": "merged"},
            {"op": "erode", "input": "merged", "output": "merged", "radius_mm": 1},
            {"op": "intersection", "inputs": ["a", "merged"], "output": "clipped"},
        ],
        {"b": "backend_b"},
    )
    expected = np.zeros_like(a)
    expected[3:6, 3, 3] = 1
    np.testing.assert_array_equal(read_mask(tmp_path, "merged"), expected)
    expected[5, 3, 3] = 0
    np.testing.assert_array_equal(read_mask(tmp_path, "clipped"), expected)
    np.testing.assert_array_equal(read_mask(tmp_path, "a"), a)
    np.testing.assert_array_equal(read_mask(tmp_path, "backend_b"), b)


@pytest.mark.parametrize("kind", ["missing", "unreadable", "shape", "affine", "4d"])
def test_failed_sequence_does_not_write_any_outputs(tmp_path, kind):
    a = np.ones((3, 3, 3))
    save_mask(tmp_path, "a", a)
    save_mask(tmp_path, "first", np.zeros_like(a))
    original_bytes = (tmp_path / "first.nii.gz").read_bytes()
    if kind == "unreadable":
        (tmp_path / "b.nii.gz").write_text("not a NIfTI")
    elif kind == "shape":
        save_mask(tmp_path, "b", np.ones((4, 3, 3)))
    elif kind == "affine":
        save_mask(tmp_path, "b", a, np.diag([2.0, 1.0, 1.0, 1.0]))
    elif kind == "4d":
        save_mask(tmp_path, "b", np.ones((3, 3, 3, 2)))
    error = {
        "missing": FileNotFoundError,
        "unreadable": nib.filebasedimages.ImageFileError,
    }.get(kind, ops.MaskGeometryError)
    with pytest.raises(error):
        run_operations(
            tmp_path,
            [
                {"op": "not", "input": "a", "output": "first"},
                {"op": "union", "inputs": ["first", "b"], "output": "merged"},
            ],
        )
    assert (tmp_path / "first.nii.gz").read_bytes() == original_bytes
    assert not (tmp_path / "merged.nii.gz").exists()
    assert not (tmp_path / ".imperandi-postprocess.json").exists()


@pytest.mark.parametrize(
    "step,match",
    [
        ({"op": "unknown", "input": "a", "output": "x"}, "unsupported operation"),
        ({"op": "not", "inputs": ["a", "b"], "output": "x"}, "exactly one"),
        ({"op": "difference", "input": "a", "output": "x"}, "at least two"),
        ({"op": "not", "inputs": "a", "output": "x"}, "non-empty list"),
        ({"op": "not", "input": "a"}, "output"),
        ({"op": "not", "input": "a", "output": "../x"}, "segmentation directory"),
        ({"op": "dilate", "input": "a", "output": "x", "radius_mm": -1}, "radius_mm"),
        (
            {"op": "dilate", "input": "a", "output": "x", "radius_mm": float("nan")},
            "radius_mm",
        ),
        ({"op": "dilate", "input": "a", "output": "x", "radius_mm": True}, "radius_mm"),
        ({"op": "erode", "input": "a", "output": "x", "iterations": 0}, "iterations"),
        ({"op": "erode", "input": "a", "output": "x", "iterations": 1.5}, "iterations"),
        (
            {"op": "not", "input": "a", "output": "x", "radius_mm": 1},
            "unsupported settings",
        ),
        (
            {"op": "largest_cc", "input": "a", "output": "x", "connectivity": 4},
            "connectivity",
        ),
        (
            {"op": "not", "input": "a", "inputs": ["a"], "output": "x"},
            "either input or inputs",
        ),
        (None, "mapping"),
    ],
)
def test_manifest_rejects_invalid_operations(step, match):
    with pytest.raises(ValueError, match=match):
        segment.validate_segmentation_config(
            {
                "modalities": {
                    "CT": {
                        "tasks": [{"task": "total", "output": "a"}],
                        "postprocess": {"operations": [step]},
                    }
                }
            }
        )


def test_validation_rejects_forward_references_and_unsupported_settings():
    with pytest.raises(ValueError, match="before it is produced"):
        ops.validate_operations(
            {
                "operations": [
                    {"op": "not", "input": "later", "output": "first"},
                    {"op": "not", "input": "a", "output": "later"},
                ]
            },
            task_outputs={"a"},
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        ops.validate_operations(
            {"operations": [], "merge_keys": ["a"]}, task_outputs={"a"}
        )
    with pytest.raises(ValueError, match="must be a list"):
        ops.validate_operations({"operations": {}}, task_outputs={"a"})


class UnexpectedBackend:
    def run(self, **kwargs):
        raise AssertionError("Existing task outputs should be reused")


def test_in_place_operation_runs_once_and_tracks_aliases(tmp_path):
    save_mask(tmp_path, "a", np.zeros((3, 3, 3)))
    config = {
        "tasks": [{"task": "total", "output": "a"}],
        "postprocess": {"operations": [{"op": "not", "input": "a", "output": "a"}]},
    }
    assert not segment._has_existing_task_outputs(tmp_path, config)
    resolved = {}
    for _ in range(2):
        segment.segment_volume(
            tmp_path / "vol.nii.gz",
            tmp_path,
            config,
            backend=UnexpectedBackend(),
            resolved_output_to_fetch=resolved,
        )
        assert read_mask(tmp_path, "a").all()
        assert segment._has_existing_task_outputs(tmp_path, config)
    assert resolved == {"a": "a"}


def test_alias_override_updates_backend_file_and_resume(tmp_path):
    save_mask(tmp_path, "backend_a", np.zeros((3, 3, 3)))
    config = {
        "tasks": [{"task": "total", "output": "a", "fetch_output": "backend_a"}],
        "postprocess": {"operations": [{"op": "not", "input": "a", "output": "a"}]},
    }
    resolved = {}
    segment.segment_volume(
        tmp_path / "vol.nii.gz",
        tmp_path,
        config,
        backend=UnexpectedBackend(),
        resolved_output_to_fetch=resolved,
    )
    assert resolved == {"a": "backend_a"}
    assert read_mask(tmp_path, "backend_a").all()
    assert not (tmp_path / "a.nii.gz").exists()
    assert segment._has_existing_task_outputs(tmp_path, config)
    segment.segment_volume(
        tmp_path / "vol.nii.gz", tmp_path, config, backend=UnexpectedBackend()
    )
    assert read_mask(tmp_path, "backend_a").all()


def test_resume_requires_all_outputs_and_matching_operations_and_sources(tmp_path):
    save_mask(tmp_path, "a", np.zeros((3, 3, 3)))
    operations = run_operations(
        tmp_path,
        [
            {"op": "not", "input": "a", "output": "first"},
            {"op": "not", "input": "first", "output": "last"},
        ],
    )
    assert ops.operations_are_current(tmp_path, operations, {})
    changed = copy.deepcopy(operations)
    changed[0]["op"] = "fill_holes"
    assert not ops.operations_are_current(tmp_path, changed, {})
    (tmp_path / "first.nii.gz").unlink()
    assert not ops.operations_are_current(tmp_path, operations, {})
    ops.apply_operations(tmp_path, operations, {})
    assert ops.operations_are_current(tmp_path, operations, {})
    save_mask(tmp_path, "a", np.ones((3, 3, 3)))
    assert not ops.operations_are_current(tmp_path, operations, {})


def test_new_task_run_and_force_recompute_operations(tmp_path):
    class Backend:
        calls = 0

        def run(self, **kwargs):
            self.calls += 1
            save_mask(tmp_path, "a", np.zeros((3, 3, 3)))

    backend = Backend()
    config = {
        "tasks": [{"task": "total", "output": "a"}],
        "postprocess": {"operations": [{"op": "not", "input": "a", "output": "a"}]},
    }
    for force in [False, True]:
        segment.segment_volume(
            tmp_path / "vol.nii.gz", tmp_path, config, backend=backend, force=force
        )
        assert read_mask(tmp_path, "a").all()
    assert backend.calls == 2


@pytest.mark.parametrize("policy", ["fail", "warn_only"])
def test_geometry_failure_policy_does_not_report_old_outputs(tmp_path, policy):
    save_mask(tmp_path, "a", np.ones((3, 3, 3)))
    save_mask(tmp_path, "b", np.ones((4, 3, 3)))
    save_mask(tmp_path, "result", np.ones((3, 3, 3)))
    config = {
        "tasks": [{"task": "total", "outputs": ["a", "b"]}],
        "postprocess": {
            "on_failure": policy,
            "operations": [{"op": "union", "inputs": ["a", "b"], "output": "result"}],
        },
    }
    resolved = {}
    if policy == "fail":
        with pytest.raises(ops.MaskGeometryError):
            segment.segment_volume(
                tmp_path / "vol.nii.gz", tmp_path, config, backend=UnexpectedBackend()
            )
    else:
        warnings = segment.segment_volume(
            tmp_path / "vol.nii.gz",
            tmp_path,
            config,
            backend=UnexpectedBackend(),
            resolved_output_to_fetch=resolved,
        )
        assert "shape or affine mismatch" in warnings[0]
        assert resolved == {"a": "a", "b": "b"}


def test_main_reads_manifest_and_exports_every_operation_output(tmp_path, monkeypatch):
    from argparse import Namespace
    from imperandi.utils.multiprocessing import MPStrategy
    from imperandi.utils import multiprocessing as mp_utils

    nifti = tmp_path / "vol.nii.gz"
    save_mask(tmp_path, "vol", np.ones((3, 3, 3)))
    save_mask(tmp_path, "backend_a", np.ones((3, 3, 3)))
    save_mask(tmp_path, "b", np.zeros((3, 3, 3)))
    config = {
        "modalities": {
            "CT": {
                "tasks": [
                    {"task": "total", "output": "a", "fetch_output": "backend_a"}
                ],
                "postprocess": {
                    "operations": [
                        {"op": "not", "input": "a", "output": "a"},
                        {"op": "union", "inputs": ["a", "b"], "output": "merged"},
                        {"op": "fill_holes", "input": "merged", "output": "final"},
                    ]
                },
            }
        }
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump({"segmentation": config}))
    csv_path = tmp_path / "input.csv"
    pd.DataFrame([{"nifti_path": str(nifti), "Modality": "CT"}]).to_csv(
        csv_path, index=False
    )
    monkeypatch.setattr(
        segment, "prefetch_totalsegmentator_models", lambda *a, **kw: None
    )
    monkeypatch.setattr(segment, "TotalSegmentatorBackend", UnexpectedBackend)
    monkeypatch.setattr(
        mp_utils,
        "decide_multiprocessing_strategy",
        lambda **kw: MPStrategy(
            mode="serial",
            start_method="spawn",
            max_workers=1,
            max_in_flight=1,
            recycle_every=0,
            env={},
            use_gpu=False,
            gpu_count=0,
            hard_timeout_supported=False,
            reasons={},
        ),
    )
    monkeypatch.setattr(mp_utils, "apply_strategy_env", lambda *a, **kw: None)
    args = Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "out.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        manifest=str(manifest_path),
        num_workers=1,
        verbose=False,
        force=False,
        start_method="spawn",
        timeout_sec=10,
    )
    segment.main(args)
    row = pd.read_csv(args.csv_path_out).iloc[0]
    for name, filename in {
        "a": "backend_a",
        "b": "b",
        "merged": "merged",
        "final": "final",
    }.items():
        assert Path(row[f"mask_{name}"]).name == f"{filename}.nii.gz"
        assert not read_mask(tmp_path, filename).any()
    assert not (tmp_path / "a.nii.gz").exists()


def test_empty_operations_disable_postprocessing():
    config = segment.validate_segmentation_config(
        {
            "modalities": {
                "CT": {
                    "tasks": [{"task": "total", "output": "a"}],
                    "postprocess": {"operations": []},
                }
            }
        }
    )
    assert segment.build_segmentation_output_maps(config) == (
        {"a": "mask_a"},
        {"a": "a"},
        set(),
    )


def test_direct_worker_normalizes_output_names_and_infers_backend_masks(tmp_path):
    class Backend:
        def run(self, **kwargs):
            save_mask(tmp_path, "a", np.zeros((3, 3, 3)))

    config = {
        "tasks": [{"task": "total"}],
        "postprocess": {
            "operations": [
                {"op": "not", "input": "mask_a.nii.gz", "output": "mask_result.nii.gz"}
            ]
        },
    }
    resolved = {}
    segment.segment_volume(
        tmp_path / "vol.nii.gz",
        tmp_path,
        config,
        backend=Backend(),
        resolved_output_to_fetch=resolved,
    )
    assert resolved == {"a": "a", "result": "result"}
    assert read_mask(tmp_path, "result").all()


def test_missing_input_always_fails_even_with_geometry_warn_only(tmp_path):
    save_mask(tmp_path, "a", np.ones((3, 3, 3, 2)))
    config = {
        "tasks": [{"task": "total", "output": "a"}],
        "postprocess": {
            "on_failure": "warn_only",
            "operations": [
                {"op": "union", "inputs": ["a", "missing"], "output": "result"}
            ],
        },
    }
    with pytest.raises(FileNotFoundError, match="missing.nii.gz"):
        segment.segment_volume(
            tmp_path / "vol.nii.gz", tmp_path, config, backend=UnexpectedBackend()
        )


@pytest.mark.parametrize("manifest", ["generic", "blueprint_manifest_example"])
def test_builtin_segmentation_manifests_validate(manifest):
    config = segment.load_segmentation_config(
        manifest, base_path=Path(segment.__file__).parent
    )
    assert set(config["modalities"]) == {"CT", "MR"}


def test_largest_component_connectivity_changes_diagonal_connections(tmp_path):
    data = np.zeros((9, 9, 9))
    data[1, 1, 1:3] = 1
    data[4, 4, 4] = data[5, 5, 5] = data[6, 6, 6] = 1
    save_mask(tmp_path, "a", data)
    run_operations(
        tmp_path,
        [
            {"op": "largest_cc", "input": "a", "output": "faces", "connectivity": 1},
            {"op": "largest_cc", "input": "a", "output": "corners", "connectivity": 3},
        ],
    )
    assert read_mask(tmp_path, "faces").sum() == 2
    assert read_mask(tmp_path, "corners").sum() == 3


@pytest.mark.parametrize(
    "postprocess",
    [
        {},
        {"merge_keys": ["a", "b"], "output": "merged"},
        {"merge_keys": ["a"], "close": False, "largest_cc": True},
        {"operations": [], "merge_keys": ["a"]},
        {"operations": [], "output": "merged"},
        {"operations": [], "radius_mm": 5},
        {"operations": [], "close": True},
        {"operations": [], "fill_holes": True},
        {"operations": [], "largest_cc": True},
    ],
)
def test_invalid_postprocess_schema_is_rejected_by_manifest_and_worker(
    tmp_path, postprocess
):
    config = {"tasks": [{"task": "total", "output": "a"}], "postprocess": postprocess}
    with pytest.raises(ValueError, match="postprocess.operations"):
        segment.validate_segmentation_config({"modalities": {"CT": config}})
    with pytest.raises(ValueError, match="postprocess.operations"):
        segment.segment_volume(
            tmp_path / "vol.nii.gz", tmp_path, config, backend=UnexpectedBackend()
        )


@pytest.mark.parametrize("op", ["union", "intersection", "xor"])
def test_single_input_logical_operations_copy_mask(tmp_path, op):
    data = np.zeros((5, 5, 5))
    data[1:4, 1:4, 1:4] = 1
    save_mask(tmp_path, "a", data)
    run_operations(tmp_path, [{"op": op, "input": "a", "output": "copy"}])
    np.testing.assert_array_equal(read_mask(tmp_path, "copy"), data)


@pytest.mark.parametrize("radius", [-1, True, float("inf"), "large"])
def test_invalid_voxel_radius_is_rejected(radius):
    with pytest.raises(ValueError, match="radius_vox"):
        ops.validate_operations(
            {
                "operations": [
                    {
                        "op": "close",
                        "input": "a",
                        "output": "a",
                        "radius_vox": radius,
                    }
                ]
            },
            task_outputs={"a"},
        )


def test_morphology_radius_units_are_mutually_exclusive():
    with pytest.raises(ValueError, match="either radius_mm or radius_vox"):
        ops.validate_operations(
            {
                "operations": [
                    {
                        "op": "dilate",
                        "input": "a",
                        "output": "a",
                        "radius_mm": 2,
                        "radius_vox": 2,
                    }
                ]
            },
            task_outputs={"a"},
        )


@pytest.mark.parametrize("close", [False, True])
@pytest.mark.parametrize("fill_holes", [False, True])
@pytest.mark.parametrize("largest_cc", [False, True])
@pytest.mark.parametrize("merged_key", ["merged", "liver"])
@pytest.mark.parametrize("zooms", [(1, 1, 1), (1, 2, 3)])
def test_sequence_merges_cleans_and_clips_source_masks(
    tmp_path, close, fill_holes, largest_cc, merged_key, zooms
):
    from scipy.ndimage import binary_closing, binary_fill_holes
    from skimage.measure import label, regionprops
    from skimage.morphology import ball

    liver = np.zeros((25, 25, 25), dtype=bool)
    liver[6:17, 6:17, 6:17] = True
    liver[10:13, 10:13, 10:13] = False
    tumor = np.zeros_like(liver)
    tumor[14:20, 8:15, 8:15] = True
    tumor[2:4, 2:4, 2:4] = True
    affine = np.diag([*zooms, 1.0])
    save_mask(tmp_path, "liver", liver, affine)
    save_mask(tmp_path, "liver_lesions", tumor, affine)

    # Independent reference using a voxel-space sphere, 26-connected components
    # and intersections to update source masks.
    radius_vox = max(max(1, round(2.5 / z)) for z in zooms)
    expected = liver | tumor
    if close:
        expected = binary_closing(expected, structure=ball(radius_vox))
    if fill_holes:
        expected = binary_fill_holes(expected)
    if largest_cc:
        labels, count = label(expected, return_num=True)
        if count > 1:
            largest = max(regionprops(labels), key=lambda r: r.area)
            expected = labels == largest.label

    steps = [{"op": "union", "inputs": ["liver", "liver_tumor"], "output": merged_key}]
    if close:
        steps.append(
            {
                "op": "close",
                "input": merged_key,
                "output": merged_key,
                "radius_vox": radius_vox,
            }
        )
    if fill_holes:
        steps.append({"op": "fill_holes", "input": merged_key, "output": merged_key})
    if largest_cc:
        steps.append({"op": "largest_cc", "input": merged_key, "output": merged_key})
    for key in ["liver", "liver_tumor"]:
        if key != merged_key:
            steps.append(
                {"op": "intersection", "inputs": [key, merged_key], "output": key}
            )
    config = {
        "tasks": [
            {
                "task": "total",
                "outputs": ["liver", "liver_tumor"],
                "fetch_outputs": ["liver", "liver_lesions"],
            }
        ],
        "postprocess": {"on_failure": "warn_only", "operations": steps},
    }
    resolved = {}
    segment.segment_volume(
        tmp_path / "vol.nii.gz",
        tmp_path,
        config,
        backend=UnexpectedBackend(),
        resolved_output_to_fetch=resolved,
    )
    np.testing.assert_array_equal(read_mask(tmp_path, merged_key), expected)
    np.testing.assert_array_equal(
        read_mask(tmp_path, "liver"),
        expected if merged_key == "liver" else liver & expected,
    )
    np.testing.assert_array_equal(
        read_mask(tmp_path, "liver_lesions"), tumor & expected
    )
    assert resolved["liver_tumor"] == "liver_lesions"
    assert not (tmp_path / "liver_tumor.nii.gz").exists()


def test_fallback_and_operandi_manifests_use_explicit_cleanup_sequences():
    fallback = segment.validate_segmentation_config(
        segment._default_segmentation_config()
    )
    path = (
        Path(__file__).resolve().parents[2] / "dataset_configs/manifests/operandi.yaml"
    )
    operandi = segment.load_segmentation_config(str(path), base_path=path.parent)
    for config in [fallback, operandi]:
        for modality in ["CT", "MR"]:
            postprocess = config["modalities"][modality]["postprocess"]
            operations = postprocess["operations"]
            assert [step["op"] for step in operations[:4]] == [
                "union",
                "close",
                "fill_holes",
                "largest_cc",
            ]
            assert all(step["op"] == "intersection" for step in operations[4:])
            assert operations[-1]["output"] == "liver_tumor"
