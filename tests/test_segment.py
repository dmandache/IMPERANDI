import sys
from pathlib import Path
import types

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import argparse
import pandas as pd
import pytest

from imperandi.process import segment as segment_module
from imperandi.utils.multiprocessing import MPStrategy
import imperandi.utils.multiprocessing as mp_utils


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


def test_load_segmentation_config_default():
    cfg = segment_module.load_segmentation_config(
        None, base_path=Path(__file__).resolve().parents[1] / "src" / "imperandi"
    )
    assert "tasks" in cfg
    assert cfg["backend"] == "totalsegmentator"


def test_load_segmentation_config_missing(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError):
        segment_module.load_segmentation_config(
            str(missing),
            base_path=Path(__file__).resolve().parents[1] / "src" / "imperandi",
        )


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


def test_resolve_merge_outputs_requires_merge_keys():
    postprocess = {"merge_outputs": ["a.nii.gz"]}
    tasks = [{"key": "a", "output": "a.nii.gz"}]
    with pytest.raises(ValueError, match="merge_keys is required"):
        segment_module.resolve_merge_outputs(postprocess, tasks)


def test_resolve_merge_outputs_raises_when_merge_key_missing():
    postprocess = {"merge_keys": ["missing_key"]}
    tasks = [{"output": "a.nii.gz"}]
    with pytest.raises(ValueError, match="unknown mask column"):
        segment_module.resolve_merge_outputs(postprocess, tasks)


def test_resolve_merge_outputs_supports_bare_and_mask_column_keys():
    postprocess = {"merge_keys": ["a", "mask_b"]}
    tasks = [{"output": "a.nii.gz"}, {"output": "b.nii.gz"}]
    assert segment_module.resolve_merge_outputs(postprocess, tasks) == [
        "a",
        "b",
    ]


def test_infer_task_fetch_outputs_supports_aliasing_backend_filename():
    task = {
        "task": "liver_lesions",
        "output": "liver_tumor.nii.gz",
        "fetch_output": "liver_lesion.nii.gz",
    }

    assert segment_module.infer_task_fetch_outputs(task) == {
        "liver_tumor": "liver_lesion"
    }


def test_segment_volume_calls_postprocess(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    tasks_config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "a", "task": "task_a", "output": "a.nii.gz", "extra": {}},
            {"key": "b", "task": "task_b", "output": "b.nii.gz", "extra": {}},
        ],
        "postprocess": {
            "merge_keys": ["a", "b"],
            "output": "merged.nii.gz",
            "radius_mm": 2.0,
            "largest_cc": True,
            "fill_holes": True,
            "close": True,
        },
    }

    backend = DummyBackend({"task_a": "a.nii.gz", "task_b": "b.nii.gz"})

    calls = {}

    def fake_clean(dir_path, mask_files, *, output_name, **kwargs):
        calls["dir_path"] = dir_path
        calls["mask_files"] = mask_files
        calls["output_name"] = output_name
        return True

    monkeypatch.setattr(segment_module, "clean_and_merge_masks", fake_clean)

    segment_module.segment_volume(
        nifti,
        tmp_path,
        tasks_config,
        verbose=False,
        force=True,
        backend=backend,
    )

    assert calls["output_name"] == "merged.nii.gz"
    assert set(calls["mask_files"]) == {"a.nii.gz", "b.nii.gz"}


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


def test_segment_volume_uses_fetch_output_alias_for_expected_and_merge_paths(
    tmp_path, monkeypatch
):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    tasks_config = {
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
            "merge_keys": ["liver", "liver_tumor"],
            "output": "merged.nii.gz",
        },
    }

    backend = DummyBackend(
        {"total": "liver.nii.gz", "liver_lesions": "liver_lesion.nii.gz"}
    )

    calls = {}

    def fake_clean(dir_path, mask_files, *, output_name, **kwargs):
        calls["dir_path"] = dir_path
        calls["mask_files"] = mask_files
        calls["output_name"] = output_name
        (Path(dir_path) / output_name).write_text("merged")
        return True

    monkeypatch.setattr(segment_module, "clean_and_merge_masks", fake_clean)

    segment_module.segment_volume(
        nifti,
        tmp_path,
        tasks_config,
        verbose=False,
        force=True,
        backend=backend,
    )

    assert calls["output_name"] == "merged.nii.gz"
    assert set(calls["mask_files"]) == {"liver.nii.gz", "liver_lesion.nii.gz"}


def test_segment_volume_skips_postprocess_when_outputs_already_checkpointed(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")
    (tmp_path / "a.nii.gz").write_text("mask")
    (tmp_path / "b.nii.gz").write_text("mask")
    (tmp_path / "merged.nii.gz").write_text("merged")

    tasks_config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "a", "task": "task_a", "output": "a.nii.gz", "extra": {}},
            {"key": "b", "task": "task_b", "output": "b.nii.gz", "extra": {}},
        ],
        "postprocess": {
            "merge_keys": ["a", "b"],
            "output": "merged.nii.gz",
        },
    }

    backend = DummyBackend({"task_a": "a.nii.gz", "task_b": "b.nii.gz"})
    merge_calls = {"count": 0}

    def fake_clean(*args, **kwargs):
        merge_calls["count"] += 1
        return True

    monkeypatch.setattr(segment_module, "clean_and_merge_masks", fake_clean)

    warnings = segment_module.segment_volume(
        nifti,
        tmp_path,
        tasks_config,
        verbose=False,
        force=False,
        backend=backend,
    )

    assert merge_calls["count"] == 0
    assert warnings == []


def test_segment_volume_runs_postprocess_when_any_task_ran(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")
    (tmp_path / "a.nii.gz").write_text("mask")
    (tmp_path / "merged.nii.gz").write_text("merged")

    tasks_config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "a", "task": "task_a", "output": "a.nii.gz", "extra": {}},
            {"key": "b", "task": "task_b", "output": "b.nii.gz", "extra": {}},
        ],
        "postprocess": {
            "merge_keys": ["a", "b"],
            "output": "merged.nii.gz",
        },
    }

    backend = DummyBackend({"task_a": "a.nii.gz", "task_b": "b.nii.gz"})
    merge_calls = {"count": 0}

    def fake_clean(*args, **kwargs):
        merge_calls["count"] += 1
        return True

    monkeypatch.setattr(segment_module, "clean_and_merge_masks", fake_clean)

    warnings = segment_module.segment_volume(
        nifti,
        tmp_path,
        tasks_config,
        verbose=False,
        force=False,
        backend=backend,
    )

    assert merge_calls["count"] == 1
    assert any("overwrite existing file" in message for message in warnings)


def test_segment_volume_warn_only_when_merge_missing(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    tasks_config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "a", "task": "task_a", "output": "a.nii.gz", "extra": {}},
        ],
        "postprocess": {
            "merge_keys": ["a"],
            "output": "merged.nii.gz",
            "on_failure": "warn_only",
        },
    }

    backend = DummyBackend({"task_a": "a.nii.gz"})
    monkeypatch.setattr(segment_module, "clean_and_merge_masks", lambda *a, **k: False)

    warnings = segment_module.segment_volume(
        nifti,
        tmp_path,
        tasks_config,
        verbose=False,
        force=True,
        backend=backend,
    )

    assert warnings
    assert "merged.nii.gz" in warnings[0]


def test_segment_volume_fail_policy_when_merge_missing(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    tasks_config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "a", "task": "task_a", "output": "a.nii.gz", "extra": {}},
        ],
        "postprocess": {
            "merge_keys": ["a"],
            "output": "merged.nii.gz",
            "on_failure": "fail",
        },
    }

    backend = DummyBackend({"task_a": "a.nii.gz"})
    monkeypatch.setattr(segment_module, "clean_and_merge_masks", lambda *a, **k: False)

    with pytest.raises(RuntimeError, match="merged.nii.gz"):
        segment_module.segment_volume(
            nifti,
            tmp_path,
            tasks_config,
            verbose=False,
            force=True,
            backend=backend,
        )


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


def test_main_writes_mask_columns(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    csv_path = tmp_path / "nifti_index.csv"
    df = pd.DataFrame([{"nifti_path": str(nifti)}])
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
        "postprocess": {"merge_keys": ["liver", "vessels"], "output": "merged.nii.gz"},
    }

    config_path = tmp_path / "tasks.json"
    config_path.write_text(json.dumps({"segmentation": config}))

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
    df = pd.DataFrame([{"nifti_path": str(nifti)}])
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
        "postprocess": {"merge_keys": ["liver", "liver_tumor"], "output": "merged.nii.gz"},
    }

    config_path = tmp_path / "tasks.json"
    config_path.write_text(json.dumps({"segmentation": config}))

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
    pd.DataFrame([{"nifti_path": str(nifti)}]).to_csv(csv_path, index=False)

    config_path = tmp_path / "tasks.json"
    config_path.write_text(
        json.dumps(
            {
                "backend": "totalsegmentator",
                "tasks": [{"task": "task_a"}],
            }
        )
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
    pd.DataFrame([{"nifti_path": str(nifti)}]).to_csv(csv_path, index=False)

    config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "liver", "task": "total", "output": "liver.nii.gz", "extra": {}},
        ],
        "postprocess": {"merge_keys": ["liver"], "output": "merged.nii.gz"},
    }

    config_path = tmp_path / "tasks.json"
    config_path.write_text(json.dumps({"segmentation": config}))

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
    assert "missing merged mask" in out_df.loc[0, "warning_message"]
    assert pd.isna(out_df.loc[0, "mask_merged"])


def test_main_single_worker_avoids_process_pool(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": str(nifti)}]).to_csv(csv_path, index=False)

    config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "liver", "task": "total", "output": "liver.nii.gz", "extra": {}},
        ],
    }
    config_path = tmp_path / "tasks.json"
    config_path.write_text(json.dumps({"segmentation": config}))

    monkeypatch.setattr(
        segment_module, "prefetch_totalsegmentator_models", lambda *a, **k: None
    )
    monkeypatch.setattr(segment_module, "tqdm", passthrough_tqdm)
    patch_strategy(monkeypatch, mode="serial", max_workers=1, max_in_flight=1)

    def fail_if_pool_used(*args, **kwargs):
        raise AssertionError("ProcessPoolExecutor should not be used in single-worker mode")

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
    pd.DataFrame([{"nifti_path": str(nifti)}]).to_csv(csv_path, index=False)
    config_path = tmp_path / "tasks.json"
    config_path.write_text(
        json.dumps(
            {
                "backend": "totalsegmentator",
                "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
            }
        )
    )

    monkeypatch.setattr(
        segment_module, "prefetch_totalsegmentator_models", lambda *a, **k: None
    )
    monkeypatch.setattr(segment_module, "tqdm", passthrough_tqdm)
    patch_strategy(monkeypatch, mode="process_pool", max_workers=1, max_in_flight=1)

    def fail_if_pool_used(*args, **kwargs):
        raise AssertionError("ProcessPoolExecutor should not be used when strategy max_workers=1")

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
    pd.DataFrame([{"nifti_path": str(nifti)}]).to_csv(csv_path, index=False)
    config_path = tmp_path / "tasks.json"
    config_path.write_text(
        json.dumps(
            {
                "backend": "totalsegmentator",
                "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
            }
        )
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
    pd.DataFrame([{"nifti_path": str(nifti)}]).to_csv(csv_path, index=False)
    config_path = tmp_path / "tasks.json"
    config_path.write_text(
        json.dumps(
            {
                "backend": "totalsegmentator",
                "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
            }
        )
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
    pd.DataFrame([{"nifti_path": p} for p in paths]).to_csv(csv_path, index=False)
    config_path = tmp_path / "tasks.json"
    config_path.write_text(
        json.dumps(
            {
                "backend": "totalsegmentator",
                "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
            }
        )
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
    pd.DataFrame([{"nifti_path": str(nifti)}]).to_csv(csv_path, index=False)
    config_path = tmp_path / "tasks.json"
    config_path.write_text(
        json.dumps(
            {
                "backend": "totalsegmentator",
                "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
            }
        )
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
    pd.DataFrame([{"nifti_path": str(nifti)}]).to_csv(csv_path, index=False)
    config_path = tmp_path / "tasks.json"
    config_path.write_text(
        json.dumps(
            {
                "backend": "totalsegmentator",
                "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
            }
        )
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
    pd.DataFrame([{"nifti_path": p} for p in paths]).to_csv(csv_path, index=False)
    config_path = tmp_path / "tasks.json"
    config_path.write_text(
        json.dumps(
            {
                "backend": "totalsegmentator",
                "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
            }
        )
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
    pd.DataFrame([{"nifti_path": str(nifti)}]).to_csv(csv_path, index=False)
    config_path = tmp_path / "tasks.json"
    config_path.write_text(
        json.dumps(
            {
                "backend": "totalsegmentator",
                "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
            }
        )
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
        raise AssertionError("ProcessPoolExecutor should not be used in subprocess fallback mode")

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


def test_main_resume_skips_completed_rows(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")
    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": str(nifti)}]).to_csv(csv_path, index=False)
    config_path = tmp_path / "tasks.json"
    config_path.write_text(
        json.dumps(
            {
                "backend": "totalsegmentator",
                "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
            }
        )
    )

    prefetch_calls = {"count": 0}

    def fake_prefetch(*args, **kwargs):
        prefetch_calls["count"] += 1

    monkeypatch.setattr(segment_module, "prefetch_totalsegmentator_models", fake_prefetch)
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
        force=False,
        start_method="spawn",
        timeout_sec=10,
        checkpoint_every_rows=1,
        checkpoint_every_sec=3600,
        resume=False,
        strict_resume=False,
    )
    segment_module.main(args)
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
            {"nifti_path": str(nifti_a)},
            {"nifti_path": str(nifti_a)},
            {"nifti_path": str(nifti_b)},
        ]
    ).to_csv(csv_path, index=False)
    config_path = tmp_path / "tasks.json"
    config_path.write_text(
        json.dumps(
            {
                "backend": "totalsegmentator",
                "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
            }
        )
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
    pd.DataFrame([{"nifti_path": str(nifti)}]).to_csv(csv_path, index=False)
    config_path = tmp_path / "tasks.json"
    config_path.write_text(
        json.dumps(
            {
                "backend": "totalsegmentator",
                "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
            }
        )
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

    config_path.write_text(
        json.dumps(
            {
                "backend": "totalsegmentator",
                "tasks": [
                    {"key": "spleen", "task": "total", "output": "spleen.nii.gz"}
                ],
            }
        )
    )
    args.resume = True
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
                "mask_liver": "preexisting-a",
                "warning_message": "warn-a",
            },
            {
                "nifti_path": str(nifti_b),
                "mask_liver": "preexisting-b",
                "warning_message": "warn-b",
            },
        ]
    ).to_csv(csv_path, index=False)
    config_path = tmp_path / "tasks.json"
    config_path.write_text(
        json.dumps(
            {
                "backend": "totalsegmentator",
                "tasks": [{"key": "liver", "task": "total", "output": "liver.nii.gz"}],
            }
        )
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
