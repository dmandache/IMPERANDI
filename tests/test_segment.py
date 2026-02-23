import sys
from pathlib import Path

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import argparse
import pandas as pd
import pytest

from imperandi.process import segment as segment_module


class DummyBackend:
    def __init__(self, outputs):
        self.outputs = outputs

    def run(self, *, input_path, output_dir, task, fast, **kwargs):
        out_name = self.outputs[task]
        (Path(output_dir) / out_name).write_text("mask")


def test_load_tasks_config_default():
    cfg = segment_module.load_tasks_config(None)
    assert "tasks" in cfg
    assert cfg["backend"] == "totalsegmentator"


def test_load_tasks_config_missing(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError):
        segment_module.load_tasks_config(missing)


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
        fast=True,
        verbose=False,
        force=True,
        backend=backend,
    )

    assert calls["output_name"] == "merged.nii.gz"
    assert set(calls["mask_files"]) == {"a.nii.gz", "b.nii.gz"}


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
        fast=False,
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
            fast=False,
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

    idx, out_dir, err, warning = segment_module.process_single_volume(
        0,
        {"nifti_path": str(nifti)},
        tasks_config,
        fast=False,
        verbose=False,
        force=True,
        backend=backend,
    )

    assert err is None
    assert warning is None
    assert out_dir == str(tmp_path)


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
        def run(self, *, input_path, output_dir, task, fast, **kwargs):
            return None

    idx, out_dir, err, warning = segment_module.process_single_volume(
        0,
        {"nifti_path": str(nifti)},
        tasks_config,
        fast=False,
        verbose=False,
        force=True,
        backend=NoWriteBackend(),
    )

    assert out_dir is None
    assert "Expected mask not produced" in err
    assert warning is None


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
    config_path.write_text(json.dumps(config))

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
    monkeypatch.setattr(segment_module, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(segment_module, "tqdm", lambda it, **kwargs: it)
    monkeypatch.setattr(
        segment_module, "prefetch_totalsegmentator_models", lambda *a, **k: None
    )

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "segmented.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        tasks_config=str(config_path),
        num_workers=1,
        fast=False,
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
    config_path.write_text(json.dumps(config))

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
    monkeypatch.setattr(segment_module, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(segment_module, "tqdm", lambda it, **kwargs: it)
    monkeypatch.setattr(
        segment_module, "prefetch_totalsegmentator_models", lambda *a, **k: None
    )

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "segmented.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        tasks_config=str(config_path),
        num_workers=1,
        fast=False,
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
    config_path.write_text(json.dumps(config))

    monkeypatch.setattr(
        segment_module, "prefetch_totalsegmentator_models", lambda *a, **k: None
    )
    monkeypatch.setattr(segment_module, "tqdm", lambda it, **kwargs: it)

    def fail_if_pool_used(*args, **kwargs):
        raise AssertionError("ProcessPoolExecutor should not be used in single-worker mode")

    monkeypatch.setattr(segment_module, "ProcessPoolExecutor", fail_if_pool_used)

    def fake_process_single_volume(
        idx, row, tasks_config, *, fast, verbose, force, backend=None
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
        tasks_config=str(config_path),
        num_workers=1,
        fast=False,
        verbose=False,
        force=False,
        start_method="spawn",
        timeout_sec=10,
    )

    segment_module.main(args)

    out_df = pd.read_csv(args.csv_path_out)
    assert out_df.loc[0, "mask_liver"].endswith("liver.nii.gz")
