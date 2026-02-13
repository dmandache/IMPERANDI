import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imperandi.process import segment as segment_module
from imperandi.process import segment_config as segment_config_module


class DummyBackend:
    def __init__(self, outputs):
        self.outputs = outputs

    def run(self, *, input_path, output_dir, task, fast, **kwargs):
        out_name = self.outputs[task]
        (Path(output_dir) / out_name).write_text("mask")


class RecordingBackend:
    def __init__(self, output_name="mask.nii.gz"):
        self.output_name = output_name
        self.calls = []

    def run(self, *, input_path, output_dir, task, fast, **kwargs):
        self.calls.append({"task": task, "fast": fast, "kwargs": dict(kwargs)})
        (Path(output_dir) / self.output_name).write_text("mask")


class MultiTaskRecordingBackend:
    def __init__(self):
        self.calls = []

    def run(self, *, input_path, output_dir, task, fast, **kwargs):
        self.calls.append({"task": task, "fast": fast, "kwargs": dict(kwargs)})
        (Path(output_dir) / f"{task}.nii.gz").write_text("mask")


def fake_process_single_volume(
    idx,
    row,
    tasks_config,
    *,
    fast,
    verbose,
    force,
    backend=None,
):
    del tasks_config, fast, verbose, force, backend
    nifti_path = Path(row["nifti_path"])
    out_dir = nifti_path.parent

    outputs_raw = str(row.get("fake_outputs", "")).strip()
    if outputs_raw:
        for name in outputs_raw.split("|"):
            if name:
                (out_dir / name).write_text("mask")

    warning_msg = row.get("fake_warning")
    error_msg = row.get("fake_error")
    if error_msg:
        return idx, None, str(error_msg), warning_msg
    return idx, str(out_dir), None, warning_msg


def fake_process_single_volume_with_sleep(
    idx,
    row,
    tasks_config,
    *,
    fast,
    verbose,
    force,
    backend=None,
):
    sleep_sec = float(row.get("fake_sleep_sec", 0) or 0)
    if sleep_sec > 0:
        time.sleep(sleep_sec)
    return fake_process_single_volume(
        idx,
        row,
        tasks_config,
        fast=fast,
        verbose=verbose,
        force=force,
        backend=backend,
    )


def fake_process_single_volume_no_result(
    idx,
    row,
    tasks_config,
    *,
    fast,
    verbose,
    force,
    backend=None,
):
    del idx, row, tasks_config, fast, verbose, force, backend
    os._exit(0)


def _run_main_with_worker(
    tmp_path,
    monkeypatch,
    *,
    config,
    rows,
    worker_fn=fake_process_single_volume,
    timeout_sec=10,
    num_workers=1,
    start_method="spawn",
):
    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    config_path = tmp_path / "tasks.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setattr(segment_module, "process_single_volume", worker_fn)
    monkeypatch.setattr(
        segment_module,
        "prefetch_totalsegmentator_models",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(segment_module, "tqdm", lambda it, **kwargs: it)

    args = argparse.Namespace(
        csv_path=str(csv_path),
        csv_path_out=str(tmp_path / "segmented.csv"),
        error_csv_path=str(tmp_path / "errors.csv"),
        tasks_config=str(config_path),
        num_workers=num_workers,
        fast=False,
        verbose=False,
        force=False,
        start_method=start_method,
        timeout_sec=timeout_sec,
    )
    segment_module.main(args)

    out_df = pd.read_csv(args.csv_path_out)
    err_path = Path(args.error_csv_path)
    err_df = pd.read_csv(err_path) if err_path.exists() else pd.DataFrame()
    return out_df, err_df, args


def test_load_tasks_config_default():
    cfg = segment_config_module.load_tasks_config(None)
    assert "tasks" in cfg
    assert cfg["backend"] == "totalsegmentator"


def test_load_tasks_config_from_manifest():
    manifest = {
        "segmentation": {
            "backend": "totalsegmentator",
            "tasks": [
                {"key": "x", "task": "total", "output": "x.nii.gz", "extra": {}},
            ],
        }
    }
    cfg = segment_config_module.load_tasks_config(None, manifest=manifest)
    assert cfg["tasks"][0]["key"] == "x"


def test_resolve_manifest_fast_default_from_segmentation_fast_overrides_cli(caplog):
    cfg = {
        "backend": "totalsegmentator",
        "fast": True,
        "tasks": [{"task": "total", "extra": {}}],
    }

    with caplog.at_level("WARNING"):
        resolved = segment_config_module.resolve_manifest_fast_default(
            cfg, cli_fast=False
        )

    assert resolved is True
    assert cfg["fast"] is True
    assert any(
        "Manifest fast setting detected" in rec.getMessage() for rec in caplog.records
    )


def test_resolve_task_fast_and_extra_from_extra_fast_overrides_default():
    task = {"task": "total", "extra": {"roi_subset": ["liver"], "fast": True}}

    resolved, extra = segment_config_module.resolve_task_fast_and_extra(
        task, task_index=0, default_fast=False
    )

    assert resolved is True
    assert extra["roi_subset"] == ["liver"]
    assert "fast" not in extra


def test_load_tasks_config_missing(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError):
        segment_config_module.load_tasks_config(missing)


def test_segment_parser_defaults_manifest_to_generic():
    parser = argparse.ArgumentParser()
    segment_module.add_segment_arguments(parser)
    args = parser.parse_args([])
    assert args.manifest == "generic"


def test_resolve_postprocess_operation_in_key_uses_mask_out_key_column():
    op = segment_config_module.resolve_postprocess_operation(
        {"in_key": "liver", "out_key": "liver_clean"},
        op_index=1,
    )
    assert op["input_keys"] == ["liver"]
    assert op["output_column"] == "mask_liver_clean"


def test_resolve_postprocess_operation_merge_keys_uses_mask_out_key_column():
    op = segment_config_module.resolve_postprocess_operation(
        {"merge_keys": ["liver", "vessels"], "out_key": "combined"},
        op_index=1,
    )
    assert op["input_keys"] == ["liver", "vessels"]
    assert op["output_column"] == "mask_combined"


def test_resolve_postprocess_operation_rejects_in_key_and_merge_keys():
    with pytest.raises(ValueError, match="either in_key or merge_keys"):
        segment_config_module.resolve_postprocess_operation(
            {"in_key": "liver", "merge_keys": ["liver"]},
            op_index=1,
        )


@pytest.mark.parametrize("legacy_key", ["output_column", "column_name", "output_col"])
def test_resolve_postprocess_operation_rejects_legacy_output_column_keys(legacy_key):
    with pytest.raises(ValueError, match="Unsupported postprocess key"):
        segment_config_module.resolve_postprocess_operation(
            {"merge_keys": ["liver"], legacy_key: "mask_custom"},
            op_index=1,
        )


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
            "out_key": "postproc",
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

    assert calls["output_name"] == "postproc.nii.gz"
    assert set(calls["mask_files"]) == {"a.nii.gz", "b.nii.gz"}


def test_segment_volume_supports_postprocess_in_key(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    tasks_config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "a", "task": "task_a", "output": "a.nii.gz", "extra": {}},
            {"key": "b", "task": "task_b", "output": "b.nii.gz", "extra": {}},
        ],
        "postprocess": {
            "in_key": "a",
            "out_key": "a_clean",
        },
    }

    backend = DummyBackend({"task_a": "a.nii.gz", "task_b": "b.nii.gz"})
    calls = {}

    def fake_clean(dir_path, mask_files, *, output_name, **kwargs):
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

    assert calls["output_name"] == "a_clean.nii.gz"
    assert calls["mask_files"] == ["a.nii.gz"]


def test_segment_volume_force_false_skips_existing_segmentation_and_postprocess(
    tmp_path, monkeypatch
):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")
    (tmp_path / "a.nii.gz").write_text("existing")
    (tmp_path / "postproc.nii.gz").write_text("existing")

    tasks_config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "a", "task": "task_a", "output": "a.nii.gz", "extra": {}},
        ],
        "postprocess": {"merge_keys": ["a"], "out_key": "postproc"},
    }

    class ShouldNotRunBackend:
        def run(self, *, input_path, output_dir, task, fast, **kwargs):
            raise AssertionError("segmentation task should be skipped when outputs exist")

    def fail_clean(*args, **kwargs):
        raise AssertionError("postprocess should be skipped when output exists")

    monkeypatch.setattr(segment_module, "clean_and_merge_masks", fail_clean)

    segment_module.segment_volume(
        nifti,
        tmp_path,
        tasks_config,
        fast=False,
        verbose=False,
        force=False,
        backend=ShouldNotRunBackend(),
    )


def test_segment_volume_force_false_skips_existing_segmentation_but_runs_missing_postprocess(
    tmp_path, monkeypatch
):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")
    (tmp_path / "a.nii.gz").write_text("existing")

    tasks_config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "a", "task": "task_a", "output": "a.nii.gz", "extra": {}},
        ],
        "postprocess": {"merge_keys": ["a"], "out_key": "postproc"},
    }

    class ShouldNotRunBackend:
        def run(self, *, input_path, output_dir, task, fast, **kwargs):
            raise AssertionError("segmentation task should be skipped when outputs exist")

    calls = {"n": 0}

    def fake_clean(dir_path, mask_files, *, output_name, **kwargs):
        del mask_files, kwargs
        calls["n"] += 1
        (Path(dir_path) / output_name).write_text("mask")
        return True

    monkeypatch.setattr(segment_module, "clean_and_merge_masks", fake_clean)

    segment_module.segment_volume(
        nifti,
        tmp_path,
        tasks_config,
        fast=False,
        verbose=False,
        force=False,
        backend=ShouldNotRunBackend(),
    )

    assert calls["n"] == 1
    assert (tmp_path / "postproc.nii.gz").exists()


def test_segment_volume_force_true_reruns_existing_segmentation_and_postprocess(
    tmp_path, monkeypatch
):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")
    (tmp_path / "a.nii.gz").write_text("existing")
    (tmp_path / "postproc.nii.gz").write_text("existing")

    tasks_config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "a", "task": "task_a", "output": "a.nii.gz", "extra": {}},
        ],
        "postprocess": {"merge_keys": ["a"], "out_key": "postproc"},
    }

    calls = {"seg": 0, "post": 0}

    class RecordingBackend:
        def run(self, *, input_path, output_dir, task, fast, **kwargs):
            del input_path, task, fast, kwargs
            calls["seg"] += 1
            (Path(output_dir) / "a.nii.gz").write_text("fresh")

    def fake_clean(dir_path, mask_files, *, output_name, **kwargs):
        del mask_files, kwargs
        calls["post"] += 1
        (Path(dir_path) / output_name).write_text("mask")
        return True

    monkeypatch.setattr(segment_module, "clean_and_merge_masks", fake_clean)

    segment_module.segment_volume(
        nifti,
        tmp_path,
        tasks_config,
        fast=False,
        verbose=False,
        force=True,
        backend=RecordingBackend(),
    )

    assert calls["seg"] == 1
    assert calls["post"] == 1


def test_segment_volume_uses_manifest_fast_and_strips_extra_fast(tmp_path):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    tasks_config = {
        "backend": "totalsegmentator",
        "tasks": [
            {
                "task": "task_a",
                "extra": {"fast": True, "roi_subset": ["liver"]},
            },
        ],
    }

    backend = RecordingBackend(output_name="liver.nii.gz")

    segment_module.segment_volume(
        nifti,
        tmp_path,
        tasks_config,
        fast=False,
        verbose=False,
        force=True,
        backend=backend,
    )

    assert len(backend.calls) == 1
    assert backend.calls[0]["fast"] is True
    assert "fast" not in backend.calls[0]["kwargs"]


def test_segment_volume_respects_per_task_fast_overrides(tmp_path):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    tasks_config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"task": "task_a", "extra": {}},
            {"task": "task_b", "extra": {"fast": True}},
        ],
    }
    backend = MultiTaskRecordingBackend()

    segment_module.segment_volume(
        nifti,
        tmp_path,
        tasks_config,
        fast=False,
        verbose=False,
        force=True,
        backend=backend,
    )

    assert [call["task"] for call in backend.calls] == ["task_a", "task_b"]
    assert [call["fast"] for call in backend.calls] == [False, True]


def test_resolve_postprocess_operations_accepts_dict_or_list():
    ops_from_dict = segment_config_module.resolve_postprocess_operations(
        {"in_key": "liver", "out_key": "liver_clean"}
    )
    assert len(ops_from_dict) == 1

    ops_from_list = segment_config_module.resolve_postprocess_operations(
        [
            {"in_key": "liver", "out_key": "liver_clean"},
            {"merge_keys": ["liver"], "out_key": "liver_combined"},
        ]
    )
    assert len(ops_from_list) == 2


def test_resolve_postprocess_operation_defaults_output_name_from_out_key():
    op = segment_config_module.resolve_postprocess_operation(
        {"merge_keys": ["liver", "tumor"], "out_key": "combined"},
        op_index=1,
    )
    assert op["output_name"] == "combined.nii.gz"
    assert op["output_column"] == "mask_combined"


def test_resolve_postprocess_operation_requires_out_key():
    with pytest.raises(ValueError, match="out_key is required"):
        segment_config_module.resolve_postprocess_operation(
            {"in_key": "liver"}, op_index=1
        )


def test_segment_volume_executes_postprocess_list_in_order_and_chains(
    tmp_path, monkeypatch
):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    tasks_config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "a", "task": "task_a", "output": "a.nii.gz", "extra": {}},
            {"key": "b", "task": "task_b", "output": "b.nii.gz", "extra": {}},
        ],
        "postprocess": [
            {"merge_keys": ["a", "b"], "out_key": "ab"},
            {"merge_keys": ["ab"], "out_key": "ab_refined"},
        ],
    }

    backend = DummyBackend({"task_a": "a.nii.gz", "task_b": "b.nii.gz"})
    calls = []

    def fake_clean(dir_path, mask_files, *, output_name, **kwargs):
        calls.append((list(mask_files), output_name))
        (Path(dir_path) / output_name).write_text("mask")
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

    assert calls[0] == (["a.nii.gz", "b.nii.gz"], "ab.nii.gz")
    assert calls[1] == (["ab.nii.gz"], "ab_refined.nii.gz")


def test_segment_volume_continues_after_warn_only_missing_dependency(
    tmp_path, monkeypatch
):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    tasks_config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "a", "task": "task_a", "output": "a.nii.gz", "extra": {}},
        ],
        "postprocess": [
            {
                "merge_keys": ["missing_key"],
                "out_key": "broken",
                "on_failure": "warn_only",
            },
            {"merge_keys": ["a"], "out_key": "ok"},
        ],
    }

    backend = DummyBackend({"task_a": "a.nii.gz"})

    def fake_clean(dir_path, mask_files, *, output_name, **kwargs):
        (Path(dir_path) / output_name).write_text("mask")
        return True

    monkeypatch.setattr(segment_module, "clean_and_merge_masks", fake_clean)

    warnings = segment_module.segment_volume(
        nifti,
        tmp_path,
        tasks_config,
        fast=False,
        verbose=False,
        force=True,
        backend=backend,
    )

    assert any("missing input key(s)" in w for w in warnings)
    assert (tmp_path / "ok.nii.gz").exists()


def test_segment_volume_fails_on_missing_dependency_when_fail_policy(tmp_path):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    tasks_config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "a", "task": "task_a", "output": "a.nii.gz", "extra": {}},
        ],
        "postprocess": [
            {"merge_keys": ["missing_key"], "out_key": "broken", "on_failure": "fail"},
        ],
    }

    backend = DummyBackend({"task_a": "a.nii.gz"})

    with pytest.raises(ValueError, match="missing input key"):
        segment_module.segment_volume(
            nifti,
            tmp_path,
            tasks_config,
            fast=False,
            verbose=False,
            force=True,
            backend=backend,
        )


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
            "out_key": "postproc",
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
    assert "postproc.nii.gz" in warnings[0]


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
            "out_key": "postproc",
            "on_failure": "fail",
        },
    }

    backend = DummyBackend({"task_a": "a.nii.gz"})
    monkeypatch.setattr(segment_module, "clean_and_merge_masks", lambda *a, **k: False)

    with pytest.raises(RuntimeError, match="postproc.nii.gz"):
        segment_module.segment_volume(
            nifti,
            tmp_path,
            tasks_config,
            fast=False,
            verbose=False,
            force=True,
            backend=backend,
        )


def test_segment_volume_continues_independent_steps_before_fail_policy_raise(
    tmp_path, monkeypatch
):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    tasks_config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "a", "task": "task_a", "output": "a.nii.gz", "extra": {}},
        ],
        "postprocess": [
            {"merge_keys": ["missing_key"], "out_key": "broken", "on_failure": "fail"},
            {"merge_keys": ["a"], "out_key": "ok"},
        ],
    }

    backend = DummyBackend({"task_a": "a.nii.gz"})

    def fake_clean(dir_path, mask_files, *, output_name, **kwargs):
        del mask_files, kwargs
        (Path(dir_path) / output_name).write_text("mask")
        return True

    monkeypatch.setattr(segment_module, "clean_and_merge_masks", fake_clean)

    with pytest.raises(ValueError, match="missing input key"):
        segment_module.segment_volume(
            nifti,
            tmp_path,
            tasks_config,
            fast=False,
            verbose=False,
            force=True,
            backend=backend,
        )

    assert (tmp_path / "ok.nii.gz").exists()


def test_segment_volume_warn_only_exception_does_not_skip_later_operations(
    tmp_path, monkeypatch
):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    tasks_config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "a", "task": "task_a", "output": "a.nii.gz", "extra": {}},
        ],
        "postprocess": [
            {"merge_keys": ["a"], "out_key": "broken", "on_failure": "warn_only"},
            {"merge_keys": ["a"], "out_key": "ok"},
        ],
    }

    backend = DummyBackend({"task_a": "a.nii.gz"})

    def fake_clean(dir_path, mask_files, *, output_name, **kwargs):
        del mask_files, kwargs
        if output_name == "broken.nii.gz":
            raise RuntimeError("merge exploded")
        (Path(dir_path) / output_name).write_text("mask")
        return True

    monkeypatch.setattr(segment_module, "clean_and_merge_masks", fake_clean)

    warnings = segment_module.segment_volume(
        nifti,
        tmp_path,
        tasks_config,
        fast=False,
        verbose=False,
        force=True,
        backend=backend,
    )

    assert any("failed with error: merge exploded" in w for w in warnings)
    assert (tmp_path / "ok.nii.gz").exists()


def test_segment_volume_fail_policy_exception_still_runs_later_operations(
    tmp_path, monkeypatch
):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    tasks_config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "a", "task": "task_a", "output": "a.nii.gz", "extra": {}},
        ],
        "postprocess": [
            {"merge_keys": ["a"], "out_key": "broken", "on_failure": "fail"},
            {"merge_keys": ["a"], "out_key": "ok"},
        ],
    }

    backend = DummyBackend({"task_a": "a.nii.gz"})

    def fake_clean(dir_path, mask_files, *, output_name, **kwargs):
        del mask_files, kwargs
        if output_name == "broken.nii.gz":
            raise RuntimeError("merge exploded")
        (Path(dir_path) / output_name).write_text("mask")
        return True

    monkeypatch.setattr(segment_module, "clean_and_merge_masks", fake_clean)

    with pytest.raises(RuntimeError, match="merge exploded"):
        segment_module.segment_volume(
            nifti,
            tmp_path,
            tasks_config,
            fast=False,
            verbose=False,
            force=True,
            backend=backend,
        )

    assert (tmp_path / "ok.nii.gz").exists()


def test_segment_volume_missing_task_key_fails_fast(tmp_path):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")
    tasks_config = {"backend": "totalsegmentator", "tasks": [{"extra": {}}]}

    with pytest.raises(ValueError, match=r"segmentation\.tasks\[0\]\.task is required"):
        segment_module.segment_volume(
            nifti,
            tmp_path,
            tasks_config,
            fast=False,
            verbose=False,
            force=True,
            backend=DummyBackend({}),
        )


def test_prefetch_missing_task_key_fails_with_same_validation():
    tasks_config = {"backend": "totalsegmentator", "tasks": [{"extra": {}}]}
    with pytest.raises(ValueError, match=r"segmentation\.tasks\[0\]\.task is required"):
        segment_module.prefetch_totalsegmentator_models(tasks_config, fast=False)


def test_segment_volume_collects_postprocess_collision_warning(tmp_path):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    tasks_config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"task": "task_a", "extra": {}},
        ],
        "postprocess": {"merge_keys": ["liver"], "out_key": "liver"},
    }
    backend = DummyBackend({"task_a": "liver.nii.gz"})

    warnings = segment_module.segment_volume(
        nifti,
        tmp_path,
        tasks_config,
        fast=False,
        verbose=False,
        force=False,
        backend=backend,
    )

    assert any("collides with an existing output" in warning for warning in warnings)


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
    assert "No segmentation masks found after task" in err
    assert warning is None


def test_main_writes_mask_columns(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")
    config = {
        "backend": "totalsegmentator",
        "tasks": [{"task": "total", "extra": {}}],
        "postprocess": {"merge_keys": ["liver", "vessels"], "out_key": "postproc"},
    }
    rows = [
        {
            "nifti_path": str(nifti),
            "fake_outputs": "liver.nii.gz|vessels.nii.gz|postproc.nii.gz",
        }
    ]
    out_df, _, _ = _run_main_with_worker(
        tmp_path, monkeypatch, config=config, rows=rows
    )

    assert "mask_liver" in out_df.columns
    assert "mask_vessels" in out_df.columns
    assert "mask_postproc" in out_df.columns
    assert out_df.loc[0, "mask_liver"].endswith("liver.nii.gz")
    assert out_df.loc[0, "mask_vessels"].endswith("vessels.nii.gz")
    assert out_df.loc[0, "mask_postproc"].endswith("postproc.nii.gz")


def test_main_writes_custom_merged_mask_column(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")
    config = {
        "backend": "totalsegmentator",
        "tasks": [{"task": "total", "extra": {}}],
        "postprocess": {"merge_keys": ["liver", "vessels"], "out_key": "combined"},
    }
    rows = [
        {
            "nifti_path": str(nifti),
            "fake_outputs": "liver.nii.gz|vessels.nii.gz|combined.nii.gz",
        }
    ]
    out_df, _, _ = _run_main_with_worker(
        tmp_path, monkeypatch, config=config, rows=rows
    )

    assert "mask_liver" in out_df.columns
    assert "mask_vessels" in out_df.columns
    assert "mask_combined" in out_df.columns
    assert "mask_merged" not in out_df.columns
    assert "mask_postproc" not in out_df.columns


def test_main_warns_and_overwrites_when_postprocess_semantic_column_collides(
    tmp_path, monkeypatch
):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")
    config = {
        "backend": "totalsegmentator",
        "tasks": [{"task": "total", "extra": {}}],
        "postprocess": {"merge_keys": ["liver"], "out_key": "mask_liver"},
    }
    rows = [
        {
            "nifti_path": str(nifti),
            "mask_mask_liver": "old_liver.nii.gz",
            "fake_outputs": "liver.nii.gz|mask_liver.nii.gz",
        }
    ]
    out_df, _, _ = _run_main_with_worker(
        tmp_path, monkeypatch, config=config, rows=rows
    )

    assert "mask_mask_liver" in out_df.columns
    assert out_df.loc[0, "mask_mask_liver"].endswith("mask_liver.nii.gz")
    warning = out_df.loc[0, "warning_message"]
    assert isinstance(warning, str)
    assert "mask column collision for mask_mask_liver" in warning


def test_main_records_warning_when_merged_mask_missing(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")
    config = {
        "backend": "totalsegmentator",
        "tasks": [{"task": "total", "extra": {}}],
        "postprocess": {"merge_keys": ["liver"], "out_key": "postproc"},
    }
    rows = [{"nifti_path": str(nifti), "fake_outputs": "liver.nii.gz"}]
    out_df, _, _ = _run_main_with_worker(
        tmp_path, monkeypatch, config=config, rows=rows
    )

    assert "warning_message" in out_df.columns
    assert pd.isna(out_df.loc[0, "warning_message"])
    assert "mask_liver" in out_df.columns
    assert out_df.loc[0, "mask_liver"].endswith("liver.nii.gz")
    assert "mask_postproc" not in out_df.columns


def test_main_writes_multiple_postprocess_columns(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")
    config = {
        "backend": "totalsegmentator",
        "tasks": [{"task": "total", "extra": {}}],
        "postprocess": [
            {"merge_keys": ["liver", "tumor"], "out_key": "combined"},
            {"merge_keys": ["combined"], "out_key": "final"},
        ],
    }
    rows = [
        {
            "nifti_path": str(nifti),
            "fake_outputs": "liver.nii.gz|tumor.nii.gz|combined.nii.gz|final.nii.gz",
        }
    ]
    out_df, _, _ = _run_main_with_worker(
        tmp_path, monkeypatch, config=config, rows=rows
    )

    assert "mask_combined" in out_df.columns
    assert "mask_final" in out_df.columns
    assert out_df.loc[0, "mask_combined"].endswith("combined.nii.gz")
    assert out_df.loc[0, "mask_final"].endswith("final.nii.gz")


def test_main_discovers_all_generated_masks_and_excludes_source_nifti(
    tmp_path, monkeypatch
):
    nifti = tmp_path / "scan.nii.gz"
    nifti.write_text("nifti")
    config = {
        "backend": "totalsegmentator",
        "tasks": [{"task": "total", "extra": {}}],
        "postprocess": {"merge_keys": ["liver"], "out_key": "postproc"},
    }
    rows = [{"nifti_path": str(nifti), "fake_outputs": "liver.nii.gz|spleen.nii.gz"}]
    out_df, _, _ = _run_main_with_worker(
        tmp_path, monkeypatch, config=config, rows=rows
    )

    assert "mask_liver" in out_df.columns
    assert "mask_spleen" in out_df.columns
    assert "mask_scan" not in out_df.columns


def test_main_hard_timeout_marks_row_and_keeps_other_success(tmp_path, monkeypatch):
    nifti_fast = tmp_path / "fast.nii.gz"
    nifti_slow = tmp_path / "slow.nii.gz"
    nifti_fast.write_text("nifti")
    nifti_slow.write_text("nifti")
    config = {"backend": "totalsegmentator", "tasks": [{"task": "total", "extra": {}}]}
    rows = [
        {
            "nifti_path": str(nifti_fast),
            "fake_outputs": "liver.nii.gz",
            "fake_sleep_sec": 0,
        },
        {
            "nifti_path": str(nifti_slow),
            "fake_outputs": "liver.nii.gz",
            "fake_sleep_sec": 10.0,
        },
    ]
    out_df, err_df, _ = _run_main_with_worker(
        tmp_path,
        monkeypatch,
        config=config,
        rows=rows,
        worker_fn=fake_process_single_volume_with_sleep,
        timeout_sec=5.0,
        num_workers=1,
    )

    fast_row = out_df[out_df["nifti_path"] == str(nifti_fast)].iloc[0]
    assert str(fast_row["mask_liver"]).endswith("liver.nii.gz")
    assert not err_df.empty
    assert any(err_df["error_message"].astype(str).str.contains("timeout"))


def test_main_records_worker_crash_when_no_result_returned(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")
    config = {"backend": "totalsegmentator", "tasks": [{"task": "total", "extra": {}}]}
    rows = [{"nifti_path": str(nifti)}]
    _, err_df, _ = _run_main_with_worker(
        tmp_path,
        monkeypatch,
        config=config,
        rows=rows,
        worker_fn=fake_process_single_volume_no_result,
        timeout_sec=10,
        num_workers=1,
    )

    assert not err_df.empty
    assert "worker crash: no result returned" in str(err_df.loc[0, "error_message"])
