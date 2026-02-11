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


def test_load_tasks_config_from_manifest():
    manifest = {
        "segmentation": {
            "backend": "totalsegmentator",
            "tasks": [
                {"key": "x", "task": "total", "output": "x.nii.gz", "extra": {}},
            ],
        }
    }
    cfg = segment_module.load_tasks_config(None, manifest=manifest)
    assert cfg["tasks"][0]["key"] == "x"


def test_load_tasks_config_missing(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError):
        segment_module.load_tasks_config(missing)


def test_segment_parser_defaults_manifest_to_generic():
    parser = argparse.ArgumentParser()
    segment_module.add_segment_arguments(parser)
    args = parser.parse_args([])
    assert args.manifest == "generic"


def test_resolve_postprocess_config_in_key_defaults_to_mask_key_column():
    keys, out_col = segment_module.resolve_postprocess_config({"in_key": "liver"})
    assert keys == ["liver"]
    assert out_col == "mask_liver"


def test_resolve_postprocess_config_merge_keys_defaults_to_mask_merged():
    keys, out_col = segment_module.resolve_postprocess_config(
        {"merge_keys": ["liver", "vessels"]}
    )
    assert keys == ["liver", "vessels"]
    assert out_col == "mask_merged"


def test_resolve_postprocess_config_rejects_in_key_and_merge_keys():
    with pytest.raises(ValueError, match="either in_key or merge_keys"):
        segment_module.resolve_postprocess_config(
            {"in_key": "liver", "merge_keys": ["liver"]}
        )


@pytest.mark.parametrize("legacy_key", ["output_column", "column_name", "output_col"])
def test_resolve_postprocess_config_rejects_legacy_output_column_keys(legacy_key):
    with pytest.raises(ValueError, match="Unsupported postprocess key"):
        segment_module.resolve_postprocess_config(
            {"merge_keys": ["liver"], legacy_key: "mask_custom"}
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
            "output": "a_clean.nii.gz",
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


def test_resolve_postprocess_operations_accepts_dict_or_list():
    ops_from_dict = segment_module.resolve_postprocess_operations({"in_key": "liver"})
    assert len(ops_from_dict) == 1

    ops_from_list = segment_module.resolve_postprocess_operations(
        [{"in_key": "liver"}, {"merge_keys": ["liver"]}]
    )
    assert len(ops_from_list) == 2


def test_resolve_postprocess_operation_defaults_output_name_from_out_key():
    op = segment_module.resolve_postprocess_operation(
        {"merge_keys": ["liver", "tumor"], "out_key": "combined"},
        op_index=1,
    )
    assert op["output_name"] == "combined.nii.gz"
    assert op["output_column"] == "mask_combined"


def test_resolve_postprocess_operation_defaults_output_name_from_column_when_missing():
    op_in_key = segment_module.resolve_postprocess_operation(
        {"in_key": "liver"}, op_index=1
    )
    assert op_in_key["output_column"] == "mask_liver"
    assert op_in_key["output_name"] == "liver.nii.gz"

    op_merge = segment_module.resolve_postprocess_operation(
        {"merge_keys": ["liver", "tumor"]},
        op_index=2,
    )
    assert op_merge["output_column"] == "mask_merged"
    assert op_merge["output_name"] == "merged.nii.gz"


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


def test_main_writes_custom_merged_mask_column(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": str(nifti)}]).to_csv(csv_path, index=False)

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
            "merge_keys": ["liver", "vessels"],
            "output": "merged.nii.gz",
            "out_key": "combined",
        },
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
    assert "mask_combined" in out_df.columns
    assert "mask_merged" not in out_df.columns
    assert out_df.loc[0, "mask_liver"].endswith("liver.nii.gz")
    assert out_df.loc[0, "mask_vessels"].endswith("vessels.nii.gz")
    assert out_df.loc[0, "mask_combined"].endswith("merged.nii.gz")


def test_main_warns_on_merged_column_collision(tmp_path, monkeypatch, caplog):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": str(nifti)}]).to_csv(csv_path, index=False)

    config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "liver", "task": "total", "output": "liver.nii.gz", "extra": {}},
        ],
        "postprocess": {
            "merge_keys": ["liver"],
            "output": "merged.nii.gz",
            "out_key": "liver",
        },
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

    with caplog.at_level("WARNING"):
        segment_module.main(args)

    assert any(
        "matches an existing task mask column" in rec.getMessage()
        for rec in caplog.records
    )

    out_df = pd.read_csv(args.csv_path_out)
    assert "mask_liver" in out_df.columns
    assert out_df.loc[0, "mask_liver"].endswith("merged.nii.gz")
    assert "mask_merged" not in out_df.columns


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


def test_main_writes_multiple_postprocess_columns(tmp_path, monkeypatch):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": str(nifti)}]).to_csv(csv_path, index=False)

    config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "liver", "task": "total", "output": "liver.nii.gz", "extra": {}},
            {
                "key": "tumor",
                "task": "liver_vessels",
                "output": "tumor.nii.gz",
                "extra": {},
            },
        ],
        "postprocess": [
            {
                "merge_keys": ["liver", "tumor"],
                "out_key": "combined",
                "output": "combined.nii.gz",
            },
            {"merge_keys": ["combined"], "out_key": "final", "output": "final.nii.gz"},
        ],
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
            (out_path / "tumor.nii.gz").write_text("mask")
            (out_path / "combined.nii.gz").write_text("mask")
            (out_path / "final.nii.gz").write_text("mask")
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
    assert "mask_combined" in out_df.columns
    assert "mask_final" in out_df.columns
    assert out_df.loc[0, "mask_combined"].endswith("combined.nii.gz")
    assert out_df.loc[0, "mask_final"].endswith("final.nii.gz")


def test_main_warns_on_postprocess_column_and_file_collisions(
    tmp_path, monkeypatch, caplog
):
    nifti = tmp_path / "vol.nii.gz"
    nifti.write_text("nifti")

    csv_path = tmp_path / "nifti_index.csv"
    pd.DataFrame([{"nifti_path": str(nifti)}]).to_csv(csv_path, index=False)

    config = {
        "backend": "totalsegmentator",
        "tasks": [
            {"key": "liver", "task": "total", "output": "liver.nii.gz", "extra": {}},
        ],
        "postprocess": [
            {"merge_keys": ["liver"], "out_key": "liver", "output": "liver.nii.gz"},
            {"merge_keys": ["liver"], "out_key": "liver", "output": "liver.nii.gz"},
        ],
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

    with caplog.at_level("WARNING"):
        segment_module.main(args)

    messages = [rec.getMessage() for rec in caplog.records]
    assert any(
        "output column 'mask_liver' matches an existing task mask column" in m
        for m in messages
    )
    assert any(
        "output column 'mask_liver' matches another postprocess operation" in m
        for m in messages
    )
    assert any(
        "output file 'liver.nii.gz' collides with an existing task/postprocess output"
        in m
        for m in messages
    )
