import os
import sys
from pathlib import Path

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

from imperandi.utils import multiprocessing as mp_utils


def test_read_int_parses_valid_and_invalid(tmp_path):
    path = tmp_path / "value.txt"
    path.write_text("42")
    assert mp_utils._read_int(str(path)) == 42

    path.write_text("not_an_int")
    assert mp_utils._read_int(str(path)) is None

    assert mp_utils._read_int(str(tmp_path / "missing.txt")) is None


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("", 0),
        ("none", 0),
        ("-1", 0),
        ("0,1", 2),
        ("GPU-abc, GPU-def", 2),
        ("0,,2", 2),
    ],
)
def test_visible_cuda_devices_from_env(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)

    assert mp_utils._visible_cuda_devices_from_env() == expected


def test_slurm_mem_mb_from_node_and_cpu(monkeypatch):
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "8192")
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "1024")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert mp_utils._slurm_mem_mb() == 8192

    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    assert mp_utils._slurm_mem_mb() == 4096


def test_nvidia_smi_gpu_count_parses_output(monkeypatch):
    monkeypatch.setattr(mp_utils.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(
        mp_utils.subprocess,
        "check_output",
        lambda *args, **kwargs: "GPU 0: A100\nGPU 1: A100\nNot a gpu line\n",
    )

    assert mp_utils._nvidia_smi_gpu_count() == 2


def test_nvidia_smi_gpu_count_handles_missing_binary(monkeypatch):
    monkeypatch.setattr(mp_utils.shutil, "which", lambda _: None)
    assert mp_utils._nvidia_smi_gpu_count() is None


def test_ram_total_mb_falls_back_to_sysconf(monkeypatch):
    monkeypatch.setattr(mp_utils, "_cgroup_mem_limit_mb", lambda: None)
    monkeypatch.setattr(mp_utils, "_slurm_mem_mb", lambda: None)

    def fake_sysconf(name):
        values = {"SC_PHYS_PAGES": 1000, "SC_PAGE_SIZE": 4096}
        return values[name]

    monkeypatch.setattr(mp_utils.os, "sysconf", fake_sysconf, raising=False)

    assert mp_utils._ram_total_mb() == int(1000 * 4096 / (1024 * 1024))


def test_decide_strategy_single_gpu_uses_subprocess_mode_for_hard_timeouts(
    monkeypatch,
):
    monkeypatch.setattr(mp_utils, "_slurm_cpus", lambda: 16)
    monkeypatch.setattr(mp_utils, "_ram_total_mb", lambda: 128000)
    monkeypatch.setattr(mp_utils, "_visible_cuda_devices_from_env", lambda: 1)
    monkeypatch.setattr(mp_utils, "_torch_gpu_count", lambda: None)
    monkeypatch.setattr(mp_utils, "_nvidia_smi_gpu_count", lambda: None)

    strategy = mp_utils.decide_multiprocessing_strategy(
        prefer_gpu=True,
        requested_workers=8,
        need_hard_timeouts=True,
    )

    assert strategy.mode == "subprocess_per_case"
    assert strategy.use_gpu is True
    assert strategy.gpu_count == 1
    assert strategy.start_method == "spawn"
    assert strategy.max_workers == 1
    assert strategy.max_in_flight == 1
    assert strategy.hard_timeout_supported is True


def test_decide_strategy_multi_gpu_clamps_workers_to_gpu_count(monkeypatch):
    monkeypatch.setattr(mp_utils, "_slurm_cpus", lambda: 32)
    monkeypatch.setattr(mp_utils, "_ram_total_mb", lambda: 256000)
    monkeypatch.setattr(mp_utils, "_visible_cuda_devices_from_env", lambda: 2)
    monkeypatch.setattr(mp_utils, "_torch_gpu_count", lambda: None)
    monkeypatch.setattr(mp_utils, "_nvidia_smi_gpu_count", lambda: None)

    strategy = mp_utils.decide_multiprocessing_strategy(
        prefer_gpu=True,
        requested_workers=16,
        need_hard_timeouts=True,
    )

    assert strategy.mode == "process_pool"
    assert strategy.max_workers == 2
    assert strategy.max_in_flight == 2
    assert strategy.reasons["max_in_flight_reason"].startswith("GPU workload")
    assert strategy.start_method == "spawn"


def test_decide_strategy_gpu_respects_explicit_max_in_flight(monkeypatch):
    monkeypatch.setattr(mp_utils, "_slurm_cpus", lambda: 10)
    monkeypatch.setattr(mp_utils, "_ram_total_mb", lambda: 80000)
    monkeypatch.setattr(mp_utils, "_visible_cuda_devices_from_env", lambda: 2)
    monkeypatch.setattr(mp_utils, "_torch_gpu_count", lambda: None)
    monkeypatch.setattr(mp_utils, "_nvidia_smi_gpu_count", lambda: None)

    strategy = mp_utils.decide_multiprocessing_strategy(
        prefer_gpu=True,
        requested_workers=2,
        requested_max_in_flight=4,
    )

    assert strategy.max_workers == 2
    assert strategy.max_in_flight == 4
    assert strategy.reasons["max_in_flight_reason"] == "explicit user override"
    assert strategy.recycle_every == 0


def test_decide_strategy_cpu_only_respects_hint_and_serial_case(monkeypatch):
    monkeypatch.setattr(mp_utils, "_slurm_cpus", lambda: 8)
    monkeypatch.setattr(mp_utils, "_ram_total_mb", lambda: 64000)
    monkeypatch.setattr(mp_utils, "_visible_cuda_devices_from_env", lambda: None)
    monkeypatch.setattr(mp_utils, "_torch_gpu_count", lambda: None)
    monkeypatch.setattr(mp_utils, "_nvidia_smi_gpu_count", lambda: None)

    pool_strategy = mp_utils.decide_multiprocessing_strategy(
        prefer_gpu=False,
        requested_workers=3,
        start_method_hint="forkserver",
        need_hard_timeouts=False,
    )
    assert pool_strategy.mode == "process_pool"
    assert pool_strategy.start_method == "forkserver"
    assert pool_strategy.max_workers == 3
    assert pool_strategy.hard_timeout_supported is False

    serial_strategy = mp_utils.decide_multiprocessing_strategy(
        prefer_gpu=False,
        requested_workers=1,
        need_hard_timeouts=False,
    )
    assert serial_strategy.mode == "serial"
    assert serial_strategy.max_workers == 1
    assert serial_strategy.max_in_flight == 1
    assert serial_strategy.recycle_every == 0


@pytest.mark.parametrize(
    "cpus,gpus,requested_workers,ram_mb,workers,threads_per_worker,compute_threads",
    [
        pytest.param(6, 2, 4, 64000, 2, 3, 2, id="gpu-cap"),
        # The existing CPU cap reserves one core, so 2 CPUs allow only 1 worker.
        pytest.param(2, 0, 2, 64000, 1, 2, 1, id="two-cpus"),
        pytest.param(3, 0, 2, 64000, 2, 1, 1, id="two-workers-minimum"),
        pytest.param(1, 0, 1, 64000, 1, 1, 1, id="single-cpu"),
        pytest.param(6, 0, 1, 64000, 1, 6, 5, id="serial"),
        pytest.param(6, 1, 4, 64000, 1, 6, 5, id="single-gpu"),
        pytest.param(8, 0, 4, 8000, 2, 4, 3, id="memory-cap"),
        pytest.param(7, 0, 3, 64000, 3, 2, 1, id="uneven-allocation"),
    ],
)
def test_decide_strategy_allocates_cpu_threads_to_resolved_workers(
    monkeypatch,
    cpus,
    gpus,
    requested_workers,
    ram_mb,
    workers,
    threads_per_worker,
    compute_threads,
):
    monkeypatch.setattr(mp_utils, "_slurm_cpus", lambda: cpus)
    monkeypatch.setattr(mp_utils, "_ram_total_mb", lambda: ram_mb)
    monkeypatch.setattr(mp_utils, "_visible_cuda_devices_from_env", lambda: gpus)
    monkeypatch.setattr(mp_utils, "_torch_gpu_count", lambda: None)
    monkeypatch.setattr(mp_utils, "_nvidia_smi_gpu_count", lambda: None)

    strategy = mp_utils.decide_multiprocessing_strategy(
        requested_workers=requested_workers,
    )

    assert strategy.max_workers == workers
    assert strategy.reasons["threads_per_worker"] == threads_per_worker
    assert strategy.reasons["compute_threads"] == compute_threads
    assert strategy.env == {
        "OMP_NUM_THREADS": str(compute_threads),
        "MKL_NUM_THREADS": str(compute_threads),
        "OPENBLAS_NUM_THREADS": str(compute_threads),
        "NUMEXPR_NUM_THREADS": str(compute_threads),
        "VECLIB_MAXIMUM_THREADS": str(compute_threads),
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(compute_threads),
        "TORCH_NUM_THREADS": str(compute_threads),
    }
    diagnostics = mp_utils.strategy_to_log_dict(strategy)
    assert diagnostics["reasons"]["threads_per_worker"] == threads_per_worker
    assert diagnostics["reasons"]["compute_threads"] == compute_threads


def test_apply_strategy_env_overwrites_existing_values(monkeypatch):
    strategy = mp_utils.MPStrategy(
        mode="serial",
        start_method="spawn",
        max_workers=1,
        max_in_flight=1,
        recycle_every=0,
        env={"OMP_NUM_THREADS": "2", "CUSTOM_TEST_ENV": "xyz"},
        use_gpu=False,
        gpu_count=0,
        hard_timeout_supported=False,
        reasons={},
    )
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("CUSTOM_TEST_ENV", raising=False)

    mp_utils.apply_strategy_env(strategy)

    assert os.environ["OMP_NUM_THREADS"] == "2"
    assert os.environ["CUSTOM_TEST_ENV"] == "xyz"


def test_strategy_to_log_dict_contains_dataclass_fields():
    strategy = mp_utils.MPStrategy(
        mode="process_pool",
        start_method="spawn",
        max_workers=2,
        max_in_flight=4,
        recycle_every=10,
        env={"OMP_NUM_THREADS": "1"},
        use_gpu=True,
        gpu_count=2,
        hard_timeout_supported=True,
        reasons={"x": 1},
    )

    data = mp_utils.strategy_to_log_dict(strategy)

    assert data["mode"] == "process_pool"
    assert data["gpu_count"] == 2
    assert data["env"]["OMP_NUM_THREADS"] == "1"
    assert data["reasons"]["x"] == 1
