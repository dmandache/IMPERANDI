from __future__ import annotations

import os
import re
import math
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class MPStrategy:
    # High-level decision
    mode: str  # "serial", "process_pool", "subprocess_per_case"
    # ProcessPoolExecutor knobs
    start_method: str  # "spawn" | "forkserver" | "fork"
    max_workers: int
    max_in_flight: int
    recycle_every: int  # recreate executor every N tasks (0 disables)
    # Env/threading knobs (apply before spawning)
    env: Dict[str, str]
    # CUDA/GPU
    use_gpu: bool
    gpu_count: int
    # Timeouts
    hard_timeout_supported: bool  # if True, prefer subprocess_per_case for real kill
    # Diagnostics
    reasons: Dict[str, Any]


def _read_int(path: str) -> Optional[int]:
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except Exception:
        return None


def _slurm_cpus() -> Optional[int]:
    # Slurm commonly sets one of these
    for k in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        v = os.environ.get(k)
        if v and v.isdigit():
            return int(v)
    return None


def _slurm_mem_mb() -> Optional[int]:
    """
    Try to infer memory limit from Slurm env vars.
    Returns MB if possible, else None.
    """
    v = os.environ.get("SLURM_MEM_PER_NODE")
    if v and v.isdigit():
        return int(v)  # already MB
    v = os.environ.get("SLURM_MEM_PER_CPU")
    if v and v.isdigit():
        cpus = _slurm_cpus() or os.cpu_count() or 1
        return int(v) * cpus  # MB
    return None


def _cgroup_mem_limit_mb() -> Optional[int]:
    """
    cgroup v2: /sys/fs/cgroup/memory.max
    cgroup v1: /sys/fs/cgroup/memory/memory.limit_in_bytes
    """
    # v2
    v2 = "/sys/fs/cgroup/memory.max"
    if os.path.exists(v2):
        try:
            s = open(v2, "r").read().strip()
            if s != "max":
                return int(int(s) / (1024 * 1024))
        except Exception:
            pass

    # v1
    v1 = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    if os.path.exists(v1):
        try:
            b = int(open(v1, "r").read().strip())
            # Some systems use a huge number for "no limit"
            if b > 0 and b < 1 << 60:
                return int(b / (1024 * 1024))
        except Exception:
            pass

    return None


def _ram_total_mb() -> int:
    # Prefer cgroup limit (containers / slurm), then slurm env, then psutil-like fallback.
    cg = _cgroup_mem_limit_mb()
    if cg:
        return cg
    sl = _slurm_mem_mb()
    if sl:
        return sl
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size / (1024 * 1024))
    except Exception:
        # last resort
        return 0


def _visible_cuda_devices_from_env() -> Optional[int]:
    """
    If CUDA_VISIBLE_DEVICES is set, count entries.
    Handles values like "0,1,2" or UUIDs, and also empty/None.
    """
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v is None:
        return None
    v = v.strip()
    if v == "" or v.lower() in {"none", "no", "-1"}:
        return 0
    # Split on commas
    parts = [p.strip() for p in v.split(",") if p.strip() != ""]
    return len(parts)


def _nvidia_smi_gpu_count() -> Optional[int]:
    """
    Best-effort GPU count via nvidia-smi. Returns None if not available.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-L"], stderr=subprocess.STDOUT, text=True, timeout=2
        )
        # Lines like: "GPU 0: ..."
        gpus = [ln for ln in out.splitlines() if ln.strip().startswith("GPU ")]
        return len(gpus)
    except Exception:
        return None


def _torch_gpu_count() -> Optional[int]:
    """
    Try torch if available (won't crash if torch isn't installed).
    """
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return None


def decide_multiprocessing_strategy(
    *,
    prefer_gpu: bool = True,
    requested_workers: Optional[int] = None,
    start_method_hint: str = "spawn",
    target_task_mem_mb: int = 2500,
    reserve_mem_frac: float = 0.30,
    max_workers_cap: int = 32,
    enable_recycling: bool = True,
    recycle_every: int = 25,
    max_in_flight_factor: int = 2,
    need_hard_timeouts: bool = True,
) -> MPStrategy:
    """
    Decide a robust multiprocessing strategy for mixed CPU/GPU medical imaging pipelines.

    Key principles (good defaults for TotalSegmentator/nnUNet-like workloads):
      - Never run >1 GPU inference worker per *physical GPU*.
      - On 1 GPU, keep GPU inference serial (max_workers=1).
      - Cap workers by memory (RAM) and CPUs.
      - Limit BLAS/OMP/ITK threads to avoid oversubscription.
      - If you truly need hard per-case timeouts, prefer subprocess-per-case.

    Parameters
    ----------
    prefer_gpu:
        If True, we assume the workload may use CUDA when available.
    requested_workers:
        User request; we will clamp to safe values.
    start_method_hint:
        "spawn" is safest with torch/CUDA.
    target_task_mem_mb:
        Approx RAM consumption per *process* (tune to your workload).
    reserve_mem_frac:
        Fraction of total RAM to reserve for OS / cache / peak buffers.
    enable_recycling:
        If True, recommend recycling executor every N tasks to mitigate leaks/fragmentation.
    recycle_every:
        Suggested recycle period (tasks).
    max_in_flight_factor:
        Bound pending futures to max_workers * factor.
    need_hard_timeouts:
        If True, we consider "subprocess_per_case" for real kill-on-timeout.

    Returns
    -------
    MPStrategy
        A strategy object with recommended settings and environment variables.
    """
    reasons: Dict[str, Any] = {}

    # CPUs
    cpu_count = _slurm_cpus() or (os.cpu_count() or 1)
    reasons["cpu_count"] = cpu_count

    # RAM
    ram_mb = _ram_total_mb()
    reasons["ram_mb"] = ram_mb

    # GPU count: prefer CUDA_VISIBLE_DEVICES -> torch -> nvidia-smi
    env_visible = _visible_cuda_devices_from_env()
    torch_cnt = _torch_gpu_count()
    smi_cnt = _nvidia_smi_gpu_count()

    # Determine gpu_count with precedence:
    # If CUDA_VISIBLE_DEVICES is set, trust that as "visible" GPUs.
    if env_visible is not None:
        gpu_count = env_visible
        reasons["gpu_count_source"] = "CUDA_VISIBLE_DEVICES"
    elif torch_cnt is not None:
        gpu_count = torch_cnt
        reasons["gpu_count_source"] = "torch"
    elif smi_cnt is not None:
        gpu_count = smi_cnt
        reasons["gpu_count_source"] = "nvidia-smi"
    else:
        gpu_count = 0
        reasons["gpu_count_source"] = "none"

    use_gpu = bool(prefer_gpu and gpu_count > 0)
    reasons["use_gpu"] = use_gpu
    reasons["gpu_count"] = gpu_count

    # Memory-based cap
    if ram_mb and ram_mb > 0:
        usable = int(ram_mb * (1.0 - reserve_mem_frac))
        mem_cap = max(1, usable // max(1, target_task_mem_mb))
    else:
        # unknown RAM; be conservative
        mem_cap = 2
    reasons["mem_cap_workers"] = mem_cap

    # CPU-based cap (leave 1 core for coordination)
    cpu_cap = max(1, cpu_count - 1)
    reasons["cpu_cap_workers"] = cpu_cap

    # GPU-based cap: 1 worker per GPU (robust default)
    gpu_cap = gpu_count if use_gpu else max_workers_cap
    # For 1 GPU: explicitly enforce 1 worker to avoid multi-process CUDA fights
    if use_gpu and gpu_count == 1:
        gpu_cap = 1
        reasons["gpu_cap_note"] = "single GPU -> force 1 worker to prevent CUDA OOM/fragmentation"

    # Combine caps
    cap = min(mem_cap, cpu_cap, gpu_cap, max_workers_cap)
    reasons["combined_cap"] = cap

    # Apply requested_workers
    if requested_workers is None:
        max_workers = cap
        reasons["requested_workers"] = None
        reasons["workers_reason"] = "auto"
    else:
        max_workers = max(1, min(int(requested_workers), cap))
        reasons["requested_workers"] = requested_workers
        reasons["workers_reason"] = "clamped_to_safe_cap"

    # Decide mode
    # If we need hard timeouts, subprocess-per-case is the only truly reliable kill mechanism.
    # But it's slower; we recommend it mainly when GPU is involved and instability/timeouts are expected.
    if use_gpu and need_hard_timeouts:
        mode = "subprocess_per_case" if max_workers == 1 else "process_pool"
        hard_timeout_supported = True
        reasons["mode_reason"] = (
            "GPU detected + hard timeouts requested: subprocess-per-case recommended for real kill"
            if mode == "subprocess_per_case"
            else "GPU detected: process_pool limited to <= GPUs"
        )
    else:
        mode = "process_pool" if max_workers > 1 else "serial"
        hard_timeout_supported = False
        reasons["mode_reason"] = "CPU-only or no hard timeouts: use pool when workers>1"

    # Start method
    start_method = start_method_hint
    if use_gpu:
        # For torch/CUDA, spawn is usually safest across environments
        start_method = "spawn"
        reasons["start_method_reason"] = "CUDA workload -> spawn for stability"
    else:
        # On Linux, forkserver can be a good compromise
        if start_method_hint not in {"spawn", "forkserver", "fork"}:
            start_method = "spawn"
        reasons["start_method_reason"] = "non-CUDA -> respect hint if valid"

    # Threads/env knobs: prevent oversubscription & reduce memory spikes
    env = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1",
    }
    # If you want PyTorch to not spawn tons of CPU threads in preprocessing
    env["TORCH_NUM_THREADS"] = "1"

    # Executor recycling (helps leaks/fragmentation, especially after long runs)
    if enable_recycling and mode == "process_pool":
        rec = max(1, int(recycle_every))
    else:
        rec = 0

    # In-flight futures bound
    max_in_flight = max(1, int(max_workers) * max(1, int(max_in_flight_factor)))

    # If mode is serial/subprocess_per_case, these are mostly irrelevant but keep consistent.
    if mode in {"serial", "subprocess_per_case"}:
        max_in_flight = 1
        if mode == "serial":
            max_workers = 1

    return MPStrategy(
        mode=mode,
        start_method=start_method,
        max_workers=max_workers,
        max_in_flight=max_in_flight,
        recycle_every=rec,
        env=env,
        use_gpu=use_gpu,
        gpu_count=gpu_count,
        hard_timeout_supported=hard_timeout_supported,
        reasons=reasons,
    )


def apply_strategy_env(strategy: MPStrategy) -> None:
    """
    Apply the environment variables recommended by decide_multiprocessing_strategy().
    Call this *before* creating any pools/executors and ideally before importing
    heavy numeric stacks if you can.
    """
    for k, v in strategy.env.items():
        os.environ.setdefault(k, v)


def strategy_to_log_dict(strategy: MPStrategy) -> Dict[str, Any]:
    """Convenience for structured logging."""
    d = asdict(strategy)
    # env can be large; keep it but it's fine for structured logs
    return d