"""segment.py
=================
Batch‑process a list of 3‑D volumes to obtain masks with a configurable
segmentation backend (default: TotalSegmentator v2).

The module supports both CLI and library usage:

1. reads a CSV containing a ``nifti_path`` column,
2. spawns a multiprocessing pool (``spawn`` context – required for
   PyTorch + CUDA),
3. runs config‑driven segmentation tasks per volume,
4. optionally merges / cleans masks, and
5. writes updated CSVs with output paths and a separate error CSV.
"""

from __future__ import annotations

import argparse
from collections import deque
import copy
import logging
import multiprocessing as mp
import os
import re
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Dict, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, binary_fill_holes
from skimage.measure import label, regionprops
from skimage.morphology import ball
from tqdm import tqdm

from imperandi.utils.misc import report_volumes  # type: ignore
from imperandi.utils.logging import setup_logging
from imperandi.utils.manifest import load_manifest
from imperandi.utils.run_state import (
    atomic_write_csv,
    atomic_write_json,
    compute_args_hash,
    fingerprint_inputs,
    load_state,
    now_epoch,
    state_matches,
)

# -----------------------------------------------------------------------------
# Configuration & logging
# -----------------------------------------------------------------------------
# Path where TotalSegmentator models are cached (edit as needed)
# os.environ.setdefault("TOTALSEG_HOME_DIR", str(Path.home() / ".totalsegmentator_v2"))

DEFAULT_TIMEOUT = 15 * 60  # seconds – hard wall per study inside the pool

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------


def load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]:
    """Return image data, affine matrix and voxel sizes (zoom)."""
    img = nib.load(str(path))
    return img.get_fdata(), img.affine, img.header.get_zooms()


def save_nifti(
    data: np.ndarray, affine: np.ndarray, out_path: Path, *, dtype=np.uint8
) -> None:
    """Write *data* to *out_path* as a NIfTI‑1 file with *dtype*."""
    img = nib.Nifti1Image(data.astype(dtype, copy=False), affine)
    nib.save(img, str(out_path))


def compute_struct_elem(zooms: Tuple[float, ...], radius_mm: float = 5.0) -> np.ndarray:
    """Create a spherical structuring element with *radius_mm* in real units."""
    radii_vox = [max(1, int(round(radius_mm / z))) for z in zooms]
    return ball(max(radii_vox))


# -----------------------------------------------------------------------------
# Mask post‑processing
# -----------------------------------------------------------------------------


def clean_and_merge_masks(
    dir_path: Path,
    mask_files: List[str],
    *,
    output_name: str,
    radius_mm: float = 5.0,
    verbose: bool = False,
    close: bool = True,
    fill_holes: bool = True,
    largest_cc: bool = True,
) -> bool:
    """Merge masks and optionally apply morphological cleanup."""

    masks: Dict[str, np.ndarray] = {}
    ref_affine: np.ndarray | None = None
    voxel_zooms: Tuple[float, ...] | None = None

    for fname in mask_files:
        src = dir_path / fname
        if not src.exists():
            logger.warning("Mask missing – skipping merge: %s", src)
            continue

        data, affine, zooms = load_nifti(src)
        if ref_affine is None:
            ref_affine, voxel_zooms = affine, zooms
        elif not np.allclose(ref_affine, affine):
            logger.error("Affine mismatch for %s – aborting merge.", src.name)
            return False
        masks[fname] = data > 0

    if not masks:
        logger.error("No valid masks found to merge in %s", dir_path)
        return False

    if len({m.shape for m in masks.values()}) > 1:
        logger.error("Mask shape mismatch in %s – aborting merge.", dir_path)
        return False

    merged = np.logical_or.reduce(list(masks.values()))
    if close:
        merged = binary_closing(
            merged, structure=compute_struct_elem(voxel_zooms, radius_mm)
        )
    if fill_holes:
        merged = binary_fill_holes(merged)

    # Keep only the largest connected component (CC)
    if largest_cc:
        labeled, n_cc = label(merged, return_num=True)
        if n_cc > 1:
            largest = max(regionprops(labeled), key=lambda r: r.area)
            merged = labeled == largest.label
            if verbose:
                logger.info(
                    f"{dir_path} : kept largest CC ({largest.area} voxels) out of {n_cc}"
                )
        elif verbose:
            logger.info(f"{dir_path} : single connected component")

    save_nifti(merged, ref_affine, dir_path / output_name)

    # Optional: overwrite the originals with their cleaned‑up version
    for fname, mask in masks.items():
        save_nifti(mask & merged, ref_affine, dir_path / fname)

    return True


# -----------------------------------------------------------------------------
# Segmentation of one 3‑D volume
# -----------------------------------------------------------------------------


class TotalSegmentatorBackend:
    """Thin wrapper for TotalSegmentator to keep dependency optional."""

    def __init__(self) -> None:
        self._ts = None

    def _ensure_imported(self) -> None:
        if self._ts is None:
            from totalsegmentator.python_api import totalsegmentator

            self._ts = totalsegmentator

    def run(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        task: str,
        fast: bool,
        **kwargs: Any,
    ) -> None:
        self._ensure_imported()
        self._ts(
            input=input_path,
            output=output_dir,
            task=task,
            fast=fast,
            quiet=True,
            verbose=False,
            **kwargs,
        )


def _default_segmentation_config() -> Dict[str, Any]:
    return {
        "backend": "totalsegmentator",
        "tasks": [
            {
                "key": "liver",
                "task": "total",
                "output": "liver.nii.gz",
                "extra": {"roi_subset_robust": ["liver"]},
            },
            {
                "key": "liver_tumor",
                "task": "liver_vessels",
                "output": "liver_tumor.nii.gz",
                "extra": {},
            },
        ],
        "postprocess": {
            "merge_keys": ["liver", "liver_tumor"],
            "output": "liver_all.nii.gz",
            "radius_mm": 5.0,
            "largest_cc": True,
            "fill_holes": True,
            "close": True,
        },
    }


def load_segmentation_config(manifest_arg: str | None, *, base_path: Path) -> Dict[str, Any]:
    """Load segmentation config from manifest, falling back to generic manifest."""
    generic_manifest = load_manifest("generic", base_path=base_path)
    generic_segmentation = generic_manifest.get("segmentation") or _default_segmentation_config()

    if not manifest_arg:
        return copy.deepcopy(generic_segmentation)

    manifest = load_manifest(manifest_arg, base_path=base_path)
    manifest_segmentation = manifest.get("segmentation")
    if manifest_segmentation:
        return copy.deepcopy(manifest_segmentation)
    if "tasks" in manifest and "backend" in manifest:
        # Backward compatibility for legacy files that were pure task configs.
        return copy.deepcopy(manifest)
    return copy.deepcopy(generic_segmentation)


def prefetch_totalsegmentator_models(
    tasks_config: Dict[str, Any], *, fast: bool
) -> None:
    """Download required TotalSegmentator weights before multiprocessing."""
    if tasks_config.get("backend", "totalsegmentator") != "totalsegmentator":
        return

    tasks = tasks_config.get("tasks", [])
    if not tasks:
        return

    task_to_id = {
        "total": [291, 292, 293, 294, 295],
        "total_fast": [297, 298],
        "total_mr": [850, 851],
        "total_fast_mr": [852, 853],
        "lung_vessels": [258],
        "cerebral_bleed": [150],
        "hip_implant": [260],
        "pleural_pericard_effusion": [315],
        "body": [299],
        "body_fast": [300],
        "body_mr": [597],
        "body_mr_fast": [598],
        "vertebrae_mr": [756],
        "head_glands_cavities": [775],
        "headneck_bones_vessels": [776],
        "head_muscles": [777],
        "headneck_muscles": [778, 779],
        "liver_vessels": [8],
        "lung_nodules": [913],
        "kidney_cysts": [789],
        "oculomotor_muscles": [351],
        "breasts": [527],
        "ventricle_parts": [552],
        "liver_segments": [570],
        "liver_segments_mr": [576],
        "craniofacial_structures": [115],
        "abdominal_muscles": [952],
        "teeth": [113],
        "trunk_cavities": [343],
        "brain_aneurysm": [615],
        "heartchambers_highres": [301],
        "appendicular_bones": [304],
        "appendicular_bones_mr": [855],
        "tissue_types": [481],
        "tissue_types_mr": [925],
        "tissue_4_types": [485],
        "vertebrae_body": [305],
        "face": [303],
        "face_mr": [856],
        "brain_structures": [409],
        "thigh_shoulder_muscles": [857],
        "thigh_shoulder_muscles_mr": [857],
        "coronary_arteries": [507],
    }

    task_names = {task["task"] for task in tasks if "task" in task}
    if not task_names:
        return

    resolved_tasks: List[str] = []
    for name in sorted(task_names):
        if name == "total" and fast:
            resolved_tasks.append("total_fast")
        elif name == "total_mr" and fast:
            resolved_tasks.append("total_fast_mr")
        elif name == "body" and fast:
            resolved_tasks.append("body_fast")
        elif name == "body_mr" and fast:
            resolved_tasks.append("body_mr_fast")
        else:
            resolved_tasks.append(name)

    missing = [name for name in resolved_tasks if name not in task_to_id]
    if missing:
        logger.warning(
            "Skipping model prefetch for unknown tasks: %s", ", ".join(missing)
        )

    task_ids: List[int] = []
    for name in resolved_tasks:
        ids = task_to_id.get(name)
        if not ids:
            continue
        task_ids.extend(ids)

    if not task_ids:
        return

    from totalsegmentator.python_api import download_pretrained_weights

    logger.info(
        "Prefetching TotalSegmentator models for tasks: %s",
        ", ".join(resolved_tasks),
    )
    for task_id in sorted(set(task_ids)):
        download_pretrained_weights(task_id)


def segment_volume(
    nifti_path: Path,
    output_dir: Path,
    tasks_config: Dict[str, Any],
    *,
    fast: bool,
    verbose: bool = False,
    force: bool = False,
    backend: TotalSegmentatorBackend | None = None,
) -> List[str]:
    """Run segmentation tasks and optional post‐processing."""
    warnings: List[str] = []
    tasks = tasks_config.get("tasks", [])
    if not tasks:
        raise ValueError("No tasks provided in config")

    backend_name = tasks_config.get("backend", "totalsegmentator")
    if backend_name != "totalsegmentator":
        raise ValueError(f"Unsupported backend: {backend_name}")

    backend = backend or TotalSegmentatorBackend()

    for task in tasks:
        task_name = task["task"]
        task_output = task["output"]
        extra = task.get("extra", {})

        # TotalSegmentator can spawn additional saving threads per process
        # (nr_thr_saving defaults to 6). In our multi-process executor this can
        # multiply aggressively and trigger worker instability on long runs.
        # Keep a conservative default unless users explicitly override it.
        extra.setdefault("nr_thr_saving", 2)

        dst = output_dir / task_output
        if dst.exists() and not force:
            if verbose:
                logger.info("Skip %s – file exists", dst)
            continue

        try:
            backend.run(
                input_path=nifti_path,
                output_dir=output_dir,
                task=task_name,
                fast=fast,
                **extra,
            )
        except Exception as exc:
            logger.error(
                "Segmentation failed on %s (%s): %s", nifti_path, task_name, exc
            )
            raise

        if not dst.exists():
            raise RuntimeError(f"Expected mask not produced: {dst}")
        if verbose:
            logger.info("Mask saved at %s", dst)

    postprocess = tasks_config.get("postprocess")
    if not postprocess:
        return warnings

    merge_keys = postprocess.get("merge_keys", [])
    if not merge_keys:
        return warnings

    key_to_output = {task["key"]: task["output"] for task in tasks}
    merge_files = [key_to_output[k] for k in merge_keys if k in key_to_output]
    if not merge_files:
        return warnings

    merged_name = postprocess.get("output", "mask_merged.nii.gz")
    dst = output_dir / merged_name
    if dst.exists() and not force:
        if verbose:
            logger.info("Skip %s – file exists", dst)
        return warnings

    on_failure = str(postprocess.get("on_failure", "warn_only")).strip().lower()
    if on_failure not in {"warn_only", "fail"}:
        raise ValueError(
            f"Invalid postprocess.on_failure='{on_failure}'. Use 'warn_only' or 'fail'."
        )

    merged_ok = clean_and_merge_masks(
        output_dir,
        merge_files,
        output_name=merged_name,
        radius_mm=float(postprocess.get("radius_mm", 5.0)),
        verbose=verbose,
        close=bool(postprocess.get("close", True)),
        fill_holes=bool(postprocess.get("fill_holes", True)),
        largest_cc=bool(postprocess.get("largest_cc", True)),
    )
    if not merged_ok or not dst.exists():
        message = f"Postprocess merge did not produce expected output: {dst}"
        if on_failure == "fail":
            raise RuntimeError(message)
        logger.warning(message)
        warnings.append(message)

    return warnings


# -----------------------------------------------------------------------------
# Worker wrapper (called in pool)
# -----------------------------------------------------------------------------


def process_single_volume(
    idx: int,
    row: Dict[str, Any],  # must be JSON‑serialisable
    tasks_config: Dict[str, Any],
    *,
    fast: bool,
    verbose: bool,
    force: bool,
    backend: TotalSegmentatorBackend | None = None,
) -> Tuple[int, str | None, str | None, str | None]:
    """Return ``(idx, output_dir|None, error_msg|None, warning_msg|None)``."""

    setup_logging(verbose=verbose)

    try:
        nifti_path = Path(row["nifti_path"])
    except KeyError:
        return idx, None, "column 'nifti_path' missing", None

    if not nifti_path.exists():
        return idx, None, "file not found", None

    try:
        warnings = segment_volume(
            nifti_path,
            nifti_path.parent,
            tasks_config,
            fast=fast,
            verbose=verbose,
            force=force,
            backend=backend,
        )
        warning_msg = " | ".join(warnings) if warnings else None
        return idx, str(nifti_path.parent), None, warning_msg
    except Exception as exc:
        # Capture full traceback for later debugging
        logger.debug("Traceback for %s:\n%s", nifti_path.name, traceback.format_exc())
        return idx, None, str(exc), None


# -----------------------------------------------------------------------------
# GPU worker pinning helpers
# -----------------------------------------------------------------------------


def _resolve_visible_gpu_tokens(gpu_count: int) -> List[str]:
    """
    Return GPU tokens suitable for CUDA_VISIBLE_DEVICES assignment.

    If CUDA_VISIBLE_DEVICES is already set (e.g. "2,3"), preserve those
    logical tokens. Otherwise, default to "0..gpu_count-1".
    """
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is not None:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if parts:
            return parts
    return [str(i) for i in range(max(0, int(gpu_count)))]


def _worker_gpu_initializer(gpu_tokens: List[str]) -> None:
    """
    Pin each worker process to a single GPU token.

    Mapping is deterministic per worker slot:
      worker_slot -> gpu_tokens[worker_slot % len(gpu_tokens)]
    """
    if not gpu_tokens:
        return

    proc = mp.current_process()
    slot_idx: int | None = None

    identity = getattr(proc, "_identity", None)
    if identity:
        try:
            slot_idx = int(identity[0]) - 1
        except Exception:
            slot_idx = None

    if slot_idx is None:
        m = re.search(r"(\d+)$", proc.name or "")
        if m:
            slot_idx = int(m.group(1)) - 1

    if slot_idx is None:
        slot_idx = 0

    token = gpu_tokens[slot_idx % len(gpu_tokens)]
    os.environ["CUDA_VISIBLE_DEVICES"] = token


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------


def add_segment_arguments(
    parser: argparse.ArgumentParser,
    include_manifest: bool = True,
    include_dry_run: bool = True,
) -> None:
    parser.add_argument(
        "csv_path_pos",
        nargs="?",
        type=str,
        default=None,
        help="Path to the input CSV file. Defaults to ./nifti_index.csv.",
    )
    parser.add_argument(
        "--csv_path",
        dest="csv_path_opt",
        type=str,
    )
    parser.add_argument(
        "--csv_path_out",
        type=str,
        required=False,
        default=None,
        help="Output CSV (default: overwrite input).",
    )
    parser.add_argument(
        "--error_csv_path",
        type=str,
        default=None,
        help="CSV for failures only (default: alongside input CSV).",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Pool size")
    parser.add_argument(
        "--fast", action="store_true", help="Use TotalSegmentator fast mode"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re‑run even if output masks already exist",
    )
    parser.add_argument(
        "--start_method",
        choices=["spawn", "fork", "forkserver"],
        default="spawn",
        help="multiprocessing start method: spawn=robust, fork=faster (Linux)",
    )
    parser.add_argument(
        "--timeout_sec",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Per-volume timeout in seconds",
    )
    parser.add_argument(
        "--checkpoint_every_rows",
        type=int,
        default=25,
        help="Flush checkpoint files every N processed rows.",
    )
    parser.add_argument(
        "--checkpoint_every_sec",
        type=int,
        default=30,
        help="Flush checkpoint files every T seconds.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from matching checkpoint state if available.",
    )
    parser.add_argument(
        "--strict_resume",
        action="store_true",
        default=False,
        help="Use content hashing for input fingerprint when resuming.",
    )
    parser.add_argument(
        "--state_path",
        type=str,
        default=None,
        help="Optional path for run state JSON.",
    )
    if include_manifest:
        parser.add_argument(
            "--manifest",
            type=str,
            default=None,
            help="Dataset manifest name or path to manifest JSON.",
        )
    if include_dry_run:
        parser.add_argument(
            "--dry-run",
            dest="dry_run",
            action="store_true",
            default=False,
            help="Print planned actions without running.",
        )


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch segmentation with TotalSegmentator v2",
        add_help=add_help,
    )
    add_segment_arguments(parser)
    return parser


def normalize_segment_args(args: argparse.Namespace) -> argparse.Namespace:
    csv_in = args.csv_path_opt if args.csv_path_opt is not None else args.csv_path_pos

    if csv_in is None:
        csv_path = Path.cwd() / "nifti_index.csv"
    else:
        csv_path = Path(csv_in)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    args.csv_path = str(csv_path.resolve())

    if not args.csv_path_out:
        args.csv_path_out = args.csv_path
    else:
        args.csv_path_out = str(Path(args.csv_path_out))

    if args.error_csv_path:
        args.error_csv_path = str(Path(args.error_csv_path))
    else:
        args.error_csv_path = str(Path(args.csv_path).parent / "seg_errors.csv")

    del args.csv_path_pos
    del args.csv_path_opt

    return args



def main(args: argparse.Namespace) -> None:
    setup_logging(verbose=getattr(args, "verbose", False))
    tasks_config = load_segmentation_config(
        getattr(args, "manifest", None),
        base_path=Path(__file__).resolve().parents[1],
    )
    prefetch_totalsegmentator_models(tasks_config, fast=args.fast)

    output_path = Path(args.csv_path_out)
    error_path = Path(args.error_csv_path)
    state_path = (
        Path(args.state_path)
        if getattr(args, "state_path", None)
        else output_path.parent / f"{output_path.stem}.segment.state.json"
    )
    checkpoint_main_path = output_path.parent / f"{output_path.stem}.segment.checkpoint.csv"
    checkpoint_err_path = error_path.parent / f"{error_path.stem}.segment.checkpoint.csv"

    exclude_hash_args = {
        "csv_path_out",
        "error_csv_path",
        "dry_run",
        "verbose",
        "resume",
        "state_path",
        "checkpoint_every_rows",
        "checkpoint_every_sec",
        "strict_resume",
    }
    args_hash = compute_args_hash(args, exclude_keys=exclude_hash_args)
    input_fp = fingerprint_inputs(
        args.csv_path, strict=bool(getattr(args, "strict_resume", False))
    )
    state = load_state(state_path)
    can_resume = bool(getattr(args, "resume", False)) and state_matches(
        state,
        command="segment",
        args_hash=args_hash,
        input_fingerprint=input_fp,
    )

    from imperandi.utils.multiprocessing import (
        apply_strategy_env,
        strategy_to_log_dict,
        decide_multiprocessing_strategy
    )
    # Decide
    strategy = decide_multiprocessing_strategy(
        prefer_gpu=True,
        requested_workers=args.num_workers,
        start_method_hint=args.start_method,
        target_task_mem_mb=3000,      # tune (TotalSegmentator can be heavy)
        need_hard_timeouts=True,
    )

    logger.info("MP strategy: %s", strategy_to_log_dict(strategy))
    effective_workers = strategy.max_workers
    effective_start_method = strategy.start_method
    effective_timeout = args.timeout_sec
    logger.info(
        "Requested MP settings: workers=%d start_method=%s timeout_sec=%d | Effective: mode=%s workers=%d start_method=%s max_in_flight=%d recycle_every=%d",
        args.num_workers,
        args.start_method,
        args.timeout_sec,
        strategy.mode,
        effective_workers,
        effective_start_method,
        strategy.max_in_flight,
        strategy.recycle_every,
    )

    # Apply env caps BEFORE pool creation
    apply_strategy_env(strategy)


    # --- read and pre‑clean CSV ------------------------------------------------
    if can_resume and checkpoint_main_path.exists():
        logger.info("Resuming segment from checkpoint: %s", checkpoint_main_path)
        df = pd.read_csv(checkpoint_main_path).copy()
    else:
        df = pd.read_csv(args.csv_path).copy()
    if "nifti_path" not in df.columns:
        unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
        if unnamed:
            df = df.drop(columns=unnamed)
    if "nifti_path" not in df.columns:
        raise KeyError("column 'nifti_path' missing")
    df = df.drop_duplicates("nifti_path").copy()
    for task in tasks_config.get("tasks", []):
        df[f"mask_{task['key']}"] = None
    if tasks_config.get("postprocess"):
        df["mask_merged"] = None
    df["warning_message"] = None

    completed_indices: set[int] = set()
    if can_resume:
        completed_indices = {
            int(i) for i in (state or {}).get("completed_indices", []) if isinstance(i, int)
        }
        logger.info("Resume enabled: %d completed rows restored from state", len(completed_indices))

    errors_by_idx: Dict[int, str] = {}
    if can_resume and checkpoint_err_path.exists():
        err_ckpt = pd.read_csv(checkpoint_err_path)
        if "idx" in err_ckpt.columns and "error_message" in err_ckpt.columns:
            for _, row in err_ckpt.iterrows():
                try:
                    errors_by_idx[int(row["idx"])] = str(row["error_message"])
                except Exception:
                    continue

    checkpoint_every_rows = max(1, int(getattr(args, "checkpoint_every_rows", 25)))
    checkpoint_every_sec = max(1, int(getattr(args, "checkpoint_every_sec", 30)))
    last_checkpoint_time = now_epoch()
    processed_since_checkpoint = 0

    def _checkpoint_write(*, force: bool = False) -> None:
        nonlocal last_checkpoint_time, processed_since_checkpoint
        elapsed = now_epoch() - last_checkpoint_time
        if not force and processed_since_checkpoint < checkpoint_every_rows and elapsed < checkpoint_every_sec:
            return

        atomic_write_csv(df, checkpoint_main_path, index=False)
        if errors_by_idx:
            err_ckpt_df = pd.DataFrame(
                [{"idx": k, "error_message": v} for k, v in sorted(errors_by_idx.items())]
            )
            atomic_write_csv(err_ckpt_df, checkpoint_err_path, index=False)
        elif checkpoint_err_path.exists():
            checkpoint_err_path.unlink()
        atomic_write_json(
            state_path,
            {
                "command": "segment",
                "args_hash": args_hash,
                "input_fingerprint": input_fp,
                "completed_indices": sorted(completed_indices),
                "updated_at_epoch": now_epoch(),
            },
        )
        last_checkpoint_time = now_epoch()
        processed_since_checkpoint = 0

    def _apply_result(
        idx: int, out_dir: str | None, err_msg: str | None, warning_msg: str | None
    ) -> None:
        nonlocal processed_since_checkpoint
        completed_indices.add(int(idx))
        processed_since_checkpoint += 1

        if out_dir:
            base = Path(out_dir)
            row_warnings: List[str] = []
            for task in tasks_config.get("tasks", []):
                mask_path = base / task["output"]
                if mask_path.exists():
                    df.at[idx, f"mask_{task['key']}"] = str(mask_path)
                else:
                    row_warnings.append(f"missing mask: {mask_path}")
            if tasks_config.get("postprocess"):
                merged_name = tasks_config["postprocess"].get(
                    "output", "mask_merged.nii.gz"
                )
                merged_path = base / merged_name
                if merged_path.exists():
                    df.at[idx, "mask_merged"] = str(merged_path)
                else:
                    row_warnings.append(f"missing merged mask: {merged_path}")
            if warning_msg:
                row_warnings.append(warning_msg)
            if row_warnings:
                df.at[idx, "warning_message"] = " | ".join(row_warnings)
            if idx in errors_by_idx:
                del errors_by_idx[idx]
        else:
            errors_by_idx[idx] = err_msg or "unknown"

        _checkpoint_write(force=False)

    # --- spawn multiprocessing pool -------------------------------------------
    try:
        ctx = mp.get_context(
            effective_start_method
        )  # 'spawn' required for torch / CUDA stability
    except ValueError:
        available = mp.get_all_start_methods()
        fallback = "spawn" if "spawn" in available else available[0]
        logger.warning(
            "Unsupported start_method=%r on this platform; falling back to %r",
            effective_start_method,
            fallback,
        )
        ctx = mp.get_context(fallback)

    def _broken_pool_message(exc: BaseException) -> str:
        return (
            "BrokenProcessPool: likely worker process died unexpectedly "
            f"({type(exc).__name__}: {exc})"
        )

    def _is_retryable(err_msg: str | None) -> bool:
        if not err_msg:
            return False
        low = err_msg.lower()
        return "worker crash" in low or "brokenprocesspool" in low

    gpu_tokens: List[str] = []
    pool_init_kwargs: Dict[str, Any] = {}
    if strategy.use_gpu and strategy.gpu_count > 0:
        gpu_tokens = _resolve_visible_gpu_tokens(strategy.gpu_count)
        if gpu_tokens:
            pool_init_kwargs = {
                "initializer": _worker_gpu_initializer,
                "initargs": (gpu_tokens,),
            }
            logger.info(
                "GPU worker pinning enabled: %d worker(s) across %d visible GPU token(s)",
                effective_workers,
                len(gpu_tokens),
            )

    def _run_rows(
        row_indices: List[int],
        *,
        progress_bar: tqdm | None = None,
        on_result: Any | None = None,
    ) -> Dict[int, Tuple[int, str | None, str | None, str | None]]:
        out: Dict[int, Tuple[int, str | None, str | None, str | None]] = {}
        if not row_indices:
            return out

        max_in_flight = max(1, int(strategy.max_in_flight))
        row_queue = deque(row_indices)
        broken_pool = False

        def _record_result(
            idx: int, result: Tuple[int, str | None, str | None, str | None]
        ) -> None:
            out[idx] = result
            if on_result is not None:
                on_result(*result)
            if progress_bar is not None:
                progress_bar.update(1)

        def _create_pool() -> ProcessPoolExecutor:
            try:
                return ProcessPoolExecutor(
                    max_workers=effective_workers,
                    mp_context=ctx,
                    **pool_init_kwargs,
                )
            except TypeError as exc:
                if pool_init_kwargs:
                    logger.warning(
                        "Executor does not support worker initializer; running without GPU pinning (%s)",
                        exc,
                    )
                return ProcessPoolExecutor(max_workers=effective_workers, mp_context=ctx)

        def _shutdown_pool(pool: ProcessPoolExecutor, force: bool) -> None:
            if force:
                # Force-kill orphaned workers after crash/timeout.
                pool.shutdown(wait=False, cancel_futures=True)
                processes = getattr(pool, "_processes", None)
                if processes:
                    for p in processes.values():
                        if p.is_alive():
                            p.kill()
                return
            # Graceful shutdown prevents collateral damage to interpreter state.
            pool.shutdown(wait=True, cancel_futures=False)

        while row_queue and not broken_pool:
            pool = _create_pool()

            futures: Dict[Any, int] = {}
            submit_started_at: Dict[Any, float] = {}
            restart_pool_for_timeout = False
            broken_pool_msg: str | None = None

            def _submit_until_limit() -> None:
                while row_queue and len(futures) < max_in_flight:
                    idx = row_queue.popleft()
                    fut = pool.submit(
                        process_single_volume,
                        idx,
                        df.loc[idx].to_dict(),
                        tasks_config,
                        fast=args.fast,
                        verbose=args.verbose,
                        force=args.force,
                    )
                    futures[fut] = idx
                    submit_started_at[fut] = time.monotonic()

            def _expire_timed_out_futures() -> bool:
                now = time.monotonic()
                timed_out_futures: List[Any] = []
                for fut, i in list(futures.items()):
                    started_at = submit_started_at[fut]
                    if now - started_at < effective_timeout:
                        continue
                    timed_out_futures.append(fut)
                    logger.warning(
                        "Row %d exceeded %ds wall time (elapsed %.1fs) – recycling worker pool",
                        i,
                        effective_timeout,
                        now - started_at,
                    )
                    _record_result(i, (i, None, f"timeout after {effective_timeout}s", None))

                if not timed_out_futures:
                    return False

                for timed_out in timed_out_futures:
                    futures.pop(timed_out)
                    submit_started_at.pop(timed_out, None)
                    timed_out.cancel()
                # Re-queue remaining in-flight rows to retry in a fresh pool.
                for pending_fut, pending_idx in list(futures.items()):
                    row_queue.appendleft(pending_idx)
                    pending_fut.cancel()
                futures.clear()
                submit_started_at.clear()
                return True

            def _collect_completed_nonblocking() -> bool:
                nonlocal broken_pool, broken_pool_msg
                completed_any = False
                for fut, i in list(futures.items()):
                    try:
                        res = fut.result(timeout=0)
                    except TimeoutError:
                        continue
                    except BrokenProcessPool as exc:
                        broken_pool = True
                        broken_pool_msg = _broken_pool_message(exc)
                        logger.error(
                            "BrokenProcessPool while collecting row %d; aborting fast: %s",
                            i,
                            exc,
                        )
                        break
                    except Exception as exc:
                        res = (i, None, f"worker crash: {type(exc).__name__}: {exc}", None)

                    futures.pop(fut, None)
                    submit_started_at.pop(fut, None)
                    _record_result(i, res)
                    _submit_until_limit()
                    completed_any = True
                return completed_any

            try:
                _submit_until_limit()
                while futures and not restart_pool_for_timeout and not broken_pool:
                    if _expire_timed_out_futures():
                        restart_pool_for_timeout = True
                        break

                    if broken_pool:
                        break
                    if not _collect_completed_nonblocking():
                        time.sleep(0.05)
            except BrokenProcessPool as exc:
                broken_pool = True
                broken_pool_msg = _broken_pool_message(exc)
                logger.error("BrokenProcessPool during completion loop; aborting fast: %s", exc)
            finally:
                _shutdown_pool(pool, force=(broken_pool or restart_pool_for_timeout))

            if broken_pool:
                msg = broken_pool_msg or "BrokenProcessPool"
                for i in list(futures.values()):
                    out[i] = (i, None, msg, None)
                for i in row_queue:
                    out[i] = (i, None, msg, None)
                break

        return out

    def _run_rows_with_recycling(
        row_indices: List[int],
        *,
        progress_bar: tqdm | None = None,
        on_result: Any | None = None,
    ) -> Dict[int, Tuple[int, str | None, str | None, str | None]]:
        if strategy.recycle_every <= 0:
            return _run_rows(row_indices, progress_bar=progress_bar, on_result=on_result)

        out: Dict[int, Tuple[int, str | None, str | None, str | None]] = {}
        chunk_size = max(1, int(strategy.recycle_every))
        for start in range(0, len(row_indices), chunk_size):
            chunk = row_indices[start : start + chunk_size]
            out.update(_run_rows(chunk, progress_bar=progress_bar, on_result=on_result))
        return out

    row_indices = [i for i in list(df.index) if i not in completed_indices]
    run_serial = strategy.mode == "serial" or effective_workers <= 1
    if strategy.mode == "subprocess_per_case":
        # logger.warning(
        #     "Strategy selected mode='subprocess_per_case', but this mode is deferred in segment; falling back to serial execution for now."
        # )
        # run_serial = True
        pass

    if run_serial:
        logger.info(
            "Running segmentation in single-worker mode (no multiprocessing pool)"
        )
        results_by_idx = {}
        for idx in tqdm(row_indices, total=len(row_indices), desc="Segment"):
            result = process_single_volume(
                idx,
                df.loc[idx].to_dict(),
                tasks_config,
                fast=args.fast,
                verbose=args.verbose,
                force=args.force,
            )
            results_by_idx[idx] = result
            _apply_result(*result)
    else:
        with tqdm(total=len(row_indices), desc="Segment") as progress_bar:
            results_by_idx = _run_rows_with_recycling(
                row_indices,
                progress_bar=progress_bar,
                on_result=_apply_result,
            )

            retry_indices = [
                i
                for i in row_indices
                if _is_retryable(results_by_idx.get(i, (i, None, None, None))[2])
            ]
            if retry_indices:
                logger.warning(
                    "Retrying %d row(s) in a fresh executor after worker crash/BrokenProcessPool",
                    len(retry_indices),
                )
                progress_bar.total = (progress_bar.total or 0) + len(retry_indices)
                progress_bar.refresh()
                retry_results = _run_rows_with_recycling(
                    retry_indices,
                    progress_bar=progress_bar,
                    on_result=_apply_result,
                )
                results_by_idx.update(retry_results)

    results: List[Tuple[int, str | None, str | None, str | None]] = [
        results_by_idx[i] for i in row_indices if i in results_by_idx
    ]

    _checkpoint_write(force=True)

    # --- write output tables ---------------------------------------------------
    atomic_write_csv(df, args.csv_path_out, index=False)
    logger.info("Wrote main table → %s", args.csv_path_out)

    if errors_by_idx:
        err_idx = sorted(errors_by_idx.keys())
        err_df = df.loc[err_idx].copy()
        err_df["error_message"] = [errors_by_idx[i] for i in err_idx]
        atomic_write_csv(err_df, args.error_csv_path, index=False)
        logger.warning("%d rows failed – see %s", len(err_df), args.error_csv_path)

        # Optional project‑specific volume report
        try:
            report_volumes(err_df)
        except Exception:
            logger.debug("report_volumes() failed – continuing")

    logger.info("Segmentation done ✔")
    atomic_write_json(
        state_path,
        {
            "command": "segment",
            "args_hash": args_hash,
            "input_fingerprint": input_fp,
            "completed_indices": sorted(completed_indices),
            "updated_at_epoch": now_epoch(),
            "finished": True,
        },
    )

    print("END: active_children:", mp.active_children(), flush=True)
    print("END: threads:", [t.name for t in threading.enumerate()], flush=True)
    time.sleep(2)
    print("END2: active_children:", mp.active_children(), flush=True)

    return


if __name__ == "__main__":
    setup_logging()
    args = build_parser().parse_args()
    args = normalize_segment_args(args)
    if getattr(args, "dry_run", False):
        logger.info("Dry run: segment")
        logger.info("%s", args)
        raise SystemExit(0)
    main(args)
