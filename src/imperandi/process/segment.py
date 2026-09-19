"""segment.py
=================
Batch‑process a list of 3‑D volumes to obtain masks with a configurable
segmentation backend (default: TotalSegmentator v2).

The module supports both CLI and library usage:

1. reads a CSV containing a ``nifti_path`` column,
2. spawns a multiprocessing pool (``spawn`` context – required for
   PyTorch + CUDA),
3. runs config‑driven segmentation tasks per volume,
4. optionally runs sequential logical / morphological mask operations, and
5. writes updated CSVs with output paths and a separate error CSV.
"""

from __future__ import annotations

import argparse
from ast import literal_eval
from collections import deque
import copy
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as distribution_version
import logging
import multiprocessing as mp
from numbers import Real
import os
from queue import Empty, Queue
import re
import sys
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
from tqdm import tqdm

from imperandi.process.mask_operations import (
    MaskGeometryError,
    apply_operations,
    operation_sources,
    operations_are_current,
    validate_operations,
)
from imperandi.utils.misc import report_volumes  # type: ignore
from imperandi.utils.logging import log_task_summary, setup_logging
from imperandi.utils.manifest import load_manifest
from imperandi.utils.checkpoint_cli import add_checkpoint_arguments
from imperandi.utils.run_state import (
    log_finished_resume_summary,
    atomic_write_csv,
    CheckpointManager,
    ensure_source_id_column,
    fingerprint_inputs,
    merge_with_existing_output,
    normalize_source_id,
    normalize_source_ids,
    prepare_resume_context,
    source_id_resume_signature,
)

# -----------------------------------------------------------------------------
# Configuration & logging
# -----------------------------------------------------------------------------
# Path where TotalSegmentator models are cached (edit as needed)
# os.environ.setdefault("TOTALSEG_HOME_DIR", str(Path.home() / ".totalsegmentator_v2"))

DEFAULT_TIMEOUT = 30 * 60  # seconds - actual worker execution time per study
DEFAULT_CHECKPOINT_EVERY_ROWS = 50
DEFAULT_CHECKPOINT_EVERY_SEC = 5 * 60  # seconds
DEFAULT_CROP_MARGIN_MM = 50.0

LIVER_LESIONS_MIN_TOTALSEGMENTATOR_VERSION = "2.13.0"
LIVER_LESIONS_TASKS = frozenset({"liver_lesions", "liver_lesions_mr"})

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerEvent:
    """Lifecycle signal emitted by a pool worker for one submitted row."""

    row_idx: int
    pid: int
    event: str
    timestamp: float
    generation: int
    gpu: str | None = None


_WORKER_EVENT_QUEUE: Any | None = None
_WORKER_GENERATION = 0

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
        **kwargs: Any,
    ) -> None:
        self._ensure_imported()
        self._ts(
            input=input_path,
            output=output_dir,
            task=task,
            quiet=True,
            verbose=False,
            **kwargs,
        )


def _default_segmentation_config() -> Dict[str, Any]:
    return {
        "backend": "totalsegmentator",
        "crop": False,
        "crop_margin_mm": DEFAULT_CROP_MARGIN_MM,
        "modalities": {
            "CT": {
                "tasks": [
                    {
                        "task": "total",
                        "extra": {
                            "roi_subset_robust": ["liver"],
                            "fastest": True,
                        },
                    },
                    {
                        "task": "liver_lesions",
                        "output": "liver_tumor",
                        "fetch_output": "liver_lesions",
                        "extra": {},
                    },
                ],
                "postprocess": {
                    "on_failure": "warn_only",
                    "operations": [
                        {
                            "op": "union",
                            "inputs": ["liver", "liver_tumor"],
                            "output": "liver_all",
                        },
                        {
                            "op": "close",
                            "input": "liver_all",
                            "output": "liver_all",
                            "radius_mm": 5.0,
                        },
                        {
                            "op": "fill_holes",
                            "input": "liver_all",
                            "output": "liver_all",
                        },
                        {
                            "op": "largest_cc",
                            "input": "liver_all",
                            "output": "liver_all",
                        },
                        {
                            "op": "intersection",
                            "inputs": ["liver", "liver_all"],
                            "output": "liver",
                        },
                        {
                            "op": "intersection",
                            "inputs": ["liver_tumor", "liver_all"],
                            "output": "liver_tumor",
                        },
                    ],
                },
            },
            "MR": {
                "tasks": [
                    {
                        "task": "total_mr",
                        "extra": {
                            "roi_subset_robust": ["liver"],
                            "fastest": True,
                        },
                    },
                    {
                        "task": "liver_lesions_mr",
                        "output": "liver_tumor",
                        "fetch_output": "liver_lesions",
                        "extra": {},
                    },
                ],
                "postprocess": {
                    "on_failure": "warn_only",
                    "operations": [
                        {
                            "op": "union",
                            "inputs": ["liver", "liver_tumor"],
                            "output": "liver_all",
                        },
                        {
                            "op": "close",
                            "input": "liver_all",
                            "output": "liver_all",
                            "radius_mm": 5.0,
                        },
                        {
                            "op": "fill_holes",
                            "input": "liver_all",
                            "output": "liver_all",
                        },
                        {
                            "op": "largest_cc",
                            "input": "liver_all",
                            "output": "liver_all",
                        },
                        {
                            "op": "intersection",
                            "inputs": ["liver", "liver_all"],
                            "output": "liver",
                        },
                        {
                            "op": "intersection",
                            "inputs": ["liver_tumor", "liver_all"],
                            "output": "liver_tumor",
                        },
                    ],
                },
            },
        },
    }


def normalize_segmentation_modality(value: Any) -> str | None:
    """Normalize one row modality to the manifest's ``CT``/``MR`` keys."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("[", "(", "{")):
            try:
                return normalize_segmentation_modality(literal_eval(text))
            except (ValueError, SyntaxError):
                pass
        label = text.upper()
        if label == "CT":
            return "CT"
        if label in {"MR", "MRI"}:
            return "MR"
        return None

    if isinstance(value, (list, tuple, set)):
        normalized = {
            modality
            for item in value
            if (modality := normalize_segmentation_modality(item)) is not None
        }
        return next(iter(normalized)) if len(normalized) == 1 else None

    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return normalize_segmentation_modality(str(value))


def _validate_resolved_segmentation_config(
    config: Mapping[str, Any], *, field: str
) -> Dict[str, Any]:
    tasks = config.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"{field}.tasks must be a non-empty list.")

    normalized_tasks: List[Dict[str, Any]] = []
    for index, raw_task in enumerate(tasks):
        task_field = f"{field}.tasks[{index}]"
        if not isinstance(raw_task, Mapping):
            raise ValueError(f"{task_field} must be a mapping.")
        task = copy.deepcopy(dict(raw_task))
        task_name = task.get("task")
        if not isinstance(task_name, str) or not task_name.strip():
            raise ValueError(f"{task_field}.task must be a non-empty string.")
        task["task"] = task_name.strip()
        extra = task.get("extra", {})
        if not isinstance(extra, Mapping):
            raise ValueError(f"{task_field}.extra must be a mapping.")
        task["extra"] = copy.deepcopy(dict(extra))
        infer_task_fetch_outputs(task)
        normalized_tasks.append(task)

    normalized: Dict[str, Any] = {"tasks": normalized_tasks}
    postprocess = config.get("postprocess")
    if postprocess is not None:
        if not isinstance(postprocess, Mapping):
            raise ValueError(f"{field}.postprocess must be a mapping.")
        normalized["postprocess"] = copy.deepcopy(dict(postprocess))
        normalized["postprocess"]["operations"] = validate_operations(
            postprocess,
            task_outputs=set(build_output_column_map(normalized_tasks)),
            field=f"{field}.postprocess",
        )
    return normalized


def validate_segmentation_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the modality-keyed segmentation manifest contract."""
    if not isinstance(config, Mapping):
        raise ValueError("segmentation must be a mapping.")
    backend = str(config.get("backend", "totalsegmentator")).strip().lower()
    if backend != "totalsegmentator":
        raise ValueError(f"Unsupported backend: {backend}")

    crop = config.get("crop", False)
    if type(crop) is not bool:
        raise ValueError("segmentation.crop must be true or false.")
    crop_margin_mm = config.get("crop_margin_mm", DEFAULT_CROP_MARGIN_MM)
    if (
        isinstance(crop_margin_mm, bool)
        or not isinstance(crop_margin_mm, Real)
        or not np.isfinite(crop_margin_mm)
        or crop_margin_mm < 0
    ):
        raise ValueError(
            "segmentation.crop_margin_mm must be finite and non-negative."
        )

    raw_modalities = config.get("modalities")
    if not isinstance(raw_modalities, Mapping) or not raw_modalities:
        raise ValueError("segmentation.modalities must be a non-empty mapping.")

    modalities: Dict[str, Dict[str, Any]] = {}
    for raw_modality, raw_config in raw_modalities.items():
        modality = normalize_segmentation_modality(raw_modality)
        if modality is None:
            raise ValueError(
                "segmentation.modalities keys must be CT, MR, or MRI; "
                f"got {raw_modality!r}."
            )
        if modality in modalities:
            raise ValueError(
                f"segmentation modality {modality!r} is configured more than once."
            )
        if not isinstance(raw_config, Mapping):
            raise ValueError(
                f"segmentation.modalities.{raw_modality} must be a mapping."
            )
        modalities[modality] = _validate_resolved_segmentation_config(
            raw_config,
            field=f"segmentation.modalities.{raw_modality}",
        )
        for task in modalities[modality]["tasks"]:
            runtime_task, _ = _resolve_runtime_task(task["task"], {})
            task_modality = "MR" if runtime_task.endswith("_mr") else "CT"
            if task_modality != modality:
                raise ValueError(
                    f"TotalSegmentator task {task['task']!r} is a "
                    f"{task_modality} model and cannot be configured under "
                    f"segmentation.modalities.{raw_modality}."
                )

    return {
        "backend": backend,
        "crop": crop,
        "crop_margin_mm": float(crop_margin_mm),
        "modalities": modalities,
    }


def resolve_segmentation_config_for_modality(
    config: Mapping[str, Any], modality: Any
) -> Dict[str, Any] | None:
    """Return the worker configuration for a row's normalized modality."""
    normalized_modality = normalize_segmentation_modality(modality)
    if normalized_modality is None:
        return None
    modality_config = config.get("modalities", {}).get(normalized_modality)
    if modality_config is None:
        return None
    return {
        "backend": config.get("backend", "totalsegmentator"),
        **copy.deepcopy(dict(modality_config)),
    }


def load_segmentation_config(
    manifest_arg: str | None, *, base_path: Path
) -> Dict[str, Any]:
    """Load segmentation config from manifest, falling back to generic manifest."""
    generic_manifest = load_manifest("generic", base_path=base_path)
    generic_segmentation = (
        generic_manifest.get("segmentation") or _default_segmentation_config()
    )

    if not manifest_arg:
        return validate_segmentation_config(generic_segmentation)

    manifest = load_manifest(manifest_arg, base_path=base_path)
    manifest_segmentation = manifest.get("segmentation")
    if manifest_segmentation:
        return validate_segmentation_config(manifest_segmentation)
    return validate_segmentation_config(generic_segmentation)


def _resolve_prefetch_task_name(task: Dict[str, Any]) -> str | None:
    task_name = str(task.get("task", "")).strip()
    if not task_name:
        return None

    extra = task.get("extra", {})
    if not isinstance(extra, dict):
        return task_name

    if bool(extra.get("fast")) or bool(extra.get("fastest")):
        fast_aliases = {
            "total": "total_fast",
            "total_mr": "total_fast_mr",
            "body": "body_fast",
            "body_mr": "body_mr_fast",
        }
        return fast_aliases.get(task_name, task_name)
    return task_name


def _resolve_runtime_task(
    task_name: str, extra: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    runtime_aliases = {
        "total_fast": "total",
        "total_fast_mr": "total_mr",
        "body_fast": "body",
        "body_mr_fast": "body_mr",
    }
    runtime_task = runtime_aliases.get(task_name, task_name)
    if runtime_task != task_name:
        extra.setdefault("fast", True)
    return runtime_task, extra


def _parse_version_tuple(raw: str) -> Tuple[int, ...]:
    parts = tuple(int(part) for part in re.findall(r"\d+", str(raw)))
    return parts or (0,)


def _version_is_at_least(current_version: str, minimum_version: str) -> bool:
    current_parts = _parse_version_tuple(current_version)
    minimum_parts = _parse_version_tuple(minimum_version)
    max_len = max(len(current_parts), len(minimum_parts))
    current_parts += (0,) * (max_len - len(current_parts))
    minimum_parts += (0,) * (max_len - len(minimum_parts))
    return current_parts >= minimum_parts


def _get_totalsegmentator_version() -> str:
    for package_name in ("TotalSegmentator", "totalsegmentator"):
        try:
            return distribution_version(package_name)
        except PackageNotFoundError:
            continue
    try:
        import totalsegmentator

        return str(getattr(totalsegmentator, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _raise_liver_lesions_version_error(current_version: str) -> None:
    raise RuntimeError(
        "task needs totalsegmentator version >= "
        f"{LIVER_LESIONS_MIN_TOTALSEGMENTATOR_VERSION}, "
        f"current version=={current_version}"
    )


def _ensure_liver_lesions_version_supported(task_names: List[str]) -> None:
    if not any(task_name in LIVER_LESIONS_TASKS for task_name in task_names):
        return

    current_version = _get_totalsegmentator_version()
    if not _version_is_at_least(
        current_version, LIVER_LESIONS_MIN_TOTALSEGMENTATOR_VERSION
    ):
        _raise_liver_lesions_version_error(current_version)


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def _normalize_output_key(raw: str) -> str:
    value = str(raw).strip()
    if value.endswith(".nii.gz"):
        value = value[: -len(".nii.gz")]
    return value


def infer_task_outputs(task: Dict[str, Any]) -> List[str]:
    """Infer normalized logical output names from one segmentation task."""
    outputs: List[str] = []
    outputs.extend(_normalize_output_key(v) for v in _as_str_list(task.get("outputs")))
    outputs.extend(_normalize_output_key(v) for v in _as_str_list(task.get("output")))
    extra = task.get("extra", {})
    if isinstance(extra, dict):
        for roi_key in ("roi_subset_robust", "roi_subset"):
            for roi in _as_str_list(extra.get(roi_key)):
                roi = roi.strip()
                if roi:
                    outputs.append(_normalize_output_key(roi))
    # dedupe but keep order
    return list(dict.fromkeys(outputs))


def infer_task_fetch_outputs(task: Dict[str, Any]) -> Dict[str, str]:
    """Map logical output keys to backend-produced filenames to fetch."""
    outputs = infer_task_outputs(task)
    fetch_outputs: List[str] = []
    fetch_outputs.extend(
        _normalize_output_key(v) for v in _as_str_list(task.get("fetch_outputs"))
    )
    fetch_outputs.extend(
        _normalize_output_key(v) for v in _as_str_list(task.get("fetch_output"))
    )
    if not fetch_outputs:
        return {output_name: output_name for output_name in outputs}
    if len(fetch_outputs) != len(outputs):
        raise ValueError(
            "task.fetch_output(s) must match task.output(s) one-to-one. "
            f"task={task.get('task', '<unknown>')!r}, "
            f"outputs={outputs}, fetch_outputs={fetch_outputs}"
        )
    return dict(zip(outputs, fetch_outputs))


def _output_to_column(output_name: str) -> str:
    value = str(output_name).strip()
    if value.endswith(".nii.gz"):
        value = value[: -len(".nii.gz")]
    return f"mask_{value or 'unnamed'}"


def _output_to_filename(output_name: str) -> str:
    value = str(output_name).strip()
    if value.endswith(".nii.gz"):
        return value
    return f"{value}.nii.gz"


def build_output_column_map(tasks: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map each unique logical task output to its ``mask_*`` CSV column."""
    output_to_column: Dict[str, str] = {}
    for task in tasks:
        for output_name in infer_task_outputs(task):
            if output_name not in output_to_column:
                output_to_column[output_name] = _output_to_column(output_name)
    return output_to_column


def build_output_fetch_map(tasks: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map logical task outputs to filenames produced by the backend."""
    output_to_fetch: Dict[str, str] = {}
    for task in tasks:
        for output_name, fetch_name in infer_task_fetch_outputs(task).items():
            if output_name not in output_to_fetch:
                output_to_fetch[output_name] = fetch_name
    return output_to_fetch


def _snapshot_nifti_files(dir_path: Path) -> Dict[str, Tuple[int, int]]:
    snapshot: Dict[str, Tuple[int, int]] = {}
    for path in sorted(dir_path.glob("*.nii.gz")):
        if not path.is_file():
            continue
        stat = path.stat()
        snapshot[path.name] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _infer_outputs_from_snapshot(
    before: Dict[str, Tuple[int, int]],
    dir_path: Path,
    *,
    exclude_names: set[str] | None = None,
) -> List[str]:
    exclude_names = exclude_names or set()
    inferred: List[str] = []
    for path in sorted(dir_path.glob("*.nii.gz")):
        if not path.is_file() or path.name in exclude_names:
            continue
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if before.get(path.name) == signature:
            continue
        inferred.append(_normalize_output_key(path.name))
    return inferred


def iter_modality_segmentation_configs(
    config: Mapping[str, Any], modalities: set[str] | None = None
):
    """Yield normalized modality and worker config pairs."""
    for modality, modality_config in config.get("modalities", {}).items():
        if modalities is not None and modality not in modalities:
            continue
        yield modality, {
            "backend": config.get("backend", "totalsegmentator"),
            **copy.deepcopy(dict(modality_config)),
        }


def build_segmentation_output_maps(
    config: Mapping[str, Any],
) -> Tuple[Dict[str, str], Dict[str, str], set[str]]:
    """Build union output maps across every configured modality."""
    output_to_column: Dict[str, str] = {}
    output_to_fetch: Dict[str, str] = {}
    postprocess_outputs: set[str] = set()
    for _, resolved in iter_modality_segmentation_configs(config):
        for output_name, column_name in build_output_column_map(
            resolved["tasks"]
        ).items():
            output_to_column.setdefault(output_name, column_name)
        for output_name, fetch_name in build_output_fetch_map(
            resolved["tasks"]
        ).items():
            existing = output_to_fetch.setdefault(output_name, fetch_name)
            if existing != fetch_name:
                raise ValueError(
                    "Logical segmentation output maps to different backend files "
                    f"across modalities: {output_name!r} -> "
                    f"{existing!r}/{fetch_name!r}."
                )
        for output_name in _postprocess_output_names(resolved.get("postprocess")):
            output_to_column.setdefault(output_name, _output_to_column(output_name))
            output_to_fetch.setdefault(output_name, output_name)
            postprocess_outputs.add(output_name)
    return output_to_column, output_to_fetch, postprocess_outputs


def _postprocess_output_names(postprocess: Mapping[str, Any] | None) -> List[str]:
    if not postprocess:
        return []
    return list(dict.fromkeys(step["output"] for step in postprocess["operations"]))


def prefetch_totalsegmentator_models(
    tasks_config: Dict[str, Any], *, modalities: set[str] | None = None
) -> None:
    """Download required TotalSegmentator weights before multiprocessing."""
    if tasks_config.get("backend", "totalsegmentator") != "totalsegmentator":
        return

    if "modalities" in tasks_config:
        tasks = [
            task
            for _, resolved in iter_modality_segmentation_configs(
                tasks_config, modalities
            )
            for task in resolved.get("tasks", [])
        ]
    else:
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
        "liver_lesions": [591],
        "liver_lesions_mr": [589],
    }

    task_names = {
        resolved_name
        for task in tasks
        for resolved_name in [_resolve_prefetch_task_name(task)]
        if resolved_name
    }
    if not task_names:
        return

    resolved_tasks = sorted(task_names)
    _ensure_liver_lesions_version_supported(resolved_tasks)

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


def _has_existing_task_outputs(output_dir: Path, tasks_config: Dict[str, Any]) -> bool:
    """Check whether all configured tasks and postprocessing can reuse outputs."""
    tasks = tasks_config.get("tasks", [])
    if not tasks:
        return False
    for task in tasks:
        outputs = infer_task_fetch_outputs(task)
        if not outputs or not all(
            (output_dir / _output_to_filename(name)).exists()
            for name in outputs.values()
        ):
            return False
    postprocess = tasks_config.get("postprocess")
    if postprocess is not None:
        operations = validate_operations(
            postprocess, task_outputs=set(build_output_column_map(tasks))
        )
        return operations_are_current(
            output_dir, operations, build_output_fetch_map(tasks)
        )
    return True


def segment_volume(
    nifti_path: Path,
    output_dir: Path,
    tasks_config: Dict[str, Any],
    *,
    verbose: bool = False,
    force: bool = False,
    crop_margin_mm: float | None = None,
    crop_mask_paths: List[Path] | None = None,
    backend: TotalSegmentatorBackend | None = None,
    resolved_output_to_fetch: Dict[str, str] | None = None,
    debug: bool = False,
    row_idx: int | None = None,
    worker_generation: int = 0,
) -> List[str]:
    """Run segmentation tasks and optional post‐processing."""
    volume_started_at = time.monotonic()
    backend_total_sec = 0.0
    postprocess_sec = 0.0
    io_sec = 0.0
    warnings: List[str] = []
    ran_any_task = False
    tasks = tasks_config.get("tasks", [])
    if not tasks:
        raise ValueError("No tasks provided in config")

    backend_name = tasks_config.get("backend", "totalsegmentator")
    if backend_name != "totalsegmentator":
        raise ValueError(f"Unsupported backend: {backend_name}")

    backend = backend or TotalSegmentatorBackend()
    output_to_column = build_output_column_map(tasks)
    output_to_fetch = build_output_fetch_map(tasks)
    postprocess = tasks_config.get("postprocess")
    operations = None
    if postprocess is not None:
        operations = validate_operations(
            postprocess, task_outputs=set(output_to_column)
        )
        postprocess = {**postprocess, "operations": operations}

    def _store_resolved_outputs() -> None:
        if resolved_output_to_fetch is None:
            return
        resolved = dict(output_to_fetch)
        for output_name in _postprocess_output_names(postprocess):
            resolved[output_name] = output_to_fetch.get(output_name, output_name)
        resolved_output_to_fetch.clear()
        resolved_output_to_fetch.update(resolved)

    for task in tasks:
        task_name = task["task"]
        task_outputs = infer_task_outputs(task)
        task_fetch_outputs = infer_task_fetch_outputs(task) if task_outputs else {}
        extra = task.get("extra", {})
        if not isinstance(extra, dict):
            extra = {}
        task_name, extra = _resolve_runtime_task(task_name, extra)

        # TotalSegmentator can spawn additional saving threads per process
        # (nr_thr_saving defaults to 6). In our multi-process executor this can
        # multiply aggressively and trigger worker instability on long runs.
        # Keep a conservative default unless users explicitly override it.
        extra.setdefault("nr_thr_saving", 2)

        io_started_at = time.monotonic()
        before_snapshot = _snapshot_nifti_files(output_dir) if not task_outputs else {}
        expected_paths = [
            output_dir / _output_to_filename(task_fetch_outputs[output_name])
            for output_name in task_outputs
        ]
        io_sec += time.monotonic() - io_started_at
        if expected_paths and all(dst.exists() for dst in expected_paths) and not force:
            if verbose:
                logger.info("Skip %s – files exist", task_name)
            continue

        try:
            if debug:
                logger.debug(
                    "row=%s task=%s start pid=%d generation=%d options=%s",
                    row_idx,
                    task_name,
                    os.getpid(),
                    worker_generation,
                    {
                        key: extra[key]
                        for key in ("roi_subset_robust", "nr_thr_resamp")
                        if key in extra
                    },
                )
            backend_started_at = time.monotonic()
            backend.run(
                input_path=nifti_path,
                output_dir=output_dir,
                task=task_name,
                **extra,
            )
            task_duration = time.monotonic() - backend_started_at
            backend_total_sec += task_duration
            if debug:
                logger.debug(
                    "row=%s task=%s done duration=%.3fs pid=%d generation=%d",
                    row_idx,
                    task_name,
                    task_duration,
                    os.getpid(),
                    worker_generation,
                )
            ran_any_task = True
        except Exception as exc:
            logger.error(
                "Segmentation failed on %s (%s): %s", nifti_path, task_name, exc
            )
            raise

        io_started_at = time.monotonic()
        if task_outputs:
            missing = [p for p in expected_paths if not p.exists()]
            if missing:
                if len(missing) == 1:
                    raise RuntimeError(f"Expected mask not produced: {missing[0]}")
                raise RuntimeError(
                    "Expected masks not produced: " + ", ".join(str(p) for p in missing)
                )
        else:
            inferred_outputs = _infer_outputs_from_snapshot(
                before_snapshot,
                output_dir,
                exclude_names={nifti_path.name},
            )
            if not inferred_outputs:
                if not tasks_config.get("postprocess"):
                    raise RuntimeError(
                        f"Could not infer outputs for task '{task_name}' from created segmentations."
                    )
                logger.warning(
                    "Task '%s' produced no newly written masks; it may have reused "
                    "existing files. Deferring validation until the requested "
                    "postprocess masks are fetched after all tasks finish.",
                    task_name,
                )
                continue
            task_outputs = inferred_outputs
            task_fetch_outputs = {
                output_name: output_name for output_name in inferred_outputs
            }
            for output_name, fetch_name in task_fetch_outputs.items():
                output_to_column.setdefault(output_name, _output_to_column(output_name))
                output_to_fetch.setdefault(output_name, fetch_name)
            if verbose:
                logger.info(
                    "Inferred outputs for %s from created segmentations: %s",
                    task_name,
                    ", ".join(task_outputs),
                )
        if verbose:
            logger.info("Masks saved for %s", task_name)
        io_sec += time.monotonic() - io_started_at

    if postprocess:
        for name in operation_sources(operations):
            output_to_fetch.setdefault(name, name)

    # Crop the source image and raw segmentation masks before post-processing,
    # so every logical and morphological operation runs on the reduced grid.
    if crop_margin_mm is not None:
        crop_started_at = time.monotonic()
        masks_to_crop = [
            output_dir / _output_to_filename(fetch_name)
            for fetch_name in dict.fromkeys(output_to_fetch.values())
        ]
        masks_to_crop.extend(crop_mask_paths or [])
        if not crop_to_segmented_organs(
            nifti_path,
            masks_to_crop,
            margin_mm=crop_margin_mm,
        ):
            warnings.append(
                f"Crop skipped for {nifti_path}: all segmentation masks are empty"
            )
        io_sec += time.monotonic() - crop_started_at

    if not postprocess:
        _store_resolved_outputs()
        if debug:
            logger.debug(
                "row=%s timing total=%.3fs backend.total=%.3fs postprocess=%.3fs io=%.3fs",
                row_idx,
                time.monotonic() - volume_started_at,
                backend_total_sec,
                postprocess_sec,
                io_sec,
            )
        return warnings

    if (
        force
        or ran_any_task
        or not operations_are_current(output_dir, operations, output_to_fetch)
    ):
        try:
            postprocess_started_at = time.monotonic()
            apply_operations(output_dir, operations, output_to_fetch)
            postprocess_sec += time.monotonic() - postprocess_started_at
        except MaskGeometryError as exc:
            if postprocess.get("on_failure", "fail") == "fail":
                raise
            message = f"Postprocess operations failed: {exc}"
            logger.warning(message)
            warnings.append(message)
            # A failed sequence must not advertise stale derived masks.
            if resolved_output_to_fetch is not None:
                resolved_output_to_fetch.clear()
                resolved_output_to_fetch.update(output_to_fetch)
            return warnings
    elif verbose:
        logger.info("Skip postprocess – completed operations and masks are unchanged")
    _store_resolved_outputs()
    if debug:
        logger.debug(
            "row=%s timing total=%.3fs backend.total=%.3fs postprocess=%.3fs io=%.3fs",
            row_idx,
            time.monotonic() - volume_started_at,
            backend_total_sec,
            postprocess_sec,
            io_sec,
        )
    return warnings


def _crop_nifti_image(
    image: nib.spatialimages.SpatialImage,
    spatial_slices: Tuple[slice, slice, slice],
) -> nib.spatialimages.SpatialImage:
    """Crop an image while keeping its original world-coordinate system."""
    full_slices = spatial_slices + (slice(None),) * (len(image.shape) - 3)
    starts = np.asarray([axis.start or 0 for axis in spatial_slices], dtype=float)
    voxel_translation = np.eye(4)
    voxel_translation[:3, 3] = starts
    cropped_affine = image.affine @ voxel_translation
    cropped = image.__class__(
        np.asanyarray(image.dataobj[full_slices]),
        cropped_affine,
        header=image.header.copy(),
        extra=image.extra.copy(),
    )

    # Preserve both NIfTI transforms and their codes. They may intentionally
    # differ, so translate each independently rather than copying one affine.
    if hasattr(image, "get_qform") and hasattr(cropped, "set_qform"):
        qform, qform_code = image.get_qform(coded=True)
        if qform is not None:
            cropped.set_qform(qform @ voxel_translation, int(qform_code))
    if hasattr(image, "get_sform") and hasattr(cropped, "set_sform"):
        sform, sform_code = image.get_sform(coded=True)
        if sform is not None:
            cropped.set_sform(sform @ voxel_translation, int(sform_code))
    return cropped


def crop_to_segmented_organs(
    nifti_path: Path,
    mask_paths: List[Path],
    *,
    margin_mm: float = DEFAULT_CROP_MARGIN_MM,
) -> bool:
    """Crop an image and its masks to their union bbox plus a physical margin.

    Returns ``False`` when every mask is empty, and ``True`` when a non-empty
    bounding box was found (including when it already spans the whole image).
    """
    if not np.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError("crop margin must be finite and non-negative")

    image = nib.load(str(nifti_path), mmap=False)
    if len(image.shape) < 3:
        raise MaskGeometryError(
            f"NIfTI image must have at least three dimensions; got {image.shape}."
        )
    spatial_shape = tuple(int(size) for size in image.shape[:3])
    mask_images: List[Tuple[Path, nib.spatialimages.SpatialImage]] = []
    lower = np.asarray(spatial_shape, dtype=int)
    upper = np.zeros(3, dtype=int)
    found_foreground = False

    unique_mask_paths: List[Path] = []
    seen_mask_paths: set[Path] = set()
    for mask_path in mask_paths:
        canonical_path = mask_path.resolve()
        if canonical_path not in seen_mask_paths:
            seen_mask_paths.add(canonical_path)
            unique_mask_paths.append(mask_path)

    for mask_path in unique_mask_paths:
        mask = nib.load(str(mask_path), mmap=False)
        if len(mask.shape) != 3:
            raise MaskGeometryError(
                f"Mask {mask_path} must be 3-D; got shape {mask.shape}."
            )
        if tuple(mask.shape) != spatial_shape or not np.allclose(
            mask.affine, image.affine
        ):
            raise MaskGeometryError(
                f"Mask {mask_path} does not share the image shape and affine."
            )
        mask_images.append((mask_path, mask))
        foreground = np.asanyarray(mask.dataobj) > 0
        if foreground.any():
            found_foreground = True
            for axis in range(3):
                other_axes = tuple(index for index in range(3) if index != axis)
                occupied = np.flatnonzero(foreground.any(axis=other_axes))
                lower[axis] = min(lower[axis], int(occupied[0]))
                upper[axis] = max(upper[axis], int(occupied[-1]) + 1)

    if not found_foreground:
        return False

    voxel_sizes = nib.affines.voxel_sizes(image.affine)[:3]
    if not np.all(np.isfinite(voxel_sizes) & (voxel_sizes > 0)):
        raise MaskGeometryError("Cropping requires finite, positive voxel sizes.")
    margin_voxels = np.ceil(float(margin_mm) / voxel_sizes).astype(int)
    lower = np.maximum(0, lower - margin_voxels)
    upper = np.minimum(np.asarray(spatial_shape), upper + margin_voxels)
    spatial_slices = tuple(
        slice(int(start), int(stop)) for start, stop in zip(lower, upper)
    )

    # Do not rewrite already-cropped files on resumed/idempotent runs.
    if np.all(lower == 0) and np.all(upper == np.asarray(spatial_shape)):
        return True

    images_to_crop = [
        (nifti_path, image),
        *(
            (path, mask)
            for path, mask in mask_images
            if path.resolve() != nifti_path.resolve()
        ),
    ]
    temporary_paths: List[Tuple[Path, Path]] = []
    try:
        for path, source in images_to_crop:
            suffix = ".nii.gz" if path.name.endswith(".nii.gz") else path.suffix
            stem = path.name[: -len(suffix)] if suffix else path.name
            temporary_path = path.with_name(f".{stem}.crop-tmp{suffix}")
            nib.save(_crop_nifti_image(source, spatial_slices), temporary_path)
            temporary_paths.append((temporary_path, path))
        for temporary_path, path in temporary_paths:
            temporary_path.replace(path)
    finally:
        for temporary_path, _ in temporary_paths:
            temporary_path.unlink(missing_ok=True)
    return True


# -----------------------------------------------------------------------------
# Worker wrapper (called in pool)
# -----------------------------------------------------------------------------


def _resource_snapshot(*, include_cuda: bool = True) -> Dict[str, float]:
    """Return lightweight process/system memory diagnostics when available."""
    snapshot: Dict[str, float] = {}
    try:
        import psutil  # type: ignore

        process = psutil.Process(os.getpid())
        snapshot["rss_mb"] = process.memory_info().rss / (1024 * 1024)
        snapshot["available_ram_mb"] = psutil.virtual_memory().available / (1024 * 1024)
    except (ImportError, OSError, RuntimeError):
        pass

    # Never import torch solely for diagnostics: importing it can initialize
    # substantial runtime state. Only inspect CUDA if the application already
    # loaded torch and CUDA is already initialized.
    torch_module = sys.modules.get("torch")
    if include_cuda and torch_module is not None:
        try:
            cuda = torch_module.cuda
            if cuda.is_initialized():
                snapshot["cuda_alloc_mb"] = cuda.memory_allocated() / (1024 * 1024)
                snapshot["cuda_reserved_mb"] = cuda.memory_reserved() / (1024 * 1024)
                snapshot["cuda_max_alloc_mb"] = cuda.max_memory_allocated() / (
                    1024 * 1024
                )
        except (AttributeError, RuntimeError):
            pass
    return snapshot


def _format_resource_snapshot(snapshot: Mapping[str, float]) -> str:
    return " ".join(f"{key}={value:.1f}MB" for key, value in snapshot.items())


def _log_multiprocessing_leftovers() -> None:
    children = mp.active_children()
    logger.debug("segmentation complete; active_children=%d", len(children))
    for child in children:
        logger.warning(
            "Child still alive after segmentation: pid=%s name=%s exitcode=%s",
            child.pid,
            child.name,
            child.exitcode,
        )
    non_daemon_threads = [
        thread.name
        for thread in threading.enumerate()
        if thread is not threading.current_thread() and not thread.daemon
    ]
    logger.debug("active_non_daemon_threads=%s", non_daemon_threads)


def _emit_worker_event(event: str, row_idx: int) -> None:
    if _WORKER_EVENT_QUEUE is None:
        return
    worker_event = WorkerEvent(
        row_idx=row_idx,
        pid=os.getpid(),
        event=event,
        timestamp=time.monotonic(),
        generation=_WORKER_GENERATION,
        gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    try:
        _WORKER_EVENT_QUEUE.put(worker_event)
    except (BrokenPipeError, EOFError, OSError):
        # A forced pool restart can tear down the parent-side queue first.
        pass


def process_single_volume(
    idx: int,
    row: Dict[str, Any],  # must be JSON‑serialisable
    tasks_config: Dict[str, Any],
    *,
    verbose: bool,
    force: bool,
    crop_margin_mm: float | None = None,
    backend: TotalSegmentatorBackend | None = None,
    debug: bool = False,
    worker_generation: int = 0,
) -> Tuple[int, str | None, str | None, str | None, Dict[str, str] | None]:
    """Return ``(idx, output_dir|None, error_msg|None, warning_msg|None, outputs|None)``."""

    setup_logging(verbose=(verbose or debug))
    total_started_at = time.monotonic()
    if debug:
        logger.debug(
            "row=%s worker.begin pid=%d parent_pid=%d generation=%d gpu=%s %s",
            idx,
            os.getpid(),
            os.getppid(),
            worker_generation,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            _format_resource_snapshot(_resource_snapshot()),
        )

    resolved_tasks_config = tasks_config
    if "modalities" in tasks_config:
        resolved_tasks_config = resolve_segmentation_config_for_modality(
            tasks_config, row.get("Modality")
        )
        if resolved_tasks_config is None:
            return idx, None, None, None, {}

    try:
        nifti_path = Path(row["nifti_path"])
    except KeyError:
        return idx, None, "column 'nifti_path' missing", None, None

    if not nifti_path.exists():
        return idx, None, "file not found", None, None

    try:
        resolved_output_to_fetch: Dict[str, str] = {}
        crop_mask_paths: List[Path] = []
        if crop_margin_mm is not None:
            for column_name, value in row.items():
                if (
                    str(column_name).startswith("mask_")
                    and isinstance(value, (str, Path))
                    and str(value).strip()
                ):
                    existing_mask_path = Path(value)
                    if existing_mask_path.exists():
                        crop_mask_paths.append(existing_mask_path)
        warnings = segment_volume(
            nifti_path,
            nifti_path.parent,
            resolved_tasks_config,
            verbose=verbose,
            force=force,
            crop_margin_mm=crop_margin_mm,
            crop_mask_paths=crop_mask_paths,
            backend=backend,
            resolved_output_to_fetch=resolved_output_to_fetch,
            debug=debug,
            row_idx=idx,
            worker_generation=worker_generation,
        )
        warning_msg = " | ".join(warnings) if warnings else None
        result = (
            idx,
            str(nifti_path.parent),
            None,
            warning_msg,
            resolved_output_to_fetch,
        )
        if debug:
            logger.debug(
                "row=%s worker.end pid=%d generation=%d total_execution=%.3fs %s",
                idx,
                os.getpid(),
                worker_generation,
                time.monotonic() - total_started_at,
                _format_resource_snapshot(_resource_snapshot()),
            )
        return result
    except Exception as exc:
        # Capture full traceback for later debugging
        logger.debug("Traceback for %s:\n%s", nifti_path.name, traceback.format_exc())
        return idx, None, str(exc), None, None


def _process_single_volume_timed(
    idx: int,
    row: Dict[str, Any],
    tasks_config: Dict[str, Any],
    *,
    verbose: bool,
    force: bool,
    crop_margin_mm: float | None = None,
    debug: bool = False,
) -> Tuple[int, str | None, str | None, str | None, Dict[str, str] | None]:
    """Signal actual worker execution boundaries around one volume."""
    _emit_worker_event("started", idx)
    try:
        return process_single_volume(
            idx,
            row,
            tasks_config,
            verbose=verbose,
            force=force,
            crop_margin_mm=crop_margin_mm,
            debug=debug,
            worker_generation=_WORKER_GENERATION,
        )
    finally:
        _emit_worker_event("finished", idx)


def _normalize_process_result(
    result: Tuple[Any, ...],
) -> Tuple[int, str | None, str | None, str | None, Dict[str, str] | None]:
    if len(result) == 4:
        idx, out_dir, err_msg, warning_msg = result
        return idx, out_dir, err_msg, warning_msg, None
    if len(result) == 5:
        idx, out_dir, err_msg, warning_msg, output_map = result
        return idx, out_dir, err_msg, warning_msg, output_map
    raise ValueError(f"Unexpected process result arity: {len(result)}")


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


def _worker_initializer(
    gpu_tokens: List[str], event_queue: Any, generation: int
) -> None:
    """Initialize GPU affinity and the lifecycle event channel in each worker."""
    global _WORKER_EVENT_QUEUE, _WORKER_GENERATION
    _WORKER_EVENT_QUEUE = event_queue
    _WORKER_GENERATION = generation
    _worker_gpu_initializer(gpu_tokens)


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------


def add_segment_arguments(
    parser: argparse.ArgumentParser,
    include_manifest: bool = True,
    include_dry_run: bool = True,
) -> None:
    """Add segmentation, multiprocessing, and resume options to a parser."""
    parser.add_argument(
        "csv_path_pos",
        nargs="?",
        type=str,
        default=None,
        help="Path to the input CSV file. Defaults to ./nifti_index.csv.",
    )
    parser.add_argument(
        "csv_path_out_pos",
        nargs="?",
        type=str,
        default=None,
        help="Optional output CSV path (positional alternative to --csv_path_out).",
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
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Log per-row worker lifecycle, timings, and lightweight resources.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re‑run even if output masks already exist",
    )
    crop_group = parser.add_mutually_exclusive_group()
    crop_group.add_argument(
        "--crop",
        dest="crop",
        action="store_true",
        default=None,
        help=(
            "Crop each NIfTI image and its masks in place to the segmented-organ "
            "bounding box after backend segmentation and before post-processing."
        ),
    )
    crop_group.add_argument(
        "--no-crop",
        dest="crop",
        action="store_false",
        help="Disable cropping even when enabled by the manifest.",
    )
    parser.add_argument(
        "--crop_margin_mm",
        "--crop-margin-mm",
        type=float,
        default=None,
        metavar="MM",
        help=(
            "Override the manifest's in-bounds physical crop margin "
            f"(default without a manifest setting: {DEFAULT_CROP_MARGIN_MM:g} mm)."
        ),
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
        help="Per-volume worker execution timeout in seconds",
    )
    parser.add_argument(
        "--max-in-flight",
        dest="max_in_flight",
        type=int,
        default=None,
        help="Maximum submitted rows; GPU default is one row per worker.",
    )
    parser.add_argument(
        "--recycle-every",
        dest="recycle_every",
        type=int,
        default=None,
        help="Scheduled pool recycle interval in terminal rows (0 disables).",
    )
    add_checkpoint_arguments(
        parser,
        default_rows=DEFAULT_CHECKPOINT_EVERY_ROWS,
        default_sec=DEFAULT_CHECKPOINT_EVERY_SEC,
    )
    if include_manifest:
        parser.add_argument(
            "--manifest",
            type=str,
            default=None,
            help="Built-in manifest name or path to manifest YAML.",
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
    """Build the standalone batch-segmentation parser."""
    parser = argparse.ArgumentParser(
        description="Batch segmentation with TotalSegmentator v2",
        add_help=add_help,
    )
    add_segment_arguments(parser)
    return parser


def normalize_segment_args(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve segmentation input, output, and error paths in-place."""
    csv_in = args.csv_path_opt if args.csv_path_opt is not None else args.csv_path_pos

    if csv_in is None:
        csv_path = Path.cwd() / "nifti_index.csv"
    else:
        csv_path = Path(csv_in)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    args.csv_path = str(csv_path.resolve())

    csv_path_out_pos = getattr(args, "csv_path_out_pos", None)
    csv_out = args.csv_path_out if args.csv_path_out else csv_path_out_pos
    if not csv_out:
        args.csv_path_out = args.csv_path
    else:
        args.csv_path_out = str(Path(csv_out))

    if args.error_csv_path:
        args.error_csv_path = str(Path(args.error_csv_path))
    else:
        args.error_csv_path = str(Path(args.csv_path).parent / "seg_errors.csv")

    crop_margin_mm = getattr(args, "crop_margin_mm", None)
    if crop_margin_mm is not None and (
        not np.isfinite(crop_margin_mm) or crop_margin_mm < 0
    ):
        raise ValueError("--crop-margin-mm must be finite and non-negative")
    if crop_margin_mm is not None:
        args.crop_margin_mm = float(crop_margin_mm)

    if hasattr(args, "timeout_sec") and int(args.timeout_sec) <= 0:
        raise ValueError("--timeout-sec must be positive")
    max_in_flight = getattr(args, "max_in_flight", None)
    if max_in_flight is not None and int(max_in_flight) <= 0:
        raise ValueError("--max-in-flight must be positive")
    recycle_every = getattr(args, "recycle_every", None)
    if recycle_every is not None and int(recycle_every) < 0:
        raise ValueError("--recycle-every must be non-negative")

    del args.csv_path_pos
    del args.csv_path_opt
    if hasattr(args, "csv_path_out_pos"):
        del args.csv_path_out_pos

    return args


def _requested_crop_margin(
    args: argparse.Namespace, segmentation_config: Mapping[str, Any]
) -> float | None:
    cli_crop = getattr(args, "crop", None)
    crop_enabled = (
        bool(segmentation_config.get("crop", False))
        if cli_crop is None
        else bool(cli_crop)
    )
    if not crop_enabled:
        return None
    cli_margin = getattr(args, "crop_margin_mm", None)
    value = (
        segmentation_config.get("crop_margin_mm", DEFAULT_CROP_MARGIN_MM)
        if cli_margin is None
        else cli_margin
    )
    margin = float(value)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("--crop-margin-mm must be finite and non-negative")
    return margin


def main(args: argparse.Namespace) -> None:
    """Run manifest-configured segmentation over every eligible cohort row."""
    debug_enabled = bool(getattr(args, "debug", False))
    setup_logging(verbose=(getattr(args, "verbose", False) or debug_enabled))
    output_path = Path(args.csv_path_out)
    error_path = Path(args.error_csv_path)
    manifest_arg = getattr(args, "manifest", None)
    tasks_config = load_segmentation_config(
        manifest_arg,
        base_path=Path(__file__).resolve().parents[1],
    )
    crop_margin_mm = _requested_crop_margin(args, tasks_config)
    source_id_signature = source_id_resume_signature(args.csv_path)
    checkpoint_signature = {
        "segmentation": tasks_config,
    }
    if source_id_signature:
        checkpoint_signature["source_id"] = source_id_signature
    resume_args = argparse.Namespace(
        **vars(args),
        checkpoint_signature=checkpoint_signature,
    )

    exclude_hash_args = {
        "csv_path_out",
        "dry_run",
        "verbose",
        "debug",
        "resume",
        "checkpoint_every_rows",
        "checkpoint_every_sec",
        "strict_resume",
        "manifest",
    }
    resume_ctx = prepare_resume_context(
        args=resume_args,
        command="segment",
        inputs=args.csv_path,
        output_path=output_path,
        error_path=error_path,
        exclude_hash_args=exclude_hash_args,
    )
    paths = resume_ctx["paths"]
    state = resume_ctx["state"]
    can_resume = resume_ctx["can_resume"]
    already_finished = resume_ctx["already_finished"]
    ckpt = CheckpointManager(paths=paths, config=resume_ctx["config"])

    if already_finished:
        logger.info(
            "Resume enabled and matching segment run already finished; skipping execution."
        )
        log_finished_resume_summary(
            logger, "Segmentation", state, paths.error_checkpoint_path
        )
        return

    from imperandi.utils.multiprocessing import (
        apply_strategy_env,
        strategy_to_log_dict,
        decide_multiprocessing_strategy,
    )

    # Decide
    requested_recycle_every = getattr(args, "recycle_every", None)
    strategy = decide_multiprocessing_strategy(
        prefer_gpu=True,
        requested_workers=args.num_workers,
        start_method_hint=args.start_method,
        target_task_mem_mb=6000,  # tune (TotalSegmentator can be heavy)
        enable_recycling=(
            requested_recycle_every is not None and requested_recycle_every > 0
        ),
        recycle_every=requested_recycle_every or 0,
        requested_max_in_flight=getattr(args, "max_in_flight", None),
        need_hard_timeouts=True,
    )

    effective_workers = strategy.max_workers
    effective_start_method = strategy.start_method
    effective_timeout = args.timeout_sec
    logger.info(
        "MP strategy: mode=%s start_method=%s workers=%d max_in_flight=%d "
        "timeout=%ds recycle_every=%d gpu_count=%d threads_per_worker=%s",
        strategy.mode,
        effective_start_method,
        effective_workers,
        strategy.max_in_flight,
        effective_timeout,
        strategy.recycle_every,
        strategy.gpu_count,
        strategy.reasons.get("threads_per_worker", "unknown"),
    )
    logger.info(
        "MP reasons: workers=%s; max_in_flight=%s; start_method=%s; recycle=%s",
        strategy.reasons.get("workers_reason", "unknown"),
        strategy.reasons.get("max_in_flight_reason", "unknown"),
        strategy.reasons.get("start_method_reason", "unknown"),
        strategy.reasons.get("recycle_reason", "unknown"),
    )
    logger.debug("Complete MP strategy: %s", strategy_to_log_dict(strategy))

    # Apply env caps BEFORE pool creation
    apply_strategy_env(strategy)

    # --- read and pre‑clean CSV ------------------------------------------------
    if can_resume and paths.main_checkpoint_path.exists():
        logger.info("Resuming segment from checkpoint: %s", paths.main_checkpoint_path)
        df = pd.read_csv(paths.main_checkpoint_path).copy()
    else:
        df = pd.read_csv(args.csv_path).copy()
    df = ensure_source_id_column(df)
    if "nifti_path" not in df.columns:
        unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
        if unnamed:
            df = df.drop(columns=unnamed)
    if "nifti_path" not in df.columns:
        raise KeyError("column 'nifti_path' missing")
    if "Modality" not in df.columns:
        raise KeyError(
            "column 'Modality' missing; modality-keyed segmentation requires it"
        )
    df = df.drop_duplicates("nifti_path").copy()
    row_modalities = df["Modality"].map(normalize_segmentation_modality)
    configured_modalities = set(tasks_config["modalities"])
    eligible_by_modality = row_modalities.isin(configured_modalities)
    active_modalities = set(row_modalities[eligible_by_modality].dropna())
    prefetch_totalsegmentator_models(
        tasks_config,
        modalities=active_modalities,
    )
    (
        output_to_column,
        output_to_fetch,
        postprocess_outputs,
    ) = build_segmentation_output_maps(tasks_config)
    for column_name in list(dict.fromkeys(output_to_column.values())):
        if column_name not in df.columns:
            df[column_name] = None
    if "warning_message" not in df.columns:
        df["warning_message"] = None
    logged_warning_keys: set[str] = set()

    completed_indices: set[str] = set()
    resume_skipped_count = 0
    if can_resume:
        completed_indices = normalize_source_ids(
            (state or {}).get("completed_indices", [])
        )
        resume_skipped_count = len(completed_indices)
        logger.info(
            "Resume enabled: %d completed rows restored from state",
            len(completed_indices),
        )

    errors_by_idx: Dict[str, str] = {}
    if can_resume and paths.error_checkpoint_path.exists():
        err_ckpt = pd.read_csv(paths.error_checkpoint_path)
        err_key = "_source_idx" if "_source_idx" in err_ckpt.columns else "idx"
        if err_key in err_ckpt.columns and "error_message" in err_ckpt.columns:
            for _, row in err_ckpt.iterrows():
                try:
                    source_idx = normalize_source_id(row[err_key])
                    if source_idx:
                        errors_by_idx[source_idx] = str(row["error_message"])
                except Exception:
                    continue

    resume_failed_count = len(completed_indices & set(errors_by_idx))

    modality_skipped_count = 0
    for idx in df.index[~eligible_by_modality]:
        source_idx = normalize_source_id(df.at[idx, "_source_idx"])
        if source_idx not in completed_indices:
            modality_skipped_count += 1
            completed_indices.add(source_idx)
        errors_by_idx.pop(source_idx, None)
    if modality_skipped_count:
        logger.info(
            "Skipped %d row(s) without a configured segmentation modality",
            modality_skipped_count,
        )

    def _checkpoint_write(*, force: bool = False) -> None:
        err_ckpt_df = (
            pd.DataFrame(
                [
                    {"_source_idx": k, "error_message": v}
                    for k, v in sorted(errors_by_idx.items())
                ]
            )
            if errors_by_idx
            else pd.DataFrame()
        )
        ckpt.flush(
            main_df=df,
            error_df=err_ckpt_df,
            completed_indices=completed_indices,
            force=force,
        )

    def _apply_result(
        idx: int,
        out_dir: str | None,
        err_msg: str | None,
        warning_msg: str | None,
        result_output_to_fetch: Dict[str, str] | None = None,
    ) -> None:
        source_idx = normalize_source_id(df.at[idx, "_source_idx"])
        completed_indices.add(source_idx)
        ckpt.mark_processed()

        if out_dir:
            base = Path(out_dir)
            row_warnings: List[str] = []
            if result_output_to_fetch:
                for output_name, fetch_name in result_output_to_fetch.items():
                    column_name = output_to_column.setdefault(
                        output_name, _output_to_column(output_name)
                    )
                    output_to_fetch.setdefault(output_name, fetch_name)
                    if column_name not in df.columns:
                        df[column_name] = None
            outputs_to_check = (
                result_output_to_fetch
                if result_output_to_fetch is not None
                else output_to_fetch
            )
            for output_name, fetch_name in outputs_to_check.items():
                column_name = output_to_column.setdefault(
                    output_name, _output_to_column(output_name)
                )
                if column_name not in df.columns:
                    df[column_name] = None
                mask_path = base / _output_to_filename(fetch_name)
                if mask_path.exists():
                    df.at[idx, column_name] = str(mask_path)
                elif output_name in postprocess_outputs:
                    row_warnings.append(f"missing postprocess mask: {mask_path}")
                else:
                    row_warnings.append(f"missing mask: {mask_path}")
            if warning_msg:
                warning_messages = [
                    message.strip()
                    for message in warning_msg.split(" | ")
                    if message and message.strip()
                ]
                for message in warning_messages:
                    if message not in logged_warning_keys:
                        logger.warning(message)
                        logged_warning_keys.add(message)
                    row_warnings.append(message)
            if row_warnings:
                df.at[idx, "warning_message"] = " | ".join(row_warnings)
            if source_idx in errors_by_idx:
                del errors_by_idx[source_idx]
        elif err_msg:
            errors_by_idx[source_idx] = err_msg or "unknown"
        elif source_idx in errors_by_idx:
            del errors_by_idx[source_idx]

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
    if strategy.use_gpu and strategy.gpu_count > 0:
        gpu_tokens = _resolve_visible_gpu_tokens(strategy.gpu_count)
        if gpu_tokens:
            logger.info(
                "GPU worker pinning enabled: %d worker(s) across %d visible GPU token(s)",
                effective_workers,
                len(gpu_tokens),
            )

    pool_generation = 0

    def _run_rows(
        row_indices: List[int],
        *,
        progress_bar: tqdm | None = None,
        on_result: Any | None = None,
        normal_shutdown_reason: str = "completed",
    ) -> Dict[
        int, Tuple[int, str | None, str | None, str | None, Dict[str, str] | None]
    ]:
        nonlocal pool_generation
        out: Dict[
            int, Tuple[int, str | None, str | None, str | None, Dict[str, str] | None]
        ] = {}
        if not row_indices:
            return out

        max_in_flight = max(1, int(strategy.max_in_flight))
        row_queue = deque(row_indices)
        broken_pool = False
        event_queue = ctx.Queue() if hasattr(ctx, "Queue") else Queue()

        def _record_result(idx: int, result: Tuple[Any, ...]) -> None:
            normalized = _normalize_process_result(result)
            out[idx] = normalized
            if on_result is not None:
                on_result(*normalized)
            if progress_bar is not None:
                progress_bar.update(1)

        def _create_pool(generation: int) -> Tuple[ProcessPoolExecutor, bool]:
            init_kwargs = {
                "initializer": _worker_initializer,
                "initargs": (gpu_tokens, event_queue, generation),
            }
            try:
                pool = ProcessPoolExecutor(
                    max_workers=effective_workers,
                    mp_context=ctx,
                    **init_kwargs,
                )
                logger.debug(
                    "pool.create generation=%d workers=%d start_method=%s parent_pid=%d %s",
                    generation,
                    effective_workers,
                    effective_start_method,
                    os.getpid(),
                    _format_resource_snapshot(
                        _resource_snapshot(include_cuda=False)
                    ),
                )
                return pool, True
            except TypeError as exc:
                logger.warning(
                    "Executor does not support the worker initializer; "
                    "using submission time only when no executor queue is possible (%s)",
                    exc,
                )
                return (
                    ProcessPoolExecutor(
                        max_workers=effective_workers, mp_context=ctx
                    ),
                    False,
                )

        def _shutdown_pool(
            pool: ProcessPoolExecutor,
            force: bool,
            *,
            generation: int,
            reason: str,
        ) -> None:
            if force:
                # Capture workers before shutdown mutates executor internals.
                processes = list((getattr(pool, "_processes", None) or {}).values())
                manager_thread = getattr(pool, "_executor_manager_thread", None)

                # Stop queueing new work and cancel pending futures without blocking.
                pool.shutdown(wait=False, cancel_futures=True)

                # Best-effort hard stop for timed-out/crashed worker pools.
                for p in processes:
                    try:
                        if p.is_alive():
                            p.terminate()
                    except Exception:
                        continue
                for p in processes:
                    try:
                        p.join(timeout=1.0)
                    except Exception:
                        continue
                for p in processes:
                    try:
                        if p.is_alive():
                            p.kill()
                    except Exception:
                        continue
                for p in processes:
                    try:
                        p.join(timeout=1.0)
                    except Exception:
                        continue

                if manager_thread is not None:
                    try:
                        manager_thread.join(timeout=3.0)
                    except Exception:
                        pass

                # Finalize executor internals so atexit does not keep waiting on
                # orphaned manager resources after forced worker termination.
                try:
                    pool.shutdown(wait=True, cancel_futures=True)
                except Exception:
                    pass
                logger.debug(
                    "pool.shutdown generation=%d reason=%s forced=true", generation, reason
                )
                return
            # Graceful shutdown prevents collateral damage to interpreter state.
            pool.shutdown(wait=True, cancel_futures=False)
            logger.debug(
                "pool.shutdown generation=%d reason=%s forced=false", generation, reason
            )

        try:
            while row_queue and not broken_pool:
                pool_generation += 1
                generation = pool_generation
                pool, worker_events_supported = _create_pool(generation)

                futures: Dict[Any, int] = {}
                submitted_at: Dict[Any, float] = {}
                started_events: Dict[int, WorkerEvent] = {}
                finished_rows: set[int] = set()
                restart_pool_for_timeout = False
                broken_pool_msg: str | None = None

                def _drain_worker_events() -> None:
                    while True:
                        try:
                            event = event_queue.get_nowait()
                        except Empty:
                            return
                        if not isinstance(event, WorkerEvent):
                            continue
                        if event.generation != generation:
                            continue
                        if event.event == "started":
                            started_events[event.row_idx] = event
                            future = next(
                                (
                                    candidate
                                    for candidate, candidate_idx in futures.items()
                                    if candidate_idx == event.row_idx
                                ),
                                None,
                            )
                            queue_wait = (
                                event.timestamp - submitted_at[future]
                                if future in submitted_at
                                else 0.0
                            )
                            logger.debug(
                                "worker_started row=%d pid=%d gpu=%s generation=%d queue_wait=%.3fs",
                                event.row_idx,
                                event.pid,
                                event.gpu,
                                event.generation,
                                max(0.0, queue_wait),
                            )
                        elif event.event == "finished":
                            finished_rows.add(event.row_idx)
                            started = started_events.get(event.row_idx)
                            execution_sec = (
                                event.timestamp - started.timestamp if started else 0.0
                            )
                            logger.debug(
                                "worker_finished row=%d pid=%d generation=%d total_execution=%.3fs",
                                event.row_idx,
                                event.pid,
                                event.generation,
                                max(0.0, execution_sec),
                            )

                def _submit_until_limit() -> None:
                    while row_queue and len(futures) < max_in_flight:
                        idx = row_queue.popleft()
                        row = df.loc[idx].to_dict()
                        resolved_config = resolve_segmentation_config_for_modality(
                            tasks_config, row.get("Modality")
                        )
                        if resolved_config is None:
                            raise RuntimeError(
                                f"No segmentation model configured for row {idx}."
                            )
                        submitted = time.monotonic()
                        fut = pool.submit(
                            _process_single_volume_timed,
                            idx,
                            row,
                            resolved_config,
                            verbose=args.verbose,
                            force=args.force,
                            crop_margin_mm=crop_margin_mm,
                            debug=debug_enabled,
                        )
                        futures[fut] = idx
                        submitted_at[fut] = submitted
                        # Compatibility fallback for executor implementations
                        # without initializers. It is only safe when no executor
                        # queue can exist; production ProcessPoolExecutor supports
                        # the event-based path above.
                        if (
                            not worker_events_supported
                            and max_in_flight <= effective_workers
                        ):
                            started_events[idx] = WorkerEvent(
                                row_idx=idx,
                                pid=-1,
                                event="started",
                                timestamp=submitted,
                                generation=generation,
                            )
                        logger.debug(
                            "submit row=%d future=%s queued=%d running_estimate=%d generation=%d",
                            idx,
                            id(fut),
                            max(0, len(futures) - len(started_events)),
                            len(started_events),
                            generation,
                        )

                def _expire_timed_out_futures() -> bool:
                    now = time.monotonic()
                    timed_out_futures: List[Any] = []
                    for fut, idx in list(futures.items()):
                        started = started_events.get(idx)
                        if started is None or idx in finished_rows:
                            continue
                        execution_sec = now - started.timestamp
                        if execution_sec < effective_timeout:
                            continue
                        timed_out_futures.append(fut)
                        wall_sec = now - submitted_at[fut]
                        logger.warning(
                            "Timeout: row=%d execution=%.1fs wall_since_submission=%.1fs "
                            "worker_pid=%s limit=%ds",
                            idx,
                            execution_sec,
                            wall_sec,
                            started.pid,
                            effective_timeout,
                        )
                        _record_result(
                            idx,
                            (
                                idx,
                                None,
                                f"execution timeout after {effective_timeout}s",
                                None,
                                None,
                            ),
                        )

                    if not timed_out_futures:
                        return False

                    timed_out_rows = [futures[fut] for fut in timed_out_futures]
                    timed_out_set = set(timed_out_rows)
                    interrupted_rows = [
                        idx
                        for idx in futures.values()
                        if idx not in timed_out_set and idx in started_events
                    ]
                    queued_rows = [
                        idx
                        for idx in futures.values()
                        if idx not in timed_out_set and idx not in started_events
                    ]
                    logger.warning(
                        "Forced worker-pool restart after execution timeout: "
                        "ProcessPoolExecutor cannot safely replace one worker. "
                        "Timed-out rows: %s; interrupted running rows: %s; queued rows: %s",
                        timed_out_rows,
                        interrupted_rows,
                        queued_rows,
                    )

                    for timed_out in timed_out_futures:
                        futures.pop(timed_out, None)
                        submitted_at.pop(timed_out, None)
                        timed_out.cancel()

                    requeued_rows = list(futures.values())
                    for pending_fut in list(futures):
                        pending_fut.cancel()
                    for pending_idx in reversed(requeued_rows):
                        row_queue.appendleft(pending_idx)
                    logger.warning("Requeued rows after forced restart: %s", requeued_rows)
                    futures.clear()
                    submitted_at.clear()
                    return True

                def _collect_completed_nonblocking() -> bool:
                    nonlocal broken_pool, broken_pool_msg
                    completed_any = False
                    for fut, idx in list(futures.items()):
                        try:
                            res = fut.result(timeout=0)
                        except TimeoutError:
                            continue
                        except BrokenProcessPool as exc:
                            broken_pool = True
                            broken_pool_msg = _broken_pool_message(exc)
                            logger.error(
                                "BrokenProcessPool while collecting row %d; aborting fast: %s",
                                idx,
                                exc,
                            )
                            break
                        except Exception as exc:
                            res = (
                                idx,
                                None,
                                f"worker crash: {type(exc).__name__}: {exc}",
                                None,
                                None,
                            )

                        futures.pop(fut, None)
                        submitted_at.pop(fut, None)
                        started_events.pop(idx, None)
                        finished_rows.discard(idx)
                        _record_result(idx, res)
                        _submit_until_limit()
                        completed_any = True
                    return completed_any

                try:
                    _submit_until_limit()
                    while futures and not restart_pool_for_timeout and not broken_pool:
                        _drain_worker_events()
                        completed_any = _collect_completed_nonblocking()
                        _drain_worker_events()
                        if _expire_timed_out_futures():
                            restart_pool_for_timeout = True
                            break
                        if not completed_any:
                            time.sleep(0.05)
                except BrokenProcessPool as exc:
                    broken_pool = True
                    broken_pool_msg = _broken_pool_message(exc)
                    logger.error(
                        "BrokenProcessPool during completion loop; aborting fast: %s",
                        exc,
                    )
                finally:
                    shutdown_reason = (
                        "timeout"
                        if restart_pool_for_timeout
                        else "crash"
                        if broken_pool
                        else normal_shutdown_reason
                    )
                    _shutdown_pool(
                        pool,
                        force=(broken_pool or restart_pool_for_timeout),
                        generation=generation,
                        reason=shutdown_reason,
                    )

                if broken_pool:
                    msg = broken_pool_msg or "BrokenProcessPool"
                    for idx in list(futures.values()):
                        out[idx] = (idx, None, msg, None, None)
                    for idx in row_queue:
                        out[idx] = (idx, None, msg, None, None)
                    break
        finally:
            try:
                event_queue.close()
                event_queue.join_thread()
            except (AttributeError, OSError, ValueError):
                pass

        return out

    def _run_rows_with_recycling(
        row_indices: List[int],
        *,
        progress_bar: tqdm | None = None,
        on_result: Any | None = None,
    ) -> Dict[
        int, Tuple[int, str | None, str | None, str | None, Dict[str, str] | None]
    ]:
        if strategy.recycle_every <= 0:
            return _run_rows(
                row_indices, progress_bar=progress_bar, on_result=on_result
            )

        out: Dict[
            int, Tuple[int, str | None, str | None, str | None, Dict[str, str] | None]
        ] = {}
        chunk_size = max(1, int(strategy.recycle_every))
        for start in range(0, len(row_indices), chunk_size):
            chunk = row_indices[start : start + chunk_size]
            is_last = start + chunk_size >= len(row_indices)
            out.update(
                _run_rows(
                    chunk,
                    progress_bar=progress_bar,
                    on_result=on_result,
                    normal_shutdown_reason=("completed" if is_last else "scheduled"),
                )
            )
            if not is_last:
                logger.info(
                    "Scheduled worker-pool recycle after %d terminal row(s)",
                    len(chunk),
                )
        return out

    row_indices = [
        i
        for i in list(df.index)
        if bool(eligible_by_modality.at[i])
        and normalize_source_id(df.at[i, "_source_idx"]) not in completed_indices
    ]
    processed_source_ids = {
        normalize_source_id(df.at[i, "_source_idx"]) for i in row_indices
    }
    existing_output_ids = set()
    if not args.force:
        for i in row_indices:
            row = df.loc[i]
            nifti_path = row.get("nifti_path")
            if not isinstance(nifti_path, (str, Path)):
                continue
            resolved_config = resolve_segmentation_config_for_modality(
                tasks_config, row.get("Modality")
            )
            if resolved_config and _has_existing_task_outputs(
                Path(nifti_path).parent, resolved_config
            ):
                existing_output_ids.add(normalize_source_id(df.at[i, "_source_idx"]))
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
            row = df.loc[idx].to_dict()
            resolved_config = resolve_segmentation_config_for_modality(
                tasks_config, row.get("Modality")
            )
            if resolved_config is None:
                raise RuntimeError(f"No segmentation model configured for row {idx}.")
            result = _normalize_process_result(
                process_single_volume(
                    idx,
                    row,
                    resolved_config,
                    verbose=args.verbose,
                    force=args.force,
                    crop_margin_mm=crop_margin_mm,
                    debug=debug_enabled,
                    worker_generation=0,
                )
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
                if _is_retryable(results_by_idx.get(i, (i, None, None, None, None))[2])
            ]
            if retry_indices:
                logger.warning(
                    "Retrying %d row(s) in a fresh executor after worker crash/BrokenProcessPool",
                    len(retry_indices),
                )
                retry_results = _run_rows_with_recycling(
                    retry_indices,
                    progress_bar=progress_bar,
                    on_result=_apply_result,
                )
                results_by_idx.update(retry_results)

    _checkpoint_write(force=True)

    # --- write output tables ---------------------------------------------------
    df_out = df.drop(columns=["_source_idx"], errors="ignore")
    df_out = merge_with_existing_output(
        df_out,
        args.csv_path_out,
        preferred_keys=["volume_id", "nifti_path", "_source_idx"],
        strict=True,
    )
    atomic_write_csv(df_out, args.csv_path_out, index=False)
    logger.info("Wrote main table → %s", args.csv_path_out)

    if errors_by_idx:
        error_messages = pd.Series(errors_by_idx, name="error_message")
        err_df = (
            df[df["_source_idx"].isin(error_messages.index)]
            .copy()
            .sort_values("_source_idx")
        )
        err_df["error_message"] = err_df["_source_idx"].map(error_messages)
        err_df = err_df.drop(columns=["_source_idx"], errors="ignore")
        atomic_write_csv(err_df, args.error_csv_path, index=False)
        logger.warning("%d rows failed – see %s", len(err_df), args.error_csv_path)

        # Optional project‑specific volume report
        try:
            report_volumes(err_df)
        except Exception:
            logger.debug("report_volumes() failed – continuing")

    ckpt.finalize_state(
        completed_indices=completed_indices,
        input_fingerprint=fingerprint_inputs(
            args.csv_path, strict=bool(getattr(args, "strict_resume", False))
        ),
    )
    run_failed_count = len(processed_source_ids & set(errors_by_idx))
    existing_output_count = len(existing_output_ids - set(errors_by_idx))
    log_task_summary(
        logger,
        "Segmentation",
        total_rows=len(df),
        processed_rows=len(row_indices),
        succeeded_rows=max(
            0, len(row_indices) - run_failed_count - existing_output_count
        ),
        skipped_rows=(
            resume_skipped_count + modality_skipped_count + existing_output_count
        ),
        failed_rows=run_failed_count,
        success_label="segmented",
        resumed_rows=resume_skipped_count - resume_failed_count + existing_output_count,
        extra_counts={
            "skipped by resume after prior failure": resume_failed_count,
            "skipped by modality": modality_skipped_count,
            "skipped with existing masks": existing_output_count,
        },
    )
    logger.info("Segmentation done ✔")
    if debug_enabled:
        _log_multiprocessing_leftovers()

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
