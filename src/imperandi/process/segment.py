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
from ast import literal_eval
from collections import deque
import copy
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version as distribution_version
import logging
import multiprocessing as mp
import os
import re
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
from imperandi.utils.logging import log_task_summary, setup_logging
from imperandi.utils.manifest import load_manifest
from imperandi.utils.checkpoint_cli import add_checkpoint_arguments
from imperandi.utils.run_state import (
    atomic_write_csv,
    CheckpointManager,
    ensure_source_id_column,
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

DEFAULT_TIMEOUT = 15 * 60  # seconds – hard wall per study inside the pool
DEFAULT_CHECKPOINT_EVERY_ROWS = 50
DEFAULT_CHECKPOINT_EVERY_SEC = 5 * 60  # seconds

LIVER_LESIONS_MIN_TOTALSEGMENTATOR_VERSION = "2.13.0"
LIVER_LESIONS_TASKS = frozenset({"liver_lesions", "liver_lesions_mr"})

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
        if fname == output_name:
            # Keep merged output intact when destination name overlaps an input.
            continue
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
                    "merge_keys": ["liver", "liver_tumor"],
                    "output": "liver_all",
                    "radius_mm": 5.0,
                    "largest_cc": True,
                    "fill_holes": True,
                    "close": True,
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
                    "merge_keys": ["liver", "liver_tumor"],
                    "output": "liver_all",
                    "radius_mm": 5.0,
                    "largest_cc": True,
                    "fill_holes": True,
                    "close": True,
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
        resolve_merge_outputs(normalized["postprocess"], normalized_tasks)
    return normalized


def validate_segmentation_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the modality-keyed segmentation manifest contract."""
    if not isinstance(config, Mapping):
        raise ValueError("segmentation must be a mapping.")
    backend = str(config.get("backend", "totalsegmentator")).strip().lower()
    if backend != "totalsegmentator":
        raise ValueError(f"Unsupported backend: {backend}")

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

    return {"backend": backend, "modalities": modalities}


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


def _warning_dedupe_key(message: str) -> str:
    if message.startswith(
        "Postprocess output will overwrite existing file and continue:"
    ):
        return "postprocess_overwrite_existing_output"
    if "matches a task output key and will override it." in message:
        return "postprocess_output_key_override"
    return message


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


def _merge_key_to_column(merge_key: str) -> str:
    key = str(merge_key).strip()
    if not key:
        return "mask_unnamed"
    if key.startswith("mask_"):
        return key
    if key.endswith(".nii.gz"):
        key = key[: -len(".nii.gz")]
    return f"mask_{key or 'unnamed'}"


def resolve_merge_outputs(
    postprocess: Dict[str, Any],
    tasks: List[Dict[str, Any]],
    *,
    output_to_column: Dict[str, str] | None = None,
) -> List[str]:
    """Resolve post-processing merge keys to known logical task outputs.

    Raises:
        ValueError: If merge keys are absent or refer to unknown mask columns.
    """
    merge_keys = _as_str_list(postprocess.get("merge_keys"))
    if not merge_keys:
        raise ValueError("postprocess.merge_keys is required for postprocess merging.")

    output_to_column = output_to_column or build_output_column_map(tasks)
    column_to_output: Dict[str, str] = {}
    for output_name, column_name in output_to_column.items():
        if column_name not in column_to_output:
            column_to_output[column_name] = output_name

    normalized_columns = [_merge_key_to_column(k) for k in merge_keys]
    missing = [col for col in normalized_columns if col not in column_to_output]
    if missing:
        raise ValueError(
            "postprocess.merge_keys references unknown mask column(s): "
            + ", ".join(missing)
            + f". merge_keys={merge_keys}, normalized={normalized_columns}, "
            + "available="
            + ", ".join(sorted(column_to_output.keys()))
        )

    return [column_to_output[column] for column in normalized_columns]


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
        if resolved.get("postprocess"):
            output_name = (
                str(resolved["postprocess"].get("output", "merged")).strip() or "merged"
            )
            output_to_column.setdefault(output_name, _output_to_column(output_name))
            output_to_fetch.setdefault(output_name, output_name)
            postprocess_outputs.add(output_name)
    return output_to_column, output_to_fetch, postprocess_outputs


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


def segment_volume(
    nifti_path: Path,
    output_dir: Path,
    tasks_config: Dict[str, Any],
    *,
    verbose: bool = False,
    force: bool = False,
    backend: TotalSegmentatorBackend | None = None,
    resolved_output_to_fetch: Dict[str, str] | None = None,
) -> List[str]:
    """Run segmentation tasks and optional post‐processing."""
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

    def _store_resolved_outputs() -> None:
        if resolved_output_to_fetch is None:
            return
        resolved = dict(output_to_fetch)
        postprocess_config = tasks_config.get("postprocess")
        if postprocess_config:
            merged_output = (
                str(postprocess_config.get("output", "merged")).strip() or "merged"
            )
            resolved[merged_output] = merged_output
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

        before_snapshot = _snapshot_nifti_files(output_dir) if not task_outputs else {}
        expected_paths = [
            output_dir / _output_to_filename(task_fetch_outputs[output_name])
            for output_name in task_outputs
        ]
        if expected_paths and all(dst.exists() for dst in expected_paths) and not force:
            if verbose:
                logger.info("Skip %s – files exist", task_name)
            continue

        try:
            backend.run(
                input_path=nifti_path,
                output_dir=output_dir,
                task=task_name,
                **extra,
            )
            ran_any_task = True
        except Exception as exc:
            logger.error(
                "Segmentation failed on %s (%s): %s", nifti_path, task_name, exc
            )
            raise

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
                raise RuntimeError(
                    f"Could not infer outputs for task '{task_name}' from created segmentations."
                )
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

    postprocess = tasks_config.get("postprocess")
    if not postprocess:
        _store_resolved_outputs()
        return warnings

    merge_files = resolve_merge_outputs(
        postprocess, tasks, output_to_column=output_to_column
    )
    if not merge_files:
        _store_resolved_outputs()
        return warnings

    merged_output = str(postprocess.get("output", "merged")).strip() or "merged"
    merged_name = _output_to_filename(merged_output)
    dst = output_dir / merged_name
    if dst.exists() and not force and not ran_any_task:
        if verbose:
            logger.info(
                "Skip postprocess – output exists and row already has task outputs: %s",
                dst,
            )
        _store_resolved_outputs()
        return warnings
    if dst.exists():
        warnings.append(
            f"Postprocess output will overwrite existing file and continue: {dst}"
        )
    if merged_output in output_to_column:
        warnings.append(
            f"Postprocess output '{merged_output}' matches a task output key and will override it."
        )

    on_failure = str(postprocess.get("on_failure", "warn_only")).strip().lower()
    if on_failure not in {"warn_only", "fail"}:
        raise ValueError(
            f"Invalid postprocess.on_failure='{on_failure}'. Use 'warn_only' or 'fail'."
        )

    merged_ok = clean_and_merge_masks(
        output_dir,
        [_output_to_filename(output_to_fetch.get(name, name)) for name in merge_files],
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

    _store_resolved_outputs()
    return warnings


# -----------------------------------------------------------------------------
# Worker wrapper (called in pool)
# -----------------------------------------------------------------------------


def process_single_volume(
    idx: int,
    row: Dict[str, Any],  # must be JSON‑serialisable
    tasks_config: Dict[str, Any],
    *,
    verbose: bool,
    force: bool,
    backend: TotalSegmentatorBackend | None = None,
) -> Tuple[int, str | None, str | None, str | None, Dict[str, str] | None]:
    """Return ``(idx, output_dir|None, error_msg|None, warning_msg|None, outputs|None)``."""

    setup_logging(verbose=verbose)

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
        warnings = segment_volume(
            nifti_path,
            nifti_path.parent,
            resolved_tasks_config,
            verbose=verbose,
            force=force,
            backend=backend,
            resolved_output_to_fetch=resolved_output_to_fetch,
        )
        warning_msg = " | ".join(warnings) if warnings else None
        return (
            idx,
            str(nifti_path.parent),
            None,
            warning_msg,
            resolved_output_to_fetch,
        )
    except Exception as exc:
        # Capture full traceback for later debugging
        logger.debug("Traceback for %s:\n%s", nifti_path.name, traceback.format_exc())
        return idx, None, str(exc), None, None


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

    del args.csv_path_pos
    del args.csv_path_opt
    if hasattr(args, "csv_path_out_pos"):
        del args.csv_path_out_pos

    return args


def main(args: argparse.Namespace) -> None:
    """Run manifest-configured segmentation over every eligible cohort row."""
    setup_logging(verbose=getattr(args, "verbose", False))
    output_path = Path(args.csv_path_out)
    error_path = Path(args.error_csv_path)
    manifest_arg = getattr(args, "manifest", None)
    manifest_config = load_manifest(
        manifest_arg or "generic",
        base_path=Path(__file__).resolve().parents[1],
    )
    tasks_config = load_segmentation_config(
        manifest_arg,
        base_path=Path(__file__).resolve().parents[1],
    )
    source_id_signature = source_id_resume_signature(args.csv_path)
    checkpoint_signature = {
        "manifest_config": manifest_config,
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
        "resume",
        "checkpoint_every_rows",
        "checkpoint_every_sec",
        "strict_resume",
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
        return

    from imperandi.utils.multiprocessing import (
        apply_strategy_env,
        strategy_to_log_dict,
        decide_multiprocessing_strategy,
    )

    # Decide
    strategy = decide_multiprocessing_strategy(
        prefer_gpu=True,
        requested_workers=args.num_workers,
        start_method_hint=args.start_method,
        target_task_mem_mb=6000,  # tune (TotalSegmentator can be heavy)
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
                    row_warnings.append(f"missing merged mask: {mask_path}")
                else:
                    row_warnings.append(f"missing mask: {mask_path}")
            if warning_msg:
                warning_messages = [
                    message.strip()
                    for message in warning_msg.split(" | ")
                    if message and message.strip()
                ]
                for message in warning_messages:
                    key = _warning_dedupe_key(message)
                    is_global_warning = key in {
                        "postprocess_overwrite_existing_output",
                        "postprocess_output_key_override",
                    }
                    if key not in logged_warning_keys:
                        logger.warning(message)
                        logged_warning_keys.add(key)
                    if is_global_warning:
                        continue
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
    ) -> Dict[
        int, Tuple[int, str | None, str | None, str | None, Dict[str, str] | None]
    ]:
        out: Dict[
            int, Tuple[int, str | None, str | None, str | None, Dict[str, str] | None]
        ] = {}
        if not row_indices:
            return out

        max_in_flight = max(1, int(strategy.max_in_flight))
        row_queue = deque(row_indices)
        broken_pool = False

        def _record_result(idx: int, result: Tuple[Any, ...]) -> None:
            normalized = _normalize_process_result(result)
            out[idx] = normalized
            if on_result is not None:
                on_result(*normalized)
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
                return ProcessPoolExecutor(
                    max_workers=effective_workers, mp_context=ctx
                )

        def _shutdown_pool(pool: ProcessPoolExecutor, force: bool) -> None:
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
                    row = df.loc[idx].to_dict()
                    resolved_config = resolve_segmentation_config_for_modality(
                        tasks_config, row.get("Modality")
                    )
                    if resolved_config is None:
                        raise RuntimeError(
                            f"No segmentation model configured for row {idx}."
                        )
                    fut = pool.submit(
                        process_single_volume,
                        idx,
                        row,
                        resolved_config,
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
                    _record_result(
                        i,
                        (i, None, f"timeout after {effective_timeout}s", None, None),
                    )

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
                        res = (
                            i,
                            None,
                            f"worker crash: {type(exc).__name__}: {exc}",
                            None,
                            None,
                        )

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
                logger.error(
                    "BrokenProcessPool during completion loop; aborting fast: %s", exc
                )
            finally:
                _shutdown_pool(pool, force=(broken_pool or restart_pool_for_timeout))

            if broken_pool:
                msg = broken_pool_msg or "BrokenProcessPool"
                for i in list(futures.values()):
                    out[i] = (i, None, msg, None, None)
                for i in row_queue:
                    out[i] = (i, None, msg, None, None)
                break

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
            out.update(_run_rows(chunk, progress_bar=progress_bar, on_result=on_result))
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
                progress_bar.total = (progress_bar.total or 0) + len(retry_indices)
                progress_bar.refresh()
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

    ckpt.finalize_state(completed_indices=completed_indices)
    run_failed_count = len(processed_source_ids & set(errors_by_idx))
    log_task_summary(
        logger,
        "Segmentation",
        total_rows=len(df),
        processed_rows=len(row_indices),
        succeeded_rows=max(0, len(row_indices) - run_failed_count),
        skipped_rows=resume_skipped_count + modality_skipped_count,
        failed_rows=run_failed_count,
        success_label="segmented",
        extra_counts={
            "skipped by resume": resume_skipped_count,
            "skipped by modality": modality_skipped_count,
        },
    )
    logger.info("Segmentation done ✔")

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
