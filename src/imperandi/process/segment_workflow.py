"""Segmentation orchestration and workflow helpers."""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from imperandi.utils.manifest import load_manifest
from imperandi.utils.misc import report_volumes  # type: ignore

from .segment_config import (
    _handle_postprocess_missing_inputs,
    _handle_postprocess_output_failure,
    _register_postprocess_out_key,
    load_tasks_config,
    resolve_manifest_fast_default,
    resolve_postprocess_operations,
    resolve_task_fast_and_extra,
    warn_postprocess_collisions,
)
from .segment_io import (
    clean_and_merge_masks,
    diff_changed_outputs,
    discover_segmentation_outputs,
    mask_column_for_output_file,
    register_output_key_map,
    snapshot_segmentation_outputs,
)

logger = logging.getLogger(__name__)

WorkerResult = Tuple[int, Optional[str], Optional[str], Optional[str]]
TASK_STATE_FILENAME = ".imperandi_segment_tasks.json"


def _build_postprocess_output_column_map(
    tasks_config: Dict[str, Any],
) -> Dict[str, str]:
    """Build ``{output_filename_lower: output_column}`` for postprocess operations."""
    mapping: Dict[str, str] = {}
    for op in resolve_postprocess_operations(tasks_config.get("postprocess")):
        output_name = str(op["output_name"]).strip().lower()
        if not output_name:
            continue
        mapping[output_name] = str(op["output_column"])
    return mapping


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


TOTALSEG_TASK_TO_MODEL_IDS: Dict[str, List[int]] = {
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


def _prefetch_variant_name(task_name: str, task_fast: bool) -> str:
    """Return the concrete prefetch variant for the selected fast mode."""
    if task_name == "total" and task_fast:
        return "total_fast"
    if task_name == "total_mr" and task_fast:
        return "total_fast_mr"
    if task_name == "body" and task_fast:
        return "body_fast"
    if task_name == "body_mr" and task_fast:
        return "body_mr_fast"
    return task_name


def _resolve_prefetch_tasks(tasks: List[Any], *, fast: bool) -> List[str]:
    """Resolve concrete task names whose weights should be prefetched."""
    resolved_tasks: List[str] = []
    for idx, task in enumerate(tasks):
        valid_task, task_name = _validate_and_get_task(task, task_index=idx)
        task_fast, _ = resolve_task_fast_and_extra(
            valid_task,
            task_index=idx,
            default_fast=fast,
            emit_warning=False,
        )
        resolved_tasks.append(_prefetch_variant_name(task_name, task_fast))

    return resolved_tasks


def prefetch_totalsegmentator_models(
    tasks_config: Dict[str, Any],
    *,
    fast: bool,
) -> None:
    """Download required TotalSegmentator weights before multiprocessing."""
    if tasks_config.get("backend", "totalsegmentator") != "totalsegmentator":
        return

    tasks = tasks_config.get("tasks", [])
    if not tasks:
        return

    resolved_tasks = _resolve_prefetch_tasks(tasks, fast=fast)
    if not resolved_tasks:
        return

    missing = [
        name for name in resolved_tasks if name not in TOTALSEG_TASK_TO_MODEL_IDS
    ]
    if missing:
        logger.warning(
            "Skipping model prefetch for unknown tasks: %s", ", ".join(missing)
        )

    task_ids: List[int] = []
    for name in resolved_tasks:
        ids = TOTALSEG_TASK_TO_MODEL_IDS.get(name)
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


def _default_process_single_volume(*args: Any, **kwargs: Any) -> WorkerResult:
    """Late-bind to `segment.process_single_volume` to avoid import cycles."""
    from . import segment as segment_module

    return segment_module.process_single_volume(*args, **kwargs)


def _validate_and_get_task(task: Any, *, task_index: int) -> Tuple[Dict[str, Any], str]:
    """Validate one task object and return ``(task_dict, task_name)``."""
    task_path = f"segmentation.tasks[{task_index}]"
    if not isinstance(task, dict):
        raise ValueError(
            f"{task_path} must be a JSON object, got {type(task).__name__}."
        )
    if "task" not in task:
        raise ValueError(f"{task_path}.task is required.")
    task_name = str(task["task"]).strip()
    if not task_name:
        raise ValueError(f"{task_path}.task cannot be empty.")
    return task, task_name


def _resolve_declared_task_outputs(task: Dict[str, Any]) -> List[str]:
    """Return declared output filenames for one task, when provided."""
    declared: List[str] = []

    output = task.get("output")
    if isinstance(output, str) and output.strip():
        declared.append(Path(output.strip()).name)

    outputs = task.get("outputs")
    if isinstance(outputs, list):
        for item in outputs:
            if isinstance(item, str) and item.strip():
                declared.append(Path(item.strip()).name)

    key = task.get("key")
    if isinstance(key, str) and key.strip():
        candidate = f"{key.strip()}.nii.gz"
        declared.append(candidate)

    unique: List[str] = []
    seen: set[str] = set()
    for name in declared:
        lower = name.lower()
        if lower in seen:
            continue
        seen.add(lower)
        unique.append(name)
    return unique


def _normalize_nifti_filename(value: str) -> str:
    """Normalize one filename to a basename with a NIfTI extension."""
    name = Path(str(value).strip()).name
    if not name:
        return ""
    lower = name.lower()
    if lower.endswith(".nii.gz") or lower.endswith(".nii"):
        return name
    return f"{name}.nii.gz"


def _dedupe_output_names(names: List[str]) -> List[str]:
    """Return output names deduplicated case-insensitively preserving order."""
    unique: List[str] = []
    seen: set[str] = set()
    for name in names:
        normalized = _normalize_nifti_filename(name)
        if not normalized:
            continue
        lower = normalized.lower()
        if lower in seen:
            continue
        seen.add(lower)
        unique.append(normalized)
    return unique


def _resolve_existing_output_name(output_dir: Path, output_name: str) -> str | None:
    """Resolve one expected output name to an existing filename in output_dir."""
    normalized = _normalize_nifti_filename(output_name)
    if not normalized:
        return None
    primary = output_dir / normalized
    if primary.exists():
        return primary.name

    lower = normalized.lower()
    alt: Path | None = None
    if lower.endswith(".nii.gz"):
        alt = output_dir / f"{normalized[: -len('.nii.gz')]}.nii"
    elif lower.endswith(".nii"):
        alt = output_dir / f"{normalized[: -len('.nii')]}.nii.gz"
    if alt is not None and alt.exists():
        return alt.name
    return None


def _resolve_existing_output_paths(
    output_dir: Path,
    output_names: List[str],
) -> List[Path]:
    """Resolve expected outputs to existing files; returns [] if any are missing."""
    found: List[Path] = []
    for raw_name in _dedupe_output_names(output_names):
        resolved_name = _resolve_existing_output_name(output_dir, raw_name)
        if resolved_name is None:
            return []
        found.append(output_dir / resolved_name)
    return found


def _infer_task_outputs_from_extra(extra: Dict[str, Any]) -> List[str]:
    """Infer expected output filenames from known TotalSegmentator task extras."""
    inferred: List[str] = []
    seen: set[str] = set()
    for key in ("roi_subset", "roi_subset_robust"):
        raw = extra.get(key)
        values: List[str] = []
        if isinstance(raw, str):
            if raw.strip():
                values = [raw.strip()]
        elif isinstance(raw, (list, tuple, set)):
            values = [str(item).strip() for item in raw if str(item).strip()]

        for value in values:
            normalized = _normalize_nifti_filename(value)
            if not normalized:
                continue
            lower = normalized.lower()
            if lower in seen:
                continue
            seen.add(lower)
            inferred.append(normalized)
    return inferred


def _segment_task_signature(
    *,
    task_name: str,
    task_fast: bool,
    extra: Dict[str, Any],
) -> str:
    """Build a stable signature for one concrete task invocation."""
    return json.dumps(
        {"task": str(task_name), "fast": bool(task_fast), "extra": dict(extra)},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _load_segment_task_state(
    output_dir: Path,
    *,
    logger_obj: logging.Logger = logger,
) -> Dict[str, List[str]]:
    """Load per-task completion state for one output directory."""
    state_path = output_dir / TASK_STATE_FILENAME
    if not state_path.exists():
        return {}
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger_obj.warning("Failed to read task state %s: %s", state_path, exc)
        return {}

    if isinstance(raw, dict) and isinstance(raw.get("completed"), dict):
        completed_raw = raw.get("completed")
    elif isinstance(raw, dict):
        completed_raw = raw
    else:
        completed_raw = None
    if not isinstance(completed_raw, dict):
        return {}

    state: Dict[str, List[str]] = {}
    for key, value in completed_raw.items():
        if not isinstance(key, str):
            continue
        raw_outputs: List[str]
        if isinstance(value, str):
            raw_outputs = [value]
        elif isinstance(value, list):
            raw_outputs = [str(item) for item in value]
        else:
            raw_outputs = []
        normalized = _dedupe_output_names(raw_outputs)
        if normalized:
            state[key] = normalized
    return state


def _save_segment_task_state(
    output_dir: Path,
    state: Dict[str, List[str]],
    *,
    logger_obj: logging.Logger = logger,
) -> None:
    """Persist per-task completion state for one output directory."""
    state_path = output_dir / TASK_STATE_FILENAME
    payload = {
        "completed": {
            key: _dedupe_output_names(value)
            for key, value in sorted(state.items())
            if _dedupe_output_names(value)
        }
    }
    tmp_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(state_path)
    except Exception as exc:
        logger_obj.warning("Failed to write task state %s: %s", state_path, exc)
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def _update_segment_task_state_entry(
    state: Dict[str, List[str]],
    *,
    task_signature: str,
    output_dir: Path,
    candidate_outputs: List[str],
) -> bool:
    """Update one state entry from candidate outputs; returns whether state changed."""
    resolved_outputs: List[str] = []
    seen: set[str] = set()
    for raw_name in candidate_outputs:
        existing_name = _resolve_existing_output_name(output_dir, raw_name)
        if existing_name is None:
            continue
        lower = existing_name.lower()
        if lower in seen:
            continue
        seen.add(lower)
        resolved_outputs.append(existing_name)

    if not resolved_outputs:
        return False
    previous = state.get(task_signature)
    if previous == resolved_outputs:
        return False
    state[task_signature] = resolved_outputs
    return True


def _coerce_nifti_path(value: Any) -> Path | None:
    """Coerce one table value into a valid path-like value, when possible."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return Path(text)


def _count_unique_segment_patients(
    df: pd.DataFrame,
    *,
    mask: pd.Series | None = None,
) -> int | None:
    """Return unique patient count when `patient_key` exists."""
    if "patient_key" not in df.columns:
        return None
    series = df.loc[mask, "patient_key"] if mask is not None else df["patient_key"]
    valid = series[series.notna()].astype(str)
    return int(valid.nunique())


def _is_segment_volume_already_processed(
    *,
    nifti_path: Path,
    tasks_config: Dict[str, Any],
    fast: bool,
) -> bool:
    """Return whether a volume would be fully skipped under `force=False`."""
    tasks = tasks_config.get("tasks", [])
    if not tasks:
        return False

    output_dir = nifti_path.parent
    key_to_output: Dict[str, str] = {}
    task_state = _load_segment_task_state(output_dir, logger_obj=logger)
    register_output_key_map(
        key_to_output,
        discover_segmentation_outputs(output_dir, nifti_path),
        warnings=None,
    )
    default_fast = resolve_manifest_fast_default(
        tasks_config,
        cli_fast=fast,
        emit_warning=False,
    )

    for idx, task in enumerate(tasks):
        valid_task, task_name = _validate_and_get_task(task, task_index=idx)
        _, extra = resolve_task_fast_and_extra(
            valid_task,
            task_index=idx,
            default_fast=default_fast,
            emit_warning=False,
        )
        task_signature = _segment_task_signature(
            task_name=task_name,
            task_fast=False,
            extra=extra,
        )
        # Recompute with concrete task_fast because signature includes fast.
        task_fast, _ = resolve_task_fast_and_extra(
            valid_task,
            task_index=idx,
            default_fast=default_fast,
            emit_warning=False,
        )
        task_signature = _segment_task_signature(
            task_name=task_name,
            task_fast=task_fast,
            extra=extra,
        )

        declared_outputs = _resolve_declared_task_outputs(valid_task)
        inferred_outputs = _infer_task_outputs_from_extra(extra)
        skip_outputs = declared_outputs or inferred_outputs
        if skip_outputs:
            skip_paths = _resolve_existing_output_paths(output_dir, skip_outputs)
            if skip_paths:
                register_output_key_map(key_to_output, skip_paths, warnings=None)
                continue

        marker_outputs = task_state.get(task_signature, [])
        if marker_outputs:
            marker_paths = _resolve_existing_output_paths(output_dir, marker_outputs)
            if marker_paths:
                register_output_key_map(key_to_output, marker_paths, warnings=None)
                continue

        if _update_segment_task_state_entry(
            task_state,
            task_signature=task_signature,
            output_dir=output_dir,
            candidate_outputs=declared_outputs + inferred_outputs,
        ):
            continue
        return False

    for op in resolve_postprocess_operations(tasks_config.get("postprocess")):
        missing_keys = [k for k in op["input_keys"] if k not in key_to_output]
        if missing_keys:
            if str(op.get("on_failure", "warn_only")).strip().lower() == "fail":
                return False
            continue

        dst = output_dir / str(op["output_name"])
        if not dst.exists():
            return False
        key_to_output[str(op["out_key"])] = str(op["output_name"])

    return True


def _estimate_segment_workload(
    df: pd.DataFrame,
    *,
    tasks_config: Dict[str, Any],
    fast: bool,
    force: bool,
) -> Dict[str, Any]:
    """Estimate already-processed vs pending segment volumes/patients."""
    total_volumes = int(len(df))
    if total_volumes == 0:
        empty_mask = pd.Series(False, index=df.index, dtype=bool)
        return {
            "already_volumes": 0,
            "pending_volumes": 0,
            "already_patients": 0 if "patient_key" in df.columns else None,
            "pending_patients": 0 if "patient_key" in df.columns else None,
            "already_series": empty_mask,
            "pending_series": empty_mask,
        }

    if force:
        already_series = pd.Series(False, index=df.index, dtype=bool)
        pending_series = ~already_series
        total_patients = _count_unique_segment_patients(df)
        return {
            "already_volumes": 0,
            "pending_volumes": total_volumes,
            "already_patients": 0 if total_patients is not None else None,
            "pending_patients": total_patients,
            "already_series": already_series,
            "pending_series": pending_series,
        }

    already_mask: List[bool] = []
    for _, row in df.iterrows():
        nifti_path = _coerce_nifti_path(row.get("nifti_path"))
        if nifti_path is None or not nifti_path.exists():
            already_mask.append(False)
            continue
        try:
            already_mask.append(
                _is_segment_volume_already_processed(
                    nifti_path=nifti_path,
                    tasks_config=tasks_config,
                    fast=fast,
                )
            )
        except Exception:
            already_mask.append(False)

    already_series = pd.Series(already_mask, index=df.index, dtype=bool)
    pending_series = ~already_series
    return {
        "already_volumes": int(already_series.sum()),
        "pending_volumes": int(pending_series.sum()),
        "already_patients": _count_unique_segment_patients(df, mask=already_series),
        "pending_patients": _count_unique_segment_patients(df, mask=pending_series),
        "already_series": already_series,
        "pending_series": pending_series,
    }


def _process_single_volume_subprocess(
    send_conn: Any,
    process_single_volume_fn: Callable[..., WorkerResult],
    idx: int,
    row: Dict[str, Any],
    tasks_config: Dict[str, Any],
    fast: bool,
    verbose: bool,
    force: bool,
) -> None:
    """Run one worker task in a child process and send one ``WorkerResult``."""
    try:
        result = process_single_volume_fn(
            idx,
            row,
            tasks_config,
            fast=fast,
            verbose=verbose,
            force=force,
        )
    except Exception as exc:
        result = (idx, None, f"worker crash: {exc}", None)

    try:
        send_conn.send(result)
    except Exception:
        pass
    finally:
        try:
            send_conn.close()
        except Exception:
            pass


def run_segment_volume_workflow(
    *,
    nifti_path: Path,
    output_dir: Path,
    tasks_config: Dict[str, Any],
    fast: bool,
    verbose: bool = False,
    force: bool = False,
    backend: TotalSegmentatorBackend | None = None,
    clean_and_merge_masks_fn: Callable[..., bool] = clean_and_merge_masks,
    logger_obj: logging.Logger = logger,
) -> List[str]:
    """Run segmentation tasks and optional postprocess for one volume."""
    tasks = tasks_config.get("tasks", [])
    if not tasks:
        raise ValueError("No tasks provided in config")
    if tasks_config.get("backend", "totalsegmentator") != "totalsegmentator":
        raise ValueError(f"Unsupported backend: {tasks_config.get('backend')}")

    warnings: List[str] = []
    key_to_output: Dict[str, str] = {}
    task_count = len(tasks)
    default_fast = resolve_manifest_fast_default(
        tasks_config,
        cli_fast=fast,
        emit_warning=False,
    )
    segment_backend = backend or TotalSegmentatorBackend()

    existing_outputs = discover_segmentation_outputs(output_dir, nifti_path)
    register_output_key_map(key_to_output, existing_outputs, warnings)

    for idx, task in enumerate(tasks):
        valid_task, task_name = _validate_and_get_task(task, task_index=idx)
        task_fast, extra = resolve_task_fast_and_extra(
            valid_task,
            task_index=idx,
            default_fast=default_fast,
            emit_warning=False,
        )

        declared_outputs = _resolve_declared_task_outputs(valid_task)
        if not force:
            skip_outputs = declared_outputs
            skip_reason = "declared outputs"

            if not skip_outputs:
                inferred_outputs = _infer_task_outputs_from_extra(extra)
                if inferred_outputs:
                    skip_outputs = inferred_outputs
                    skip_reason = "inferred outputs"

            if skip_outputs:
                skip_paths = _resolve_existing_output_paths(output_dir, skip_outputs)
            else:
                skip_paths = []

            if skip_paths:
                if verbose:
                    logger_obj.info(
                        "Skip task %s - %s already exist: %s",
                        task_name,
                        skip_reason,
                        ", ".join(path.name for path in skip_paths),
                    )
                register_output_key_map(key_to_output, skip_paths, warnings)
                continue

            # Fallback for single-task configs where output stem equals task name.
            # In multi-task configs this can incorrectly skip later tasks when an
            # earlier task produced a same-named file.
            if task_count == 1:
                existing_by_task_name = key_to_output.get(task_name)
                if existing_by_task_name:
                    if verbose:
                        logger_obj.info(
                            "Skip task %s - output key already mapped to %s",
                            task_name,
                            existing_by_task_name,
                        )
                    continue

        before_snapshot = snapshot_segmentation_outputs(output_dir, nifti_path)
        try:
            segment_backend.run(
                input_path=nifti_path,
                output_dir=output_dir,
                task=task_name,
                fast=task_fast,
                **extra,
            )
        except Exception as exc:
            logger_obj.error(
                "Segmentation failed on %s (%s): %s", nifti_path, task_name, exc
            )
            raise

        after_files = discover_segmentation_outputs(output_dir, nifti_path)
        changed_outputs = diff_changed_outputs(
            before_snapshot,
            snapshot_segmentation_outputs(output_dir, nifti_path),
        )
        if not changed_outputs:
            if not after_files:
                raise RuntimeError(
                    f"No segmentation masks found after task '{task_name}'."
                )
            message = f"No new or updated segmentation outputs detected for task '{task_name}'."
            logger_obj.warning(message)
            warnings.append(message)
        elif verbose:
            logger_obj.info(
                "Task %s produced/updated %d output(s): %s",
                task_name,
                len(changed_outputs),
                ", ".join(changed_outputs),
            )
        register_output_key_map(key_to_output, after_files, warnings)

    postprocess_ops = resolve_postprocess_operations(tasks_config.get("postprocess"))
    if not postprocess_ops:
        return warnings

    warn_postprocess_collisions(
        {
            mask_column_for_output_file(Path(filename))
            for filename in key_to_output.values()
        },
        set(key_to_output.values()),
        postprocess_ops,
        warnings=warnings,
    )

    fail_policy_errors: List[Exception] = []
    for op in postprocess_ops:
        try:
            missing_keys = [k for k in op["input_keys"] if k not in key_to_output]
            if missing_keys:
                _handle_postprocess_missing_inputs(op, missing_keys, warnings)
                continue

            merge_files = [key_to_output[k] for k in op["input_keys"]]
            dst = output_dir / str(op["output_name"])
            if dst.exists() and not force:
                if verbose:
                    logger_obj.info("Skip %s - file exists", dst)
                _register_postprocess_out_key(key_to_output, op, warnings)
                continue

            merged_ok = clean_and_merge_masks_fn(
                output_dir,
                merge_files,
                output_name=str(op["output_name"]),
                radius_mm=float(op["radius_mm"]),
                verbose=verbose,
                close=bool(op["close"]),
                fill_holes=bool(op["fill_holes"]),
                largest_cc=bool(op["largest_cc"]),
            )
            if not merged_ok or not dst.exists():
                _handle_postprocess_output_failure(op, dst, warnings)
                continue
            _register_postprocess_out_key(key_to_output, op, warnings)
        except Exception as exc:
            if str(op.get("on_failure", "warn_only")).strip().lower() == "fail":
                fail_policy_errors.append(exc)
                logger_obj.error("%s", exc)
            else:
                message = (
                    f"Postprocess operation {op['index']} failed with error: {exc}"
                )
                logger_obj.warning(message)
                warnings.append(message)
            continue

    if fail_policy_errors:
        if len(fail_policy_errors) > 1:
            logger_obj.error(
                "Postprocess encountered %d fail-policy errors; raising first.",
                len(fail_policy_errors),
            )
        raise fail_policy_errors[0]

    return warnings


def run_segment_batch_workflow(
    *,
    args: Any,
    manifest_base_path: Path,
    process_single_volume_fn: Callable[
        ..., WorkerResult
    ] = _default_process_single_volume,
    prefetch_totalsegmentator_models_fn: Callable[
        ..., None
    ] = prefetch_totalsegmentator_models,
    process_pool_executor_cls: Callable[..., Any] = ProcessPoolExecutor,
    as_completed_fn: Callable[..., Any] = as_completed,
    tqdm_fn: Callable[..., Any] = tqdm,
    report_volumes_fn: Callable[[pd.DataFrame], Any] = report_volumes,
    logger_obj: logging.Logger = logger,
) -> None:
    """Run CSV-level multiprocessing segmentation workflow."""
    # Kept for backward-compatible seams; scheduler no longer depends on futures.
    _ = process_pool_executor_cls, as_completed_fn

    manifest = load_manifest(
        getattr(args, "manifest", None),
        base_path=manifest_base_path,
    )
    tasks_config = load_tasks_config(
        Path(args.tasks_config) if args.tasks_config else None,
        manifest=manifest,
    )
    effective_fast = resolve_manifest_fast_default(
        tasks_config,
        cli_fast=args.fast,
        emit_warning=True,
    )
    prefetch_totalsegmentator_models_fn(tasks_config, fast=effective_fast)
    postprocess_output_column_map = _build_postprocess_output_column_map(tasks_config)

    df = pd.read_csv(args.csv_path).copy()
    if "nifti_path" not in df.columns:
        unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
        if unnamed:
            df = df.drop(columns=unnamed)
    if "nifti_path" not in df.columns:
        raise KeyError("column 'nifti_path' missing")
    df = df.drop_duplicates("nifti_path").copy()
    df["warning_message"] = None
    if args.num_workers < 1:
        raise ValueError("num_workers must be >= 1")

    workload = _estimate_segment_workload(
        df,
        tasks_config=tasks_config,
        fast=effective_fast,
        force=bool(args.force),
    )
    already_patients = workload["already_patients"]
    pending_patients = workload["pending_patients"]
    if already_patients is not None and pending_patients is not None:
        logger_obj.info(
            "Segment workload: %d patient(s) already processed (%d volume(s)); "
            "%d patient(s) will be processed now (%d volume(s)).",
            int(already_patients),
            int(workload["already_volumes"]),
            int(pending_patients),
            int(workload["pending_volumes"]),
        )
    else:
        logger_obj.info(
            "Segment workload: %d volume(s) already processed; "
            "%d volume(s) will be processed now.",
            int(workload["already_volumes"]),
            int(workload["pending_volumes"]),
        )

    already_series = workload["already_series"]
    pending_series = workload["pending_series"]

    precompleted_results: List[WorkerResult] = []
    for idx, row in df.loc[already_series].iterrows():
        source_nifti = _coerce_nifti_path(row.get("nifti_path"))
        if source_nifti is None:
            precompleted_results.append(
                (int(idx), None, "missing nifti_path for pre-skipped row", None)
            )
            continue
        precompleted_results.append((int(idx), str(source_nifti.parent), None, None))

    ctx = mp.get_context(args.start_method)
    pending = deque((idx, row.to_dict()) for idx, row in df.loc[pending_series].iterrows())
    total_jobs = len(pending)
    progress_iter = iter(tqdm_fn(range(total_jobs), total=total_jobs, desc="Segment"))
    results: List[WorkerResult] = list(precompleted_results)
    completed_rows: set[int] = {int(result[0]) for result in precompleted_results}
    active: Dict[int, Dict[str, Any]] = {}

    def _mark_completed(result: WorkerResult) -> None:
        row_idx = int(result[0])
        if row_idx in completed_rows:
            return
        completed_rows.add(row_idx)
        results.append(result)
        try:
            next(progress_iter)
        except StopIteration:
            pass

    while pending or active:
        while pending and len(active) < int(args.num_workers):
            row_idx, row_dict = pending.popleft()
            recv_conn, send_conn = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=_process_single_volume_subprocess,
                args=(
                    send_conn,
                    process_single_volume_fn,
                    int(row_idx),
                    row_dict,
                    tasks_config,
                    effective_fast,
                    bool(args.verbose),
                    bool(args.force),
                ),
            )
            try:
                proc.start()
            except Exception as exc:
                try:
                    recv_conn.close()
                except Exception:
                    pass
                try:
                    send_conn.close()
                except Exception:
                    pass
                _mark_completed((int(row_idx), None, f"worker crash: {exc}", None))
                continue

            send_conn.close()
            active[int(row_idx)] = {
                "process": proc,
                "conn": recv_conn,
                "started_at": time.monotonic(),
            }

        loop_made_progress = False
        now = time.monotonic()
        for row_idx, state in list(active.items()):
            proc = state["process"]
            conn = state["conn"]
            conn_broken = False

            try:
                has_message = conn.poll()
            except (BrokenPipeError, OSError):
                has_message = False
                conn_broken = True

            if has_message:
                try:
                    result = conn.recv()
                except (EOFError, BrokenPipeError, OSError):
                    result = (row_idx, None, "worker crash: no result returned", None)
                try:
                    conn.close()
                except Exception:
                    pass
                proc.join(timeout=0)
                active.pop(row_idx, None)
                _mark_completed(result)
                loop_made_progress = True
                continue

            elapsed = now - float(state["started_at"])
            if proc.is_alive() and elapsed > float(args.timeout_sec):
                logger_obj.warning(
                    "Row %d exceeded %ss - killed", row_idx, args.timeout_sec
                )
                proc.terminate()
                proc.join(timeout=1.0)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=1.0)
                try:
                    conn.close()
                except Exception:
                    pass
                active.pop(row_idx, None)
                _mark_completed((row_idx, None, "timeout", None))
                loop_made_progress = True
                continue

            if not proc.is_alive():
                proc.join(timeout=0)
                try:
                    if conn_broken:
                        result = (
                            row_idx,
                            None,
                            "worker crash: no result returned",
                            None,
                        )
                    elif conn.poll(0.05):
                        result = conn.recv()
                    else:
                        result = (
                            row_idx,
                            None,
                            "worker crash: no result returned",
                            None,
                        )
                except (EOFError, BrokenPipeError, OSError):
                    result = (row_idx, None, "worker crash: no result returned", None)
                try:
                    conn.close()
                except Exception:
                    pass
                active.pop(row_idx, None)
                _mark_completed(result)
                loop_made_progress = True

        if active and not loop_made_progress:
            time.sleep(0.05)

    errors: List[Dict[str, Any]] = []
    for idx, out_dir, err_msg, warning_msg in results:
        if out_dir:
            base = Path(out_dir)
            source_nifti = Path(df.at[idx, "nifti_path"])
            row_warnings: List[str] = []

            discovered_masks = discover_segmentation_outputs(base, source_nifti)
            if not discovered_masks:
                row_warnings.append(f"no segmentation masks found in {base}")

            for mask_path in discovered_masks:
                mask_col = postprocess_output_column_map.get(
                    mask_path.name.lower(), mask_column_for_output_file(mask_path)
                )
                if mask_col not in df.columns:
                    df[mask_col] = None
                prev = df.at[idx, mask_col]
                if pd.notna(prev) and str(prev) != str(mask_path):
                    row_warnings.append(
                        f"mask column collision for {mask_col}: {prev} -> {mask_path}"
                    )
                df.at[idx, mask_col] = str(mask_path)

            if warning_msg:
                row_warnings.append(warning_msg)
            if row_warnings:
                df.at[idx, "warning_message"] = " | ".join(row_warnings)
        else:
            errors.append({"idx": idx, "error_message": err_msg or "unknown"})

    df.to_csv(args.csv_path_out, index=False)
    logger_obj.info("Wrote main table -> %s", args.csv_path_out)

    if errors:
        err_idx = [r["idx"] for r in errors]
        err_df = df.loc[err_idx].copy()
        err_df["error_message"] = [r["error_message"] for r in errors]
        err_df.to_csv(args.error_csv_path, index=False)
        logger_obj.warning("%d rows failed - see %s", len(err_df), args.error_csv_path)
        try:
            report_volumes_fn(err_df)
        except Exception:
            logger_obj.debug("report_volumes() failed - continuing")

    logger_obj.info("All done")
