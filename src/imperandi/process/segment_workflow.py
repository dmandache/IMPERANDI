"""Class-based segmentation orchestration and workflow helpers."""

from __future__ import annotations

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
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


def _build_postprocess_output_column_map(tasks_config: Dict[str, Any]) -> Dict[str, str]:
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
        if not isinstance(task, dict):
            raise ValueError(
                f"segmentation.tasks[{idx}] must be a JSON object, got {type(task).__name__}."
            )
        if "task" not in task:
            continue

        task_name = str(task["task"])
        task_fast, _ = resolve_task_fast_and_extra(
            task,
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

    missing = [name for name in resolved_tasks if name not in TOTALSEG_TASK_TO_MODEL_IDS]
    if missing:
        logger.warning("Skipping model prefetch for unknown tasks: %s", ", ".join(missing))

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


@dataclass(frozen=True)
class VolumeSeams:
    """Runtime seams used by `_SegmentVolumePipeline`."""

    clean_and_merge_masks: Callable[..., bool]
    logger: logging.Logger

    @classmethod
    def default(cls) -> "VolumeSeams":
        return cls(
            clean_and_merge_masks=clean_and_merge_masks,
            logger=logger,
        )


@dataclass(frozen=True)
class BatchSeams:
    """Runtime seams used by `_SegmentBatchRunner`."""

    process_single_volume: Callable[..., WorkerResult]
    prefetch_totalsegmentator_models: Callable[..., None]
    process_pool_executor_cls: Callable[..., Any]
    as_completed: Callable[..., Any]
    tqdm: Callable[..., Any]
    report_volumes: Callable[[pd.DataFrame], Any]
    logger: logging.Logger

    @classmethod
    def default(cls) -> "BatchSeams":
        return cls(
            process_single_volume=_default_process_single_volume,
            prefetch_totalsegmentator_models=prefetch_totalsegmentator_models,
            process_pool_executor_cls=ProcessPoolExecutor,
            as_completed=as_completed,
            tqdm=tqdm,
            report_volumes=report_volumes,
            logger=logger,
        )


class _SegmentVolumePipeline:
    """Coordinator for one volume segmentation + postprocess run."""

    def __init__(
        self,
        *,
        nifti_path: Path,
        output_dir: Path,
        tasks_config: Dict[str, Any],
        fast: bool,
        verbose: bool = False,
        force: bool = False,
        backend: TotalSegmentatorBackend | None = None,
        seams: VolumeSeams | None = None,
    ) -> None:
        self.nifti_path = nifti_path
        self.output_dir = output_dir
        self.tasks_config = tasks_config
        self.fast = fast
        self.verbose = verbose
        self.force = force
        self.backend = backend
        self.seams = seams or VolumeSeams.default()

    def run(self) -> List[str]:
        """Run all configured tasks and optional postprocess steps."""
        warnings: List[str] = []
        default_fast = resolve_manifest_fast_default(
            self.tasks_config, cli_fast=self.fast, emit_warning=False
        )
        tasks = self._validated_tasks()
        backend = self.backend or TotalSegmentatorBackend()
        key_to_output: Dict[str, str] = {}

        existing_outputs = discover_segmentation_outputs(self.output_dir, self.nifti_path)
        register_output_key_map(key_to_output, existing_outputs, warnings)

        for idx, task in enumerate(tasks):
            self._run_single_task(
                backend=backend,
                task=task,
                task_index=idx,
                default_fast=default_fast,
                key_to_output=key_to_output,
                warnings=warnings,
            )

        self._run_postprocess_steps(key_to_output=key_to_output, warnings=warnings)
        return warnings

    def _validated_tasks(self) -> List[Any]:
        tasks = self.tasks_config.get("tasks", [])
        if not tasks:
            raise ValueError("No tasks provided in config")

        backend_name = self.tasks_config.get("backend", "totalsegmentator")
        if backend_name != "totalsegmentator":
            raise ValueError(f"Unsupported backend: {backend_name}")
        return tasks

    def _run_single_task(
        self,
        *,
        backend: TotalSegmentatorBackend,
        task: Any,
        task_index: int,
        default_fast: bool,
        key_to_output: Dict[str, str],
        warnings: List[str],
    ) -> None:
        if not isinstance(task, dict):
            raise ValueError(
                f"segmentation.tasks[{task_index}] must be a JSON object, got {type(task).__name__}."
            )
        task_name = task["task"]
        task_fast, extra = resolve_task_fast_and_extra(
            task,
            task_index=task_index,
            default_fast=default_fast,
            emit_warning=False,
        )

        before_snapshot = snapshot_segmentation_outputs(self.output_dir, self.nifti_path)
        try:
            backend.run(
                input_path=self.nifti_path,
                output_dir=self.output_dir,
                task=task_name,
                fast=task_fast,
                **extra,
            )
        except Exception as exc:
            self.seams.logger.error(
                "Segmentation failed on %s (%s): %s", self.nifti_path, task_name, exc
            )
            raise

        after_files = discover_segmentation_outputs(self.output_dir, self.nifti_path)
        after_snapshot = snapshot_segmentation_outputs(self.output_dir, self.nifti_path)
        changed_outputs = diff_changed_outputs(before_snapshot, after_snapshot)
        if not changed_outputs:
            if not after_files:
                raise RuntimeError(f"No segmentation masks found after task '{task_name}'.")
            message = (
                f"No new or updated segmentation outputs detected for task '{task_name}'."
            )
            if self.force:
                raise RuntimeError(message)
            self.seams.logger.warning(message)
            warnings.append(message)
        elif self.verbose:
            self.seams.logger.info(
                "Task %s produced/updated %d output(s): %s",
                task_name,
                len(changed_outputs),
                ", ".join(changed_outputs),
            )

        register_output_key_map(key_to_output, after_files, warnings)

    def _run_postprocess_steps(
        self, *, key_to_output: Dict[str, str], warnings: List[str]
    ) -> None:
        postprocess_ops = resolve_postprocess_operations(self.tasks_config.get("postprocess"))
        if not postprocess_ops:
            return

        existing_columns = {
            mask_column_for_output_file(Path(filename))
            for filename in key_to_output.values()
        }
        existing_output_names = set(key_to_output.values())
        warn_postprocess_collisions(existing_columns, existing_output_names, postprocess_ops)

        for op in postprocess_ops:
            missing_keys = [k for k in op["input_keys"] if k not in key_to_output]
            if missing_keys:
                _handle_postprocess_missing_inputs(op, missing_keys, warnings)
                continue

            merge_files = [key_to_output[k] for k in op["input_keys"]]
            dst = self.output_dir / str(op["output_name"])
            if dst.exists() and not self.force:
                if self.verbose:
                    self.seams.logger.info("Skip %s - file exists", dst)
                _register_postprocess_out_key(key_to_output, op, warnings)
                continue

            merged_ok = self.seams.clean_and_merge_masks(
                self.output_dir,
                merge_files,
                output_name=str(op["output_name"]),
                radius_mm=float(op["radius_mm"]),
                verbose=self.verbose,
                close=bool(op["close"]),
                fill_holes=bool(op["fill_holes"]),
                largest_cc=bool(op["largest_cc"]),
            )
            if not merged_ok or not dst.exists():
                _handle_postprocess_output_failure(op, dst, warnings)
                continue
            _register_postprocess_out_key(key_to_output, op, warnings)


class _SegmentBatchRunner:
    """Coordinator for CSV-level multiprocessing segmentation runs."""

    def __init__(
        self,
        *,
        args: Any,
        manifest_base_path: Path,
        seams: BatchSeams | None = None,
    ) -> None:
        self.args = args
        self.manifest_base_path = manifest_base_path
        self.seams = seams or BatchSeams.default()

    def run(self) -> None:
        tasks_config, effective_fast = self._load_config_and_prefetch()
        postprocess_output_column_map = _build_postprocess_output_column_map(tasks_config)
        df = self._load_dataframe()
        results = self._run_pool(df, tasks_config, effective_fast)
        errors = self._consolidate_results(df, results, postprocess_output_column_map)
        self._write_output_tables(df, errors)
        self.seams.logger.info("All done")

    def _load_config_and_prefetch(self) -> Tuple[Dict[str, Any], bool]:
        manifest = load_manifest(
            getattr(self.args, "manifest", None),
            base_path=self.manifest_base_path,
        )
        tasks_config = load_tasks_config(
            Path(self.args.tasks_config) if self.args.tasks_config else None,
            manifest=manifest,
        )
        effective_fast = resolve_manifest_fast_default(
            tasks_config,
            cli_fast=self.args.fast,
            emit_warning=True,
        )
        self.seams.prefetch_totalsegmentator_models(tasks_config, fast=effective_fast)
        return tasks_config, effective_fast

    def _load_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.args.csv_path).copy()
        if "nifti_path" not in df.columns:
            unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
            if unnamed:
                df = df.drop(columns=unnamed)
        if "nifti_path" not in df.columns:
            raise KeyError("column 'nifti_path' missing")
        df = df.drop_duplicates("nifti_path").copy()
        df["warning_message"] = None
        return df

    def _run_pool(
        self,
        df: pd.DataFrame,
        tasks_config: Dict[str, Any],
        effective_fast: bool,
    ) -> List[WorkerResult]:
        ctx = mp.get_context(self.args.start_method)
        results: List[WorkerResult] = []
        with self.seams.process_pool_executor_cls(
            max_workers=self.args.num_workers,
            mp_context=ctx,
        ) as pool:
            futures = {
                pool.submit(
                    self.seams.process_single_volume,
                    idx,
                    row.to_dict(),
                    tasks_config,
                    fast=effective_fast,
                    verbose=self.args.verbose,
                    force=self.args.force,
                ): idx
                for idx, row in df.iterrows()
            }

            for fut in self.seams.tqdm(
                self.seams.as_completed(futures), total=len(futures), desc="Segment"
            ):
                i = futures[fut]
                try:
                    res = fut.result(timeout=self.args.timeout_sec)
                except TimeoutError:
                    self.seams.logger.warning(
                        "Row %d exceeded %ds - killed", i, self.args.timeout_sec
                    )
                    fut.cancel()
                    res = (i, None, "timeout", None)
                except Exception as exc:
                    res = (i, None, f"worker crash: {exc}", None)
                results.append(res)

            pool.shutdown(wait=False, cancel_futures=True)
            processes = getattr(pool, "_processes", None)
            if processes:
                for proc in processes.values():
                    if proc.is_alive():
                        proc.kill()

        return results

    def _consolidate_results(
        self,
        df: pd.DataFrame,
        results: List[WorkerResult],
        postprocess_output_column_map: Dict[str, str],
    ) -> List[Dict[str, Any]]:
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
                    mapped_column = postprocess_output_column_map.get(mask_path.name.lower())
                    if mapped_column:
                        mask_col = mapped_column
                    else:
                        mask_col = mask_column_for_output_file(mask_path)
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

        return errors

    def _write_output_tables(self, df: pd.DataFrame, errors: List[Dict[str, Any]]) -> None:
        df.to_csv(self.args.csv_path_out, index=False)
        self.seams.logger.info("Wrote main table -> %s", self.args.csv_path_out)

        if errors:
            err_idx = [r["idx"] for r in errors]
            err_df = df.loc[err_idx].copy()
            err_df["error_message"] = [r["error_message"] for r in errors]
            err_df.to_csv(self.args.error_csv_path, index=False)
            self.seams.logger.warning(
                "%d rows failed - see %s", len(err_df), self.args.error_csv_path
            )
            try:
                self.seams.report_volumes(err_df)
            except Exception:
                self.seams.logger.debug("report_volumes() failed - continuing")
