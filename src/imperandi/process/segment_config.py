"""Configuration and postprocess helpers for segmentation workflows."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _default_tasks_config() -> Dict[str, Any]:
    """Compute the default config value."""
    return {
        "backend": "totalsegmentator",
        "tasks": [
            {
                "task": "total",
                "extra": {"roi_subset_robust": ["liver"]},
            },
            {
                "task": "liver_vessels",
                "extra": {},
            },
        ],
        "postprocess": {
            "merge_keys": ["liver", "liver_tumor"],
            "out_key": "merged",
            "radius_mm": 5.0,
            "largest_cc": True,
            "fill_holes": True,
            "close": True,
        },
    }


def load_tasks_config(
    path: Path | None,
    *,
    manifest: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Load segmentation tasks from JSON, then manifest, then builtin defaults."""
    if path is None:
        manifest_segmentation = (manifest or {}).get("segmentation")
        if manifest_segmentation is not None:
            if not isinstance(manifest_segmentation, dict):
                raise ValueError("Manifest key 'segmentation' must be a JSON object.")
            return copy.deepcopy(manifest_segmentation)
        return _default_tasks_config()

    if not path.exists():
        raise FileNotFoundError(f"Tasks config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_bool_flag(value: Any, *, field: str) -> bool:
    """Coerce a manifest/CLI boolean-like value into ``bool``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"{field} must be boolean-like, got {value!r}.")


def resolve_manifest_fast_default(
    tasks_config: Dict[str, Any],
    cli_fast: bool,
    *,
    emit_warning: bool = True,
) -> bool:
    """Resolve global default fast value from CLI and optional ``segmentation.fast``."""
    resolved = bool(cli_fast)
    if "fast" not in tasks_config:
        return resolved

    manifest_fast = _coerce_bool_flag(
        tasks_config["fast"],
        field="segmentation.fast",
    )
    if emit_warning:
        logger.warning(
            "Manifest fast setting detected at segmentation.fast=%s; "
            "overriding CLI --fast=%s.",
            manifest_fast,
            bool(cli_fast),
        )
    return manifest_fast


def resolve_task_fast_and_extra(
    task: Dict[str, Any],
    *,
    task_index: int,
    default_fast: bool,
    emit_warning: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """Resolve per-task fast flag and return sanitized ``extra`` kwargs."""
    task_path = f"segmentation.tasks[{task_index}]"
    task_name = str(task.get("task", f"task_{task_index}"))

    if "fast" in task:
        task_fast: bool | None = _coerce_bool_flag(
            task["fast"], field=f"{task_path}.fast"
        )
    else:
        task_fast = None

    extra_raw = task.get("extra", {})
    if extra_raw is None:
        extra: Dict[str, Any] = {}
    elif isinstance(extra_raw, dict):
        extra = dict(extra_raw)
    else:
        raise ValueError(f"{task_path}.extra must be a JSON object when provided.")

    if "fast" in extra:
        extra_fast: bool | None = _coerce_bool_flag(
            extra.pop("fast"),
            field=f"{task_path}.extra.fast",
        )
    else:
        extra_fast = None

    resolved_fast = default_fast
    resolved_source = "default"
    if task_fast is not None:
        resolved_fast = task_fast
        resolved_source = f"{task_path}.fast"
    elif extra_fast is not None:
        resolved_fast = extra_fast
        resolved_source = f"{task_path}.extra.fast"

    if task_fast is not None and extra_fast is not None and task_fast != extra_fast:
        if emit_warning:
            logger.warning(
                "Conflicting fast settings for task '%s' (%s.fast=%s, %s.extra.fast=%s). "
                "Using %s.fast.",
                task_name,
                task_path,
                task_fast,
                task_path,
                extra_fast,
                task_path,
            )
        resolved_fast = task_fast
        resolved_source = f"{task_path}.fast"

    if emit_warning and resolved_source != "default":
        logger.warning(
            "Per-task fast override for task '%s' from %s=%s " "(default fast=%s).",
            task_name,
            resolved_source,
            resolved_fast,
            default_fast,
        )

    return resolved_fast, extra


def _mask_column_from_out_key(out_key: str) -> str:
    """Build output DataFrame column name from an ``out_key`` value."""
    key = str(out_key).strip()
    if not key:
        raise ValueError("postprocess.out_key cannot be empty")
    return f"mask_{key}"


def _normalized_out_key(out_key: str) -> str:
    """Normalize out_key for file-name derivation."""
    key = str(out_key).strip()
    return key


def normalize_postprocess_operations(postprocess: Any) -> List[Dict[str, Any]]:
    """Normalize postprocess config into an ordered list of operation objects."""
    if postprocess is None:
        return []
    if isinstance(postprocess, dict):
        return [postprocess]
    if isinstance(postprocess, list):
        for idx, op in enumerate(postprocess, start=1):
            if not isinstance(op, dict):
                raise ValueError(
                    f"postprocess[{idx}] must be a JSON object, got {type(op).__name__}."
                )
        return postprocess
    raise ValueError(
        "postprocess must be a JSON object or list of objects, "
        f"got {type(postprocess).__name__}."
    )


def resolve_postprocess_operation(
    op: Dict[str, Any], *, op_index: int
) -> Dict[str, Any]:
    """Resolve and validate one postprocess operation."""
    legacy_keys = [
        k for k in ("output", "output_column", "column_name", "output_col") if k in op
    ]
    if legacy_keys:
        raise ValueError(
            "Unsupported postprocess key(s) in operation "
            f"{op_index}: {', '.join(legacy_keys)}. Use postprocess.out_key."
        )

    in_key = str(op.get("in_key", "")).strip()
    raw_merge_keys = op.get("merge_keys")
    if raw_merge_keys is None:
        merge_keys: List[str] = []
    elif isinstance(raw_merge_keys, list):
        merge_keys = [str(k).strip() for k in raw_merge_keys if str(k).strip()]
    else:
        raise ValueError(
            f"postprocess operation {op_index}: merge_keys must be a list when provided."
        )

    if in_key and merge_keys:
        raise ValueError(
            f"postprocess operation {op_index}: provide either in_key or merge_keys, not both."
        )

    if in_key:
        input_keys = [in_key]
    elif merge_keys:
        input_keys = merge_keys
    else:
        raise ValueError(
            f"postprocess operation {op_index}: include either in_key or a non-empty merge_keys list."
        )

    out_key = str(op.get("out_key", "")).strip()
    if not out_key:
        raise ValueError(f"postprocess operation {op_index}: out_key is required.")

    if out_key.lower().endswith(".nii.gz") or out_key.lower().endswith(".nii"):
        raise ValueError(
            f"postprocess operation {op_index}: out_key must not include a NIfTI extension."
        )
    if "/" in out_key or "\\" in out_key:
        raise ValueError(
            f"postprocess operation {op_index}: out_key must be a file stem, not a path."
        )

    output_column = _mask_column_from_out_key(out_key)
    normalized_out_key = _normalized_out_key(out_key)
    output_name = f"{normalized_out_key}.nii.gz"

    on_failure = str(op.get("on_failure", "warn_only")).strip().lower()
    if on_failure not in {"warn_only", "fail"}:
        raise ValueError(
            f"postprocess operation {op_index}: invalid on_failure='{on_failure}'. "
            "Use 'warn_only' or 'fail'."
        )

    if not normalized_out_key:
        raise ValueError(
            f"postprocess operation {op_index}: could not resolve a non-empty output key."
        )

    return {
        "index": op_index,
        "input_keys": input_keys,
        "out_key": normalized_out_key,
        "output_column": output_column,
        "output_name": output_name,
        "radius_mm": float(op.get("radius_mm", 5.0)),
        "close": bool(op.get("close", True)),
        "fill_holes": bool(op.get("fill_holes", True)),
        "largest_cc": bool(op.get("largest_cc", True)),
        "on_failure": on_failure,
    }


def resolve_postprocess_operations(postprocess: Any) -> List[Dict[str, Any]]:
    """Resolve postprocess config into a validated ordered operation list."""
    operations = normalize_postprocess_operations(postprocess)
    return [
        resolve_postprocess_operation(op, op_index=i)
        for i, op in enumerate(operations, start=1)
    ]


def warn_postprocess_collisions(
    existing_columns: set[str],
    existing_output_names: set[str],
    ops: List[Dict[str, Any]],
    warnings: List[str] | None = None,
) -> None:
    """Emit warnings for postprocess column/path collisions and continue."""
    seen_columns = set(existing_columns)
    seen_outputs = set(existing_output_names)

    for op in ops:
        col = str(op["output_column"])
        out_name = str(op["output_name"])
        prefix = f"Postprocess operation {op['index']}: "

        if col in seen_columns:
            msg = (
                prefix + f"output column '{col}' matches an existing mask column; "
                "paths may be overwritten."
            )
            logger.warning(msg)
            if warnings is not None:
                warnings.append(msg)
        seen_columns.add(col)

        if out_name in seen_outputs:
            msg = (
                prefix + f"output file '{out_name}' collides with an existing output; "
                "last writer wins."
            )
            logger.warning(msg)
            if warnings is not None:
                warnings.append(msg)
        seen_outputs.add(out_name)


def get_postprocess_columns_and_outputs(
    tasks_config: Dict[str, Any],
) -> Tuple[List[str], List[Tuple[str, str]], List[Dict[str, Any]]]:
    """Return postprocess columns and ``(column, output_name)`` pairs in op order."""
    ops = resolve_postprocess_operations(tasks_config.get("postprocess"))
    if not ops:
        return [], [], []

    warn_postprocess_collisions(set(), set(), ops)
    columns = [str(op["output_column"]) for op in ops]
    pairs = [(str(op["output_column"]), str(op["output_name"])) for op in ops]
    return columns, pairs, ops


def _handle_postprocess_missing_inputs(
    op: Dict[str, Any], missing_keys: List[str], warnings: List[str]
) -> None:
    """Handle missing input keys for one operation."""
    message = f"Postprocess operation {op['index']} missing input key(s): " + ", ".join(
        sorted(set(missing_keys))
    )
    if op["on_failure"] == "fail":
        raise ValueError(message)
    logger.warning(message)
    warnings.append(message)


def _handle_postprocess_output_failure(
    op: Dict[str, Any], dst: Path, warnings: List[str]
) -> None:
    """Handle missing postprocess output."""
    message = (
        f"Postprocess operation {op['index']} did not produce expected output: {dst}"
    )
    if op["on_failure"] == "fail":
        raise RuntimeError(message)
    logger.warning(message)
    warnings.append(message)


def _register_postprocess_out_key(
    key_to_output: Dict[str, str], op: Dict[str, Any], warnings: List[str]
) -> None:
    """Register postprocess out_key for downstream chained operations."""
    out_key = str(op["out_key"])
    output_name = str(op["output_name"])
    if out_key in key_to_output and key_to_output[out_key] != output_name:
        message = (
            f"Postprocess operation {op['index']} out_key '{out_key}' overrides an existing "
            "key mapping; downstream operations will use the latest mapping."
        )
        logger.warning(message)
        warnings.append(message)
    key_to_output[out_key] = output_name
