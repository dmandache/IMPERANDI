"""Ordered, manifest-configured operations on binary NIfTI masks."""

from __future__ import annotations

from collections.abc import Mapping
import json
from numbers import Real
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage

_ALIASES = {
    "or": "union",
    "and": "intersection",
    "subtract": "difference",
    "dilation": "dilate",
    "erosion": "erode",
    "opening": "open",
    "closing": "close",
    "largest_component": "largest_cc",
}
_LOGICAL = {"union", "intersection", "difference", "xor"}
_MORPHOLOGY = {"dilate", "erode", "open", "close"}
_UNARY = _MORPHOLOGY | {"not", "fill_holes", "largest_cc"}
_STATE_FILENAME = ".imperandi-postprocess.json"


class MaskGeometryError(ValueError):
    """Masks used together do not occupy the same voxel grid."""


def normalize_mask_key(value: Any, *, field: str) -> str:
    """Accept logical names, mask_* columns, and .nii.gz filenames."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty mask name.")
    key = value.strip().removesuffix(".nii.gz").removeprefix("mask_")
    if not key or key in {".", ".."} or "/" in key or "\\" in key:
        raise ValueError(f"{field} must name a mask in the segmentation directory.")
    return key


def validate_operations(
    postprocess: Mapping[str, Any],
    *,
    task_outputs: set[str],
    field: str = "postprocess",
) -> list[dict[str, Any]]:
    """Validate an explicit operation list without mutating the manifest."""
    if not isinstance(postprocess, Mapping):
        raise ValueError(f"{field} must be a mapping with an operations list.")
    if "operations" not in postprocess:
        raise ValueError(
            f"{field}.operations is required and must be a list of sequential mask operations."
        )
    unknown = set(postprocess) - {"operations", "on_failure"}
    if unknown:
        raise ValueError(
            f"{field}.operations cannot be combined with other postprocess settings: "
            + ", ".join(sorted(map(str, unknown)))
        )
    if postprocess.get("on_failure", "fail") not in ("fail", "warn_only"):
        raise ValueError(f"{field}.on_failure must be 'fail' or 'warn_only'.")
    raw_operations = postprocess["operations"]
    if not isinstance(raw_operations, list):
        raise ValueError(f"{field}.operations must be a list.")

    operations = []
    for index, raw in enumerate(raw_operations):
        step_field = f"{field}.operations[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{step_field} must be a mapping.")
        op = raw.get("op")
        if not isinstance(op, str):
            raise ValueError(f"{step_field}.op must be an operation name.")
        op = _ALIASES.get(op.strip().lower(), op.strip().lower())
        if op not in _LOGICAL | _UNARY:
            raise ValueError(f"{step_field}.op has unsupported operation {op!r}.")
        allowed = {"op", "input", "inputs", "output"}
        if op in _MORPHOLOGY:
            allowed |= {"radius_mm", "radius_vox", "iterations"}
        if op in {"fill_holes", "largest_cc"}:
            allowed.add("connectivity")
        if set(raw) - allowed:
            raise ValueError(
                f"{step_field} has unsupported settings for {op}: "
                + ", ".join(sorted(map(str, set(raw) - allowed)))
            )
        if "input" in raw and "inputs" in raw:
            raise ValueError(f"{step_field} must use either input or inputs.")
        inputs = [raw["input"]] if "input" in raw else raw.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise ValueError(f"{step_field}.inputs must be a non-empty list.")
        if op in _UNARY and len(inputs) != 1:
            raise ValueError(f"{step_field}: {op} requires exactly one input.")
        if op == "difference" and len(inputs) < 2:
            raise ValueError(f"{step_field}: {op} requires at least two inputs.")
        step = {
            "op": op,
            "inputs": [
                normalize_mask_key(value, field=f"{step_field}.inputs")
                for value in inputs
            ],
            "output": normalize_mask_key(
                raw.get("output"), field=f"{step_field}.output"
            ),
        }
        if op in _MORPHOLOGY:
            if "radius_mm" in raw and "radius_vox" in raw:
                raise ValueError(
                    f"{step_field} must use either radius_mm or radius_vox."
                )
            radius_key = "radius_vox" if "radius_vox" in raw else "radius_mm"
            radius = raw.get(radius_key, 1.0)
            if (
                isinstance(radius, bool)
                or not isinstance(radius, Real)
                or not np.isfinite(radius)
                or radius < 0
            ):
                raise ValueError(
                    f"{step_field}.{radius_key} must be finite and non-negative."
                )
            iterations = raw.get("iterations", 1)
            if type(iterations) is not int or iterations < 1:
                raise ValueError(f"{step_field}.iterations must be a positive integer.")
            step.update({radius_key: float(radius), "iterations": iterations})
        if op in {"fill_holes", "largest_cc"}:
            connectivity = raw.get("connectivity", 1 if op == "fill_holes" else 3)
            if type(connectivity) is not int or connectivity not in {1, 2, 3}:
                raise ValueError(f"{step_field}.connectivity must be 1, 2, or 3.")
            step["connectivity"] = connectivity
        operations.append(step)

    # Backend outputs may be undeclared; references to later steps, however,
    # must not accidentally load a stale result from disk.
    produced = {step["output"] for step in operations}
    available = set(task_outputs)
    for index, step in enumerate(operations):
        for key in step["inputs"]:
            if key in produced and key not in available and key != step["output"]:
                raise ValueError(
                    f"{field}.operations[{index}] references {key!r} before it is produced."
                )
        available.add(step["output"])
    return operations


def operation_sources(operations: list[dict[str, Any]]) -> list[str]:
    """Return masks loaded from disk, excluding results of preceding steps."""
    produced: set[str] = set()
    sources = []
    for step in operations:
        sources.extend(key for key in step["inputs"] if key not in produced)
        produced.add(step["output"])
    return list(dict.fromkeys(sources))


def _state(
    directory: Path,
    operations: list[dict[str, Any]],
    output_to_fetch: Mapping[str, str],
) -> dict[str, Any]:
    sources = {
        key: output_to_fetch.get(key, key) for key in operation_sources(operations)
    }
    paths = {f"{name}.nii.gz" for name in sources.values()} | {
        f"{output_to_fetch.get(step['output'], step['output'])}.nii.gz"
        for step in operations
    }
    files = {}
    for name in sorted(paths):
        stat = (directory / name).stat()
        files[name] = [stat.st_mtime_ns, stat.st_size]
    return {"version": 2, "operations": operations, "sources": sources, "files": files}


def operations_are_current(directory, operations, output_to_fetch) -> bool:
    """Reuse only a completed sequence whose operations and files still match."""
    if not operations:
        return True
    try:
        recorded = json.loads((directory / _STATE_FILENAME).read_text(encoding="utf-8"))
        return recorded == _state(directory, operations, output_to_fetch)
    except (OSError, ValueError):
        return False


def apply_operations(
    directory: Path,
    operations: list[dict[str, Any]],
    output_to_fetch: Mapping[str, str],
) -> None:
    """Compute the full sequence before writing any named outputs.

    Inputs are binary (positive voxels are foreground). Each step updates its
    named output for subsequent steps; other inputs remain unchanged.
    """
    if not operations:
        return
    masks = {}
    # Fetch every source before doing any work, so missing inputs always fail.
    for key in operation_sources(operations):
        path = directory / f"{output_to_fetch.get(key, key)}.nii.gz"
        img = nib.load(str(path))
        data = np.asanyarray(img.dataobj) > 0
        masks[key] = (data, img.affine, img.header.get_zooms())

    for key, (data, _, _) in masks.items():
        if data.ndim != 3:
            raise MaskGeometryError(
                f"Mask {key!r} must be 3-D; got shape {data.shape}."
            )

    for index, step in enumerate(operations):
        inputs = [masks[key] for key in step["inputs"]]
        mask, affine, zooms = inputs[0]
        for other, other_affine, _ in inputs[1:]:
            if mask.shape != other.shape or not np.allclose(affine, other_affine):
                raise MaskGeometryError(
                    f"postprocess.operations[{index}] ({step['op']}): "
                    f"mask shape or affine mismatch for {step['inputs']}."
                )
        op = step["op"]
        if op in {"union", "intersection", "xor"}:
            reduce_op = {
                "union": np.logical_or,
                "intersection": np.logical_and,
                "xor": np.logical_xor,
            }[op]
            result = mask.copy()
            for other, _, _ in inputs[1:]:
                reduce_op(result, other, out=result)
        elif op == "difference":
            result = mask.copy()
            for other, _, _ in inputs[1:]:
                result &= ~other
        elif op == "not":
            result = ~mask
        elif op in _MORPHOLOGY:
            from skimage.morphology import (
                isotropic_closing,
                isotropic_dilation,
                isotropic_erosion,
                isotropic_opening,
            )

            functions = {
                "dilate": isotropic_dilation,
                "erode": isotropic_erosion,
                "open": isotropic_opening,
                "close": isotropic_closing,
            }
            spacing = np.asarray(
                (1, 1, 1) if "radius_vox" in step else zooms, dtype=float
            )
            if spacing.shape != (3,) or not np.all(
                np.isfinite(spacing) & (spacing > 0)
            ):
                raise MaskGeometryError(
                    "Morphology requires three finite, positive voxel sizes."
                )
            radius = step.get("radius_vox", step.get("radius_mm"))
            iterations = step["iterations"]
            phases = [op]
            if iterations > 1 and op in {"open", "close"}:
                phases = ["erode", "dilate"] if op == "open" else ["dilate", "erode"]
            result = mask.copy()
            if radius > 0:
                for phase in phases:
                    for _ in range(iterations):
                        # Distance transforms of an empty foreground can produce
                        # spurious corner voxels during dilation.
                        if not result.any():
                            break
                        result = functions[phase](result, radius=radius, spacing=spacing)
        elif op == "fill_holes":
            result = ndimage.binary_fill_holes(
                mask,
                structure=ndimage.generate_binary_structure(3, step["connectivity"]),
            )
        else:  # largest_cc
            labels, count = ndimage.label(
                mask,
                structure=ndimage.generate_binary_structure(3, step["connectivity"]),
            )
            if count:
                sizes = np.bincount(labels.ravel())
                sizes[0] = 0
                result = labels == sizes.argmax()
            else:
                result = mask.copy()
        masks[step["output"]] = (result, affine, zooms)

    # Repeated output names save the final version only.
    for key in dict.fromkeys(step["output"] for step in operations):
        mask, affine, _ = masks[key]
        nib.save(
            nib.Nifti1Image(mask.astype(np.uint8), affine),
            directory / f"{output_to_fetch.get(key, key)}.nii.gz",
        )
    state_path = directory / _STATE_FILENAME
    temporary_path = state_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(_state(directory, operations, output_to_fetch), sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(state_path)
