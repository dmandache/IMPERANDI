"""Ordered, mask-aware intensity processing of scalar 3-D NIfTI volumes."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from imperandi.utils.checkpoint_cli import add_checkpoint_arguments
from imperandi.utils.logging import log_task_summary, setup_logging
from imperandi.utils.manifest import load_manifest
from imperandi.utils.misc import print_args
from imperandi.utils.run_state import (
    CheckpointManager,
    atomic_write_csv,
    atomic_write_json,
    fingerprint_file,
    prepare_resume_context,
)

logger = logging.getLogger(__name__)
ALGORITHM_VERSION = 1
METHODS = {
    "bias_correction": ("n4",),
    "contrast": ("percentile", "window", "gamma", "histogram"),
    "normalization": ("zscore", "robust_zscore", "minmax", "percentile"),
}
METHOD_ALIASES = {
    "z-score": "zscore",
    "z_score": "zscore",
    "robust-zscore": "robust_zscore",
}
STEP_DEFAULTS = {
    ("bias_correction", "n4"): {
        "iterations": [50, 50, 30, 20],
        "shrink_factor": 2,
        "convergence_threshold": 0.001,
    },
    ("contrast", "percentile"): {"percentiles": [1.0, 99.0]},
    ("contrast", "window"): {},
    ("contrast", "gamma"): {"gamma": 1.0},
    ("contrast", "histogram"): {"bins": 256},
    ("normalization", "zscore"): {},
    ("normalization", "robust_zscore"): {},
    ("normalization", "minmax"): {"output_range": [0.0, 1.0]},
    ("normalization", "percentile"): {
        "percentiles": [1.0, 99.0],
        "output_range": [0.0, 1.0],
    },
}


def _mapping(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _unknown_keys(value: dict, allowed, label: str) -> None:
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(
            f"Unknown {label} options: {', '.join(sorted(map(str, unknown)))}"
        )


def _number(value, label: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return float(value)


def _integer(value, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _pair(value, label: str, *, percentiles: bool = False) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain two numbers")
    low, high = [_number(v, label) for v in value]
    if low >= high or (percentiles and not 0 <= low < high <= 100):
        raise ValueError(
            f"Invalid {label}: expected increasing bounds"
            + (" in [0, 100]" if percentiles else "")
        )
    return [low, high]


def validate_step(raw: dict) -> dict:
    """Validate one step and materialize all algorithm defaults."""
    raw = _mapping(raw, "postprocessing step")
    kind = raw.get("type")
    method = raw.get("method", "n4" if kind == "bias_correction" else None)
    if not isinstance(kind, str) or kind not in METHODS or not isinstance(method, str):
        raise ValueError("Each step requires a supported type and method")
    method = METHOD_ALIASES.get(method, method)
    if method not in METHODS[kind]:
        raise ValueError(f"Unsupported {kind} method: {method!r}")
    defaults = STEP_DEFAULTS[kind, method]
    allowed = {"type", "method", "name", *defaults}
    if method == "window":
        allowed.update({"lower", "upper"})
    _unknown_keys(raw, allowed, f"{kind}.{method}")
    step = {**deepcopy(defaults), **raw, "type": kind, "method": method}
    if "name" in step and (
        not isinstance(step["name"], str) or not step["name"].strip()
    ):
        raise ValueError("Step name must be a non-empty string")
    if "percentiles" in step:
        step["percentiles"] = _pair(
            step["percentiles"], "percentiles", percentiles=True
        )
    if "output_range" in step:
        step["output_range"] = _pair(step["output_range"], "output_range")
    if method == "window":
        step["lower"], step["upper"] = _pair(
            [step.get("lower"), step.get("upper")], "window"
        )
    if method == "gamma":
        step["gamma"] = _number(step["gamma"], "gamma", positive=True)
    if method == "histogram":
        step["bins"] = _integer(step["bins"], "bins", 2)
    if method == "n4":
        iterations = step["iterations"]
        if not isinstance(iterations, list) or not iterations:
            raise ValueError("N4 iterations must be a non-empty list")
        step["iterations"] = [_integer(v, "N4 iterations") for v in iterations]
        step["shrink_factor"] = _integer(step["shrink_factor"], "shrink_factor")
        step["convergence_threshold"] = _number(
            step["convergence_threshold"], "convergence_threshold", positive=True
        )
    return step


def _validate_profile(raw: dict) -> dict:
    raw = _mapping(raw, "postprocessing profile")
    _unknown_keys(raw, {"steps", "mask", "mask_column", "outside_mask"}, "profile")
    profile = {
        "mask": "nonzero",
        "mask_column": None,
        "outside_mask": "preserve",
        **raw,
    }
    if profile["mask"] not in ("all", "nonzero", "positive"):
        raise ValueError("mask must be all, nonzero, or positive")
    if profile["outside_mask"] not in ("preserve", "zero"):
        raise ValueError("outside_mask must be preserve or zero")
    column = profile["mask_column"]
    if column is not None and (not isinstance(column, str) or not column.strip()):
        raise ValueError("mask_column must be a non-empty string")
    steps = profile.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("image_postprocessing requires a non-empty steps list")
    profile["steps"] = [validate_step(step) for step in steps]
    bias_positions = [
        i
        for i, step in enumerate(profile["steps"])
        if step["type"] == "bias_correction"
    ]
    if bias_positions and bias_positions != [0]:
        raise ValueError(
            "Bias correction must occur once, before contrast and normalization"
        )
    return profile


def validate_config(raw: dict) -> dict:
    """Validate global steps or complete per-modality profiles (no implicit merge)."""
    raw = _mapping(raw, "image_postprocessing")
    _unknown_keys(
        raw,
        {"version", "modalities", "steps", "mask", "mask_column", "outside_mask"},
        "image_postprocessing",
    )
    if type(raw.get("version", 1)) is not int or raw.get("version", 1) != 1:
        raise ValueError("image_postprocessing.version must be 1")
    if "modalities" not in raw:
        return {
            "version": 1,
            **_validate_profile({k: v for k, v in raw.items() if k != "version"}),
        }
    if set(raw) - {"version", "modalities"}:
        raise ValueError("Use either global steps or modalities, not both")
    modalities = _mapping(raw["modalities"], "image_postprocessing.modalities")
    if not modalities:
        raise ValueError("image_postprocessing.modalities must not be empty")
    profiles = {}
    for key, profile in modalities.items():
        modality = str(key).strip().upper()
        modality = "MR" if modality == "MRI" else modality
        if modality not in ("CT", "MR") or modality in profiles:
            raise ValueError(f"Unsupported or duplicate modality: {key!r}")
        profiles[modality] = _validate_profile(profile)
    return {"version": 1, "modalities": profiles}


def profile_for_row(config: dict, row) -> dict | None:
    if "modalities" not in config:
        return {k: v for k, v in config.items() if k != "version"}
    modality = str(row.get("Modality", "")).strip().upper()
    return config["modalities"].get("MR" if modality == "MRI" else modality)


def _load_sitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError(
            "N4 bias correction requires optional dependencies. Install with: pip install 'imperandi[postprocess]'"
        ) from exc
    return sitk


def _n4(
    data: np.ndarray, mask: np.ndarray, affine: np.ndarray, step: dict
) -> np.ndarray:
    sitk = _load_sitk()
    if np.any(data[mask] <= 0):
        raise ValueError(
            "N4 requires positive intensities inside the mask; use a positive or explicit foreground mask"
        )
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    direction = affine[:3, :3] / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError(
            "N4 requires an orthogonal image grid; sheared affines are unsupported"
        )
    image = sitk.GetImageFromArray(
        np.asarray(data, dtype=np.float32).transpose(2, 1, 0)
    )
    image.SetSpacing(tuple(spacing))
    ras_to_lps = np.diag([-1.0, -1.0, 1.0])
    image.SetDirection(tuple((ras_to_lps @ direction).ravel()))
    image.SetOrigin(tuple(ras_to_lps @ affine[:3, 3]))
    mask_image = sitk.GetImageFromArray(mask.astype(np.uint8).transpose(2, 1, 0))
    mask_image.CopyInformation(image)
    shrink = step["shrink_factor"]
    if any(size // shrink < 4 for size in data.shape):
        raise ValueError(
            "N4 needs at least 4 voxels per axis after shrinking; reduce shrink_factor"
        )
    small_image = sitk.Shrink(image, [shrink] * 3)
    small_mask = sitk.Shrink(mask_image, [shrink] * 3)
    if not np.any(sitk.GetArrayViewFromImage(small_mask)):
        raise ValueError("N4 mask is empty after shrinking; reduce shrink_factor")
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetNumberOfThreads(1)
    corrector.SetMaximumNumberOfIterations(step["iterations"])
    corrector.SetConvergenceThreshold(step["convergence_threshold"])
    corrector.Execute(small_image, small_mask)
    # Estimate on the smaller image, then evaluate the field on the original grid.
    log_field = sitk.GetArrayFromImage(
        corrector.GetLogBiasFieldAsImage(image)
    ).transpose(2, 1, 0)
    return data[mask] / np.exp(log_field[mask])


def _stats(values: np.ndarray) -> dict:
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def process_array(
    data: np.ndarray, affine: np.ndarray, profile: dict, mask: np.ndarray | None = None
) -> tuple[np.ndarray, dict]:
    """Process selected voxels; the original selection remains fixed across steps.

    All input voxels must be finite. An explicit mask replaces automatic masking.
    Empty masks fail; zero-variance normalization maps selected voxels to zero
    (or the lower output bound for range normalization).
    """
    profile = _validate_profile(profile)
    if np.iscomplexobj(data):
        raise ValueError("Postprocessing requires real scalar intensities")
    if profile["mask_column"] and mask is None:
        raise ValueError("A mask array is required when mask_column is configured")
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 3 or not data.size:
        raise ValueError("Postprocessing supports non-empty scalar 3-D images only")
    if not np.all(np.isfinite(data)):
        raise ValueError("Input image contains NaN or infinite intensities")
    affine = np.asarray(affine, dtype=float)
    if (
        affine.shape != (4, 4)
        or not np.all(np.isfinite(affine))
        or abs(np.linalg.det(affine[:3, :3])) < 1e-12
    ):
        raise ValueError("Input image has an invalid spatial affine")
    if mask is None:
        mask = {
            "all": lambda: np.ones(data.shape, dtype=bool),
            "nonzero": lambda: data != 0,
            "positive": lambda: data > 0,
        }[profile["mask"]]()
    else:
        mask = np.asarray(mask)
        if mask.shape != data.shape or not np.all(np.isfinite(mask)):
            raise ValueError("Mask must have the image shape and finite values")
        mask = mask > 0
    if not mask.any():
        raise ValueError("Postprocessing mask is empty")
    result = data.copy()
    report = {"before": _stats(result[mask]), "steps": []}
    for step in profile["steps"]:
        values = result[mask]
        kind, method = step["type"], step["method"]
        details = {"type": kind, "method": method}
        if kind == "bias_correction":
            transformed = _n4(result, mask, affine, step)
        elif kind == "contrast":
            if method in ("percentile", "window"):
                low, high = (
                    np.percentile(values, step["percentiles"])
                    if method == "percentile"
                    else (step["lower"], step["upper"])
                )
                transformed = np.clip(values, low, high)
                details.update(lower=float(low), upper=float(high))
            elif method == "gamma":
                low, high = values.min(), values.max()
                transformed = (
                    values
                    if high == low
                    else low
                    + ((values - low) / (high - low)) ** step["gamma"] * (high - low)
                )
            else:
                if values.min() == values.max():
                    transformed = np.zeros_like(values)
                else:
                    counts, edges = np.histogram(values, bins=step["bins"])
                    cdf = counts.cumsum() / values.size
                    transformed = np.interp(values, (edges[:-1] + edges[1:]) / 2, cdf)
        else:
            if method == "zscore":
                center, scale = values.mean(), values.std()
            elif method == "robust_zscore":
                center = np.median(values)
                scale = 1.4826 * np.median(np.abs(values - center))
            else:
                low, high = (
                    np.percentile(values, step["percentiles"])
                    if method == "percentile"
                    else (values.min(), values.max())
                )
                center, scale = low, high - low
                values = np.clip(values, low, high)
            degenerate = bool(scale == 0)
            transformed = (
                np.zeros_like(values) if degenerate else (values - center) / scale
            )
            if "output_range" in step:
                low_out, high_out = step["output_range"]
                transformed = transformed * (high_out - low_out) + low_out
            details.update(
                center=float(center), scale=float(scale), degenerate=degenerate
            )
            if degenerate:
                logger.warning(
                    "%s has zero intensity spread; using a constant output", method
                )
        if not np.all(np.isfinite(transformed)):
            raise ValueError(f"{kind}.{method} produced non-finite intensities")
        result[mask] = transformed
        report["steps"].append(details)
    if profile["outside_mask"] == "zero":
        result[~mask] = 0
    if np.any(np.abs(result) > np.finfo(np.float32).max):
        raise ValueError("Processed intensities exceed the float32 output range")
    result = result.astype(np.float32)
    report["after"] = _stats(result[mask].astype(np.float64))
    return result, report


def _atomic_save_image(image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=".postprocess-", suffix=".nii.gz", dir=path.parent
    )
    os.close(fd)
    temporary = Path(name)
    try:
        nib.save(image, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_scalar_image(path: Path):
    if not path.name.lower().endswith((".nii", ".nii.gz")):
        raise ValueError(f"Expected a NIfTI .nii or .nii.gz file: {path}")
    image = nib.load(path)
    if len(image.shape) != 3 or np.issubdtype(
        image.get_data_dtype(), np.complexfloating
    ):
        raise ValueError("Postprocessing supports real scalar 3-D NIfTI images only")
    return image


def process_volume(
    input_path: Path,
    output_dir: Path,
    profile: dict,
    *,
    mask_path: Path | None = None,
    resume: bool = True,
    force: bool = False,
    strict: bool = False,
) -> tuple[Path, Path, bool]:
    """Write a derived image and provenance, or reuse a verified matching artifact."""
    profile = _validate_profile(profile)
    if profile["mask_column"] and mask_path is None:
        raise ValueError("A mask path is required when mask_column is configured")
    input_path = Path(input_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {input_path}")
    if mask_path is not None:
        mask_path = Path(mask_path).resolve()
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask does not exist: {mask_path}")
    versions = {
        package: importlib.metadata.version(package) for package in ("numpy", "nibabel")
    }
    if any(step["type"] == "bias_correction" for step in profile["steps"]):
        versions["SimpleITK"] = _load_sitk().Version_VersionString()
    signature = {
        "algorithm_version": ALGORITHM_VERSION,
        "input": fingerprint_file(input_path, strict=strict),
        "mask": fingerprint_file(mask_path, strict=strict) if mask_path else None,
        "profile": profile,
        "versions": versions,
    }
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[
        :20
    ]
    stem = (
        input_path.name[:-7]
        if input_path.name.lower().endswith(".nii.gz")
        else input_path.stem
    )
    destination = Path(output_dir).resolve() / f"{stem[:80]}-{digest}" / "image.nii.gz"
    sidecar = destination.with_name("postprocess.json")
    if destination.resolve() in {input_path, mask_path}:
        raise ValueError("Derived output must not overwrite the source image or mask")
    existing = None
    if sidecar.is_file():
        try:
            existing = json.loads(sidecar.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    if (
        resume
        and not force
        and isinstance(existing, dict)
        and existing.get("signature") == signature
        and existing.get("output") == fingerprint_file(destination, strict=strict)
    ):
        return destination, sidecar, True
    image = _read_scalar_image(input_path)
    mask = None
    if mask_path:
        mask_image = _read_scalar_image(mask_path)
        if image.shape != mask_image.shape or not np.allclose(
            image.affine, mask_image.affine, rtol=0, atol=1e-5
        ):
            raise ValueError(
                "Mask geometry must match the image shape and affine; resample the mask first"
            )
        mask = mask_image.get_fdata()
    processed, report = process_array(image.get_fdata(), image.affine, profile, mask)
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    header["cal_min"], header["cal_max"] = float(processed.min()), float(
        processed.max()
    )
    output = image.__class__(processed, image.affine, header=header)
    # Keep both coordinate systems and their codes, including code=0.
    output.set_qform(image.get_qform(), int(image.header["qform_code"]))
    output.set_sform(image.get_sform(), int(image.header["sform_code"]))
    _atomic_save_image(output, destination)
    atomic_write_json(
        sidecar,
        {
            "signature": signature,
            "output": fingerprint_file(destination, strict=strict),
            "statistics": report,
        },
    )
    return destination, sidecar, False


def add_postprocess_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "csv_path_pos", nargs="?", help="Input cohort CSV (default: ./nifti_index.csv)."
    )
    parser.add_argument(
        "output_dir_pos",
        nargs="?",
        help="Derived image root (default: <csv_dir>/POSTPROCESSED).",
    )
    parser.add_argument("--csv_path", "--csv-path", dest="csv_path_opt")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir_opt")
    parser.add_argument(
        "--csv_path_out",
        "--csv-path-out",
        help="Default: <csv_dir>/nifti_index_postprocessed.csv.",
    )
    parser.add_argument(
        "--error_csv_path",
        "--error-csv-path",
        help="Default: <csv_dir>/postprocess_errors.csv.",
    )
    parser.add_argument(
        "--manifest",
        help="Built-in name or YAML path with image_postprocessing settings.",
    )
    parser.add_argument(
        "--bias-correction",
        "--bias_correction",
        choices=("none", *METHODS["bias_correction"]),
    )
    parser.add_argument("--contrast", choices=("none", *METHODS["contrast"]))
    parser.add_argument(
        "--normalization",
        type=lambda value: METHOD_ALIASES.get(value, value),
        choices=("none", *METHODS["normalization"]),
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=[1.0, 99.0],
    )
    parser.add_argument("--window", type=float, nargs=2, metavar=("LOW", "HIGH"))
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--mask", choices=("all", "nonzero", "positive"))
    parser.add_argument(
        "--mask-column",
        "--mask_column",
        help="CSV column with foreground-mask paths; overrides automatic masking.",
    )
    parser.add_argument("--outside-mask", choices=("preserve", "zero"))
    parser.add_argument(
        "--force", action="store_true", help="Recompute matching derived images."
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the effective configuration without writing files.",
    )
    add_checkpoint_arguments(parser, default_rows=50, default_sec=300)


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, add_help=add_help)
    add_postprocess_arguments(parser)
    return parser


def normalize_postprocess_args(args: argparse.Namespace) -> argparse.Namespace:
    csv_path = (
        Path(args.csv_path_opt or args.csv_path_pos or "nifti_index.csv")
        .expanduser()
        .resolve()
    )
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Not a CSV file: {csv_path}")
    args.csv_path = str(csv_path)
    args.output_dir = str(
        Path(
            args.output_dir_opt
            or args.output_dir_pos
            or csv_path.parent / "POSTPROCESSED"
        )
        .expanduser()
        .resolve()
    )
    args.csv_path_out = str(
        Path(args.csv_path_out or csv_path.parent / "nifti_index_postprocessed.csv")
        .expanduser()
        .resolve()
    )
    args.error_csv_path = str(
        Path(args.error_csv_path or csv_path.parent / "postprocess_errors.csv")
        .expanduser()
        .resolve()
    )
    if len({args.csv_path, args.csv_path_out, args.error_csv_path}) != 3:
        raise ValueError("Input, output, and error CSV paths must be distinct")
    if any(
        Path(path).suffix.lower() != ".csv"
        for path in (args.csv_path_out, args.error_csv_path)
    ):
        raise ValueError("Output and error tables must use .csv paths")
    for key in ("csv_path_pos", "csv_path_opt", "output_dir_pos", "output_dir_opt"):
        delattr(args, key)
    manifest = load_manifest(
        args.manifest, base_path=Path(__file__).resolve().parents[1]
    )
    raw = manifest.get("image_postprocessing")
    explicit_methods = any(getattr(args, kind) is not None for kind in METHODS)
    if explicit_methods:
        if raw is not None:
            logger.warning(
                "CLI method flags replace manifest image_postprocessing profiles"
            )
        steps = []
        for kind in METHODS:
            method = getattr(args, kind)
            if method is None or method == "none":
                continue
            step = {"type": kind, "method": method}
            if method == "percentile":
                step["percentiles"] = args.percentiles
            if method == "window":
                if args.window is None:
                    raise ValueError("--contrast window requires --window LOW HIGH")
                step["lower"], step["upper"] = args.window
            if method == "gamma":
                step["gamma"] = args.gamma
            steps.append(step)
        raw = {"steps": steps}
    if raw is None:
        raise ValueError(
            "Supply CLI processing methods or a manifest image_postprocessing section"
        )
    raw = deepcopy(_mapping(raw, "image_postprocessing"))
    profiles = (
        _mapping(raw["modalities"], "modalities").values()
        if "modalities" in raw
        else [raw]
    )
    for profile in profiles:
        _mapping(profile, "profile")
        for key in ("mask", "mask_column", "outside_mask"):
            if getattr(args, key) is not None:
                profile[key] = getattr(args, key)
    args.postprocessing_config = validate_config(raw)
    return args


def _row_path(value, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing {label} path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def main(args: argparse.Namespace) -> int:
    """Process a cohort, retaining failed/skipped rows and returning 1 on failures.

    Resume verifies each image's sidecar rather than trusting completed CSV rows,
    so missing/changed artifacts are rebuilt and previous failures are retried.
    """
    config = args.postprocessing_config
    # Keep identifiers such as '001' and 'NA' intact in the derived index.
    df = pd.read_csv(args.csv_path, dtype=str, keep_default_na=False)
    if "nifti_path" not in df:
        raise ValueError("Input CSV requires a nifti_path column")
    if "modalities" in config and "Modality" not in df:
        raise ValueError(
            "Modality column is required for modality-specific postprocessing"
        )
    selected_profiles = [profile_for_row(config, row) for _, row in df.iterrows()]
    for profile in selected_profiles:
        if profile:
            if profile["mask_column"] and profile["mask_column"] not in df:
                raise ValueError(f"Mask column missing: {profile['mask_column']}")
            if any(step["type"] == "bias_correction" for step in profile["steps"]):
                _load_sitk()
    base = Path(args.csv_path).parent
    mask_columns = {
        profile["mask_column"]
        for profile in selected_profiles
        if profile and profile["mask_column"]
    }
    # All path columns remain usable when the derived index is written elsewhere.
    for column in df.columns:
        if column in {
            "nifti_path",
            "source_nifti_path",
            *mask_columns,
        } or column.startswith("mask_"):
            df[column] = df[column].map(
                lambda value: (
                    str(_row_path(value, base, column))
                    if isinstance(value, str) and value.strip()
                    else value
                )
            )
    df["source_nifti_path"] = df["nifti_path"]
    for column in (
        "nifti_path",
        "postprocess_status",
        "postprocess_error",
        "postprocess_provenance",
    ):
        df[column] = (
            df[column].astype(object)
            if column == "nifti_path"
            else pd.Series("", index=df.index, dtype=object)
        )
    context = prepare_resume_context(
        args=args,
        command="postprocess",
        inputs=args.csv_path,
        output_path=args.csv_path_out,
        error_path=args.error_csv_path,
    )
    checkpoint = CheckpointManager(paths=context["paths"], config=context["config"])
    completed = set()
    errors = []
    error_columns = ["idx", "source_nifti_path", "error"]
    counts = {"processed": 0, "resumed": 0, "skipped": 0, "failed": 0}

    def flush(force=False):
        checkpoint.flush(
            main_df=df,
            error_df=pd.DataFrame(errors, columns=error_columns),
            completed_indices=completed,
            force=force,
        )

    try:
        for idx, profile in tqdm(
            enumerate(selected_profiles),
            total=len(df),
            desc="Postprocess",
            unit="volume",
        ):
            if profile is None:
                df.at[idx, "postprocess_status"] = "skipped"
                counts["skipped"] += 1
            else:
                try:
                    source = _row_path(df.at[idx, "source_nifti_path"], base, "image")
                    mask_path = (
                        _row_path(df.at[idx, profile["mask_column"]], base, "mask")
                        if profile["mask_column"]
                        else None
                    )
                    output, sidecar, reused = process_volume(
                        source,
                        Path(args.output_dir),
                        profile,
                        mask_path=mask_path,
                        resume=args.resume,
                        force=args.force,
                        strict=args.strict_resume,
                    )
                    df.at[idx, "nifti_path"] = str(output)
                    df.at[idx, "postprocess_provenance"] = str(sidecar)
                    status = "resumed" if reused else "processed"
                    df.at[idx, "postprocess_status"] = status
                    counts[status] += 1
                except Exception as exc:
                    logger.warning("Postprocessing failed for row %s: %s", idx, exc)
                    df.at[idx, "postprocess_status"] = "failed"
                    df.at[idx, "postprocess_error"] = str(exc)
                    # A failed derivative must not silently become an unprocessed input.
                    df.at[idx, "nifti_path"] = ""
                    errors.append(
                        {
                            "idx": idx,
                            "source_nifti_path": df.at[idx, "source_nifti_path"],
                            "error": str(exc),
                        }
                    )
                    counts["failed"] += 1
            completed.add(str(idx))
            checkpoint.mark_processed()
            flush()
    finally:
        flush(force=True)
    atomic_write_csv(df, args.csv_path_out)
    atomic_write_csv(pd.DataFrame(errors, columns=error_columns), args.error_csv_path)
    checkpoint.finalize_state(completed_indices=completed)
    log_task_summary(
        logger,
        "Image postprocessing",
        total_rows=len(df),
        processed_rows=counts["processed"] + counts["failed"],
        succeeded_rows=counts["processed"],
        failed_rows=counts["failed"],
        skipped_rows=counts["skipped"] + counts["resumed"],
        resumed_rows=counts["resumed"],
    )
    logger.info("Wrote postprocessed table -> %s", args.csv_path_out)
    return 1 if errors else 0


def run(args: argparse.Namespace) -> int:
    """Shared entry point for the top-level and standalone module CLIs."""
    args = normalize_postprocess_args(args)
    if args.dry_run:
        print_args(args)
        return 0
    return main(args)


if __name__ == "__main__":
    args = build_parser().parse_args()
    setup_logging(verbose=args.verbose)
    try:
        exit_code = run(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        exit_code = 2
    raise SystemExit(exit_code)
