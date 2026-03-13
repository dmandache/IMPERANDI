from __future__ import annotations

import argparse
import itertools
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from imperandi.process import _registration_common as reg_common
from imperandi.process.registration import (
    OrganNormalizeConfig,
    normalize_image_and_masks,
    parse_spacing_csv_value,
    save_transform_artifacts,
)
from imperandi.utils.checkpoint_cli import add_checkpoint_arguments
from imperandi.utils.misc import print_args
from imperandi.utils.run_state import (
    CheckpointManager,
    atomic_write_csv,
    merge_with_existing_output,
    prepare_resume_context,
)

logger = logging.getLogger(__name__)
DEFAULT_CHECKPOINT_EVERY_ROWS = 50
DEFAULT_CHECKPOINT_EVERY_SEC = 5 * 60
TEMPLATE_MODE_SINGLE_SAMPLE = "single_sample"
TEMPLATE_MODE_MEAN_SHAPE = "mean_shape"
TEMPLATE_MODE_PRINCIPAL_VECTORS = "principal_vectors"
TEMPLATE_MODE_CHOICES = (
    TEMPLATE_MODE_SINGLE_SAMPLE,
    TEMPLATE_MODE_MEAN_SHAPE,
    TEMPLATE_MODE_PRINCIPAL_VECTORS,
)
DEFAULT_TEMPLATE_MODE = TEMPLATE_MODE_MEAN_SHAPE
DEFAULT_ORGAN_PRINCIPAL_VECTORS: dict[str, list[list[float]]] = {
    "liver": [
        [-0.83771703, 0.52332983, 0.15606429],
        [0.52885744, 0.70617388, 0.4707741],
        [-0.13616161, -0.47691124, 0.86834076],
    ]
}


def add_register_population_arguments(
    parser: argparse.ArgumentParser,
    include_dry_run: bool = True,
) -> None:
    parser.add_argument(
        "csv_path_pos",
        nargs="?",
        type=str,
        default=None,
        help="Path to input CSV with `nifti_path` and organ mask columns. Defaults to ./nifti_index.csv.",
    )
    parser.add_argument(
        "csv_path_out_pos",
        nargs="?",
        type=str,
        default=None,
        help="Optional output CSV path (positional alternative to --csv_path_out).",
    )
    parser.add_argument("--csv_path", dest="csv_path_opt", type=str)
    parser.add_argument(
        "--csv_path_out",
        type=str,
        default=None,
        help=(
            "Path to save the population-registration CSV. "
            "Defaults to <csv_dir>/<csv_stem>_registered_population.csv."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory used for template artifacts and optional registered outputs.",
    )
    parser.add_argument(
        "--error_csv_path",
        type=str,
        default=None,
        help="Path to save failed rows (default: <csv_dir>/register_population_errors.csv).",
    )
    parser.add_argument(
        "--log_csv_path",
        type=str,
        default=None,
        help="Path to save row-level registration logs (default: <csv_dir>/register_population_log.csv).",
    )
    parser.add_argument(
        "--organ",
        type=str,
        default="liver",
        help="Organ name used to resolve the default mask column (default: liver).",
    )
    parser.add_argument(
        "--mask_column",
        type=str,
        default=None,
        help="Explicit organ mask column to use instead of the default mask_<organ>.",
    )
    parser.add_argument(
        "--template_sample_size",
        type=int,
        default=reg_common.DEFAULT_TEMPLATE_SAMPLE_SIZE,
        help=(
            "Maximum number of valid rows used for reference selection/building "
            "(mean_shape and principal_vectors modes)."
        ),
    )
    parser.add_argument(
        "--template_mode",
        type=str,
        default=DEFAULT_TEMPLATE_MODE,
        choices=list(TEMPLATE_MODE_CHOICES),
        help=(
            "Reference-building mode: "
            "'single_sample' (single exemplar), "
            "'mean_shape' (compute mean organ shape and use it as reference), "
            "'principal_vectors' (align masks to principal axes reference)."
        ),
    )
    parser.add_argument(
        "--template_source_idx",
        type=int,
        default=None,
        help=(
            "Optional source index used as explicit reference sample in "
            "--template_mode single_sample."
        ),
    )
    parser.add_argument(
        "--principal_vectors",
        type=str,
        default=None,
        help=(
            "Optional target principal axes for --template_mode principal_vectors, "
            "formatted as 9 comma-separated floats "
            "(v1x,v1y,v1z,v2x,v2y,v2z,v3x,v3y,v3z). "
            "If omitted, liver uses built-in default vectors; other organs use "
            "population average-shape eigenvectors."
        ),
    )
    parser.add_argument(
        "--template_seed",
        type=int,
        default=reg_common.DEFAULT_TEMPLATE_SEED,
        help="Random seed used when sampling template candidates.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=reg_common.DEFAULT_NUM_WORKERS,
        help="Number of concurrent row workers used for registration.",
    )
    parser.add_argument(
        "--pad_mm",
        type=float,
        default=reg_common.DEFAULT_PAD_MM,
        help="Reserved registration padding in mm for future template refinements.",
    )
    parser.add_argument(
        "--save_registered_outputs",
        action="store_true",
        default=False,
        help="Also resample and save registered images and masks and rewrite their paths in the output CSV.",
    )
    parser.add_argument(
        "--normalize_registered_outputs",
        action="store_true",
        default=False,
        help=(
            "Apply organ extraction + geometry normalization to registered outputs "
            "for downstream inter-patient comparability."
        ),
    )
    parser.add_argument(
        "--normalize_crop_mode",
        type=str,
        default="margin",
        choices=["tight", "margin", "full"],
        help="Organ extraction mode used during normalization.",
    )
    parser.add_argument(
        "--normalize_margin_mm",
        type=float,
        default=10.0,
        help="Organ crop margin in mm when --normalize_crop_mode=margin.",
    )
    parser.add_argument(
        "--normalize_without_background",
        action="store_true",
        default=False,
        help="Remove background outside organ support during normalization.",
    )
    parser.add_argument(
        "--normalize_spacing",
        type=str,
        default=None,
        help=(
            "Target spacing as sx,sy,sz (for example 1.5,1.5,1.5). "
            "Defaults to template spacing when normalization is enabled."
        ),
    )
    parser.add_argument(
        "--normalize_orientation",
        type=str,
        default="LPS",
        help="Target canonical orientation during normalization (default: LPS).",
    )
    parser.add_argument(
        "--disable_normalize_center_organ",
        action="store_true",
        default=False,
        help="Disable organ-centering origin adjustment in normalized outputs.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    add_checkpoint_arguments(
        parser,
        default_rows=DEFAULT_CHECKPOINT_EVERY_ROWS,
        default_sec=DEFAULT_CHECKPOINT_EVERY_SEC,
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
        description=(
            "Build a population liver reference and align cohort rows to it "
            "(single sample, mean shape, or principal vectors)."
        ),
        add_help=add_help,
    )
    add_register_population_arguments(parser)
    return parser


def normalize_register_population_args(args: argparse.Namespace) -> argparse.Namespace:
    csv_in = args.csv_path_opt if args.csv_path_opt is not None else args.csv_path_pos
    csv_path = Path(csv_in) if csv_in else (Path.cwd() / "nifti_index.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Not a CSV file: {csv_path}")

    csv_path = csv_path.resolve()
    args.csv_path = str(csv_path)
    args.output_dir = str(Path(args.output_dir))
    args.organ = str(args.organ).strip().lower()
    args.mask_column = reg_common.resolve_mask_column(
        organ=args.organ,
        mask_column=args.mask_column,
    )
    args.template_sample_size = max(1, int(args.template_sample_size))
    args.template_seed = int(args.template_seed)
    args.num_workers = max(1, int(args.num_workers))
    args.pad_mm = float(args.pad_mm)
    args.normalize_crop_mode = str(
        getattr(args, "normalize_crop_mode", "margin")
    ).strip().lower()
    args.normalize_margin_mm = float(getattr(args, "normalize_margin_mm", 10.0))
    if args.normalize_margin_mm < 0.0:
        raise ValueError("--normalize_margin_mm must be >= 0")
    args.normalize_without_background = bool(
        getattr(args, "normalize_without_background", False)
    )
    args.normalize_spacing = parse_spacing_csv_value(
        getattr(args, "normalize_spacing", None)
    )
    raw_orientation = str(getattr(args, "normalize_orientation", "")).strip()
    args.normalize_orientation = raw_orientation if raw_orientation else None
    args.disable_normalize_center_organ = bool(
        getattr(args, "disable_normalize_center_organ", False)
    )
    args.normalize_registered_outputs = bool(
        getattr(args, "normalize_registered_outputs", False)
    )
    if args.normalize_registered_outputs:
        args.save_registered_outputs = True
    args.template_mode = str(
        getattr(args, "template_mode", DEFAULT_TEMPLATE_MODE)
    ).strip().lower()
    if args.template_mode == "median_samples":
        args.template_mode = TEMPLATE_MODE_MEAN_SHAPE
    if args.template_mode not in TEMPLATE_MODE_CHOICES:
        raise ValueError(
            f"Unsupported --template_mode '{args.template_mode}'. "
            f"Expected one of: {', '.join(TEMPLATE_MODE_CHOICES)}"
        )

    raw_template_source_idx = getattr(args, "template_source_idx", None)
    args.template_source_idx = (
        None if raw_template_source_idx is None else int(raw_template_source_idx)
    )

    args.principal_vectors = _parse_principal_vectors(
        getattr(args, "principal_vectors", None)
    )
    if (
        args.template_source_idx is not None
        and args.template_mode != TEMPLATE_MODE_SINGLE_SAMPLE
    ):
        raise ValueError(
            "--template_source_idx is only supported with --template_mode single_sample"
        )
    if (
        args.principal_vectors is not None
        and args.template_mode != TEMPLATE_MODE_PRINCIPAL_VECTORS
    ):
        raise ValueError(
            "--principal_vectors is only supported with --template_mode principal_vectors"
        )

    csv_path_out_pos = getattr(args, "csv_path_out_pos", None)
    csv_out = args.csv_path_out if args.csv_path_out else csv_path_out_pos
    if csv_out:
        args.csv_path_out = str(Path(csv_out))
    else:
        args.csv_path_out = str(
            csv_path.parent / f"{csv_path.stem}_registered_population.csv"
        )

    if args.error_csv_path:
        args.error_csv_path = str(Path(args.error_csv_path))
    else:
        args.error_csv_path = str(csv_path.parent / "register_population_errors.csv")

    if args.log_csv_path:
        args.log_csv_path = str(Path(args.log_csv_path))
    else:
        args.log_csv_path = str(csv_path.parent / "register_population_log.csv")

    del args.csv_path_pos
    del args.csv_path_opt
    if hasattr(args, "csv_path_out_pos"):
        del args.csv_path_out_pos
    return args


def parse_arguments() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    args = normalize_register_population_args(args)
    logger.info("Running %s with args: %s", Path(__file__).name, args)
    return args


def _template_paths(args: argparse.Namespace) -> dict[str, Path]:
    template_dir = Path(args.output_dir) / "template"
    return {
        "template_dir": template_dir,
        "reference_image_path": template_dir / "template_reference.nii.gz",
        "mask_path": template_dir / f"{args.mask_column}.nii.gz",
        "principal_vectors_path": template_dir / "principal_vectors.json",
    }


def _parse_principal_vectors(value: Any) -> list[list[float]] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if len(parts) != 9:
            raise ValueError(
                "--principal_vectors must contain exactly 9 comma-separated floats"
            )
        try:
            values = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(
                "--principal_vectors must contain numeric values only"
            ) from exc
        matrix = np.asarray(values, dtype=float).reshape(3, 3)
    else:
        matrix = np.asarray(value, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError(
                "--principal_vectors must be a 3x3 matrix or 9 comma-separated floats"
            )
    matrix = _canonicalize_eigen_axes(matrix)
    return matrix.tolist()


def _resolve_template_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "template_mode", DEFAULT_TEMPLATE_MODE)).strip().lower()
    if mode == "median_samples":
        return TEMPLATE_MODE_MEAN_SHAPE
    if mode not in TEMPLATE_MODE_CHOICES:
        return DEFAULT_TEMPLATE_MODE
    return mode


def _project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("Expected a 3x3 matrix")
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def _canonicalize_eigen_axes(matrix: np.ndarray) -> np.ndarray:
    rotation = _project_to_rotation(matrix)
    for axis_index in range(3):
        col = rotation[:, axis_index]
        anchor = int(np.argmax(np.abs(col)))
        if col[anchor] < 0:
            rotation[:, axis_index] *= -1.0
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1.0
    return rotation


def _best_eigenvector_permutation(
    reference_axes: np.ndarray,
    moving_axes: np.ndarray,
) -> tuple[int, int, int]:
    reference_axes = np.asarray(reference_axes, dtype=float)
    moving_axes = np.asarray(moving_axes, dtype=float)
    if reference_axes.shape != (3, 3) or moving_axes.shape != (3, 3):
        raise ValueError("Expected 3x3 axes matrices")
    best_perm: tuple[int, int, int] = (0, 1, 2)
    best_score = -np.inf
    for perm in itertools.permutations(range(3)):
        score = 0.0
        for axis_index in range(3):
            ref_axis = reference_axes[:, axis_index]
            moving_axis = moving_axes[:, perm[axis_index]]
            ref_norm = float(np.linalg.norm(ref_axis))
            moving_norm = float(np.linalg.norm(moving_axis))
            if ref_norm <= 0.0 or moving_norm <= 0.0:
                continue
            score += abs(float(np.dot(ref_axis, moving_axis)) / (ref_norm * moving_norm))
        if score > best_score:
            best_score = score
            best_perm = (int(perm[0]), int(perm[1]), int(perm[2]))
    return best_perm


def _match_eigenvector_basis(
    reference_axes: np.ndarray,
    moving_axes: np.ndarray,
) -> np.ndarray:
    reference_axes = np.asarray(reference_axes, dtype=float)
    moving_axes = np.asarray(moving_axes, dtype=float)
    perm = _best_eigenvector_permutation(reference_axes, moving_axes)
    matched = moving_axes[:, list(perm)].copy()
    for axis_index in range(3):
        if float(np.dot(reference_axes[:, axis_index], matched[:, axis_index])) < 0.0:
            matched[:, axis_index] *= -1.0
    return _project_to_rotation(matched)


def _default_principal_vectors_for_organ(organ: str) -> np.ndarray | None:
    key = str(organ or "").strip().lower()
    matrix = DEFAULT_ORGAN_PRINCIPAL_VECTORS.get(key)
    if matrix is None:
        return None
    return _canonicalize_eigen_axes(np.asarray(matrix, dtype=float))


def _mask_physical_points(mask, *, sitk_module) -> np.ndarray:
    values = sitk_module.GetArrayViewFromImage(mask) > 0
    voxel_zyx = np.argwhere(values)
    if voxel_zyx.size == 0:
        raise ValueError("mask has no positive voxels")
    voxel_xyz = voxel_zyx[:, ::-1].astype(np.float64, copy=False)
    spacing = np.asarray(mask.GetSpacing(), dtype=np.float64)
    origin = np.asarray(mask.GetOrigin(), dtype=np.float64)
    direction = np.asarray(mask.GetDirection(), dtype=np.float64).reshape(3, 3)
    scaled = voxel_xyz * spacing[None, :]
    return origin[None, :] + scaled @ direction.T


def _mask_principal_frame(mask, *, sitk_module) -> dict[str, np.ndarray]:
    points = _mask_physical_points(mask, sitk_module=sitk_module)
    centroid = points.mean(axis=0)
    centered = points - centroid[None, :]
    covariance = centered.T @ centered
    covariance /= max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]
    axes = _canonicalize_eigen_axes(axes)
    return {
        "centroid_mm": centroid.astype(float),
        "axes": axes.astype(float),
    }


def _reference_from_sampled_frames(
    frames: list[dict[str, np.ndarray]],
    *,
    fallback_axes: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if not frames:
        raise ValueError("frames must not be empty")

    anchor_axes = np.asarray(
        fallback_axes if fallback_axes is not None else frames[0]["axes"],
        dtype=float,
    )
    aligned_axes: list[np.ndarray] = []
    centroids: list[np.ndarray] = []
    for frame in frames:
        axes = _match_eigenvector_basis(
            anchor_axes,
            np.asarray(frame["axes"], dtype=float),
        )
        aligned_axes.append(axes)
        centroids.append(np.asarray(frame["centroid_mm"], dtype=float))

    mean_axes = np.mean(np.stack(aligned_axes, axis=0), axis=0)
    mean_axes = _canonicalize_eigen_axes(mean_axes)
    centroid = np.median(np.stack(centroids, axis=0), axis=0)
    return {
        "centroid_mm": centroid.astype(float),
        "axes": mean_axes.astype(float),
    }


def _principal_frame_transform(
    *,
    moving_frame: dict[str, np.ndarray],
    reference_frame: dict[str, Any],
    sitk_module,
):
    moving_axes = np.asarray(moving_frame["axes"], dtype=float)
    moving_center = np.asarray(moving_frame["centroid_mm"], dtype=float)
    reference_axes = np.asarray(reference_frame["axes"], dtype=float)
    reference_center = np.asarray(reference_frame["centroid_mm"], dtype=float)

    moving_axes = _match_eigenvector_basis(reference_axes, moving_axes)
    rotation = reference_axes @ moving_axes.T
    rotation = _project_to_rotation(rotation)
    translation = reference_center - rotation @ moving_center

    transform = sitk_module.AffineTransform(3)
    transform.SetMatrix(rotation.reshape(-1).tolist())
    transform.SetTranslation(translation.tolist())
    return transform


def _sampled_rows_or_error(
    df: pd.DataFrame,
    *,
    args: argparse.Namespace,
    sitk_module,
) -> list[dict[str, Any]]:
    sampled_rows = reg_common.sample_valid_rows_for_template(
        df,
        mask_column=args.mask_column,
        sample_size=args.template_sample_size,
        seed=args.template_seed,
        sitk_module=sitk_module,
    )
    if not sampled_rows:
        raise RuntimeError(
            f"No valid rows found for template creation using column '{args.mask_column}'."
        )
    logger.info(
        "Template sampling collected %d valid rows (mode=%s, sample_size=%d, seed=%d).",
        len(sampled_rows),
        _resolve_template_mode(args),
        int(args.template_sample_size),
        int(args.template_seed),
    )
    return sampled_rows


def _build_single_sample_template(
    df: pd.DataFrame,
    *,
    args: argparse.Namespace,
    sitk_module,
) -> dict[str, Any]:
    template_source_idx = getattr(args, "template_source_idx", None)
    if template_source_idx is None:
        sampled_rows = _sampled_rows_or_error(df, args=args, sitk_module=sitk_module)
        exemplar = reg_common.choose_median_exemplar(sampled_rows)
    else:
        selected = df.loc[df["_source_idx"] == int(template_source_idx)]
        if selected.empty:
            raise RuntimeError(
                f"--template_source_idx={template_source_idx} not found in input rows."
            )
        row = selected.iloc[0]
        nifti_path = row.get("nifti_path")
        mask_path = row.get(args.mask_column)
        if not reg_common._is_existing_path(nifti_path):
            raise RuntimeError(
                f"--template_source_idx={template_source_idx} has invalid nifti_path: {nifti_path}"
            )
        if not reg_common._is_existing_path(mask_path):
            raise RuntimeError(
                f"--template_source_idx={template_source_idx} has invalid {args.mask_column}: {mask_path}"
            )
        try:
            reg_common.mask_metrics(str(mask_path), sitk_module=sitk_module)
        except Exception as exc:
            raise RuntimeError(
                f"--template_source_idx={template_source_idx} failed validation: {exc}"
            ) from exc
        exemplar = {
            "source_idx": int(row["_source_idx"]),
            "nifti_path": str(nifti_path),
            "mask_path": str(mask_path),
        }

    exemplar_image = reg_common.read_image(exemplar["nifti_path"], sitk_module)
    exemplar_mask = reg_common.read_binary_mask(
        exemplar["mask_path"],
        reference_image=exemplar_image,
        sitk_module=sitk_module,
    )
    paths = _template_paths(args)
    reg_common.write_image(
        exemplar_image,
        paths["reference_image_path"],
        sitk_module=sitk_module,
    )
    reg_common.write_image(
        exemplar_mask,
        paths["mask_path"],
        sitk_module=sitk_module,
    )
    logger.info(
        "Built single-sample template from source_idx=%s -> reference=%s mask=%s",
        exemplar["source_idx"],
        paths["reference_image_path"],
        paths["mask_path"],
    )
    return {
        "template_mode": TEMPLATE_MODE_SINGLE_SAMPLE,
        "template_source_idx": int(exemplar["source_idx"]),
        "reference_image_path": str(paths["reference_image_path"]),
        "mask_path": str(paths["mask_path"]),
        "sample_count": 1,
    }


def _build_mean_shape_template(
    df: pd.DataFrame,
    *,
    args: argparse.Namespace,
    sitk_module,
) -> dict[str, Any]:
    sampled_rows = _sampled_rows_or_error(df, args=args, sitk_module=sitk_module)
    exemplar = reg_common.choose_median_exemplar(sampled_rows)
    mean_metrics = reg_common.compute_mean_metrics(sampled_rows)
    exemplar_image = reg_common.read_image(exemplar["nifti_path"], sitk_module)
    exemplar_mask = reg_common.read_binary_mask(
        exemplar["mask_path"],
        reference_image=exemplar_image,
        sitk_module=sitk_module,
    )

    aligned_arrays = [
        sitk_module.GetArrayFromImage(
            sitk_module.Cast(exemplar_mask > 0, sitk_module.sitkFloat32)
        )
    ]
    for sampled in sampled_rows:
        if int(sampled["source_idx"]) == int(exemplar["source_idx"]):
            continue
        try:
            moving_mask = reg_common.read_binary_mask(
                sampled["mask_path"],
                sitk_module=sitk_module,
            )
            rigid_tx = reg_common.rigid_register_mask_pair(
                fixed_mask=exemplar_mask,
                moving_mask=moving_mask,
                sitk_module=sitk_module,
            )
            aligned_mask = reg_common.resample_like(
                exemplar_image,
                moving_mask,
                tx=rigid_tx,
                interp=sitk_module.sitkNearestNeighbor,
                default=0,
                pixel_id=sitk_module.sitkUInt8,
                sitk_module=sitk_module,
            )
            aligned_arrays.append(
                sitk_module.GetArrayFromImage(
                    sitk_module.Cast(aligned_mask > 0, sitk_module.sitkFloat32)
                )
            )
        except Exception as exc:
            logger.warning(
                "Skipping sampled template row %s after registration error: %s",
                sampled["source_idx"],
                exc,
            )

    mean_array = np.mean(np.stack(aligned_arrays, axis=0), axis=0).astype(np.float32)
    mean_image = reg_common.image_from_array_like(
        mean_array,
        reference_image=exemplar_image,
        sitk_module=sitk_module,
        cast_to=sitk_module.sitkFloat32,
    )
    template_mask = reg_common.threshold_to_largest_component(
        mean_image,
        reference_image=exemplar_image,
        threshold=0.5,
        sitk_module=sitk_module,
    )

    paths = _template_paths(args)
    reg_common.write_image(
        exemplar_image,
        paths["reference_image_path"],
        sitk_module=sitk_module,
    )
    reg_common.write_image(
        template_mask,
        paths["mask_path"],
        sitk_module=sitk_module,
    )
    logger.info(
        (
            "Built mean-shape template from %d aligned samples "
            "(anchor_source_idx=%s) -> reference=%s mask=%s"
        ),
        len(aligned_arrays),
        exemplar["source_idx"],
        paths["reference_image_path"],
        paths["mask_path"],
    )
    return {
        "template_mode": TEMPLATE_MODE_MEAN_SHAPE,
        "template_source_idx": int(exemplar["source_idx"]),
        "reference_image_path": str(paths["reference_image_path"]),
        "mask_path": str(paths["mask_path"]),
        "sample_count": len(aligned_arrays),
        "template_mean_metrics": mean_metrics,
        "template_reference_kind": "mean_shape",
    }


def _compute_average_shape_principal_frame(
    sampled_rows: list[dict[str, Any]],
    *,
    exemplar: dict[str, Any],
    exemplar_image,
    exemplar_mask,
    sitk_module,
) -> tuple[dict[str, np.ndarray], int]:
    aligned_arrays = [
        sitk_module.GetArrayFromImage(
            sitk_module.Cast(exemplar_mask > 0, sitk_module.sitkFloat32)
        )
    ]
    for sampled in sampled_rows:
        if int(sampled["source_idx"]) == int(exemplar["source_idx"]):
            continue
        try:
            moving_mask = reg_common.read_binary_mask(
                sampled["mask_path"],
                sitk_module=sitk_module,
            )
            rigid_tx = reg_common.rigid_register_mask_pair(
                fixed_mask=exemplar_mask,
                moving_mask=moving_mask,
                sitk_module=sitk_module,
            )
            aligned_mask = reg_common.resample_like(
                exemplar_image,
                moving_mask,
                tx=rigid_tx,
                interp=sitk_module.sitkNearestNeighbor,
                default=0,
                pixel_id=sitk_module.sitkUInt8,
                sitk_module=sitk_module,
            )
            aligned_arrays.append(
                sitk_module.GetArrayFromImage(
                    sitk_module.Cast(aligned_mask > 0, sitk_module.sitkFloat32)
                )
            )
        except Exception as exc:
            logger.warning(
                "Skipping average-shape principal frame row %s after registration error: %s",
                sampled["source_idx"],
                exc,
            )
    mean_array = np.mean(np.stack(aligned_arrays, axis=0), axis=0).astype(np.float32)
    mean_image = reg_common.image_from_array_like(
        mean_array,
        reference_image=exemplar_image,
        sitk_module=sitk_module,
        cast_to=sitk_module.sitkFloat32,
    )
    mean_mask = reg_common.threshold_to_largest_component(
        mean_image,
        reference_image=exemplar_image,
        threshold=0.5,
        sitk_module=sitk_module,
    )
    return _mask_principal_frame(mean_mask, sitk_module=sitk_module), len(aligned_arrays)


def _build_principal_vectors_template(
    df: pd.DataFrame,
    *,
    args: argparse.Namespace,
    sitk_module,
) -> dict[str, Any]:
    sampled_rows = _sampled_rows_or_error(df, args=args, sitk_module=sitk_module)
    exemplar = reg_common.choose_median_exemplar(sampled_rows)
    exemplar_image = reg_common.read_image(exemplar["nifti_path"], sitk_module)
    exemplar_mask = reg_common.read_binary_mask(
        exemplar["mask_path"],
        reference_image=exemplar_image,
        sitk_module=sitk_module,
    )

    frames: list[dict[str, np.ndarray]] = []
    for sampled in sampled_rows:
        try:
            sampled_mask = reg_common.read_binary_mask(
                sampled["mask_path"],
                sitk_module=sitk_module,
            )
            frames.append(_mask_principal_frame(sampled_mask, sitk_module=sitk_module))
        except Exception as exc:
            logger.warning(
                "Skipping sampled principal frame for row %s: %s",
                sampled["source_idx"],
                exc,
            )
    if not frames:
        raise RuntimeError("Unable to infer principal vectors from sampled masks.")

    reference = _reference_from_sampled_frames(frames)
    vectors_source = "sampled"
    derived_from_rows = len(frames)
    try:
        average_frame, average_rows = _compute_average_shape_principal_frame(
            sampled_rows,
            exemplar=exemplar,
            exemplar_image=exemplar_image,
            exemplar_mask=exemplar_mask,
            sitk_module=sitk_module,
        )
        reference["centroid_mm"] = np.asarray(average_frame["centroid_mm"], dtype=float)
        reference["axes"] = np.asarray(average_frame["axes"], dtype=float)
        vectors_source = "average_shape"
        derived_from_rows = int(average_rows)
    except Exception as exc:
        logger.warning(
            "Falling back to sampled principal frames after average-shape failure: %s",
            exc,
        )

    provided_principal_vectors = _parse_principal_vectors(
        getattr(args, "principal_vectors", None)
    )
    if provided_principal_vectors is not None:
        provided_axes = np.asarray(provided_principal_vectors, dtype=float)
        reference["axes"] = _canonicalize_eigen_axes(provided_axes)
        vectors_source = "user"
    elif (
        default_axes := _default_principal_vectors_for_organ(getattr(args, "organ", ""))
    ) is not None:
        reference["axes"] = default_axes
        vectors_source = f"default_{str(args.organ).strip().lower()}"

    paths = _template_paths(args)
    reg_common.write_image(
        exemplar_image,
        paths["reference_image_path"],
        sitk_module=sitk_module,
    )
    reg_common.write_image(
        exemplar_mask,
        paths["mask_path"],
        sitk_module=sitk_module,
    )
    payload = {
        "centroid_mm": reference["centroid_mm"].astype(float).tolist(),
        "axes": reference["axes"].astype(float).tolist(),
        "derived_from_rows": derived_from_rows,
        "source": vectors_source,
    }
    paths["template_dir"].mkdir(parents=True, exist_ok=True)
    paths["principal_vectors_path"].write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    logger.info(
        (
            "Built principal-vectors template from %d sampled frames "
            "(anchor_source_idx=%s, vectors_source=%s) -> reference=%s mask=%s vectors=%s"
        ),
        len(frames),
        exemplar["source_idx"],
        payload["source"],
        paths["reference_image_path"],
        paths["mask_path"],
        paths["principal_vectors_path"],
    )

    return {
        "template_mode": TEMPLATE_MODE_PRINCIPAL_VECTORS,
        "template_source_idx": int(exemplar["source_idx"]),
        "reference_image_path": str(paths["reference_image_path"]),
        "mask_path": str(paths["mask_path"]),
        "sample_count": len(frames),
        "principal_vectors_path": str(paths["principal_vectors_path"]),
        "principal_reference": {
            "centroid_mm": payload["centroid_mm"],
            "axes": payload["axes"],
        },
    }


def _build_population_template(
    df: pd.DataFrame,
    *,
    args: argparse.Namespace,
    sitk_module,
) -> dict[str, Any]:
    template_mode = _resolve_template_mode(args)
    logger.info("Building population template with mode=%s", template_mode)
    logger.debug(
        (
            "Population template settings: mask_column=%s, sample_size=%s, "
            "seed=%s, explicit_source_idx=%s."
        ),
        args.mask_column,
        args.template_sample_size,
        args.template_seed,
        getattr(args, "template_source_idx", None),
    )
    if template_mode == TEMPLATE_MODE_SINGLE_SAMPLE:
        return _build_single_sample_template(df, args=args, sitk_module=sitk_module)
    if template_mode == TEMPLATE_MODE_PRINCIPAL_VECTORS:
        return _build_principal_vectors_template(df, args=args, sitk_module=sitk_module)
    return _build_mean_shape_template(df, args=args, sitk_module=sitk_module)


def _row_log_columns() -> list[str]:
    return [
        "_source_idx",
        "patient_key",
        "population_register_template_source_idx",
        "population_register_template_mode",
        "population_register_mask_column",
        "population_register_stage",
        "population_register_dice_before",
        "population_register_dice_after",
        "population_register_transform_path",
        "population_register_transform_metadata_path",
        "population_normalization_applied",
        "population_normalization_metadata_path",
        "population_register_status",
        "population_register_error_message",
        *reg_common.POPULATION_MATRIX_COLUMNS,
    ]


def _normalization_config_for_row(
    *,
    args: argparse.Namespace,
    template_image,
) -> OrganNormalizeConfig:
    spacing = getattr(args, "normalize_spacing", None)
    if spacing is None and bool(getattr(args, "normalize_registered_outputs", False)):
        spacing = tuple(float(v) for v in template_image.GetSpacing())
    return OrganNormalizeConfig(
        crop_mode=str(getattr(args, "normalize_crop_mode", "margin")),
        margin_mm=float(getattr(args, "normalize_margin_mm", 10.0)),
        keep_background=not bool(getattr(args, "normalize_without_background", False)),
        spacing=spacing,
        orientation=getattr(args, "normalize_orientation", "LPS"),
        center_organ=not bool(getattr(args, "disable_normalize_center_organ", False)),
    )


def _register_population_row(
    row: dict[str, Any],
    *,
    template_info: dict[str, Any],
    args: argparse.Namespace,
    path_columns: list[str],
) -> dict[str, Any]:
    sitk_module = reg_common._load_register_dependencies()
    source_idx = int(row.get("_source_idx", -1))
    template_mode = _resolve_template_mode(args)
    stage = (
        "principal_vectors"
        if template_mode == TEMPLATE_MODE_PRINCIPAL_VECTORS
        else "rigid"
    )
    base_updates: dict[str, Any] = {
        "population_register_template_source_idx": int(
            template_info["template_source_idx"]
        ),
        "population_register_template_mode": template_mode,
        "population_register_mask_column": args.mask_column,
        "population_register_stage": stage,
        "population_register_transform_path": None,
        "population_register_transform_metadata_path": None,
        "population_normalization_applied": False,
        "population_normalization_metadata_path": None,
        "population_register_status": "error",
        "population_register_error_message": None,
    }

    nifti_path = row.get("nifti_path")
    mask_path = row.get(args.mask_column)
    if not reg_common._is_existing_path(nifti_path):
        logger.warning(
            "Row source_idx=%s failed validation: invalid nifti_path=%s",
            source_idx,
            nifti_path,
        )
        return {
            "source_idx": source_idx,
            "updates": {
                **base_updates,
                "population_register_error_message": f"invalid nifti_path: {nifti_path}",
            },
            "error_message": f"invalid nifti_path: {nifti_path}",
        }
    if not reg_common._is_existing_path(mask_path):
        logger.warning(
            "Row source_idx=%s failed validation: invalid %s=%s",
            source_idx,
            args.mask_column,
            mask_path,
        )
        return {
            "source_idx": source_idx,
            "updates": {
                **base_updates,
                "population_register_error_message": (
                    f"invalid {args.mask_column}: {mask_path}"
                ),
            },
            "error_message": f"invalid {args.mask_column}: {mask_path}",
        }

    try:
        template_image = reg_common.read_image(
            template_info["reference_image_path"],
            sitk_module,
        )
        template_mask = reg_common.read_binary_mask(
            template_info["mask_path"],
            reference_image=template_image,
            sitk_module=sitk_module,
        )
        moving_image = reg_common.read_image(str(nifti_path), sitk_module)
        moving_mask = reg_common.read_binary_mask(
            str(mask_path),
            reference_image=moving_image,
            sitk_module=sitk_module,
        )
        dice_before = reg_common.dice_coeff(
            template_mask,
            moving_mask,
            sitk_module=sitk_module,
        )
        if template_mode == TEMPLATE_MODE_PRINCIPAL_VECTORS:
            principal_reference = template_info.get("principal_reference")
            if principal_reference is None:
                raise RuntimeError(
                    "principal_vectors mode selected but principal reference is missing"
                )
            moving_frame = _mask_principal_frame(moving_mask, sitk_module=sitk_module)
            rigid_tx = _principal_frame_transform(
                moving_frame=moving_frame,
                reference_frame=principal_reference,
                sitk_module=sitk_module,
            )
        else:
            rigid_tx = reg_common.rigid_register_mask_pair(
                fixed_mask=template_mask,
                moving_mask=moving_mask,
                sitk_module=sitk_module,
            )
        dice_after = reg_common.dice_coeff(
            template_mask,
            moving_mask,
            sitk_module=sitk_module,
            tx=rigid_tx,
        )
        row_dir = reg_common.build_row_output_dir(args.output_dir, source_idx)
        updates = {
            **base_updates,
            **reg_common.transform_to_flat_3x4(rigid_tx),
            "population_register_dice_before": dice_before,
            "population_register_dice_after": dice_after,
            "population_register_status": "ok",
            "population_register_error_message": None,
        }
        if args.save_registered_outputs:
            rewritten_paths = reg_common.warp_row_files(
                row,
                row_dir=row_dir,
                path_columns=path_columns,
                reference_image=template_image,
                transform=rigid_tx,
                sitk_module=sitk_module,
            )
            updates.update(rewritten_paths)
            tx_artifact = save_transform_artifacts(
                row_dir=row_dir,
                transform=rigid_tx,
                sitk_module=sitk_module,
                prefix=f"population_{source_idx}_to_template",
                metadata={
                    "source_idx": int(source_idx),
                    "template_source_idx": int(template_info["template_source_idx"]),
                    "template_mode": template_mode,
                    "stage": stage,
                    "dice_before": float(dice_before),
                    "dice_after": float(dice_after),
                },
            )
            updates.update(
                {
                    "population_register_transform_path": tx_artifact.transform_path,
                    "population_register_transform_metadata_path": (
                        tx_artifact.metadata_path
                    ),
                }
            )

            if bool(getattr(args, "normalize_registered_outputs", False)):
                normalized_dir = row_dir / "normalized"
                warped_image = reg_common.read_image(
                    str(rewritten_paths["nifti_path"]),
                    sitk_module,
                )
                warped_organ_mask = reg_common.read_binary_mask(
                    str(rewritten_paths[args.mask_column]),
                    reference_image=warped_image,
                    sitk_module=sitk_module,
                )
                extra_masks: dict[str, Any] = {}
                for column_name in path_columns:
                    if column_name == "nifti_path" or column_name == args.mask_column:
                        continue
                    warped_path = rewritten_paths.get(column_name)
                    if not reg_common._is_existing_path(warped_path):
                        continue
                    extra_masks[column_name] = reg_common.read_binary_mask(
                        str(warped_path),
                        reference_image=warped_image,
                        sitk_module=sitk_module,
                    )
                normalize_cfg = _normalization_config_for_row(
                    args=args,
                    template_image=template_image,
                )
                (
                    normalized_image,
                    normalized_organ_mask,
                    normalized_extra_masks,
                    normalize_metadata,
                ) = normalize_image_and_masks(
                    image=warped_image,
                    organ_mask=warped_organ_mask,
                    masks_by_name=extra_masks,
                    config=normalize_cfg,
                    sitk_module=sitk_module,
                )
                normalized_dir.mkdir(parents=True, exist_ok=True)
                normalized_paths: dict[str, str] = {}
                image_out = reg_common.build_output_path(
                    normalized_dir,
                    column_name="nifti_path",
                    source_path=rewritten_paths.get("nifti_path"),
                )
                reg_common.write_image(
                    normalized_image,
                    image_out,
                    sitk_module=sitk_module,
                )
                normalized_paths["nifti_path"] = str(image_out)
                organ_out = reg_common.build_output_path(
                    normalized_dir,
                    column_name=args.mask_column,
                    source_path=rewritten_paths.get(args.mask_column),
                )
                reg_common.write_image(
                    normalized_organ_mask,
                    organ_out,
                    sitk_module=sitk_module,
                )
                normalized_paths[args.mask_column] = str(organ_out)
                for column_name, mask_image in normalized_extra_masks.items():
                    out_path = reg_common.build_output_path(
                        normalized_dir,
                        column_name=column_name,
                        source_path=rewritten_paths.get(column_name),
                    )
                    reg_common.write_image(
                        mask_image,
                        out_path,
                        sitk_module=sitk_module,
                    )
                    normalized_paths[column_name] = str(out_path)
                normalization_metadata_path = normalized_dir / "normalization.json"
                normalization_metadata_path.write_text(
                    json.dumps(
                        {
                            "source_idx": int(source_idx),
                            "template_source_idx": int(template_info["template_source_idx"]),
                            "normalize_config": normalize_metadata,
                            "normalized_paths": normalized_paths,
                        },
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=True,
                    ),
                    encoding="utf-8",
                )
                updates.update(normalized_paths)
                updates.update(
                    {
                        "population_normalization_applied": True,
                        "population_normalization_metadata_path": str(
                            normalization_metadata_path
                        ),
                    }
                )
                logger.debug(
                    (
                        "Population row source_idx=%s wrote normalized outputs "
                        "(paths=%d, extra_masks=%d) into %s."
                    ),
                    source_idx,
                    len(normalized_paths),
                    len(normalized_extra_masks),
                    normalized_dir,
                )
        logger.debug(
            (
                "Row source_idx=%s registered successfully "
                "(mode=%s, stage=%s, dice_before=%.4f, dice_after=%.4f)."
            ),
            source_idx,
            template_mode,
            stage,
            float(dice_before),
            float(dice_after),
        )
        return {"source_idx": source_idx, "updates": updates, "error_message": None}
    except Exception as exc:
        logger.warning(
            "Row source_idx=%s registration failed (mode=%s): %s",
            source_idx,
            template_mode,
            exc,
        )
        return {
            "source_idx": source_idx,
            "updates": {
                **base_updates,
                "population_register_error_message": str(exc),
            },
            "error_message": str(exc),
        }


def _apply_row_result(
    df: pd.DataFrame,
    idx: int,
    *,
    result: dict[str, Any],
    errors_by_idx: dict[int, dict[str, Any]],
) -> None:
    source_idx = int(df.at[idx, "_source_idx"])
    for key, value in result["updates"].items():
        df.at[idx, key] = value
    if result["error_message"]:
        error_row = df.loc[idx].to_dict()
        error_row["error_message"] = result["error_message"]
        errors_by_idx[source_idx] = error_row
    elif source_idx in errors_by_idx:
        del errors_by_idx[source_idx]


def _build_log_df(df: pd.DataFrame) -> pd.DataFrame:
    columns = [col for col in _row_log_columns() if col in df.columns]
    return df[columns].copy()


def _template_state_payload(
    template_info: dict[str, Any],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "template_mode": _resolve_template_mode(args),
        "template_source_idx": int(template_info["template_source_idx"]),
        "template_mask_path": template_info["mask_path"],
        "template_reference_image_path": template_info["reference_image_path"],
    }
    if template_info.get("principal_vectors_path"):
        payload["principal_vectors_path"] = template_info["principal_vectors_path"]
    if template_info.get("template_mean_metrics"):
        payload["template_mean_metrics"] = template_info["template_mean_metrics"]
    if template_info.get("template_reference_kind"):
        payload["template_reference_kind"] = template_info["template_reference_kind"]
    payload["save_registered_outputs"] = bool(
        getattr(args, "save_registered_outputs", False)
    )
    payload["normalize_registered_outputs"] = bool(
        getattr(args, "normalize_registered_outputs", False)
    )
    payload["normalize_crop_mode"] = str(getattr(args, "normalize_crop_mode", "margin"))
    payload["normalize_margin_mm"] = float(getattr(args, "normalize_margin_mm", 10.0))
    payload["normalize_without_background"] = bool(
        getattr(args, "normalize_without_background", False)
    )
    payload["normalize_spacing"] = (
        list(getattr(args, "normalize_spacing", None))
        if getattr(args, "normalize_spacing", None) is not None
        else None
    )
    payload["normalize_orientation"] = getattr(args, "normalize_orientation", None)
    payload["disable_normalize_center_organ"] = bool(
        getattr(args, "disable_normalize_center_organ", False)
    )
    return payload


def main(args: argparse.Namespace) -> None:
    output_path = Path(args.csv_path_out)
    error_path = Path(args.error_csv_path)
    exclude_hash_args = {
        "csv_path_out",
        "error_csv_path",
        "log_csv_path",
        "dry_run",
        "verbose",
        "resume",
        "checkpoint_every_rows",
        "checkpoint_every_sec",
        "strict_resume",
    }
    resume_ctx = prepare_resume_context(
        args=args,
        command="register_population",
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
            "Resume enabled and matching register-population run already finished; skipping execution."
        )
        return

    sitk_module = reg_common._load_register_dependencies()
    if can_resume and paths.main_checkpoint_path.exists():
        logger.info(
            "Resuming register-population from checkpoint: %s",
            paths.main_checkpoint_path,
        )
        df = pd.read_csv(paths.main_checkpoint_path).copy()
    else:
        df = pd.read_csv(args.csv_path).copy()
        df["_source_idx"] = df.index.astype(int)
    if "_source_idx" not in df.columns:
        df["_source_idx"] = df.index.astype(int)
    if "nifti_path" not in df.columns:
        raise KeyError("column 'nifti_path' missing")
    if args.mask_column not in df.columns:
        raise KeyError(f"column '{args.mask_column}' missing")
    logger.info(
        "Loaded population table with %d rows (mask_column=%s).",
        len(df),
        args.mask_column,
    )

    source_df = pd.read_csv(args.csv_path).copy()
    source_df["_source_idx"] = source_df.index.astype(int)
    template_info = _build_population_template(
        source_df,
        args=args,
        sitk_module=sitk_module,
    )
    logger.info(
        "Template ready (mode=%s, source_idx=%s, sample_count=%s).",
        template_info.get("template_mode"),
        template_info.get("template_source_idx"),
        template_info.get("sample_count"),
    )
    template_payload = _template_state_payload(template_info, args=args)
    path_columns = ["nifti_path", *reg_common.get_mask_columns(df)]

    completed_indices: set[int] = set()
    if can_resume:
        completed_indices = {
            int(i)
            for i in (state or {}).get("completed_indices", [])
            if isinstance(i, int)
        }
    errors_by_idx: dict[int, dict[str, Any]] = {}
    if can_resume and paths.error_checkpoint_path.exists():
        err_ckpt = pd.read_csv(paths.error_checkpoint_path)
        for _, row in err_ckpt.iterrows():
            if "_source_idx" in row:
                try:
                    errors_by_idx[int(row["_source_idx"])] = row.to_dict()
                except Exception:
                    pass

    def _checkpoint_write(*, force: bool = False) -> None:
        err_df = (
            pd.DataFrame(list(errors_by_idx.values()))
            if errors_by_idx
            else pd.DataFrame()
        )
        ckpt.flush(
            main_df=df,
            error_df=err_df,
            completed_indices=completed_indices,
            force=force,
            extra_state=template_payload,
        )

    pending_indices = [
        idx
        for idx in df.index.tolist()
        if int(df.at[idx, "_source_idx"]) not in completed_indices
    ]
    logger.info(
        "Processing %d pending rows (%d already completed via resume state).",
        len(pending_indices),
        len(completed_indices),
    )
    row_payloads = [
        (idx, df.loc[idx].to_dict())
        for idx in pending_indices
    ]

    if args.num_workers > 1 and len(row_payloads) > 1:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_map = {
                executor.submit(
                    _register_population_row,
                    row_dict,
                    template_info=template_info,
                    args=args,
                    path_columns=path_columns,
                ): idx
                for idx, row_dict in row_payloads
            }
            for future in tqdm(
                as_completed(future_map),
                total=len(future_map),
                desc="RegisterPopulation",
                unit="row",
            ):
                idx = future_map[future]
                result = future.result()
                _apply_row_result(df, idx, result=result, errors_by_idx=errors_by_idx)
                completed_indices.add(int(df.at[idx, "_source_idx"]))
                ckpt.mark_processed()
                _checkpoint_write(force=False)
    else:
        for idx, row_dict in tqdm(
            row_payloads,
            total=len(row_payloads),
            desc="RegisterPopulation",
            unit="row",
        ):
            result = _register_population_row(
                row_dict,
                template_info=template_info,
                args=args,
                path_columns=path_columns,
            )
            _apply_row_result(df, idx, result=result, errors_by_idx=errors_by_idx)
            completed_indices.add(int(df.at[idx, "_source_idx"]))
            ckpt.mark_processed()
            _checkpoint_write(force=False)

    _checkpoint_write(force=True)
    status_counts = (
        df["population_register_status"].value_counts(dropna=False).to_dict()
        if "population_register_status" in df.columns
        else {}
    )
    logger.info("Registration status summary: %s", status_counts)
    df_out = df.drop(columns=["_source_idx"], errors="ignore")
    df_out = merge_with_existing_output(
        df_out,
        args.csv_path_out,
        preferred_keys=["nifti_path", "patient_key"],
        strict=True,
    )
    atomic_write_csv(df_out, args.csv_path_out, index=False)
    logger.info("Wrote main table -> %s", args.csv_path_out)

    log_df = _build_log_df(df)
    atomic_write_csv(log_df, args.log_csv_path, index=False)
    logger.info("Wrote log table -> %s", args.log_csv_path)

    if errors_by_idx:
        df_err = pd.DataFrame(list(errors_by_idx.values())).drop(
            columns=["_source_idx"], errors="ignore"
        )
        atomic_write_csv(df_err, args.error_csv_path, index=False)
        logger.warning("%d rows failed -> %s", len(df_err), args.error_csv_path)

    ckpt.finalize_state(
        completed_indices=completed_indices,
        extra_state=template_payload,
    )
    logger.info("Population registration done")


if __name__ == "__main__":
    args = parse_arguments()
    if getattr(args, "dry_run", False):
        logger.info("Dry run: register-population")
        print_args(args)
        raise SystemExit(0)
    main(args)
