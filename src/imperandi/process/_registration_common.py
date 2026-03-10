from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import ndimage

logger = logging.getLogger(__name__)

DEFAULT_PAD_MM = 25.0
DEFAULT_BSPLINE_CTRL_SPACING_MM = 90.0
DEFAULT_BAND_MM = 15.0
DEFAULT_TEMPLATE_SAMPLE_SIZE = 128
DEFAULT_TEMPLATE_SEED = 0
DEFAULT_NUM_WORKERS = 2

POPULATION_MATRIX_COLUMNS = [
    "population_tx_r00",
    "population_tx_r01",
    "population_tx_r02",
    "population_tx_r10",
    "population_tx_r11",
    "population_tx_r12",
    "population_tx_r20",
    "population_tx_r21",
    "population_tx_r22",
    "population_tx_t0",
    "population_tx_t1",
    "population_tx_t2",
]

PHASE_ALIASES = {
    "portal": "portal",
    "portal_venous": "portal",
    "venous": "portal",
    "arterial": "arteriel",
    "arterial_late": "arteriel",
    "arteriel": "arteriel",
}


def _load_register_dependencies():
    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The 'register-population' and 'register-intra-patient' commands "
            "require optional dependencies. Install with: pip install -e .[register]"
        ) from exc
    return sitk


def resolve_mask_column(*, organ: str, mask_column: str | None) -> str:
    normalized_organ = str(organ or "").strip().lower()
    if not normalized_organ:
        raise ValueError("--organ must not be empty")
    if mask_column is not None and str(mask_column).strip():
        return str(mask_column).strip()
    return f"mask_{normalized_organ}"


def get_mask_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("mask_")]


def _is_existing_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    path = value.strip()
    return bool(path) and Path(path).exists()


def infer_nifti_suffix(path: str | Path | None) -> str:
    if path is None:
        return ".nii.gz"
    name = Path(path).name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    suffix = Path(path).suffix
    return suffix if suffix else ".nii.gz"


def build_row_output_dir(output_dir: str | Path, source_idx: int) -> Path:
    return Path(output_dir) / "rows" / str(int(source_idx))


def build_output_path(
    row_dir: str | Path,
    *,
    column_name: str,
    source_path: str | None,
) -> Path:
    suffix = infer_nifti_suffix(source_path)
    return Path(row_dir) / f"{column_name}{suffix}"


def normalize_phase_value(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if not isinstance(value, str):
        value = str(value)
    normalized = value.strip().lower()
    if not normalized:
        return None
    return PHASE_ALIASES.get(normalized, normalized)


def infer_phase_from_row(row: Mapping[str, Any]) -> str | None:
    phase = normalize_phase_value(row.get("phase"))
    if phase:
        return phase
    return normalize_phase_value(row.get("totalseg_phase"))


def _array_sum(values: Any) -> float:
    if hasattr(values, "sum"):
        return float(values.sum())
    try:
        return float(sum(values))
    except TypeError:
        return float(values)


def mask_has_voxels(mask, sitk_module) -> bool:
    return bool(_array_sum(sitk_module.GetArrayViewFromImage(mask)) > 0)


def same_grid(a, b) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and a.GetSpacing() == b.GetSpacing()
        and a.GetOrigin() == b.GetOrigin()
        and a.GetDirection() == b.GetDirection()
    )


def identity_transform(sitk_module):
    return sitk_module.Transform(3, sitk_module.sitkIdentity)


def read_image(path: str, sitk_module):
    return sitk_module.ReadImage(str(path))


def read_binary_mask(path: str, *, reference_image=None, sitk_module):
    mask = read_image(path, sitk_module)
    mask = sitk_module.Cast(mask > 0, sitk_module.sitkUInt8)
    if reference_image is not None and not same_grid(mask, reference_image):
        mask = resample_like(
            reference_image,
            mask,
            tx=None,
            interp=sitk_module.sitkNearestNeighbor,
            default=0,
            pixel_id=sitk_module.sitkUInt8,
            sitk_module=sitk_module,
        )
    return mask


def resample_like(
    reference_image,
    image,
    *,
    tx=None,
    interp=None,
    default=0.0,
    pixel_id=None,
    sitk_module,
):
    if pixel_id is None:
        pixel_id = image.GetPixelID()
    if tx is None:
        tx = identity_transform(sitk_module)
    if interp is None:
        interp = sitk_module.sitkLinear
    return sitk_module.Resample(
        image,
        reference_image,
        tx,
        interp,
        default,
        pixel_id,
    )


def dice_coeff(fixed_mask, moving_mask, *, sitk_module, tx=None) -> float:
    fixed = sitk_module.Cast(fixed_mask > 0, sitk_module.sitkUInt8)
    moving = sitk_module.Cast(moving_mask > 0, sitk_module.sitkUInt8)
    if tx is not None or not same_grid(fixed, moving):
        moving = resample_like(
            fixed,
            moving,
            tx=tx,
            interp=sitk_module.sitkNearestNeighbor,
            default=0,
            pixel_id=sitk_module.sitkUInt8,
            sitk_module=sitk_module,
        )
    overlap = sitk_module.LabelOverlapMeasuresImageFilter()
    overlap.Execute(fixed, moving)
    return float(overlap.GetDiceCoefficient())


def copy_image_information(image, *, reference_image) -> Any:
    if hasattr(image, "CopyInformation"):
        image.CopyInformation(reference_image)
    return image


def image_from_array_like(array: np.ndarray, *, reference_image, sitk_module, cast_to=None):
    image = sitk_module.GetImageFromArray(array)
    image = copy_image_information(image, reference_image=reference_image)
    if cast_to is not None:
        image = sitk_module.Cast(image, cast_to)
    return image


def largest_connected_component(mask_array: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask_array, dtype=bool)
    if not binary.any():
        return binary.astype(np.uint8)
    labels, count = ndimage.label(binary)
    if count <= 1:
        return binary.astype(np.uint8)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest = int(np.argmax(counts))
    return (labels == largest).astype(np.uint8)


def threshold_to_largest_component(probability_image, *, reference_image, threshold: float, sitk_module):
    array = sitk_module.GetArrayFromImage(probability_image)
    mask_array = largest_connected_component(array >= float(threshold))
    return image_from_array_like(
        mask_array.astype(np.uint8),
        reference_image=reference_image,
        sitk_module=sitk_module,
        cast_to=sitk_module.sitkUInt8,
    )


def write_image(image, path: str | Path, *, sitk_module) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk_module.WriteImage(image, str(out_path))


def copy_row_files(
    row: Mapping[str, Any],
    *,
    row_dir: str | Path,
    path_columns: Iterable[str],
) -> dict[str, str]:
    out: dict[str, str] = {}
    target_dir = Path(row_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for column in path_columns:
        value = row.get(column)
        if not _is_existing_path(value):
            continue
        destination = build_output_path(
            target_dir,
            column_name=column,
            source_path=str(value),
        )
        shutil.copy2(str(value), destination)
        out[column] = str(destination)
    return out


def warp_row_files(
    row: Mapping[str, Any],
    *,
    row_dir: str | Path,
    path_columns: Iterable[str],
    reference_image,
    transform,
    sitk_module,
) -> dict[str, str]:
    out: dict[str, str] = {}
    target_dir = Path(row_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for column in path_columns:
        value = row.get(column)
        if not _is_existing_path(value):
            continue
        image = read_image(str(value), sitk_module)
        if column == "nifti_path":
            interp = sitk_module.sitkLinear
            default = 0.0
            pixel_id = image.GetPixelID()
        else:
            interp = sitk_module.sitkNearestNeighbor
            default = 0
            pixel_id = sitk_module.sitkUInt8
        warped = resample_like(
            reference_image,
            image,
            tx=transform,
            interp=interp,
            default=default,
            pixel_id=pixel_id,
            sitk_module=sitk_module,
        )
        destination = build_output_path(
            target_dir,
            column_name=column,
            source_path=str(value),
        )
        write_image(warped, destination, sitk_module=sitk_module)
        out[column] = str(destination)
    return out


def mask_metrics(mask_path: str, *, sitk_module) -> dict[str, float]:
    mask = read_binary_mask(mask_path, sitk_module=sitk_module)
    if not mask_has_voxels(mask, sitk_module):
        raise ValueError(f"mask is empty: {mask_path}")

    spacing = tuple(float(v) for v in mask.GetSpacing())
    voxel_volume_mm3 = float(np.prod(spacing))
    voxel_count = float(_array_sum(sitk_module.GetArrayViewFromImage(mask)))

    stats = sitk_module.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    if not stats.HasLabel(1):
        raise ValueError(f"mask has no positive label: {mask_path}")

    bbox = stats.GetBoundingBox(1)
    bbox_sizes_vox = [float(v) for v in bbox[3:6]]
    bbox_sizes_mm = [bbox_sizes_vox[i] * spacing[i] for i in range(3)]
    return {
        "volume_ml": (voxel_count * voxel_volume_mm3) / 1000.0,
        "bbox_x_mm": bbox_sizes_mm[0],
        "bbox_y_mm": bbox_sizes_mm[1],
        "bbox_z_mm": bbox_sizes_mm[2],
    }


def rigid_register_mask_pair(
    *,
    fixed_mask,
    moving_mask,
    sitk_module,
):
    fixed_mask = sitk_module.Cast(fixed_mask > 0, sitk_module.sitkUInt8)
    moving_mask = sitk_module.Cast(moving_mask > 0, sitk_module.sitkUInt8)
    initializer = sitk_module.CenteredTransformInitializer(
        fixed_mask,
        moving_mask,
        sitk_module.Euler3DTransform(),
        sitk_module.CenteredTransformInitializerFilter.MOMENTS,
    )
    fixed_dm = sitk_module.SignedMaurerDistanceMap(fixed_mask, True, False, True)
    moving_dm = sitk_module.SignedMaurerDistanceMap(moving_mask, True, False, True)

    registration = sitk_module.ImageRegistrationMethod()
    registration.SetMetricAsMeanSquares()
    registration.SetInterpolator(sitk_module.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=200,
        relaxationFactor=0.5,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initializer, inPlace=False)
    return registration.Execute(fixed_dm, moving_dm)


def bspline_register_mask_pair(
    *,
    fixed_mask,
    moving_mask,
    initial_transform,
    sitk_module,
    band_mm: float,
    ctrl_spacing_mm: float,
):
    fixed_mask = sitk_module.Cast(fixed_mask > 0, sitk_module.sitkUInt8)
    moving_mask = sitk_module.Cast(moving_mask > 0, sitk_module.sitkUInt8)

    fixed_dm = sitk_module.SignedMaurerDistanceMap(fixed_mask, True, False, True)
    moving_dm = sitk_module.SignedMaurerDistanceMap(moving_mask, True, False, True)
    fixed_dm = sitk_module.Clamp(
        fixed_dm,
        lowerBound=-float(band_mm),
        upperBound=float(band_mm),
    )
    moving_dm = sitk_module.Clamp(
        moving_dm,
        lowerBound=-float(band_mm),
        upperBound=float(band_mm),
    )

    physical_lengths = [
        max(1.0, (size - 1) * spacing)
        for size, spacing in zip(fixed_mask.GetSize(), fixed_mask.GetSpacing())
    ]
    mesh_size = [
        max(1, int(round(length / max(1.0, float(ctrl_spacing_mm)))))
        for length in physical_lengths
    ]
    bspline = sitk_module.BSplineTransformInitializer(fixed_mask, mesh_size, order=3)

    registration = sitk_module.ImageRegistrationMethod()
    registration.SetMetricAsMeanSquares()
    registration.SetInterpolator(sitk_module.sitkLinear)
    registration.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-3,
        convergenceWindowSize=5,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2])
    registration.SetSmoothingSigmasPerLevel([2, 1])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetMovingInitialTransform(initial_transform)
    registration.SetInitialTransform(bspline, inPlace=True)
    registration.Execute(fixed_dm, moving_dm)

    composite = sitk_module.CompositeTransform(3)
    composite.AddTransform(initial_transform)
    composite.AddTransform(bspline)
    return composite


def transform_to_flat_3x4(transform) -> dict[str, float]:
    origin = np.asarray(transform.TransformPoint((0.0, 0.0, 0.0)), dtype=float)
    basis_points = [
        np.asarray(transform.TransformPoint((1.0, 0.0, 0.0)), dtype=float),
        np.asarray(transform.TransformPoint((0.0, 1.0, 0.0)), dtype=float),
        np.asarray(transform.TransformPoint((0.0, 0.0, 1.0)), dtype=float),
    ]
    columns = [point - origin for point in basis_points]
    matrix = np.column_stack(columns)
    flat_values = {
        "population_tx_r00": float(matrix[0, 0]),
        "population_tx_r01": float(matrix[0, 1]),
        "population_tx_r02": float(matrix[0, 2]),
        "population_tx_r10": float(matrix[1, 0]),
        "population_tx_r11": float(matrix[1, 1]),
        "population_tx_r12": float(matrix[1, 2]),
        "population_tx_r20": float(matrix[2, 0]),
        "population_tx_r21": float(matrix[2, 1]),
        "population_tx_r22": float(matrix[2, 2]),
        "population_tx_t0": float(origin[0]),
        "population_tx_t1": float(origin[1]),
        "population_tx_t2": float(origin[2]),
    }
    return flat_values


def choose_median_exemplar(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not metric_rows:
        raise ValueError("metric_rows must not be empty")

    features = np.asarray(
        [
            [
                float(row["volume_ml"]),
                float(row["bbox_x_mm"]),
                float(row["bbox_y_mm"]),
                float(row["bbox_z_mm"]),
            ]
            for row in metric_rows
        ],
        dtype=float,
    )
    medians = np.median(features, axis=0)
    distances = np.linalg.norm(features - medians[None, :], axis=1)
    best_index = int(np.argmin(distances))
    return metric_rows[best_index]


def sample_valid_rows_for_template(
    df: pd.DataFrame,
    *,
    mask_column: str,
    sample_size: int,
    seed: int,
    sitk_module,
) -> list[dict[str, Any]]:
    indices = [int(i) for i in df.index.tolist()]
    rng = np.random.default_rng(int(seed))
    if len(indices) > 1:
        indices = [int(i) for i in rng.permutation(indices)]

    samples: list[dict[str, Any]] = []
    for idx in indices:
        row = df.loc[idx]
        nifti_path = row.get("nifti_path")
        mask_path = row.get(mask_column)
        if not _is_existing_path(nifti_path) or not _is_existing_path(mask_path):
            continue
        try:
            metrics = mask_metrics(str(mask_path), sitk_module=sitk_module)
        except Exception as exc:
            logger.debug("Skipping template candidate row=%s: %s", idx, exc)
            continue
        metrics.update(
            {
                "df_index": idx,
                "source_idx": int(row.get("_source_idx", idx)),
                "nifti_path": str(nifti_path),
                "mask_path": str(mask_path),
            }
        )
        samples.append(metrics)
        if len(samples) >= max(1, int(sample_size)):
            break

    return samples


def to_numeric_sort_series(series: pd.Series, *, missing: float = math.inf) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(missing)
