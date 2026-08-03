"""Pair-table based medical image registration with per-pair error isolation."""

from __future__ import annotations

import hashlib
import re
import traceback
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class RegistrationOutput:
    pair_id: str
    transform_path: Path
    registered_image_path: Path
    metric_value: float


def _safe_pair_stem(pair_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", pair_id).strip("._") or "pair"
    if stem != pair_id or ".." in stem:
        digest = hashlib.sha1(pair_id.encode()).hexdigest()[:8]
        stem = f"{stem.replace('..', '_')}_{digest}"
    return stem


def _sitk():
    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Registration requires SimpleITK. Install IMPERANDI with its "
            "registration/segment dependencies."
        ) from exc
    return sitk


def _registration_method(sitk):
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.REGULAR)
    method.SetMetricSamplingPercentage(0.2)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=200,
        gradientMagnitudeTolerance=1e-8,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return method


def register_pair(
    *,
    pair_id: str,
    fixed_path: str | Path,
    moving_path: str | Path,
    output_dir: str | Path,
    transform: str = "rigid_affine",
) -> RegistrationOutput:
    """Register one moving image and always save the forward transform."""
    sitk = _sitk()
    fixed = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)
    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    method = _registration_method(sitk)
    method.SetInitialTransform(initial, inPlace=True)
    method.Execute(fixed, moving)
    final = initial

    if transform in {"rigid_affine", "deformable"}:
        affine = sitk.AffineTransform(3)
        affine.SetMatrix(final.GetMatrix())
        affine.SetTranslation(final.GetTranslation())
        affine.SetCenter(final.GetCenter())
        method = _registration_method(sitk)
        method.SetInitialTransform(affine, inPlace=True)
        method.Execute(fixed, moving)
        final = affine

    if transform == "deformable":
        mesh_size = [max(1, int(size / 64)) for size in fixed.GetSize()]
        bspline = sitk.BSplineTransformInitializer(fixed, mesh_size, order=3)
        method = sitk.ImageRegistrationMethod()
        method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        method.SetMetricSamplingStrategy(method.REGULAR)
        method.SetMetricSamplingPercentage(0.2)
        method.SetInterpolator(sitk.sitkLinear)
        method.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=50,
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=500,
            costFunctionConvergenceFactor=1e7,
        )
        method.SetMovingInitialTransform(final)
        method.SetInitialTransformAsBSpline(
            bspline, inPlace=True, scaleFactors=[1, 2, 4]
        )
        method.Execute(fixed, moving)
        final = sitk.CompositeTransform([final, bspline])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_stem = _safe_pair_stem(pair_id)
    transform_path = output_dir / f"{pair_stem}.tfm"
    image_path = output_dir / f"{pair_stem}_registered.nii.gz"
    sitk.WriteTransform(final, str(transform_path))
    registered = sitk.Resample(
        moving,
        fixed,
        final,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID(),
    )
    sitk.WriteImage(registered, str(image_path))
    return RegistrationOutput(
        pair_id=pair_id,
        transform_path=transform_path,
        registered_image_path=image_path,
        metric_value=float(method.GetMetricValue()),
    )


def register_pairs(
    pairs: pd.DataFrame,
    volumes: pd.DataFrame,
    *,
    output_dir: str | Path,
    transform: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process an explicit pair table without aborting after row failures."""
    required = {"moving_volume_id"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"Registration pair table is missing: {sorted(missing)}")
    if not {"fixed_volume_id", "fixed_nifti_path"} & set(pairs.columns):
        raise ValueError(
            "Registration pair table requires fixed_volume_id or fixed_nifti_path"
        )
    if "volume_id" not in volumes.columns or "nifti_path" not in volumes.columns:
        raise ValueError("Registration requires volume_id and nifti_path columns")
    lookup = volumes.set_index("volume_id")["nifti_path"].to_dict()
    output_columns = [
        "pair_id",
        "fixed_volume_id",
        "moving_volume_id",
        "registration_transform_path",
        "registered_nifti_path",
        "registration_metric",
    ]
    error_columns = [
        "pair_id",
        "fixed_volume_id",
        "moving_volume_id",
        "error_message",
        "traceback",
    ]
    outputs = []
    errors = []
    seen_pair_ids = set()
    for ordinal, (_, row) in enumerate(pairs.iterrows()):
        raw_pair_id = row.get("pair_id")
        pair_id = (
            f"pair_{ordinal:06d}"
            if pd.isna(raw_pair_id) or not str(raw_pair_id).strip()
            else str(raw_pair_id).strip()
        )
        if pair_id in seen_pair_ids:
            raise ValueError(f"Registration pair_id must be unique: {pair_id!r}")
        seen_pair_ids.add(pair_id)
        fixed_id = row.get("fixed_volume_id", pd.NA)
        moving_id = row["moving_volume_id"]
        try:
            fixed_path = row.get("fixed_nifti_path")
            if pd.isna(fixed_path) or not str(fixed_path).strip():
                fixed_path = lookup[fixed_id]
            moving_path = row.get("moving_nifti_path")
            if pd.isna(moving_path) or not str(moving_path).strip():
                moving_path = lookup[moving_id]
            result = register_pair(
                pair_id=pair_id,
                fixed_path=fixed_path,
                moving_path=moving_path,
                output_dir=output_dir,
                transform=transform,
            )
            outputs.append(
                {
                    "pair_id": pair_id,
                    "fixed_volume_id": fixed_id,
                    "moving_volume_id": moving_id,
                    "registration_transform_path": str(result.transform_path),
                    "registered_nifti_path": str(result.registered_image_path),
                    "registration_metric": result.metric_value,
                }
            )
        # A pair table is a batch boundary: record any backend or I/O failure
        # for this pair and continue processing the remaining independent pairs.
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "pair_id": pair_id,
                    "fixed_volume_id": fixed_id,
                    "moving_volume_id": moving_id,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    return (
        pd.DataFrame(outputs, columns=output_columns),
        pd.DataFrame(errors, columns=error_columns),
    )
