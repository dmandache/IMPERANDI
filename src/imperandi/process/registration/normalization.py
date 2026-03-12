from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from imperandi.process import _registration_common as reg_common


@dataclass(frozen=True)
class OrganNormalizeConfig:
    crop_mode: str = "margin"
    margin_mm: float = 10.0
    keep_background: bool = True
    spacing: tuple[float, float, float] | None = None
    orientation: str | None = "LPS"
    center_organ: bool = True


def parse_spacing_csv_value(value: str | None) -> tuple[float, float, float] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("--normalize_spacing must contain 3 comma-separated values")
    spacing = tuple(float(part) for part in parts)
    if any(v <= 0.0 for v in spacing):
        raise ValueError("--normalize_spacing values must be strictly positive")
    return spacing


def _mask_bbox(mask, *, sitk_module):
    stats = sitk_module.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk_module.Cast(mask > 0, sitk_module.sitkUInt8))
    if not stats.HasLabel(1):
        return None
    bbox = stats.GetBoundingBox(1)
    index = np.asarray(bbox[:3], dtype=int)
    size = np.asarray(bbox[3:6], dtype=int)
    return index, size


def _crop_to_roi(image, *, index: np.ndarray, size: np.ndarray, sitk_module):
    roi = sitk_module.RegionOfInterestImageFilter()
    roi.SetIndex([int(v) for v in index.tolist()])
    roi.SetSize([int(v) for v in size.tolist()])
    return roi.Execute(image)


def _compute_crop_roi(
    *,
    image,
    organ_mask,
    config: OrganNormalizeConfig,
    sitk_module,
) -> tuple[np.ndarray, np.ndarray] | None:
    if config.crop_mode == "full":
        return None
    bbox = _mask_bbox(organ_mask, sitk_module=sitk_module)
    if bbox is None:
        return None
    index, size = bbox
    if config.crop_mode == "tight":
        margin_vox = np.zeros(3, dtype=int)
    else:
        spacing = np.asarray(organ_mask.GetSpacing(), dtype=float)
        margin_vox = np.ceil(float(config.margin_mm) / spacing).astype(int)
    image_size = np.asarray(image.GetSize(), dtype=int)
    start = np.maximum(0, index - margin_vox)
    stop = np.minimum(image_size, index + size + margin_vox)
    crop_size = np.maximum(1, stop - start)
    return start, crop_size


def _resample_to_spacing(
    image,
    *,
    spacing: tuple[float, float, float],
    interp: int,
    default_value: float,
    pixel_id: int,
    sitk_module,
):
    current_spacing = np.asarray(image.GetSpacing(), dtype=float)
    current_size = np.asarray(image.GetSize(), dtype=int)
    target_spacing = np.asarray(spacing, dtype=float)
    target_size = np.maximum(
        1,
        np.round((current_size * current_spacing) / target_spacing).astype(int),
    )
    resampler = sitk_module.ResampleImageFilter()
    resampler.SetOutputSpacing([float(v) for v in target_spacing.tolist()])
    resampler.SetSize([int(v) for v in target_size.tolist()])
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(interp)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetOutputPixelType(pixel_id)
    return resampler.Execute(image)


def _center_origin_from_mask(image, organ_mask, *, sitk_module):
    stats = sitk_module.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk_module.Cast(organ_mask > 0, sitk_module.sitkUInt8))
    if not stats.HasLabel(1):
        return image, organ_mask, None
    centroid = np.asarray(stats.GetCentroid(1), dtype=float)
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    size = np.asarray(image.GetSize(), dtype=float)
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    center_index = (size - 1.0) / 2.0
    offset = direction @ (center_index * spacing)
    new_origin = centroid - offset

    image_out = sitk_module.Image(image)
    image_out.SetOrigin([float(v) for v in new_origin.tolist()])
    mask_out = sitk_module.Image(organ_mask)
    mask_out.SetOrigin([float(v) for v in new_origin.tolist()])
    return image_out, mask_out, centroid.tolist()


def normalize_image_and_masks(
    *,
    image,
    organ_mask,
    masks_by_name: dict[str, Any],
    config: OrganNormalizeConfig,
    sitk_module,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Normalize organ-centric geometry for inter-patient comparisons."""
    if not reg_common.same_grid(image, organ_mask):
        organ_mask = reg_common.resample_like(
            image,
            organ_mask,
            tx=None,
            interp=sitk_module.sitkNearestNeighbor,
            default=0,
            pixel_id=sitk_module.sitkUInt8,
            sitk_module=sitk_module,
        )
    organ_mask = sitk_module.Cast(organ_mask > 0, sitk_module.sitkUInt8)

    oriented_image = image
    oriented_organ_mask = organ_mask
    oriented_masks: dict[str, Any] = {
        key: sitk_module.Cast(value > 0, sitk_module.sitkUInt8)
        for key, value in masks_by_name.items()
        if value is not None
    }
    applied_orientation = None
    if config.orientation:
        oriented_image = sitk_module.DICOMOrient(oriented_image, str(config.orientation))
        oriented_organ_mask = sitk_module.DICOMOrient(
            oriented_organ_mask,
            str(config.orientation),
        )
        oriented_masks = {
            key: sitk_module.DICOMOrient(mask, str(config.orientation))
            for key, mask in oriented_masks.items()
        }
        applied_orientation = str(config.orientation)

    crop_roi = _compute_crop_roi(
        image=oriented_image,
        organ_mask=oriented_organ_mask,
        config=config,
        sitk_module=sitk_module,
    )
    if crop_roi is not None:
        start, size = crop_roi
        oriented_image = _crop_to_roi(
            oriented_image,
            index=start,
            size=size,
            sitk_module=sitk_module,
        )
        oriented_organ_mask = _crop_to_roi(
            oriented_organ_mask,
            index=start,
            size=size,
            sitk_module=sitk_module,
        )
        oriented_masks = {
            key: _crop_to_roi(mask, index=start, size=size, sitk_module=sitk_module)
            for key, mask in oriented_masks.items()
        }

    if not config.keep_background:
        organ_float = sitk_module.Cast(oriented_organ_mask > 0, sitk_module.sitkFloat32)
        oriented_image = sitk_module.Multiply(
            sitk_module.Cast(oriented_image, sitk_module.sitkFloat32),
            organ_float,
        )
        oriented_image = sitk_module.Cast(oriented_image, image.GetPixelID())

    if config.spacing is not None:
        oriented_image = _resample_to_spacing(
            oriented_image,
            spacing=config.spacing,
            interp=sitk_module.sitkLinear,
            default_value=0.0,
            pixel_id=oriented_image.GetPixelID(),
            sitk_module=sitk_module,
        )
        oriented_organ_mask = _resample_to_spacing(
            oriented_organ_mask,
            spacing=config.spacing,
            interp=sitk_module.sitkNearestNeighbor,
            default_value=0,
            pixel_id=sitk_module.sitkUInt8,
            sitk_module=sitk_module,
        )
        oriented_masks = {
            key: _resample_to_spacing(
                mask,
                spacing=config.spacing,
                interp=sitk_module.sitkNearestNeighbor,
                default_value=0,
                pixel_id=sitk_module.sitkUInt8,
                sitk_module=sitk_module,
            )
            for key, mask in oriented_masks.items()
        }

    centered_centroid_mm = None
    if config.center_organ:
        oriented_image, oriented_organ_mask, centered_centroid_mm = _center_origin_from_mask(
            oriented_image,
            oriented_organ_mask,
            sitk_module=sitk_module,
        )
        if centered_centroid_mm is not None:
            for key, mask in list(oriented_masks.items()):
                mask_out = sitk_module.Image(mask)
                mask_out.SetOrigin(oriented_organ_mask.GetOrigin())
                oriented_masks[key] = mask_out

    metadata = {
        "crop_mode": config.crop_mode,
        "margin_mm": float(config.margin_mm),
        "keep_background": bool(config.keep_background),
        "spacing": list(config.spacing) if config.spacing is not None else None,
        "orientation": applied_orientation,
        "center_organ": bool(config.center_organ),
        "centered_centroid_mm": centered_centroid_mm,
    }
    return oriented_image, oriented_organ_mask, oriented_masks, metadata
