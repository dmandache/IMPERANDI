from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from imperandi.process import _registration_common as reg_common


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConsensusConfig:
    rule: str = "majority"
    majority_threshold: float = 0.5
    min_component_voxels: int = 10
    use_elastic_registration: bool = True
    band_mm: float = reg_common.DEFAULT_BAND_MM
    bspline_ctrl_spacing_mm: float = reg_common.DEFAULT_BSPLINE_CTRL_SPACING_MM


@dataclass(frozen=True)
class TumorComponent:
    component_id: int
    volume_vox: int
    volume_ml: float
    centroid_x_mm: float
    centroid_y_mm: float
    centroid_z_mm: float
    bbox_x_min: int
    bbox_y_min: int
    bbox_z_min: int
    bbox_x_size: int
    bbox_y_size: int
    bbox_z_size: int


@dataclass
class ConsensusVisitResult:
    patient_key: str
    visit_key: str
    reference_source_idx: int
    consensus_mask: Any
    components: list[TumorComponent]
    aligned_mask_count: int
    transform_metadata_by_source_idx: dict[int, dict[str, Any]]


def _choose_reference_row(
    rows: list[dict[str, Any]],
    *,
    phase_column: str,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for row in rows:
        phase_value = reg_common.normalize_phase_value(row.get(phase_column))
        phase_rank = 0 if phase_value == "portal" else 1
        ranked.append((phase_rank, int(row.get("_source_idx", -1)), row))
    ranked.sort(key=lambda item: (item[0], item[1]))
    reference = ranked[0][2]
    logger.debug(
        "Selected consensus reference source_idx=%s phase=%s from %d rows.",
        int(reference.get("_source_idx", -1)),
        reg_common.normalize_phase_value(reference.get(phase_column)),
        len(rows),
    )
    return reference


def _apply_consensus_rule(
    *,
    aligned_masks: list[np.ndarray],
    config: ConsensusConfig,
) -> np.ndarray:
    if not aligned_masks:
        raise ValueError("aligned_masks must not be empty")
    stack = np.stack([mask.astype(bool) for mask in aligned_masks], axis=0)
    votes = stack.sum(axis=0)
    rule = str(config.rule).strip().lower()
    if rule == "union":
        consensus = votes > 0
    elif rule == "intersection":
        consensus = votes == stack.shape[0]
    else:
        threshold = max(1, int(math.ceil(config.majority_threshold * stack.shape[0])))
        consensus = votes >= threshold
    if int(config.min_component_voxels) > 1:
        labels, count = ndimage.label(consensus)
        if count > 0:
            kept = np.zeros_like(consensus, dtype=bool)
            component_sizes = np.bincount(labels.ravel())
            for component_id in range(1, count + 1):
                if component_sizes[component_id] >= int(config.min_component_voxels):
                    kept |= labels == component_id
            consensus = kept
    return consensus.astype(np.uint8)


def _extract_components(
    mask_array: np.ndarray,
    *,
    reference_image: Any,
    sitk_module: Any,
) -> list[TumorComponent]:
    labels, count = ndimage.label(mask_array > 0)
    if count <= 0:
        return []
    objects = ndimage.find_objects(labels)
    spacing = tuple(float(v) for v in reference_image.GetSpacing())
    voxel_volume_ml = float(np.prod(spacing) / 1000.0)
    components: list[TumorComponent] = []
    for component_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        component = labels[slc] == component_id
        volume_vox = int(component.sum())
        component_coords_local = np.argwhere(component)
        local_origin = np.asarray([slc[0].start, slc[1].start, slc[2].start], dtype=float)
        centroid_zyx = component_coords_local.mean(axis=0) + local_origin
        centroid_xyz = centroid_zyx[::-1]
        centroid_mm = reference_image.TransformContinuousIndexToPhysicalPoint(
            tuple(float(v) for v in centroid_xyz.tolist())
        )
        z_slice, y_slice, x_slice = slc
        components.append(
            TumorComponent(
                component_id=int(component_id),
                volume_vox=volume_vox,
                volume_ml=float(volume_vox * voxel_volume_ml),
                centroid_x_mm=float(centroid_mm[0]),
                centroid_y_mm=float(centroid_mm[1]),
                centroid_z_mm=float(centroid_mm[2]),
                bbox_x_min=int(x_slice.start),
                bbox_y_min=int(y_slice.start),
                bbox_z_min=int(z_slice.start),
                bbox_x_size=int(x_slice.stop - x_slice.start),
                bbox_y_size=int(y_slice.stop - y_slice.start),
                bbox_z_size=int(z_slice.stop - z_slice.start),
            )
        )
    return components


def _mask_from_array(mask_array: np.ndarray, *, reference_image: Any, sitk_module: Any):
    return reg_common.image_from_array_like(
        mask_array.astype(np.uint8),
        reference_image=reference_image,
        sitk_module=sitk_module,
        cast_to=sitk_module.sitkUInt8,
    )


def _align_tumor_mask_to_reference(
    row: Mapping[str, Any],
    *,
    reference_row: Mapping[str, Any],
    reference_image: Any,
    reference_organ_mask: Any | None,
    tumor_mask_column: str,
    organ_mask_column: str | None,
    config: ConsensusConfig,
    sitk_module: Any,
) -> tuple[Any | None, dict[str, Any]]:
    source_idx = int(row.get("_source_idx", -1))
    tumor_path = row.get(tumor_mask_column)
    if not reg_common._is_existing_path(tumor_path):
        logger.debug(
            "Skipping consensus alignment for source_idx=%s: missing %s=%s.",
            source_idx,
            tumor_mask_column,
            tumor_path,
        )
        return None, {"status": "skipped", "reason": f"missing {tumor_mask_column}"}

    moving_tumor = reg_common.read_binary_mask(str(tumor_path), sitk_module=sitk_module)
    moving_tumor = reg_common.resample_like(
        reference_image,
        moving_tumor,
        tx=None,
        interp=sitk_module.sitkNearestNeighbor,
        default=0,
        pixel_id=sitk_module.sitkUInt8,
        sitk_module=sitk_module,
    )

    if int(reference_row.get("_source_idx", -1)) == source_idx:
        logger.debug(
            "Consensus source_idx=%s is the reference row; using reference-stage identity.",
            source_idx,
        )
        return moving_tumor, {"status": "ok", "stage": "reference", "source_idx": source_idx}

    if (
        organ_mask_column is None
        or not reg_common._is_existing_path(row.get(organ_mask_column))
        or not reg_common._is_existing_path(reference_row.get(organ_mask_column))
    ):
        logger.debug(
            (
                "Consensus source_idx=%s falling back to identity because organ masks "
                "are unavailable for registration."
            ),
            source_idx,
        )
        return moving_tumor, {
            "status": "ok",
            "stage": "identity",
            "source_idx": source_idx,
            "note": "organ mask unavailable, skipped registration",
        }

    moving_organ = reg_common.read_binary_mask(
        str(row.get(organ_mask_column)),
        sitk_module=sitk_module,
    )
    if reference_organ_mask is None:
        reference_organ_mask = reg_common.read_binary_mask(
            str(reference_row.get(organ_mask_column)),
            reference_image=reference_image,
            sitk_module=sitk_module,
        )

    rigid_tx = reg_common.rigid_register_mask_pair(
        fixed_mask=reference_organ_mask,
        moving_mask=moving_organ,
        sitk_module=sitk_module,
    )
    stage = "rigid"
    final_tx = rigid_tx
    dice_before = reg_common.dice_coeff(
        reference_organ_mask,
        moving_organ,
        sitk_module=sitk_module,
    )
    dice_after_rigid = reg_common.dice_coeff(
        reference_organ_mask,
        moving_organ,
        sitk_module=sitk_module,
        tx=rigid_tx,
    )
    dice_after_elastic: float | None = None
    if config.use_elastic_registration:
        try:
            elastic_tx = reg_common.bspline_register_mask_pair(
                fixed_mask=reference_organ_mask,
                moving_mask=moving_organ,
                initial_transform=rigid_tx,
                sitk_module=sitk_module,
                band_mm=float(config.band_mm),
                ctrl_spacing_mm=float(config.bspline_ctrl_spacing_mm),
            )
            dice_after_elastic = reg_common.dice_coeff(
                reference_organ_mask,
                moving_organ,
                sitk_module=sitk_module,
                tx=elastic_tx,
            )
            if dice_after_elastic >= dice_after_rigid:
                final_tx = elastic_tx
                stage = "bspline"
        except Exception:
            # MVP fallback to rigid is explicit and deterministic.
            stage = "rigid"
            logger.debug(
                "Consensus elastic stage failed for source_idx=%s; keeping rigid transform.",
                source_idx,
            )

    aligned_tumor = reg_common.resample_like(
        reference_image,
        moving_tumor,
        tx=final_tx,
        interp=sitk_module.sitkNearestNeighbor,
        default=0,
        pixel_id=sitk_module.sitkUInt8,
        sitk_module=sitk_module,
    )
    metadata = {
        "status": "ok",
        "source_idx": source_idx,
        "stage": stage,
        "dice_before": float(dice_before),
        "dice_after_rigid": float(dice_after_rigid),
        "dice_after_elastic": (
            None if dice_after_elastic is None else float(dice_after_elastic)
        ),
    }
    logger.debug(
        (
            "Consensus aligned source_idx=%s against reference_source_idx=%s "
            "with stage=%s dice_before=%.4f dice_after_rigid=%.4f "
            "dice_after_elastic=%s."
        ),
        source_idx,
        int(reference_row.get("_source_idx", -1)),
        stage,
        float(dice_before),
        float(dice_after_rigid),
        (
            "None"
            if dice_after_elastic is None
            else f"{float(dice_after_elastic):.4f}"
        ),
    )
    return aligned_tumor, metadata


def build_visit_consensus(
    rows: list[dict[str, Any]],
    *,
    patient_key: str,
    visit_key: str,
    tumor_mask_column: str,
    organ_mask_column: str | None,
    image_column: str = "nifti_path",
    phase_column: str = "phase",
    config: ConsensusConfig,
    sitk_module: Any,
) -> ConsensusVisitResult:
    if not rows:
        raise ValueError("rows must not be empty")
    reference_row = _choose_reference_row(rows, phase_column=phase_column)
    ref_idx = int(reference_row.get("_source_idx", -1))
    logger.debug(
        (
            "Building visit consensus for patient=%s visit=%s "
            "with %d rows, reference_source_idx=%s, rule=%s."
        ),
        patient_key,
        visit_key,
        len(rows),
        ref_idx,
        config.rule,
    )
    ref_image_path = reference_row.get(image_column)
    if not reg_common._is_existing_path(ref_image_path):
        raise ValueError(f"invalid reference image path: {ref_image_path}")
    reference_image = reg_common.read_image(str(ref_image_path), sitk_module)
    reference_organ_mask = None
    if organ_mask_column and reg_common._is_existing_path(reference_row.get(organ_mask_column)):
        reference_organ_mask = reg_common.read_binary_mask(
            str(reference_row.get(organ_mask_column)),
            reference_image=reference_image,
            sitk_module=sitk_module,
        )

    aligned_arrays: list[np.ndarray] = []
    metadata_by_source_idx: dict[int, dict[str, Any]] = {}
    for row in rows:
        source_idx = int(row.get("_source_idx", -1))
        aligned_mask, metadata = _align_tumor_mask_to_reference(
            row,
            reference_row=reference_row,
            reference_image=reference_image,
            reference_organ_mask=reference_organ_mask,
            tumor_mask_column=tumor_mask_column,
            organ_mask_column=organ_mask_column,
            config=config,
            sitk_module=sitk_module,
        )
        metadata_by_source_idx[source_idx] = metadata
        if aligned_mask is None:
            continue
        aligned_arrays.append(sitk_module.GetArrayFromImage(aligned_mask) > 0)

    if not aligned_arrays:
        empty = np.zeros(
            tuple(int(v) for v in reference_image.GetSize())[::-1],
            dtype=np.uint8,
        )
        consensus_mask = _mask_from_array(
            empty,
            reference_image=reference_image,
            sitk_module=sitk_module,
        )
        return ConsensusVisitResult(
            patient_key=str(patient_key),
            visit_key=str(visit_key),
            reference_source_idx=ref_idx,
            consensus_mask=consensus_mask,
            components=[],
            aligned_mask_count=0,
            transform_metadata_by_source_idx=metadata_by_source_idx,
        )

    consensus_array = _apply_consensus_rule(
        aligned_masks=[arr.astype(np.uint8) for arr in aligned_arrays],
        config=config,
    )
    consensus_mask = _mask_from_array(
        consensus_array,
        reference_image=reference_image,
        sitk_module=sitk_module,
    )
    components = _extract_components(
        consensus_array,
        reference_image=reference_image,
        sitk_module=sitk_module,
    )
    logger.debug(
        (
            "Built visit consensus for patient=%s visit=%s "
            "(aligned_mask_count=%d, component_count=%d)."
        ),
        patient_key,
        visit_key,
        len(aligned_arrays),
        len(components),
    )
    return ConsensusVisitResult(
        patient_key=str(patient_key),
        visit_key=str(visit_key),
        reference_source_idx=ref_idx,
        consensus_mask=consensus_mask,
        components=components,
        aligned_mask_count=len(aligned_arrays),
        transform_metadata_by_source_idx=metadata_by_source_idx,
    )
