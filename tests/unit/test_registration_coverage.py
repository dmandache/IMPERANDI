"""Synthetic completeness, confidence and per-voxel observation contracts."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from imperandi.process.registration import RegistrationConfig, register_cohort
from imperandi.process.registration import alignment
from imperandi.process.registration.alignment import (
    assess_organ_mask,
    overlap_qc,
    registration_confidence,
)
from imperandi.process.registration.cohort import fuse_tumors
from imperandi.process.registration.reporting import build_qc

sitk = pytest.importorskip("SimpleITK")


@pytest.fixture(autouse=True)
def single_thread():
    previous = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    yield
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(previous)


def complete_mask():
    z, y, x = np.indices((40, 40, 40))
    mask = sitk.GetImageFromArray(
        (((x - 20) / 12) ** 2 + ((y - 20) / 10) ** 2 + ((z - 20) / 14) ** 2 < 1).astype(
            np.uint8
        )
    )
    mask.SetSpacing((1.2, 1.5, 2.0))
    mask.SetOrigin((15, -20, 3))
    mask.SetDirection((0, -1, 0, 1, 0, 0, 0, 0, 1))
    return mask


def partial_mask(mask):
    return sitk.RegionOfInterest(mask, [40, 40, 5], [0, 0, 17])


def save_row(tmp_path, name, mask, phase):
    path = tmp_path / (name + ".nii.gz")
    sitk.WriteImage(mask, str(path))
    return dict(
        patient_key="p1",
        study_id="v1",
        Modality="CT",
        phase=phase,
        nifti_path=str(path),
        mask_liver=str(path),
        mask_liver_tumor=str(path),
    )


def test_complete_partial_and_fragmented_mask_qc():
    config = RegistrationConfig()
    mask = complete_mask()
    complete = assess_organ_mask(mask, config)
    partial = assess_organ_mask(partial_mask(mask), config)
    assert complete.status == "complete"
    assert partial.status == "partial"
    assert partial.boundary_contact == ((False, False), (False, False), (True, True))
    assert partial.volume_mm3 < complete.volume_mm3
    assert partial.largest_component_fraction == 1
    assert partial.bbox_index[2] == 0 and partial.bbox_index[5] == 5
    values = np.zeros((40, 40, 40), np.uint8)
    values[5:10, 5:10, 5:10] = 1
    values[25:30, 25:30, 25:30] = 1
    fragmented = assess_organ_mask(sitk.GetImageFromArray(values), config)
    assert fragmented.status == "invalid"
    assert fragmented.largest_component_fraction == 0.5
    assert "fragmented" in fragmented.reasons


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("side", ["low", "high"])
def test_internal_straight_side_is_partial(axis, side):
    mask = complete_mask()
    values = sitk.GetArrayFromImage(mask)
    cut = [slice(None)] * 3
    cut[axis] = slice(None, 20) if side == "low" else slice(21, None)
    values[tuple(cut)] = 0
    truncated = sitk.GetImageFromArray(values)
    truncated.CopyInformation(mask)

    quality = assess_organ_mask(truncated, RegistrationConfig())

    assert quality.status == "partial"
    assert "straight_side" in quality.reasons
    assert quality.boundary_contact == ((False, False),) * 3


def test_small_flat_tip_is_not_truncation():
    mask = complete_mask()
    values = sitk.GetArrayFromImage(mask)
    values[:8] = 0
    rounded = sitk.GetImageFromArray(values)
    rounded.CopyInformation(mask)
    quality = assess_organ_mask(rounded, RegistrationConfig())
    assert quality.status == "complete"
    assert "straight_side" not in quality.reasons


def test_internal_straight_side_uses_partial_registration(monkeypatch):
    fixed = complete_mask()
    values = sitk.GetArrayFromImage(fixed)
    values[:20] = 0
    moving = sitk.GetImageFromArray(values)
    moving.CopyInformation(fixed)
    monkeypatch.setattr(
        alignment,
        "initialize_pca",
        Mock(side_effect=AssertionError("PCA must be skipped")),
    )
    result = alignment.register_pair(fixed, moving, RegistrationConfig(iterations=5))
    assert result.stages["pca"]["status"] == "skipped_partial_coverage"
    with pytest.raises(ValueError, match="Partial organ masks are disabled"):
        alignment.register_pair(
            fixed, moving, RegistrationConfig(allow_partial_organs=False)
        )


def test_partial_fov_perfect_alignment_uses_common_dice(monkeypatch):
    fixed = complete_mask()
    moving = partial_mask(fixed)
    metrics = overlap_qc(fixed, moving, sitk.Euler3DTransform())
    assert metrics["dice_full"] < 0.5
    assert metrics["dice_common_fov"] == 1
    monkeypatch.setattr(
        alignment,
        "initialize_pca",
        Mock(side_effect=AssertionError("PCA must be skipped")),
    )
    result = alignment.register_pair(
        fixed, moving, RegistrationConfig(iterations=5, min_dice=0.8)
    )
    assert result.confidence == "ok_partial_coverage"
    assert result.overlap["dice_common_fov"] == 1
    assert result.stages["pca"]["status"] == "skipped_partial_coverage"
    assert result.organ_volume_ratio > 2


def test_equal_volume_does_not_establish_confidence_or_allow_elastic(monkeypatch):
    fixed = complete_mask()
    moving = sitk.Image(fixed)
    moving.SetOrigin(tuple(np.array(fixed.GetOrigin()) + [25, 0, 0]))
    optimizer = Mock()
    optimizer.GetOptimizerStopConditionDescription.return_value = "frozen for test"
    optimizer.GetOptimizerIteration.return_value = 0
    monkeypatch.setattr(sitk, "ImageRegistrationMethod", lambda: optimizer)
    monkeypatch.setattr(
        alignment, "initialize_pca", lambda *args: sitk.Euler3DTransform()
    )
    elastic = Mock(side_effect=AssertionError("Poor overlap must block elastic"))
    monkeypatch.setattr(alignment, "elastic_refine", elastic)
    result = alignment.register_pair(
        fixed, moving, RegistrationConfig(min_dice=0, elastic=True)
    )
    assert result.organ_volume_ratio == 1
    assert result.confidence == "low_confidence"
    assert result.stages["elastic"]["status"] == "skipped_low_dice"
    elastic.assert_not_called()


def test_partial_reference_priority_and_qc(tmp_path):
    complete = complete_mask()
    rows = [
        save_row(tmp_path, "partial", partial_mask(complete), "PORTAL_VENOUS"),
        save_row(tmp_path, "complete", complete, "ARTERIAL"),
    ]
    config = RegistrationConfig(iterations=5, method="majority")
    out, errors = register_cohort(pd.DataFrame(rows), tmp_path / "out", config)
    assert errors.empty
    assert out.loc[0, "registration_reference_id"] == out.loc[1, "registration_scan_id"]
    assert out.loc[0, "registration_status"] == "ok_partial_coverage"
    assert out.registration_consensus_contributors.eq(2).all()
    qc = build_qc(out, errors, config)
    assert qc.loc[0, "registration_dice_common_fov"] == 1
    assert qc.loc[0, "registration_completeness"] == "partial"
    events = [
        json.loads(line)
        for line in Path(out.loc[0, "registration_log_path"]).read_text().splitlines()
    ]
    assert events[1]["anatomical_qc"]["registration_consensus_contributors"] == 2


def test_low_confidence_scan_excluded_from_consensus(tmp_path):
    mask = complete_mask()
    rows = [
        save_row(tmp_path, "a", mask, "PORTAL_VENOUS"),
        save_row(tmp_path, "b", mask, "ARTERIAL"),
    ]

    def uncertain(*args):
        return alignment.TransformResult(
            sitk.Euler3DTransform(), "rigid", 1, 1, [], confidence="low_confidence"
        )

    out, errors = register_cohort(
        pd.DataFrame(rows),
        tmp_path / "out",
        RegistrationConfig(method="majority"),
        pair_registration=uncertain,
    )
    assert errors.empty
    assert out.loc[1, "registration_status"] == "low_confidence"
    assert out.registration_consensus_contributors.eq(1).all()
    assert out.registration_consensus_excluded.eq(1).all()
    assert out.loc[1, "tumor_consensus_input_status"] == "excluded_low_confidence"
    assert pd.isna(out.loc[1, "reg_tumor_native_path"])


@pytest.mark.parametrize("method", ["majority", "union", "intersection"])
def test_unobserved_voxels_are_unknown(method):
    def image(values):
        return sitk.GetImageFromArray(np.array(values, np.uint8).reshape(1, 1, -1))

    masks = [image([1, 1, 1, 0]), image([0, 0, 1, 1])]
    coverage = [image([1, 1, 1, 0]), image([0, 1, 1, 0])]
    result = fuse_tumors(masks, coverage, method=method)
    assert sitk.GetArrayFromImage(result.probability).ravel().tolist() == [1, 0.5, 1, 0]
    assert sitk.GetArrayFromImage(result.observation_count).ravel().tolist() == [
        1,
        2,
        2,
        0,
    ]
    assert sitk.GetArrayFromImage(result.coverage).ravel().tolist() == [1, 1, 1, 0]
    assert sitk.GetArrayFromImage(result.mask).ravel().tolist() == [
        1,
        int(method == "union"),
        1,
        0,
    ]


def test_tiny_common_fov_is_low_confidence():
    metrics = dict(
        dice_full=0.01,
        dice_common_fov=1,
        common_fov_fraction=0.001,
        common_foreground_voxels=10,
    )
    assert (
        registration_confidence(metrics, True, RegistrationConfig()) == "low_confidence"
    )


@pytest.mark.parametrize(
    "name",
    [
        "elastic_min_dice",
        "min_confidence_dice",
        "min_common_fov_fraction",
        "min_largest_component_fraction",
    ],
)
@pytest.mark.parametrize("value", [-1, 0, 1.1, float("nan"), float("inf"), True])
def test_quality_threshold_validation(name, value):
    with pytest.raises(ValueError, match=name):
        RegistrationConfig(**{name: value})


def test_partial_masks_can_be_disabled(tmp_path):
    mask = complete_mask()
    rows = [
        save_row(tmp_path, "partial", partial_mask(mask), "PORTAL_VENOUS"),
        save_row(tmp_path, "complete", mask, "ARTERIAL"),
    ]
    out, errors = register_cohort(
        pd.DataFrame(rows),
        tmp_path / "out",
        RegistrationConfig(allow_partial_organs=False),
    )
    assert out.loc[0, "registration_status"] == "invalid_organ_mask"
    assert out.loc[0, "registration_skip_reason"] == "partial_organs_disabled"
    assert errors.stage.tolist() == ["organ_qc"]


def test_staple_missing_observations_are_explicitly_restricted():
    def image(values):
        return sitk.GetImageFromArray(np.array(values, np.uint8).reshape(1, 1, -1))

    masks = [image([1, 1, 0, 0]), image([1, 1, 0, 1])]
    coverages = [image([1, 1, 1, 0]), image([0, 1, 1, 1])]
    result = fuse_tumors(masks, coverages, method="staple")
    assert result.support_policy == "common_fov_only"
    assert sitk.GetArrayFromImage(result.coverage).ravel().tolist() == [0, 1, 1, 0]
    with pytest.raises(ValueError, match="support"):
        fuse_tumors(masks, [image([1, 0, 0, 0]), image([0, 0, 0, 1])], method="staple")
