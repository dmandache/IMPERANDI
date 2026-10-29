"""Stage chains, objective guards, and transactional transform acceptance."""

from dataclasses import asdict

import numpy as np
import pytest

from imperandi.process.registration import alignment as a, pipeline as p
from imperandi.process.registration.config import RegistrationConfig, StageConfig

sitk = pytest.importorskip("SimpleITK")


@pytest.fixture(autouse=True)
def one_thread():
    previous = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    yield
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(previous)


@pytest.fixture
def images():
    z, y, x = np.indices((20, 24, 28))
    mask = sitk.GetImageFromArray(
        (((x - 13) / 8) ** 2 + ((y - 11) / 7) ** 2 + ((z - 9) / 5) ** 2 < 1).astype(
            np.uint8
        )
    )
    intensity = sitk.GetImageFromArray(
        (x + 2 * y + 3 * z + 20 * np.sin(x / 3) * np.cos(y / 4)).astype(np.float32)
    )
    return mask, intensity


class Optimizer:
    def GetOptimizerStopConditionDescription(self):
        return "test"

    def GetOptimizerIteration(self):
        return 1


def affine(shift):
    tx = sitk.AffineTransform(3)
    tx.SetTranslation((shift, 0, 0))
    return tx


def elastic(mask, shift):
    bspline = sitk.BSplineTransformInitializer(mask, [1, 1, 1])
    values = np.zeros(bspline.GetNumberOfParameters())
    values[: len(values) // 3] = shift
    bspline.SetParameters(values)
    return sitk.CompositeTransform([affine(0), bspline])


def setup_scores(monkeypatch, values):
    scores = iter(values)
    monkeypatch.setattr(p.PairContext, "score", lambda *args: next(scores))
    monkeypatch.setattr(a, "initialize_pca", lambda *args: sitk.Euler3DTransform())
    monkeypatch.setattr(
        p.PairContext,
        "compare_mi",
        lambda self, transforms, metric: (
            [0.1 + tx.TransformPoint((10, 10, 10))[0] / 100 for tx in transforms],
            100,
        ),
    )


def test_rejected_affine_is_not_input_to_next_stage(monkeypatch, images):
    mask, intensity = images
    setup_scores(monkeypatch, [0.5, 0.5, 0.4, 0.5])
    observed = []

    def first(context, state, stage):
        return p.StageResult(affine(2), Optimizer())

    def second(context, state, stage):
        observed.append(state.transform.TransformPoint((0, 0, 0)))
        return p.StageResult(affine(1), Optimizer())

    monkeypatch.setitem(p.EXECUTORS, "mask_affine", first)
    monkeypatch.setitem(p.EXECUTORS, "mi_affine", second)
    result = a.register_pair(
        mask,
        mask,
        RegistrationConfig(stages=[{"type": "mask_affine"}, {"type": "mi_affine"}]),
        intensity,
        intensity,
    )
    assert observed == [(0, 0, 0)]
    assert result.stages["mask_affine"]["status"] == "rejected_worse_dice"
    assert result.stage == "mi_affine"


def test_mask_skip_does_not_skip_requested_mi(monkeypatch, images):
    mask, intensity = images
    setup_scores(monkeypatch, [1, 1, 0.999])
    monkeypatch.setitem(
        p.EXECUTORS, "mask_affine", lambda *args: pytest.fail("mask should skip")
    )
    monkeypatch.setitem(
        p.EXECUTORS, "mi_affine", lambda *args: p.StageResult(affine(1), Optimizer())
    )
    result = a.register_pair(
        mask,
        mask,
        RegistrationConfig(stages=[{"type": "mask_affine"}, {"type": "mi_affine"}]),
        intensity,
        intensity,
    )
    assert result.stages["mask_affine"]["status"] == "skipped_acceptable_dice"
    assert result.stage == "mi_affine"
    assert result.stages["mi_affine"]["dice_guard_reference"] == 1


@pytest.mark.parametrize("failure", ["dice", "qc", "inverse"])
def test_elastic_failure_preserves_previous_elastic_inverse(
    monkeypatch, images, failure
):
    mask, intensity = images
    first, second = elastic(mask, 0.2), elastic(mask, 0.4)
    saved_inverse = affine(-0.2)
    setup_scores(monkeypatch, [0.5, 0.5, 0.6, 0.59 if failure == "dice" else 0.6])
    monkeypatch.setitem(
        p.EXECUTORS, "mask_elastic", lambda *args: p.StageResult(first, Optimizer())
    )

    def next_stage(context, state, stage):
        assert state.transform is first
        assert state.inverse is saved_inverse
        return p.StageResult(second, Optimizer())

    monkeypatch.setitem(p.EXECUTORS, "mi_elastic", next_stage)

    def qc(tx, *args):
        return None, {
            "deformation_qc_passed": not (tx is second and failure == "qc"),
            "jacobian_min": 1,
            "jacobian_max": 1,
        }

    def inverse(tx, *args):
        if tx is second:
            assert failure == "inverse"
            raise ValueError("inverse failed")
        return saved_inverse

    monkeypatch.setattr(a, "elastic_deformation_qc", qc)
    monkeypatch.setattr(a, "invert_mask_elastic", inverse)
    result = a.register_pair(
        mask,
        mask,
        RegistrationConfig(stages=[{"type": "mask_elastic"}, {"type": "mi_elastic"}]),
        intensity,
        intensity,
    )
    assert result.reference_to_scan is first
    assert result.scan_to_reference is saved_inverse
    assert result.stage == "mask_elastic"
    assert (
        result.stages["mi_elastic"]["status"]
        == {
            "dice": "rejected_worse_dice",
            "qc": "rejected_deformation_qc",
            "inverse": "failed",
        }[failure]
    )


def test_later_mask_stage_preserves_accepted_mi(monkeypatch, images):
    mask, intensity = images
    setup_scores(monkeypatch, [0.5, 0.5, 0.5, 0.7])
    monkeypatch.setitem(
        p.EXECUTORS, "mi_affine", lambda *args: p.StageResult(affine(2), Optimizer())
    )
    monkeypatch.setitem(
        p.EXECUTORS, "mask_affine", lambda *args: p.StageResult(affine(1), Optimizer())
    )
    result = a.register_pair(
        mask,
        mask,
        RegistrationConfig(stages=[{"type": "mi_affine"}, {"type": "mask_affine"}]),
        intensity,
        intensity,
    )
    assert result.stage == "mi_affine"
    assert result.stages["mask_affine"]["status"] == "rejected_mi_guard"


def test_mi_losses_cannot_accumulate_across_stages(monkeypatch, images):
    mask, intensity = images
    setup_scores(monkeypatch, [0.8, 0.8, 0.799, 0.797])
    monkeypatch.setitem(
        p.EXECUTORS, "mi_affine", lambda *args: p.StageResult(affine(0.2), Optimizer())
    )
    monkeypatch.setitem(
        p.EXECUTORS,
        "mi_elastic",
        lambda *args: p.StageResult(elastic(mask, 0.4), Optimizer()),
    )
    result = a.register_pair(
        mask,
        mask,
        RegistrationConfig(stages=[{"type": "mi_affine"}, {"type": "mi_elastic"}]),
        intensity,
        intensity,
    )
    assert result.stage == "mi_affine"
    assert result.stages["mi_elastic"]["dice_guard_reference"] == 0.8
    assert result.stages["mi_elastic"]["status"] == "rejected_worse_dice"


@pytest.mark.parametrize("name", ["mi_affine", "mi_elastic"])
def test_explicit_mi_requires_images_and_nonconstant_roi(images, name):
    mask, _ = images
    config = RegistrationConfig(stages=[{"type": name}])
    with pytest.raises(ValueError, match="intensity images"):
        a.register_pair(mask, mask, config)
    constant = sitk.Cast(mask * 0, sitk.sitkFloat32)
    result = a.register_pair(mask, mask, config, constant, constant)
    assert result.stages[name]["status"] == "failed"
    assert "nonconstant" in result.stages[name]["error"]


def test_mask_only_chain_does_not_evaluate_mi(monkeypatch, images):
    mask, _ = images
    monkeypatch.setattr(
        a, "mutual_information_score", lambda *args, **kwargs: pytest.fail("MI called")
    )
    config = RegistrationConfig(
        stages=[
            {"type": "mask_elastic", "run_if": "always", "optimizer": {"iterations": 1}}
        ]
    )
    result = a.register_pair(mask, mask, config)
    assert result.stages["mask_elastic"]["status"] != "failed"
    assert result.dice_after == 1


def test_failed_elastic_optimizer_retains_qc_diagnostics(monkeypatch, images):
    mask, _ = images

    def fail(fixed, moving, initial, config, diagnostics, **kwargs):
        diagnostics["optimizer_qc_stop_requested"] = True
        diagnostics["optimizer_qc_stop_reasons"] = [
            "maximum_displacement_exceeds_limit"
        ]
        raise RuntimeError("optimizer did not stop")

    monkeypatch.setattr(a, "mask_elastic_refine", fail)
    result = a.register_pair(
        mask,
        mask,
        RegistrationConfig(stages=[{"type": "mask_elastic", "run_if": "always"}]),
    )
    detail = result.stages["mask_elastic"]
    assert detail["status"] == "failed"
    assert detail["optimizer_qc_stop_requested"]
    assert detail["optimizer_qc_stop_reasons"] == ["maximum_displacement_exceeds_limit"]
    assert result.stage == "identity"


def test_elastic_refinement_reuses_mesh_and_does_not_double_initialization(
    monkeypatch, images
):
    mask, intensity = images
    config = RegistrationConfig(elastic_optimizer_iterations=1)
    dm = a.distance_map(mask, padding_mm=25, band_mm=15)
    initial = elastic(dm, 0.4)
    linear = initial.GetNthTransform(0)
    linear.SetMatrix((1.1, 0.1, 0, 0, 0.9, 0, 0, 0, 1))
    initial = sitk.CompositeTransform([linear, initial.GetNthTransform(1)])
    before = tuple(initial.GetNthTransform(1).GetParameters())
    factory = sitk.ImageRegistrationMethod

    class NoUpdates:
        def __init__(self):
            self.inner = factory()

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def SetInitialTransform(self, transform, **kwargs):
            assert tuple(transform.GetParameters()) == before
            return self.inner.SetInitialTransform(transform, **kwargs)

        def Execute(self, *args):
            pass

    monkeypatch.setattr(sitk, "ImageRegistrationMethod", NoUpdates)
    tx, _ = a.mi_elastic_refine(
        dm,
        dm,
        initial,
        config,
        stage=StageConfig.resolve({"type": "mi_elastic"}, config),
        fixed_image=intensity,
        moving_image=intensity,
        fixed_organ=mask,
        moving_organ=mask,
    )
    assert tx.GetNumberOfTransforms() == 2
    point = (13, 11, 9)
    assert tx.TransformPoint(point) == pytest.approx(initial.TransformPoint(point))
    assert tuple(initial.GetNthTransform(1).GetParameters()) == before
    assert tx.TransformPoint(point) != pytest.approx(
        initial.GetNthTransform(1).TransformPoint(linear.TransformPoint(point))
    )


def test_mi_comparison_uses_same_support_and_is_repeatable(images):
    mask, intensity = images
    context = p.PairContext(
        mask, mask, RegistrationConfig(), intensity, intensity, False
    )
    metric = StageConfig.resolve({"type": "mi_affine"}, context.config).metric
    transforms = [affine(0), affine(1)]
    scores, count = context.compare_mi(transforms, metric)
    again, same_count = context.compare_mi(transforms, metric)
    assert count == same_count
    assert scores == pytest.approx(again, abs=1e-12)
    assert scores[0] > scores[1]


def test_partial_support_excludes_unobserved_candidate_background(images):
    mask, intensity = images
    moving_mask, moving_image = mask[:20, :, :], intensity[:20, :, :]
    context = p.PairContext(
        mask, moving_mask, RegistrationConfig(), intensity, moving_image, True
    )
    support, count = context.support([affine(0), affine(4)])
    values = sitk.GetArrayFromImage(support)
    assert count == 16 * mask.GetSize()[1] * mask.GetSize()[2]
    assert values[:, :, :16].all()
    assert not values[:, :, 16:].any()
    assert np.isfinite(context.dice_on_support(affine(0), support))
    assert np.isfinite(context.dice_on_support(affine(4), support))
    with pytest.raises(ValueError, match="support"):
        context.support([affine(100)])


def test_affine_refinement_updates_complete_mapping_without_composing_twice(
    monkeypatch, images
):
    mask, intensity = images
    initial = affine(2)
    initial.SetCenter((13, 11, 9))
    initial.SetMatrix((1.1, 0.1, 0, 0, 0.9, 0, 0, 0, 1))
    parameters = initial.GetParameters()
    factory = sitk.ImageRegistrationMethod

    class NoUpdates:
        def __init__(self):
            self.inner = factory()

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def Execute(self, *args):
            pass

    monkeypatch.setattr(sitk, "ImageRegistrationMethod", NoUpdates)
    candidate, _ = a.mi_refine(
        intensity, intensity, mask, mask, initial, RegistrationConfig(), affine=True
    )
    point = (12, 10, 8)
    assert candidate.TransformPoint(point) == pytest.approx(
        initial.TransformPoint(point)
    )
    assert initial.GetParameters() == parameters
    assert candidate is not initial


def test_resolved_plan_round_trip_and_independent_nested_settings():
    raw = [
        {
            "type": "mi_affine",
            "pyramid": {"shrink_factors": [2, 1], "smoothing_sigmas_mm": [1, 0]},
        }
    ]
    config = RegistrationConfig(stages=raw)
    assert RegistrationConfig.from_mapping(asdict(config)) == config
    raw[0]["pyramid"]["shrink_factors"].clear()
    assert config.stages[0].pyramid.shrink_factors == [2, 1]


@pytest.mark.parametrize(
    "stages",
    [
        None,
        {},
        "mask_affine",
        ["mask_affine"],
        [{"type": "unknown"}],
        [{"type": "mask_rigid"}] * 2,
        [{"type": "mask_affine"}, {"type": "mask_rigid"}],
        [{"type": "mask_elastic"}, {"type": "mi_affine"}],
        [{"type": "mask_affine", "metric": {}}],
        [{"type": "mask_affine", "acceptance": {"minimum_mi_improvement": 0}}],
        [{"type": "mi_affine", "acceptance": {"minimum_dice_improvement": 0}}],
        [{"type": "mi_affine", "metric": {"sampling_seed": 0}}],
        [{"type": "mi_affine", "metric": {"histogram_bins": True}}],
        [{"type": "mi_elastic", "optimizer": {"iterations": 101}}],
        [{"type": "mask_affine", "run_if": {"dice_below": float("nan")}}],
        [{"type": "mask_affine", "optimizer": {"iterations": True}}],
        [{"type": "mi_affine", "metric": {"typo": 1}}],
    ],
)
def test_invalid_stage_plans_are_rejected(stages):
    with pytest.raises(ValueError):
        RegistrationConfig(stages=stages)


@pytest.mark.parametrize(
    "key",
    [
        "enable_affine_stage",
        "enable_elastic_stage",
        "early_stop_organ_dice",
        "elastic_trigger_organ_dice",
        "minimum_stage_dice_improvement",
        "minimum_stage_mi_improvement",
        "maximum_mi_stage_dice_decrease",
    ],
)
def test_removed_settings_have_no_compatibility_path(key):
    with pytest.raises(ValueError, match="Unknown registration settings"):
        RegistrationConfig.from_mapping({key: True})


def test_empty_plan_runs_initialization_only(images):
    mask, _ = images
    moving = sitk.Image(mask)
    moving.SetOrigin((3, 0, 0))
    result = a.register_pair(mask, moving, RegistrationConfig(stages=[]))
    assert result.stage == "pca"
    assert result.dice_after == 1
    assert result.stages["mask_rigid"]["status"] == "skipped_not_configured"


def test_real_mi_elastic_improves_internal_alignment_with_unchanged_mask_dice():
    z, y, x = np.indices((24, 28, 32))
    mask = sitk.GetImageFromArray(
        ((((x - 15) / 10) ** 2 + ((y - 13) / 9) ** 2 + ((z - 11) / 7) ** 2) < 1).astype(
            np.uint8
        )
    )
    intensity = sitk.GetImageFromArray(
        (
            np.exp(-((x - 13) ** 2 + (y - 13) ** 2 + (z - 10) ** 2) / 20)
            + 0.5 * np.exp(-((x - 19) ** 2 + (y - 16) ** 2 + (z - 13) ** 2) / 12)
        ).astype(np.float32)
    )
    moving = sitk.Resample(
        intensity,
        intensity,
        sitk.TranslationTransform(3, [-0.4, 0, 0]),
        sitk.sitkLinear,
        0,
    )
    config = RegistrationConfig(
        stages=[
            {
                "type": "mi_elastic",
                "optimizer": {"iterations": 15},
                "pyramid": {"shrink_factors": [1], "smoothing_sigmas_mm": [0]},
            }
        ]
    )
    result = a.register_pair(mask, mask, config, intensity, moving)
    assert result.stage == "mi_elastic"
    assert result.dice_before == result.dice_after == 1
    assert result.stages["mi_elastic"]["mutual_information_improvement"] > 0.01
    assert result.stages["mi_elastic"]["deformation_qc_passed"]
    assert result.reference_to_scan.GetNumberOfTransforms() == 2
    point = (13, 13, 10)
    assert result.scan_to_reference.TransformPoint(
        result.reference_to_scan.TransformPoint(point)
    ) == pytest.approx(point, abs=0.01)


@pytest.mark.parametrize("output_space", ["moving", "reference"])
def test_mi_elastic_cohort_outputs_and_stage_reports(tmp_path, images, output_space):
    import json
    import pandas as pd
    from imperandi.process.registration import register_cohort

    mask, intensity = images
    rows = []
    for index in range(2):
        image_path, mask_path = (
            tmp_path / f"image{index}.nii.gz",
            tmp_path / f"mask{index}.nii.gz",
        )
        sitk.WriteImage(intensity, str(image_path))
        sitk.WriteImage(mask, str(mask_path))
        rows.append(
            {
                "patient_key": "p",
                "study_id": "s",
                "Modality": "CT",
                "nifti_path": str(image_path),
                "mask_liver_path": str(mask_path),
            }
        )
    tx = elastic(mask, 0.1)
    config = RegistrationConfig(
        stages=[{"type": "mi_elastic"}], output_space=output_space
    )
    diagnostics = {}
    inverse = a.invert_mask_elastic(tx, mask, diagnostics)
    overlap = a.overlap_qc(mask, mask, tx)
    result = a.TransformResult(
        tx,
        "mi_elastic",
        1,
        1,
        [],
        {
            "mi_elastic": {
                "dice": 1,
                "status": "accepted",
                "accepted": True,
                "selected": True,
            }
        },
        {"mi_elastic": tx},
        inverse,
        "ok",
        overlap,
        1,
    )
    output, errors = register_cohort(
        pd.DataFrame(rows),
        tmp_path / "out",
        config,
        pair_registration=lambda *args: result,
    )
    assert errors.empty
    registered = output[output.registration_status != "reference"].iloc[0]
    assert registered.registration_selected_stage == "mi_elastic"
    detail = json.loads(registered.registration_stage_details)
    assert detail["mi_elastic"]["accepted"]
    actual = sitk.ReadImage(registered.mask_liver_path)
    assert actual.GetSize() == mask.GetSize()
    assert set(np.unique(sitk.GetArrayFromImage(actual))) <= {0, 1}
    if output_space == "reference":
        published = sitk.ReadImage(registered.nifti_path)
        expected = a.resample(intensity, intensity, tx, label=False)
        assert np.allclose(
            sitk.GetArrayFromImage(published), sitk.GetArrayFromImage(expected)
        )
