"""Transactional execution of an explicit registration stage plan."""

from dataclasses import asdict, dataclass, field
import time

import numpy as np

from . import alignment as a
from .config import REGISTRATION_STAGES


@dataclass
class AcceptedState:
    transform: object
    inverse: object
    dice: float
    stage: str
    peak_transform: object
    peak_dice: float
    mi_guard: tuple | None = None


@dataclass
class StageResult:
    transform: object
    optimizer: object
    diagnostics: dict = field(default_factory=dict)


class StageExecutionError(ValueError):
    """Retain optimizer diagnostics when execution cannot return a candidate."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass
class PairContext:
    fixed_organ: object
    moving_organ: object
    config: object
    fixed_image: object | None
    moving_image: object | None
    partial: bool
    fixed_dm: object | None = None
    moving_dm: object | None = None

    def distances(self):
        if self.fixed_dm is None:
            padding = self.config.distance_map_crop_padding_mm
            if any(s.model == "elastic" for s in self.config.stages):
                padding = max(
                    padding,
                    self.config.organ_boundary_band_half_width_mm
                    + self.config.maximum_elastic_displacement_mm,
                )
            self.fixed_dm, self.moving_dm = (
                a.distance_map(
                    mask,
                    padding_mm=padding,
                    band_mm=self.config.organ_boundary_band_half_width_mm,
                )
                for mask in (self.fixed_organ, self.moving_organ)
            )
        return self.fixed_dm, self.moving_dm

    def score(self, transform):
        if not self.partial:
            return a.dice(self.fixed_organ, self.moving_organ, transform)
        metrics = a.overlap_qc(self.fixed_organ, self.moving_organ, transform)
        if (
            metrics["common_fov_fraction"]
            < self.config.minimum_common_field_of_view_fraction
            or metrics["common_foreground_voxels"] < 4
        ):
            return 0.0
        return metrics["dice_common_fov"]

    def support(self, transforms, metric=None):
        """One fixed physical support for every transform in a comparison."""
        sitk = a.backend()
        support = a.coverage_image(self.fixed_organ)
        moving_coverage = a.coverage_image(self.moving_organ)
        identity = sitk.Transform(3, sitk.sitkIdentity)
        if metric is not None:
            support &= a.metric_roi(self.fixed_organ, self.config, metric)
            support &= a.resample(
                a.coverage_image(self.fixed_image), self.fixed_organ, identity
            )
            moving_roi = a.metric_roi(self.moving_organ, self.config, metric)
        for transform in transforms:
            support &= a.resample(moving_coverage, self.fixed_organ, transform)
            if metric is not None:
                support &= a.resample(moving_roi, self.fixed_organ, transform)
                support &= a.resample(
                    a.coverage_image(self.moving_image), self.fixed_organ, transform
                )
        count = int(np.count_nonzero(sitk.GetArrayViewFromImage(support)))
        # The anatomical ROI can be much smaller than the image. Its minimum
        # sample count is separate from the whole-image common-FOV requirement.
        if count < (32 if metric is not None else 4):
            raise ValueError("Insufficient common evaluation support")
        if (
            metric is None
            and count / np.prod(self.fixed_organ.GetSize())
            < self.config.minimum_common_field_of_view_fraction
        ):
            raise ValueError("Insufficient common field of view")
        return support, count

    def dice_on_support(self, transform, support):
        sitk = a.backend()
        region = sitk.GetArrayViewFromImage(support) > 0
        fixed = (sitk.GetArrayViewFromImage(self.fixed_organ) > 0) & region
        warped = a.resample(self.moving_organ, self.fixed_organ, transform)
        moving = (sitk.GetArrayViewFromImage(warped) > 0) & region
        denominator = int(fixed.sum() + moving.sum())
        if denominator < 4:
            raise ValueError("Insufficient common anatomical foreground")
        return 2.0 * np.count_nonzero(fixed & moving) / denominator

    def compare_mi(self, transforms, metric):
        sitk = a.backend()
        support, count = self.support(transforms, metric)
        region = sitk.GetArrayViewFromImage(support) > 0
        identity = sitk.Transform(3, sitk.sitkIdentity)
        fixed = a.resample(self.fixed_image, self.fixed_organ, identity, label=False)
        values = sitk.GetArrayViewFromImage(fixed)[region]
        if not np.isfinite(values).all() or np.ptp(values) <= 0:
            raise ValueError("MI requires finite, nonconstant fixed ROI intensities")
        scores = []
        for transform in transforms:
            warped = a.resample(
                self.moving_image, self.fixed_organ, transform, label=False
            )
            values = sitk.GetArrayViewFromImage(warped)[region]
            if not np.isfinite(values).all() or np.ptp(values) <= 0:
                raise ValueError(
                    "MI requires finite, nonconstant moving ROI intensities"
                )
            scores.append(
                a.mutual_information_score(
                    self.fixed_image,
                    self.moving_image,
                    self.fixed_organ,
                    self.moving_organ,
                    transform,
                    self.config,
                    metric=metric,
                    support=support,
                )
            )
        return scores, count


def _linear(context, state, stage):
    if stage.uses_mi:
        tx, optimizer = a.mi_refine(
            context.fixed_image,
            context.moving_image,
            context.fixed_organ,
            context.moving_organ,
            state.transform,
            context.config,
            affine=True,
            stage=stage,
        )
    else:
        function = (
            a.mask_rigid_refine if stage.model == "rigid" else a.mask_affine_refine
        )
        tx, optimizer = function(
            *context.distances(), state.transform, context.config, stage=stage
        )
    return StageResult(tx, optimizer)


def _elastic(context, state, stage):
    diagnostics = {}
    function = a.mi_elastic_refine if stage.uses_mi else a.mask_elastic_refine
    try:
        tx, optimizer = function(
            *context.distances(),
            state.transform,
            context.config,
            diagnostics,
            stage=stage,
            fixed_image=context.fixed_image,
            moving_image=context.moving_image,
            fixed_organ=context.fixed_organ,
            moving_organ=context.moving_organ,
        )
    except (RuntimeError, ValueError) as exc:
        raise StageExecutionError(str(exc), diagnostics) from exc
    return StageResult(tx, optimizer, diagnostics)


EXECUTORS = {
    "mask_rigid": _linear,
    "mask_affine": _linear,
    "mi_affine": _linear,
    "mask_elastic": _elastic,
    "mi_elastic": _elastic,
}


def _acceptance(context, state, stage, candidate, candidate_dice, detail):
    config = stage.acceptance
    input_dice, compared_dice, peak_dice = state.dice, candidate_dice, state.peak_dice
    if context.partial:
        transforms = [state.transform, candidate, state.peak_transform]
        support, count = context.support(transforms)
        input_dice, compared_dice, peak_dice = [
            context.dice_on_support(tx, support) for tx in transforms
        ]
        detail.update(
            comparison_support_voxels=count,
            comparison_input_dice=input_dice,
            comparison_dice=compared_dice,
        )
    detail.update(dice=candidate_dice, dice_improvement=compared_dice - input_dice)
    if stage.uses_mi:
        scores, count = context.compare_mi([state.transform, candidate], stage.metric)
        before, after = scores
        detail.update(
            selection_metric="mattes_mutual_information",
            input_mutual_information=before,
            mutual_information=after,
            mutual_information_improvement=after - before,
            minimum_required_mi_improvement=config.minimum_mi_improvement,
            dice_guard_reference=peak_dice,
            maximum_allowed_dice_decrease=config.maximum_dice_decrease,
            mi_support_voxels=count,
        )
        return a._mi_candidate_rejection(
            compared_dice,
            peak_dice,
            before,
            after,
            config.minimum_mi_improvement,
            config.maximum_dice_decrease,
        )
    detail["minimum_required_dice_improvement"] = config.minimum_dice_improvement
    if compared_dice < input_dice - a.DICE_TOLERANCE:
        return "rejected_worse_dice"
    if compared_dice - input_dice < config.minimum_dice_improvement - a.DICE_TOLERANCE:
        return "rejected_insufficient_improvement"
    if state.mi_guard is not None:
        guard_tx, metric = state.mi_guard
        (before, after), count = context.compare_mi([guard_tx, candidate], metric)
        detail.update(
            mi_guard_reference=before,
            mi_guard_candidate=after,
            mi_guard_support_voxels=count,
        )
        if after < before - a.MI_TOLERANCE:
            return "rejected_mi_guard"
    return None


def register_pair(
    fixed_organ, moving_organ, config, fixed_image=None, moving_image=None
):
    sitk = a.backend()
    if any(s.uses_mi for s in config.stages) and (
        fixed_image is None or moving_image is None
    ):
        raise ValueError(
            "MI registration stages require fixed and moving intensity images"
        )
    fixed_qc, moving_qc = [
        a.assess_organ_mask(mask, config) for mask in (fixed_organ, moving_organ)
    ]
    if "invalid" in (fixed_qc.status, moving_qc.status):
        raise ValueError("Invalid organ mask for registration")
    partial = fixed_qc.partial or moving_qc.partial
    if partial and not config.accept_partial_organ_masks:
        raise ValueError("Partial organ masks are disabled")
    context = PairContext(
        fixed_organ, moving_organ, config, fixed_image, moving_image, partial
    )
    identity = sitk.Euler3DTransform()
    before = context.score(identity)
    state = AcceptedState(
        identity, identity.GetInverse(), before, "baseline", identity, before
    )
    details = {
        name: {
            "dice": None,
            "status": "skipped_not_configured",
            "accepted": False,
            "selected": False,
        }
        for name in REGISTRATION_STAGES
    }
    details["baseline"].update(
        dice=before,
        status="accepted",
        accepted=True,
        selected=True,
        stage_order=[s.type for s in config.stages],
    )
    transforms = {"baseline": identity}
    warnings = []
    initializer = "geometry" if partial else "pca"
    details["pca" if partial else "geometry"]["status"] = (
        "skipped_partial_coverage" if partial else "skipped_complete_organ"
    )
    started = time.perf_counter()
    try:
        tx = (
            sitk.CenteredTransformInitializer(
                fixed_organ,
                moving_organ,
                sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
            if partial
            else a.initialize_pca(
                fixed_organ, moving_organ, config.maximum_pca_rotation_degrees
            )
        )
        value = context.score(tx)
        transforms[initializer] = tx
        detail = details[initializer]
        detail.update(
            dice=value,
            input_dice=state.dice,
            dice_improvement=value - state.dice,
            minimum_required_dice_improvement=config.initializer_minimum_dice_improvement,
            fallback_stage=state.stage,
        )
        if not np.isfinite(value):
            raise ValueError("Nonfinite initialization Dice")
        if (
            value - state.dice
            >= config.initializer_minimum_dice_improvement - a.DICE_TOLERANCE
        ):
            inverse = tx.GetInverse()
            details[state.stage]["selected"] = False
            state = AcceptedState(tx, inverse, value, initializer, tx, value)
            detail.update(status="accepted", accepted=True, selected=True)
        else:
            detail["status"] = "rejected_insufficient_improvement"
    except (RuntimeError, ValueError) as exc:
        a._record_stage_failure(
            details, warnings, initializer, exc, state.dice, state.stage
        )
    finally:
        details[initializer]["elapsed_seconds"] = time.perf_counter() - started

    dice_metric = "dice_common_fov" if partial else "dice_full"
    for stage in config.stages:
        detail = details[stage.type]
        detail.update(
            input_stage=state.stage,
            input_dice=state.dice,
            fallback_stage=state.stage,
            resolved_config=asdict(stage),
            selection_metric=(
                "mattes_mutual_information" if stage.uses_mi else dice_metric
            ),
            run_condition_metric=dice_metric,
            dice_guard_metric=dice_metric,
            metric=(
                "mattes_mutual_information"
                if stage.uses_mi
                else "mean_squares_signed_distance"
            ),
            model=stage.model,
            backend_version=sitk.Version_VersionString(),
            threads=sitk.ProcessObject.GetGlobalDefaultNumberOfThreads(),
        )
        if (
            stage.run_if.dice_below is not None
            and state.dice >= stage.run_if.dice_below
        ):
            detail.update(
                status="skipped_acceptable_dice", required_dice=stage.run_if.dice_below
            )
            continue
        started = time.perf_counter()
        try:
            if stage.uses_mi:
                context.compare_mi([state.transform], stage.metric)
            optimization_started = time.perf_counter()
            try:
                result = EXECUTORS[stage.type](context, state, stage)
            finally:
                detail["optimization_elapsed_seconds"] = (
                    time.perf_counter() - optimization_started
                )
            tx = result.transform
            detail.update(result.diagnostics)
            detail.update(
                optimizer_stop=result.optimizer.GetOptimizerStopConditionDescription(),
                optimizer_iteration=int(result.optimizer.GetOptimizerIteration()),
                optimizer_iteration_limit=stage.optimizer.iterations,
            )
            if stage.model == "elastic":
                detail.update(
                    algorithm="BSplineTransform",
                    control_point_spacing_mm=config.elastic_control_point_spacing_mm,
                )
            transforms[stage.type] = tx
            value = context.score(tx)
            if not np.isfinite(value):
                raise ValueError("Nonfinite registration Dice")
            overlap = a.overlap_qc(fixed_organ, moving_organ, tx)
            detail.update(overlap)
            if (
                overlap["common_fov_fraction"]
                < config.minimum_common_field_of_view_fraction
                or overlap["common_foreground_voxels"] < 4
            ):
                detail["status"] = "rejected_insufficient_overlap"
                continue
            rejection = _acceptance(context, state, stage, tx, value, detail)
            if rejection:
                detail["status"] = rejection
                continue
            if stage.model == "elastic":
                domain, _ = context.distances()
                field_started = time.perf_counter()
                deformation, qc = a.elastic_deformation_qc(tx, domain, config)
                detail.update(qc)
                detail["field_qc_elapsed_seconds"] = time.perf_counter() - field_started
                determinant = float(
                    np.linalg.det(
                        np.asarray(tx.GetNthTransform(0).GetMatrix()).reshape(3, 3)
                    )
                )
                detail.update(
                    full_jacobian_min=determinant * qc["jacobian_min"],
                    full_jacobian_max=determinant * qc["jacobian_max"],
                )
                if not qc["deformation_qc_passed"]:
                    detail["status"] = "rejected_deformation_qc"
                    continue
                inverse = a.invert_mask_elastic(
                    tx,
                    domain,
                    detail,
                    deformation,
                    sitk.Cast(domain <= 0, sitk.sitkUInt8),
                )
            else:
                a._validate_linear_transform(tx)
                inverse = tx.GetInverse()
            # No accepted state (including cached inverse or guard) is mutated
            # until all candidate evaluations and inversion have succeeded.
            details[state.stage]["selected"] = False
            state = AcceptedState(
                tx,
                inverse,
                value,
                stage.type,
                tx if value > state.peak_dice else state.peak_transform,
                max(value, state.peak_dice),
                (tx, stage.metric) if stage.uses_mi else state.mi_guard,
            )
            detail.update(status="accepted", accepted=True, selected=True)
        except (RuntimeError, ValueError) as exc:
            if isinstance(exc, StageExecutionError):
                detail.update(exc.diagnostics)
            a._record_stage_failure(
                details, warnings, stage.type, exc, state.dice, state.stage
            )
        finally:
            detail["elapsed_seconds"] = time.perf_counter() - started

    for name, tx in transforms.items():
        details[name].update(a.overlap_qc(fixed_organ, moving_organ, tx))
        details[name].setdefault("selection_metric", dice_metric)
    overlap = a.overlap_qc(fixed_organ, moving_organ, state.transform)
    return a.TransformResult(
        state.transform,
        "identity" if state.stage == "baseline" else state.stage,
        before,
        state.dice,
        warnings,
        details,
        transforms,
        state.inverse,
        a.registration_confidence(overlap, partial, config),
        overlap,
        fixed_qc.volume_mm3 / moving_qc.volume_mm3,
    )
