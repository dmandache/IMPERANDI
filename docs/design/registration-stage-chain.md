# Configurable registration stage chains

Status: implemented with an explicit stage list and no backward compatibility.
See [the workflow guide](../source/workflow.md) for the configuration contract.

Use an ordered `registration.stages` list. Each stage selects an optimization
signal (organ masks or image mutual information) and a transform model (rigid,
affine, or elastic B-spline). Each step starts from the last accepted transform;
a failed or rejected candidate leaves that transform and its inverse intact.

## Behavior before this change

The previous implementation in `alignment.py` ran:

```text
baseline -> PCA or geometry -> mask_rigid -> mask_affine -> mi_affine -> mask_elastic
```

- Complete organs use PCA initialization; partial organs use geometry translation.
- Mask stages minimize mean-square differences between signed-distance maps.
  Candidate acceptance uses organ Dice, with common-FOV scoring for partial masks.
- `enable_affine_stage` controls MI affine only. Mask affine is always eligible.
  `MI_AFFINE_STAGE_ENABLED = False` currently blocks MI affine even when requested.
  The workflow documentation does not describe this extra gate.
- `enable_elastic_stage` enables mask B-spline refinement. There is no MI elastic
  implementation. MI computed during mask elastic is diagnostic, not its objective.
- Global Dice early stopping skips every later stage, including MI. The elastic
  Dice trigger is an additional gate.
- Linear stages optimize a copy of the accepted linear transform. Elastic adds
  one B-spline after prewarping the moving distance map with that linear transform.
- Inversion and deformation QC assume a linear transform plus one B-spline.
- MI optimization uses unseeded regular sampling. The low-level pair API substitutes
  masks when intensity images are omitted; explicitly requested MI must not do that.
- Mask elastic currently requires successful MI diagnostic evaluation. Optional
  intensity diagnostics should not make a mask-only registration fail.

## Stage choices and useful chains

| Stage | Optimization input and objective | Acceptance |
| --- | --- | --- |
| `mask_rigid` | Organ signed-distance maps; mean squares; rigid transform | Dice improvement |
| `mask_affine` | Organ signed-distance maps; mean squares; affine transform | Dice improvement and linear plausibility |
| `mi_affine` | Original intensities in an anatomical ROI; Mattes MI; affine transform | MI improvement, Dice guard, linear plausibility |
| `mask_elastic` | Organ signed-distance maps; mean squares; B-spline | Dice improvement, deformation and inverse QC |
| `mi_elastic` | Intensities in an anatomical ROI; Mattes MI; B-spline | MI improvement, Dice guard, deformation and inverse QC |

All chains retain baseline evaluation and automatic initialization before the
listed stages. Organ masks remain required for initialization, ROI definition,
and anatomical QC, even for a chain containing only MI optimization stages.
Tumor masks never influence optimization or acceptance.

Suggested configurations to evaluate on representative datasets:

| Purpose | Ordered stages |
| --- | --- |
| Existing conservative default | `mask_rigid -> mask_affine` |
| Organ-shape refinement | `mask_rigid -> mask_affine -> mask_elastic` |
| Intensity-led affine alignment | `mask_rigid -> mi_affine` |
| Shape then intensity refinement | `mask_rigid -> mask_affine -> mi_affine -> mi_elastic` |
| Shape correction followed by local intensity refinement | `mask_rigid -> mask_affine -> mask_elastic -> mi_elastic` |

Mask objectives constrain organ shape but do not directly align internal image
features. MI uses intensity correspondence, but its success still needs anatomical
checks. Neither signal should be declared universally better. Keep elastic stages
opt-in until dataset benchmarks justify enabling them.

## Configuration contract

Example of the manifest syntax:

```yaml
registration:
  output_space: moving
  stages:
    - type: mask_rigid
    - type: mask_affine
    - type: mi_affine
      run_if: always
      metric:
        roi: organ_band
        histogram_bins: 50
        sampling_strategy: regular
        sampling_fraction: 0.2
        sampling_seed: 42
      acceptance:
        minimum_mi_improvement: 0.0
        maximum_dice_decrease: 0.002
    - type: mask_elastic
      run_if:
        dice_below: 0.90
    - type: mi_elastic
      run_if: always
      metric:
        roi: organ
      optimizer:
        iterations: 25

  # Shared by both elastic objectives; one deformation domain and mesh per pair.
  elastic_control_point_spacing_mm: 90.0
  maximum_elastic_displacement_p95_mm: 8.0
  maximum_elastic_displacement_mm: 12.0
  minimum_elastic_jacobian_determinant: 0.5
  maximum_elastic_jacobian_determinant: 2.0
```

This full example demonstrates all stages; it is not a new default. Omit a stage
to disable it. A supplied list replaces the default chain. An empty list requests
baseline plus initialization only.

Introduce typed `StageConfig`, `MetricConfig`, `OptimizerConfig`, `PyramidConfig`,
`RunCondition`, and `AcceptanceConfig` dataclasses. A registry maps each stage type
to its signal, model, executor, and defaults. Do not expose arbitrary combinations
of SimpleITK metrics and optimizers in the first version.

Per-stage options:

- `run_if`: `always` or `{dice_below: threshold}`. Evaluate against the current
  accepted Dice immediately before the stage. A skip continues to the next stage.
- `pyramid`: paired `shrink_factors` and `smoothing_sigmas_mm`, validated with the
  existing coarse-to-fine rules. Inherit existing linear/elastic schedules.
- `optimizer.iterations`: inherit 100 for linear and 25 for elastic stages;
  retain the existing elastic iteration bound. Initially retain existing optimizer
  algorithms and learning-rate defaults.
- `metric`: MI histogram bins, sampling strategy/fraction/seed, and `roi` selected
  from `organ_band` or `organ`. Inherit the existing 15 mm boundary-band width.
  Default both MI stages to `organ_band`; opt into `organ` for internal features.
- `acceptance`: mask stages use `minimum_dice_improvement`; MI stages use
  `minimum_mi_improvement` and `maximum_dice_decrease`. Elastic QC remains mandatory.

Default run conditions in explicit-list mode are Dice below 0.95 for mask linear
stages, Dice below 0.90 for mask elastic, and `always` for MI stages. An explicit
`always` permits refinement even at high Dice. There is no global Dice short
circuit in this mode: organ overlap alone cannot establish MI convergence.

Validate unknown keys, finite numeric ranges, booleans used as numbers, and
inapplicable options before processing images. Stage types must be unique in v1,
preserving existing stage-keyed reports. Enforce this ordering:

```text
optional mask_rigid -> zero or more affine stages -> zero or more elastic stages
```

Allow either ordering of mask and MI within the affine or elastic block. Reject
rigid after affine and any linear stage after elastic. Identity or the selected
initializer is a valid linear base if affine stages are omitted or rejected.
Repeated stages and changing B-spline meshes mid-chain are deferred.

## Transform chaining

Keep every stored transform in the current pull direction:
`reference physical point -> moving physical point`. Resampling moving data on
the reference grid uses this mapping directly.

For linear stages, copy the current linear transform into the requested model
and optimize its parameters. An MI affine initialized from mask affine returns
the updated complete affine `A_new`; do not compose `A_old` with `A_new`.

Represent the elastic state as `T(x) = A(B(x))`, where `A` is the frozen accepted
linear mapping and `B` is one B-spline in reference coordinates. In SimpleITK,
`CompositeTransform([A, B])` applies `B` first. This ordering follows the
[SimpleITK CompositeTransform contract](https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1CompositeTransform.html).

The first elastic stage initializes `B` to identity. Subsequent elastic stages
clone and refine the accepted `B`, preserving its domain and mesh. Thus mask
elastic followed by MI elastic changes the objective while refining the same
deformation. It does not append another residual and does not reset to identity.

Preserve the existing prewarp strategy initially: prepare moving distance maps
and, when needed, moving intensities in the reference optimization domain using
`A`. Optimize `B` against these caches. Do not prewarp using `A o B` and then
initialize optimization with `B`, which would apply the accepted deformation twice.
Carry a prewarped coverage mask so synthetic background is excluded from metrics.
The domain must cover the configured organ ROI and deformation margin and remain
fixed across elastic stages.

Final acceptance metrics use original source data under the full candidate
transform. Final published images and labels are resampled from original inputs
with the selected transform, avoiding repeated interpolation of output images.
Retain the existing signed-distance organ-transfer path and nearest-neighbor
resampling for categorical labels where currently used.

Keep `T^-1 = B^-1 o A^-1`: invert the linear component exactly and the B-spline
field numerically with the existing expanded support and round-trip checks.
Commit the forward transform and validated inverse together. A later failure
must preserve the previous elastic inverse, not clear it.

## Acceptance, reproducibility, and rollback

Separate the optimizer objective from candidate acceptance:

1. Snapshot the accepted state; build a candidate using independent transform copies.
2. Validate finite parameters, overlap, and linear plausibility as applicable.
3. Evaluate the candidate and its input on comparable data support.
4. Apply the stage's acceptance rule and any retained MI guard.
5. For elastic candidates, validate deformation and inverse consistency.
6. Commit all state atomically, or record a reason and retain the snapshot.

Mask stages must improve Dice by their configured delta (existing defaults:
0.002 linear, 0.005 elastic). MI stages must improve independently evaluated
Mattes MI by at least the configured delta and numerical tolerance; report MI
with higher values meaning better correspondence. Do not compare optimizer
values from different pyramid levels or different objectives.

Maintain the peak Dice among accepted transforms. Every MI candidate must stay
within `maximum_dice_decrease` of that peak, preventing small permitted losses
from accumulating across MI affine and MI elastic.

After an accepted MI stage, retain its evaluation ROI and settings as a guard
for later mask stages. Require no MI regression beyond numerical tolerance on
that support. This makes an explicitly ordered `mi_affine -> mask_elastic` chain
preserve its accepted intensity alignment. Record guard failures separately from
failure to improve Dice. Reset the guard to the latest accepted MI stage.

For partial coverage, retain common-FOV Dice and minimum overlap checks. Evaluate
input and candidate on the same physical support for their comparison: intersect
the input/candidate valid coverage and relevant anatomical ROIs. Reject inadequate
support before comparing metrics; keep full/common-FOV Dice and coverage diagnostics
separate. Peak Dice guards require rescoring the stored accepted anatomical
checkpoint on this support, rather than comparing scores from different domains.
Never treat cropped-away voxels as matching background.

Use seeded optimizer sampling and deterministic full-resolution acceptance
evaluation with the same histogram settings and support for input and candidate.
Preflight MI images and nonconstant ROI intensities; reject insufficient valid
samples with an explicit reason. `REGULAR` sampling also involves randomness,
and masks can substantially reduce valid samples, as documented in the
[SimpleITK registration overview](https://simpleitk.readthedocs.io/en/main/registrationOverview.html).
Record the seed, backend version, and thread count; use one thread for strict
reproducibility tests.

Apply elastic limits to the complete current `B`, including after a second
elastic stage. Preserve optimizer checkpoint rollback, final field QC, and
inverse round-trip validation. Report residual and full-transform Jacobians
separately; the existing 0.5--2.0 limits describe the residual. These sampled
checks are safeguards, not a mathematical proof of global invertibility.

Missing intensity images are an input error for an explicitly configured MI
stage. Mask-only chains must work without intensities. Optional MI diagnostics
on mask steps may be unavailable without rejecting an otherwise valid candidate;
an active MI preservation guard is mandatory, so failure to evaluate it rejects
that candidate. Recoverable optimizer/QC failures continue from accepted state.
Keep final minimum accepted Dice and consensus eligibility as separate policies.

## Implementation

Implementation responsibilities:

1. **Configuration and runner:** add typed stage parsing and a resolved plan in
   `config.py`; extract the cascade loop into `pipeline.py`. Keep optimization
   primitives in `alignment.py`. Introduce `PairContext` for source data/caches,
   `AcceptedState` for transforms/inverse/guard checkpoints, and `StageResult`
   for candidate diagnostics. Executors propose candidates; the runner commits.
2. **Existing stages:** register mask rigid, mask affine, MI affine, and mask
   elastic. Remove the source-code MI gate and the legacy cascade.
3. **Elastic objective switching:** extract shared B-spline setup, callbacks,
   field QC, and inversion; implement MI elastic by changing optimization inputs
   and metric while reusing the accepted B-spline and the same QC path.
4. **Integration:** resolve the plan before launching workers; expose it in
   dry-run/log output. Update `register.py`, `reporting.py`, `cohort.py`, manifests,
   and workflow documentation. Extend stage columns/artifacts with `mi_elastic`.
5. **Resume:** fingerprint the fully resolved ordered plan and defaults in
   `runtime.py`; bump `REGISTRATION_SCHEMA`. Derive retryable enabled stages from
   that plan, replacing `_has_failed_stage` checks of legacy booleans.

If `stages` is absent, use `mask_rigid -> mask_affine`. A supplied list replaces
that default; an empty list runs baseline plus initialization. Removed global
flags, stopping thresholds, and stage-acceptance maps are unknown settings.
There is no compatibility adapter or hidden MI gate. Shared numeric settings
supply defaults, and per-stage overrides win. Use the manifest for new chains;
the former affine/elastic CLI enable switches have been removed.

Record requested/resolved stage order, input accepted stage, objective/model,
resolved parameters, run/skip reason, before/after metrics, support size, optimizer
stop reason, elapsed time, QC/inversion results, rejection/failure reason, and
fallback stage. Distinguish `accepted` from `selected`: several stages may have
been accepted, but only the final accepted state is selected. Reports use the explicit stage names directly.

## Verification and rollout

Extend existing registration tests with behavioral checks:

- Config validation, explicit-list replacement, ordering, rejection of removed settings,
  per-stage defaults/overrides, and resolved-plan checkpoint invalidation.
- A rejected affine or elastic candidate leaves the next stage's input and
  cached inverse unchanged; MI at high Dice still runs when requested.
- Noncommuting rotation/translation plus deformation verifies transform order;
  affine refinement does not double-apply its initialization.
- Mask elastic followed by MI elastic reuses one B-spline and validates total
  residual displacement; it never bypasses limits by splitting deformation.
- Known synthetic intensity structure inside similar organ masks demonstrates
  MI improvement with flat/slightly decreasing Dice. Both MI stages respect the
  peak-Dice guard; later mask stages respect the retained MI guard.
- Partial FOV, anisotropic spacing, nonidentity direction, low overlap, constant
  intensities, and small ROIs exercise metric-support and inversion failures.
- Mask-only calls succeed without MI inputs or successful diagnostic MI scoring.
- Both output spaces, stage artifacts, logs, retry handling, and resume work for
  accepted and rejected MI elastic candidates.

Run focused unit tests first, then existing registration integration tests.
Benchmark proposed chains on held-out CT phase and MR sequence pairs, comparing
organ/surface alignment, independent landmarks where available, MI, deformation
QC, runtime, and rejection rates. Use tumor alignment only as independent QC,
never as a tuning objective. Decide defaults from these results; do not change
the default chain merely because the new configuration supports more stages.
