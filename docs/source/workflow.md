# Workflow

IMPERANDI passes a cohort table from one stage to the next. Image data remains
on disk; CSV path columns preserve the link between each source volume and its
derived artifacts.

```text
DICOM roots / archives
        |
        v
 parse ------> dicom_index.csv
        |
        v
 clean ------> dicom_index_clean.csv
        |
        v
 convert ----> nifti_index.csv with paths + NIfTI images
        |
        v
 segment ----> nifti_index.csv with mask_* paths + NIfTI masks
        |
        v
 phase ------> canonical phase + provenance
        |
        v
 radiomics --> feature columns
```

## Parse

`parse` discovers files below one or more roots, reads the default selected
DICOM tags plus any supplied with `--tags`, and resolves `patient_key`,
`study_id`, and `series_id`.

ID source modes are:

- `auto` (default): prefer configured DICOM tags and fall back to path parts.
- `tags`: derive IDs from DICOM tags.
- `path`: derive IDs from the expected patient/study/series directory layout.

ZIP, TAR, TAR.GZ, and TGZ inputs can be read without manually unpacking the
whole dataset. Archive recursion is bounded by `--archive_max_depth`; temporary
materialization is removed unless `--keep_archive_cache` is set.

## Clean

`clean` groups instances into volumes, standardizes dates and times, orders
exams/acquisitions, and rejects unsuitable data. The implemented filters cover
non-CT data, localizers and secondary images, non-axial geometry, noise/body
region patterns, implausible volume length, pixel spacing, and slice thickness.
Missing geometry is generally retained for later review rather than silently
treated as a failure.

The built-in accepted reconstructed length is 30–1700 mm. Change the
volume-scope filter in `cleaning.steps` when a protocol calls for different
bounds.

## Convert

`convert` materializes each curated series and delegates DICOM-to-NIfTI
conversion to `dicom2nifti`. It works in parallel and records per-volume
failures separately. Its input needs a `dicom_path` representation produced by
ingest; output rows receive `nifti_path`.

## Segment, phase, and radiomics

`segment` dispatches each row to manifest-defined CT or MR TotalSegmentator
tasks using `Modality`, followed by modality-specific mask post-processing.
`phase` resolves the manifest's ordered ontology, metadata-rule, and
TotalSegmentator strategies, invoking prediction only where earlier strategies
did not resolve a phase.
`radiomics` computes PyRadiomics features for every `mask_*` column, including
an organ-minus-tumor strategy when paired organ and tumor masks are present.

Phase can run immediately after conversion or after segmentation. Segmentation
must precede radiomics, and a phase-filtered radiomics manifest must receive a
table containing both `mask_*` columns and the canonical `phase` column.

## Registration

Install `imperandi[registration]` to enable `imperandi register`. Run it on a
segmented cohort with curated `phase` and, for MR, `mri_sequence` columns:

```bash
imperandi register --csv_path nifti_index_phased.csv \
  --csv_path_out nifti_index_registered.csv --output_dir registration \
  --manifest generic --organ_consensus_method anchor --tumor_consensus_method anchor \
  --num_workers 4
```

Registration accepts only the current configuration names; unknown settings are
rejected. Python callers use `RegistrationConfig` and `register_cohort` from
`imperandi.process.registration`. Custom `pair_registration` callbacks receive
`(fixed_organ, moving_organ, config, fixed_image, moving_image)` and return
`alignment.TransformResult`.

The manifest's ordered `registration.grouping_columns` list defines group identity.
The default is `[patient_key, study_id, Modality]`; a dataset can instead use,
for example, `[patient_key, exam_stage]`. Saved artifacts use these grouping
values as folders in the configured order. For `[patient_key, exam_stage]`, a
group with values `patient001` and `BASELINE` is saved under
`<output_dir>/patient001/BASELINE/`, with a `registration.jsonl` group log and
a subfolder per scan. Before recomputing an existing group, registration warns
that its outputs will be overwritten and clears that group's folder and all
subfolders, including artifacts from previous output spaces or removed scans.
Groups reused on resume retain their files.
Path separators and unsafe characters in grouping values are percent-encoded;
ordinary values retain their spelling, with `MRI` normalized to `MR` for
`Modality`, matching group identity. `Modality` has no special grouping
rule: include it for separate modality groups or omit it to register modalities
together. It applies an ordered liver-first cascade: baseline, PCA for complete
organs (geometry initialization instead for partial organs), signed-distance
mask rigid, signed-distance mask affine, optional boundary-band
mutual-information (MI) affine, and optional mask-driven B-spline refinement. The
low-level rigid MI refinement remains
available through `mi_refine(..., affine=False)`, but is not part of the active
cascade.
Default masks are `mask_liver` and
`mask_liver_tumor`; masks must match their native image geometry.

Organ consensus choices are `anchor` and `majority`. Organ masks are transferred
as unclamped signed-distance fields in physical millimetres with linear
interpolation, then thresholded once on the destination grid. This transfer field
is separate from the cropped, 15 mm-clamped distance map used by the optimizer.
Coverage and tumor labels continue to use nearest-neighbour interpolation.
`anchor` transfers the selected reference organ distance field. `majority` takes
the upper median of the aligned distance values from contributors that observe a
voxel, then thresholds the fused field once. The upper median preserves the
existing strict-majority treatment of even ties while giving a 50% breakdown
point: when at least three masks observe a voxel, one arbitrary outlier cannot
move the fused boundary beyond the majority surfaces. Voxels outside every
registered candidate's observed FOV retain that scan's source organ annotation.

Tumor consensus choices are `anchor`, `majority`, `intersection`, `union`, and `staple`.
Valid complete organs take reference priority over partial organs. Boundary contact
or a broad straight cut face marks a segmentation as partial, even when the cut
lies inside the image. The straight-face heuristic checks both ends of each image
axis on the largest connected component: the end section must occupy at least
25% of the largest section and 90% of the adjacent inward section (at least four
voxels). Small rounded tips are excluded; cuts oblique to the image axes are not
detected by this heuristic. Within each configured group and completeness class,
`reference_selection_priority` applies criteria in list order: later criteria break ties
in earlier criteria. The default list first prefers CT over MR when both occur
in one group, then orders phase, MR sequence, pixel spacing, slice thickness,
and segmented organ volume. Criteria whose columns are absent simply tie.
Complete ties use deterministic scan IDs. Dataset manifests can change this order.
Anchor uses the reference tumor mask when available, otherwise the next valid
mask in the same completeness and criterion order. Missing, nonexistent, and empty
masks are omitted.
Every eligible nonempty mask contributes to non-anchor fusion, so
select independent sequences/phases upstream to avoid duplicate reconstruction
votes. Whole-FOV annotations are assumed. For majority, union, and intersection,
each voxel uses only contributors observing that location: probability is tumor
votes divided by observed contributors. Majority uses
`> consensus_probability_threshold` (ties at 0.5 are negative); union needs one
positive vote; intersection needs every observed vote positive. No observations
means unknown, never a negative vote.
STAPLE cannot model missing observations with this backend: it is restricted to
the common FOV of all contributors and uses
`>= consensus_probability_threshold`. No common support
produces an explicit consensus error. Coverage, observation counts, and
probability images remain in memory. Fusion is limited to the reference grid;
if all scans are partial, anatomy outside that grid cannot be reconstructed.

The stage loads `generic` by default. Use `--manifest generic` for a built-in
manifest or `--manifest dataset_configs/manifests/operandi.yaml` for a file.
CLI `--organ_consensus_method`, `--tumor_consensus_method`, and `--output_space`
override manifest values. Configure the ordered optimization chain under
`registration.stages`. The default is mask rigid followed by mask affine;
an explicit list replaces it, and an empty list runs initialization only.

```yaml
registration:
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
      run_if: {dice_below: 0.90}
    - type: mi_elastic
      metric: {roi: organ}
      optimizer: {iterations: 25}
```

Include only the steps needed. Rigid must precede affine, and all linear stages
must precede elastic stages. Mask and MI steps may appear in either order within
an affine or elastic block. Stage types must be unique. Omitted stages record
`skipped_not_configured`. Removed enable/disable flags, global early stopping,
and global stage-acceptance maps are rejected; there is no compatibility mode.

Each stage starts from the last accepted transform. Mask steps require Dice
improvement (`acceptance.minimum_dice_improvement`: 0.002 for linear, 0.005 for
elastic). MI steps require deterministic intensity MI improvement, with at most
`acceptance.maximum_dice_decrease` (0.002) below the best accepted anatomical Dice.
Zero `minimum_mi_improvement` still requires improvement above numerical tolerance.
Later mask steps must preserve the latest accepted MI stage's intensity score.
Partial pairs compare candidates on shared physical coverage; QC also retains
full-organ and common-FOV Dice. Tumor masks never drive registration.

`run_if: always` forces an attempted refinement regardless of Dice. Alternatively,
`run_if: {dice_below: 0.90}` skips just that step at or above its threshold.
Mask linear steps default to 0.95; mask elastic defaults to 0.90. MI steps default
to always, so high organ Dice does not suppress requested intensity alignment.
Initialization requires `initializer_minimum_dice_improvement` (default 0.001).

Each step can override `optimizer.iterations` and `pyramid` with
`shrink_factors` / `smoothing_sigmas_mm`. Shared linear defaults are 100 iterations
per level, elastic defaults are 25 (maximum 100). MI `metric.roi` is `organ_band`
by default, or `organ` for internal features. Sampling strategies are `none`,
`regular`, or `random`, with a positive fixed seed for sampled strategies.
MI stages require actual intensity images and finite, nonconstant ROI intensities;
mask-only chains do not evaluate MI. Failed candidates retain the last accepted
forward transform and inverse. Logs record accepted versus finally selected
stages, resolved settings, evaluation support, metric deltas, and rejection reasons.

Optimizer resolution is configured independently with
`linear_shrink_factors` / `linear_smoothing_sigmas_mm` and
`elastic_shrink_factors` / `elastic_smoothing_sigmas_mm`. Both default to shrink
factors `[4, 2, 1]` and smoothing sigmas `[2.0, 1.0, 0.0]` mm. Linear settings
apply to mask-rigid, mask-affine, and MI linear refinement; elastic settings
apply to mask and MI B-spline refinement. The iteration limits above apply per level.
See [Registration pyramid levels](manifests.md#registration-pyramid-levels) for
validation rules, single-level runs, and a coarser schedule to benchmark.

`reference_selection_priority` is one ordered criterion list applied within
every group, with exactly one column per item. Include `Modality` as an ordinary criterion
when cross-modality groups need a preferred reference. For example:

```yaml
registration:
  grouping_columns: [patient_key, exam_stage]
  reference_selection_priority:
    - Modality: [MR, CT]
    - mri_sequence: [T1, T2]
    - phase: [PORTAL_VENOUS]
    - PixelSpacingXY: min
    - SliceThickness: min
    - registration_organ_volume_mm3: max
```

A categorical list orders its values from most to least preferred; matching
ignores case and surrounding whitespace. Unlisted and missing values tie after
the listed categories. `min` and `max` compare finite numeric values, including
numbers stored as CSV strings. Missing columns, malformed numbers, booleans, and
nonfinite numbers rank last for either numeric direction. Spacing and thickness
criteria use the cohort columns. `registration_organ_volume_mm3` is recomputed
from each native organ mask and voxel spacing during QC; previous-run values are
discarded. Dry runs use available metadata without loading images, so their
provisional ordering cannot account for organ completeness or volume.

In the example, T1 outranks T2 even if the T2 scan has smaller pixels. Move
`registration_organ_volume_mm3: max` to the first criterion to favor the largest
organ before sequence, phase, or resolution, within the same completeness class.
Its position controls its influence; there is no separate volume cutoff.
The supplied list replaces the default globally. An empty list uses only
completeness and scan ID. Combine repeated scalar preferences into one list,
for example `- mri_sequence: [T1, T2]`,
and split compound selectors into separate criteria in the desired order.
Scalar categorical selectors such as `- mri_sequence: T1` are rejected.

Series with a missing or empty organ mask are excluded from their registration
group and recorded as `skipped`; they do not produce error-table rows. Unreadable
or otherwise invalid masks remain failures.

Anatomical QC is separate from geometry/format validation. It records physical
volume (mm³), foreground bounding box (index start/size in x/y/z), boundary
contact (low/high per image axis), and the largest face-connected component's
fraction. Foreground within `partial_mask_boundary_margin_mm` (default 1 mm) of
an FOV edge is considered potentially partial. This is a heuristic: no edge contact does
not prove completeness, and anatomical defects away from the edge may escape it.
Fewer than four voxels, an all-foreground FOV, or largest-component fraction
below `minimum_largest_component_fraction` (0.8) yield `invalid_organ_mask`.

`accept_partial_organ_masks: true` permits partial masks; false excludes them.
Every pair evaluates baseline Dice first. Complete masks then try PCA; partial
pairs use geometry translation instead of PCA. Geometry initialization is not
run for complete-organ pairs. PCA candidates whose principal rotation exceeds
`maximum_pca_rotation_degrees` (default 45 degrees) are discarded, while centered
translation remains available as a safe fallback. The configured stages then
run in list order, subject only to each stage's own run condition.

Both elastic objectives refine one B-spline on a fixed reference domain and mesh.
A mask-elastic step followed by MI-elastic refines the accepted deformation with
an intensity objective; it does not append another warp. The accepted linear
mapping stays fixed. Final output data are resampled from original sources.

Mask elastic optimizes signed-distance maps inside a shell extending
`organ_boundary_band_half_width_mm` (default 15 mm) on either side of the organ
boundary. MI elastic uses the configured intensity ROI. Shared control-point
spacing defaults to 90 mm. Deformation QC applies to the complete B-spline:
displacement p95 at most 8 mm, maximum displacement at most 12 mm, and residual
Jacobians within 0.5 to 2.0, with no nonpositive values. Full-transform Jacobian
extrema are reported separately. Both objectives require a numerically validated
inverse. Failure at any point retains the previously accepted transform and inverse.
These sampled checks and thresholds need validation for the dataset and organ.

MI optimization uses seeded sampling; acceptance recomputes full-resolution MI
without sampling on identical input/candidate support. Sparse or constant-intensity
ROIs fail explicitly. A single worker thread removes thread-order variability
for reproducibility checks.
Both full/reference-grid
Dice and common-FOV Dice are retained. Partial stage selection uses common-FOV
Dice; complete pairs retain full Dice. The fixed/moving
volume ratio is diagnostic, never sufficient evidence of successful alignment.

Overlap below `minimum_accepted_organ_dice` (0.5), common-FOV fraction
below `minimum_common_field_of_view_fraction` (0.05 of reference voxels), or
fewer than four summed foreground voxels in common support yields
`low_confidence`. Reliable pairs
are `ok` or `ok_partial_coverage`; errors/rejected pairs are `failed`.
References retain the legacy `reference` status and expose confidence separately.
Low-confidence, invalid, and failed registrations do not contribute to or receive
consensus. Accepted registrations with organ Dice below
`minimum_consensus_dice` (0.8) still receive the registered organ and tumor masks,
but their organ and tumor annotations do not contribute to either consensus.
Partial pairs use common-FOV Dice for both thresholds; complete pairs use full
Dice. `minimum_consensus_dice` must be at least
`minimum_accepted_organ_dice`. These thresholds are conservative starting points
and need validation for the dataset and organ; none establish clinical
segmentation quality.

Registration writes new NIfTI files and never modifies the source images or masks.
By default, `registration.output_space: moving` (or `--output_space moving`) keeps
`nifti_path` on the original scan and replaces `mask_liver` with the configured
organ consensus transferred to that scan's native grid
(`reg_organ_native_path`). Near the observed-coverage
boundary, consensus and source distance fields are blended over
`organ_coverage_blend_width_mm` (default 3 mm) and thresholded once. Consensus is
never inferred outside observed coverage; those voxels retain the source mask.
Successful tumor consensus replaces
`mask_liver_tumor` with the common tumor mask on the same native grid
(`reg_tumor_native_path`). Original paths are preserved as `source_mask_liver`
and `source_mask_liver_tumor` (in general, `source_<configured_mask_column>`).
A rerun uses these source columns as its input masks; it never feeds previous
consensus back into fusion. Failed stages retain their original canonical paths,
so inspect `registration_status` and `consensus_status` before downstream analysis.
Existing radiomics now consumes the native-space registered masks automatically.

Set `registration.output_space: reference` (or `--output_space reference`) to
publish every successful scan on its group's selected common reference grid.
In this mode `nifti_path`, `mask_liver`, and `mask_liver_tumor` point to
`reg_nifti_path`, `reg_organ_path`, and `reg_tumor_common_path`, respectively,
and therefore always have matching geometry. The original image is retained in
`source_nifti_path`; the original mask columns continue to use the `source_`
prefix described above. Reference-space mode writes one registered image and
the selected organ/tumor result per successful row, while moving-space mode
does not write registered images.

To keep each source organ segmentation, pass `--preserve_source_organ_mask` or set
`registration.preserve_source_organ_mask: true` in the manifest. Registration still
aligns organs and maps tumor consensus. With moving-space output, the canonical
organ path remains the source path and no `reg_organ_native_path` artifact is
written. With reference-space output, the scan's own organ segmentation is
resampled to the common grid instead of being replaced with the organ consensus.
Tumor consensus is constrained to that organ when
`clip_tumor_consensus_to_organ` is enabled. Use `--replace_source_organ_mask` to
override the manifest and restore organ transfer.

Only artifacts for the selected output space are persisted. Transforms, coverage,
and probability images remain in-memory intermediates.

`register_qc.csv` contains one row per scan, including failures, with organ
`dice_baseline`, `dice_pca`, `dice_geometry`, `dice_mask_rigid`,
`dice_mask_affine`, `dice_mi_affine`, `dice_mask_elastic`, `dice_mi_elastic`, and
`dice_selected`. It also records the effective `organ_consensus` and
`tumor_consensus` settings.
Additional `registration_*` fields include organ completeness/QC, volume ratio,
`dice_full`, `dice_common_fov`, common-FOV fraction, confidence, consensus
contributor/exclusion counts and reasons, and support policy. The same fields
appear under `anatomical_qc` in structured scan logs. Stage details include both
Dice metrics and the selection metric; skipped stages retain the Dice and reason
that caused an early stop or conditional skip.
Final-organ QC reports face-connected component count and component sizes in
mm3, fragment count and sizes, final volume and source-relative volume change,
and residual coverage-seam voxels. No component is removed or repaired
automatically; any future repair must be enabled independently of these checks.
When both the reference and moving tumor masks are available, corresponding
`tumor_dice_*` fields provide diagnostic overlap without influencing transform
selection.
Candidate scores are retained even when a simpler stage wins; unexecuted or
failed optimizations have blank Dice and explicit stage status. Reference scans
have baseline/selected Dice 1 and optimization stages marked `not_run`.
The table links configured group/scan/reference IDs, source and output mask paths,
selected stage, optimizer diagnostics, warnings, timing, and errors.
Use `--qc_csv_path` to choose its location. QC is refreshed at checkpoint
boundaries and reconstructed on resume if the final table is missing.

Each group has one structured `registration.jsonl` log, linked by
`registration_log_path`. Output rows link the single task-level QC table through
`registration_qc_path`. Console and JSONL logs record the configured grouping
columns and their values once per group. Later events use the
series position and group total (for example `series=1/6`) plus phase/sequence,
references, and stage scores/statuses. JSONL stores one context event per group
and one result event per series, with stage results nested under that series.
Logs omit file paths and source series
IDs; those values remain available in the QC and error tables for audit and resume.
Global errors remain in `register_errors.csv`; timeouts
and other worker failures also appear in the global QC table.

Organ distance maps are cropped around their foreground with configurable
`distance_map_crop_padding_mm` and clamped to
`organ_boundary_band_half_width_mm`. By default, native tumor
consensus masks are constrained to the registered organ; manifests can disable
this with `clip_tumor_consensus_to_organ: false`.

The command checkpoints complete configured groups and resumes by default.
It tracks changes to the input CSV, referenced images/masks, resolved manifest,
and algorithm settings. Each reused group must also have unchanged output artifacts.
Deleting or modifying a group's outputs reruns that group; changed input data or
settings invalidate the run. `--strict_resume` verifies referenced image, mask,
and output contents rather than relying only on file metadata. Failed groups remain recorded on resume;
use `--retry_failed` to retry them, or `--force`/`--no_resume` to recompute all groups.
Fresh work replaces the recomputed groups' artifact directories after an
overwrite warning; native inputs remain unchanged. Registration refuses to clear
an output group folder containing source inputs or reached through symlinks.

`--num_workers` bounds concurrent groups (default 1), `--threads_per_worker`
sets SimpleITK threads (default 1), and `--timeout_sec` imposes a hard per-group
limit (default 900 seconds including process startup). A timed-out group is
recorded as failed. `--start_method` defaults to `spawn`; workers exit after
each group. One worker with timeout 0 runs in-process. `--dry-run` validates
configuration and cohort identities without loading images or writing files.
Checkpoint row/time thresholds are checked between completed groups and while
waiting for subprocesses; an interrupted group is recomputed on restart.

The library API `register_cohort` remains a fresh-work image-processing interface
with replaceable registration and fusion callables. The command's runner wraps
it with scheduling and checkpoints. Registration does not perform deformable
alignment. Cross-modality or longitudinal grouping is controlled explicitly by
`grouping_columns` and should only be enabled for anatomically comparable volumes.

## Checkpoints and resume

`parse`, `convert`, `segment`, `phase`, `register`, and `radiomics` checkpoint long runs.
Resume is enabled by default when the saved command state and input fingerprint
match. Common controls are:

- `--checkpoint_every_rows N`: flush after N processed rows.
- `--checkpoint_every_sec T`: flush after T seconds.
- `--no_resume`: ignore matching checkpoint state and start a fresh run.
- `--strict_resume`: hash input contents instead of relying on the lightweight
  fingerprint; this is safer but slower on large inputs.

Changing material arguments or task-relevant inputs invalidates an incompatible
checkpoint. For cohort CSVs that are enriched in place, resume fingerprints are
semantic rather than based on the CSV file timestamp: each stage fingerprints only
the columns and referenced artifacts it consumes. Columns written by unrelated
downstream stages therefore do not invalidate an earlier completed stage.

Completed rows also retain row-level fingerprints. If only some relevant rows change,
a compatible completed run reuses unchanged rows and recomputes only the affected
rows. With `--strict_resume`, referenced image and mask files are content-hashed;
the default mode tracks their path and existence without coupling resume to file
modification timestamps.

Do not manually edit checkpoint/state files while a command is running.

The `phase` command fingerprints the columns used by its configured strategies,
including rule results and phase applicability. After a completed run, it can
reuse unchanged rows and reevaluate rows whose relevant inputs changed. Adding
unrelated CSV columns does not invalidate phase inference. Image identity uses
the NIfTI path and existence by default; use `--strict_resume` to detect changed
image contents at the same path. Invalidated predictions are cleared before
reevaluation so that a failed prediction cannot reuse a stale phase.

Curation exports retain the original rule evidence and use the manifest's text
column precedence for selection. `phase_text_column` and `phase_text_value`
identify the selected metadata evidence; unresolved `phase_reason` includes the
rule's unresolved explanation. MRI dynamic reasons identify the order template
and its canonical `time`, `acquisition_order`, or series-local volume rank.
Raw DICOM clocks and bolus metadata do not calibrate phase delays. Wide QC cells
include volume identifiers and phase provenance.
Selection scores rank candidates; they are not calibrated phase confidence.
After contextual MRI phase inference, eligible diagnostic T1 acquisitions without
acquisition-level perfusion or postcontrast evidence default to NATIVE, including
single acquisitions. These assumptions record confidence `fallback` and source
`native_no_perfusion_evidence`; they do not establish dynamic phase context.
Within the MRI delayed slot, candidates timed at 5-10 minutes (inclusive) are
preferred before the usual score and coverage ranking. Timing comes from an
unambiguous duration in the phase-evidence text, otherwise from `time_delta`.
If no candidate is in that interval, the usual ranking applies. This preference
does not change phase classification or its duration windows.

## Example Slurm batch script

For scheduled runs, a single Slurm job can execute the full pipeline with
explicit paths between stages:

```bash
#!/bin/bash
#SBATCH --job-name=imperandi-pipeline
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# Activate the environment where IMPERANDI is installed.
source /path/to/venv/bin/activate

PROJECT_ROOT=/path/to/project
DICOM_ROOT=/path/to/dicom
TABLE_DIR="$PROJECT_ROOT/tables"
NIFTI_DIR="$PROJECT_ROOT/nifti"
MANIFEST="$PROJECT_ROOT/site-a.yaml"
WORKERS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p "$TABLE_DIR" "$NIFTI_DIR"

imperandi ingest \
  --root_path "$DICOM_ROOT" \
  --output_dir "$TABLE_DIR" \
  --manifest "$MANIFEST"

imperandi convert \
  --csv_path "$TABLE_DIR/dicom_index_clean.csv" \
  --output_dir "$NIFTI_DIR" \
  --csv_path_out "$TABLE_DIR/nifti_index.csv" \
  --num_workers "$WORKERS"

imperandi segment \
  --csv_path "$TABLE_DIR/nifti_index.csv" \
  --csv_path_out "$TABLE_DIR/nifti_index_segmented.csv" \
  --manifest "$MANIFEST" \
  --num_workers "$WORKERS"

imperandi phase \
  --csv_path "$TABLE_DIR/nifti_index_segmented.csv" \
  --csv_path_out "$TABLE_DIR/nifti_index_phased.csv" \
  --manifest "$MANIFEST"

imperandi radiomics \
  --csv_path "$TABLE_DIR/nifti_index_phased.csv" \
  --csv_path_out "$TABLE_DIR/nifti_index_radiomics.csv" \
  --manifest "$MANIFEST"
```

Adjust Slurm resources for your cohort size and manifest complexity. On
re-submission, matching checkpoints let long stages continue instead of
starting from scratch unless you pass `--no_resume`.

## Operational recommendations

- Keep raw DICOM roots read-only.
- Store each stage's CSV under versioned or run-specific paths.
- Start with a small cohort and `--num_workers 1` when validating a manifest.
- Preserve error CSVs and logs with the corresponding output table.
- Use explicit output paths in automation instead of relying on defaults.



### MRI combined dynamic labels

MRI resolves named phases and combined dynamic profiles before numeric durations.
For example, `axial t1 caipi fs art port 90s` uses the ART-PORT profile rather than
assigning every volume portal from `90s`. Durations supply phase evidence only
when no phase/profile text matches, except that an unambiguous hepatobiliary
duration overrides generic delayed wording: `tardif 2h` is HEPATOBILIARY,
while `tardif 5min` remains DELAYED.

ART-PORT uses ARTERIAL for the first acquisition and
PORTAL_VENOUS for the last. ART/PORT/LATE uses ARTERIAL for the first and DELAYED
for the last; its middle acquisition is PORTAL_VENOUS only for a three-acquisition
profile. Other intermediate acquisitions remain unresolved from profile order.
An isolated combined label remains unresolved.

Across series, endpoint assignment requires a complete, distinct acquisition-order
or clock field. Studies, sequence families, planes and Dixon components remain
separate. Acquisition clock alone supplies order, not a post-injection delay.
