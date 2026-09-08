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
 convert ----> nifti_index.csv + NIfTI images
        |
        v
 segment ----> mask_* paths
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
  --manifest generic --method anchor --num_workers 4
```

The stage groups by `patient_key`, `study_id`, and normalized `Modality`
(CT or MR/MRI). Use `--visit_column visit_id` when your dataset supplies a
different visit identifier. It applies physical-space PCA initialization,
rigid organ alignment, and optional `--affine` refinement. Default masks are
`mask_liver` and `mask_liver_tumor`; masks must match their native image geometry.

Consensus methods are `anchor`, `majority`, `intersection`, `union`, and `staple`.
CT references prefer portal venous, arterial, delayed, then native phases;
MR prefers T1, T2, then DWI. Unlisted scans rank last; ties use deterministic
scan ordering.
Anchor uses the reference tumor mask when available, otherwise the next valid
mask in priority order. Missing masks are omitted; readable empty masks vote
negative. Every available accepted scan contributes to non-anchor fusion, so
select independent sequences/phases upstream to avoid duplicate reconstruction
votes. Whole-volume annotations are assumed. Fusion is limited to common
observed coverage, which is saved separately; outside-coverage zeros are unknown.
Majority saves vote fractions and uses `> 0.5` (ties negative); STAPLE saves
estimated probabilities and uses `>= 0.5`.

The stage loads `generic` by default. Use `--manifest generic` for a built-in
manifest or `--manifest dataset_configs/manifests/operandi.yaml` for a file.
CLI `--method`, `--affine`/`--no_affine`, and `--visit_column` override manifest
values. An optional manifest `registration` mapping accepts `visit_column`,
`organ_column`, `tumor_column`, `method`, `affine`, `affine_min_dice`,
`iterations`, `min_dice`, `threshold`, and `reference_priority`. Affine
refinement runs only when enabled and the best PCA/rigid organ Dice is at least
`affine_min_dice` (default 0.9). Otherwise QC records `skipped_low_dice` and
retains the best preceding transform. `reference_priority` maps CT/MR to ordered lists
of column/value selectors, such as `CT: [{phase: PORTAL_VENOUS}]` or
`MR: [{mri_sequence: T1}]`. Defaults are 100 iterations, minimum organ Dice 0.1,
and threshold 0.5. These initial QC settings require dataset validation.

`nifti_path` always points to the original scan. Registration writes new NIfTI
files and never modifies the original masks. In the output CSV, successful
organ registrations replace `mask_liver` with the reference organ mask transferred
to that scan's native grid (`reg_organ_native_path`). Successful consensus replaces
`mask_liver_tumor` with the common tumor mask on the same native grid
(`reg_tumor_native_path`). Original paths are preserved as `source_mask_liver`
and `source_mask_liver_tumor` (in general, `source_<configured_mask_column>`).
A rerun uses these source columns as its input masks; it never feeds previous
consensus back into fusion. Failed stages retain their original canonical paths,
so inspect `registration_status` and `consensus_status` before downstream analysis.
Existing radiomics now consumes the native-space registered masks automatically.

Derived `reg_*` columns also retain reference-space images/organs, common tumor
masks, coverage, probabilities, and both transform directions.
`reg_reference_to_scan_path` maps reference points to native scan points for
resampling onto the reference grid; its inverse transfers reference masks back.

`registration_qc.csv` contains one row per scan, including failures, with organ
`dice_baseline`, `dice_pca`, `dice_rigid`, `dice_affine`, and `dice_selected`.
Candidate scores are retained even when a simpler stage wins; unexecuted or
failed optimizations have blank Dice and explicit stage status. Reference scans
have baseline/selected Dice 1 and optimization stages marked `not_run`.
The table links patient/visit/scan/reference IDs, source and output mask paths,
transforms, selected stage, optimizer diagnostics, warnings, timing, and errors.
Use `--qc_csv_path` to choose its location. QC is refreshed at checkpoint
boundaries and reconstructed on resume if the final table is missing.

Each group also has `qc.csv`, `group.json`, and a structured `registration.jsonl`
log, linked by `registration_qc_path`, `registration_report_path`, and
`registration_log_path`. Console and JSONL logs record patient ID, date, visit
order, configured visit value, and modality once per group. Later events use the
series position and group total (for example `series=1/6`) plus phase/sequence,
references, and stage scores/statuses. JSONL stores one context event per group
and one result event per series, with stage results nested under that series.
Logs omit file paths and source series
IDs; those values remain available in tables and reports for audit and resume.
Global errors remain in `registration_errors.csv`; timeouts
and other worker failures also appear in the global QC table.

The command checkpoints complete visit/modality groups and resumes by default.
It tracks changes to the input CSV, referenced images/masks, resolved manifest,
and algorithm settings. Each reused group must also have unchanged output artifacts.
Deleting or modifying a group's outputs reruns that group; changed input data or
settings invalidate the run. `--strict_resume` verifies referenced image, mask,
and output contents rather than relying only on file metadata. Failed groups remain recorded on resume;
use `--retry_failed` to retry them, or `--force`/`--no_resume` to recompute all groups.
Fresh work gets separate artifact directories; native inputs remain unchanged.

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
it with scheduling and checkpoints. Registration does not perform deformable,
cross-modality, or cross-visit alignment.

## Checkpoints and resume

`parse`, `convert`, `segment`, `phase`, `register`, and `radiomics` checkpoint long runs.
Resume is enabled by default when the saved command state and input fingerprint
match. Common controls are:

- `--checkpoint_every_rows N`: flush after N processed rows.
- `--checkpoint_every_sec T`: flush after T seconds.
- `--no_resume`: ignore matching checkpoint state and start a fresh run.
- `--strict_resume`: hash input contents instead of relying on the lightweight
  fingerprint; this is safer but slower on large inputs.

Changing material arguments or inputs invalidates an incompatible checkpoint.
Do not manually edit checkpoint/state files while a command is running.

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
