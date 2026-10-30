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
 segment ----> nifti_index.csv with mask_*_path paths + NIfTI masks
        |
        v
 phase ------> canonical phase + provenance
        |
        v
 register ---> optional alignment and consensus masks
        |
        v
 radiomics --> feature columns
```

## Choose the stages and cohort

Ingest and conversion produce a NIfTI cohort. Stop there if downstream tools
only need images. Segmentation supplies masks for radiomics or registration;
phase prediction is optional when metadata already resolves the needed phases.
Quality-control viewers can inspect any table with image paths.

Cleaning with a `phase_curation` step and the `phase` command also export a
selected cohort. The main CSV retains all surviving rows; `<input_stem>_curated.csv`
contains the best eligible CT volume per phase or MR volume per sequence/phase
slot in each exam. Choose the main CSV to process all candidates, or the selected
CSV to process representative volumes. The examples use the main CSV.

Use the most recent table containing the paths needed by the next stage. A
selection exported during ingest has DICOM paths; a selection exported after
segmentation can also carry NIfTI and mask paths. The wide `qc_*.csv` tables are
for review, not pipeline input. See [Outputs](outputs.md#curation-and-selection-exports)
for filenames and [best-series selection](manifests.md#best-series-per-exam)
for eligibility and ranking rules.

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
unsupported modalities (the built-in manifest accepts CT, MR, and MRI),
localizers and secondary images, non-axial geometry, noise/body
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
`radiomics` computes PyRadiomics features for every `mask_*_path` column, including
an organ-minus-tumor strategy when paired organ and tumor masks are present.

Phase can run immediately after conversion or after segmentation. Segmentation
must precede radiomics, and a phase-filtered radiomics manifest must receive a
table containing both `mask_*_path` columns and the canonical `phase` column.

## Phase evidence and selection

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

See [MRI dynamic acquisitions](manifests.md#mri-dynamic-acquisitions) for
combined labels, timing, and acquisition-order rules.

## Registration

Registration is optional. After segmentation and phase curation, `register`
aligns organs within configured groups and can transfer consensus masks to each
scan's native grid or a common reference grid. Feed its output CSV to radiomics
when features should use those masks. See [Registration](registration.md) for
installation, grouping, output spaces, and quality checks.

## Checkpoints and resume

`parse`, `convert`, `segment`, `phase`, `register`, and `radiomics` checkpoint long runs.
Resume is enabled by default when the saved command state and input fingerprint
match. Common controls are:

- `--checkpoint_every_rows N`: flush after N processed rows.
- `--checkpoint_every_sec T`: flush after T seconds.
- `--no_resume`: ignore matching checkpoint state and start a fresh run.
- `--strict_resume`: hash input contents instead of relying on the lightweight
  fingerprint; this is safer but slower on large inputs.
- `--retry_failed`: retry recorded failures while preserving successful results, even after a completed run. Supported by `convert`, `segment`, `phase`, `radiomics`, and `registration`. By default, resume skips previous failures. Successful retries clear their error records.

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
