# Workflow

IMPERANDI passes a cohort table from one stage to the next. Image data remains
on disk; CSV path columns preserve the link between each source volume and its
derived artifacts.

```text
DICOM roots / archives
        │
        ▼
 parse ─────► dicom_index.csv
        │
        ▼
 clean ─────► dicom_index_clean.csv
        │
        ▼
 convert ───► nifti_index.csv + NIfTI images
        │
        ├────────► phase ─────► totalseg_* metadata
        │
        ▼
 segment ───► mask_* paths
        │
        ▼
 radiomics ─► feature columns
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

The default accepted reconstructed length is 30–1700 mm. Override it with
`--volume-length-min-mm` and `--volume-length-max-mm` when a protocol calls for
different bounds.

## Convert

`convert` materializes each curated series and delegates DICOM-to-NIfTI
conversion to `dicom2nifti`. It works in parallel and records per-volume
failures separately. Its input needs a `dicom_path` representation produced by
ingest; output rows receive `nifti_path`.

## Segment, phase, and radiomics

`segment` runs manifest-defined TotalSegmentator tasks and optional mask
post-processing. `phase` runs TotalSegmentator's CT contrast-phase extractor.
`radiomics` computes PyRadiomics features for every `mask_*` column, including
an organ-minus-tumor strategy when paired organ and tumor masks are present.

Phase can run immediately after conversion. Segmentation must precede
radiomics because radiomics requires one or more mask columns.

## Checkpoints and resume

`parse`, `convert`, `segment`, `phase`, and `radiomics` checkpoint long runs.
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
MANIFEST="$PROJECT_ROOT/site-a.json"
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
  --csv_path_out "$TABLE_DIR/nifti_index_phased.csv"

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

