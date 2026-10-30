# Quickstart

This example keeps raw DICOM data, tabular indexes, and generated NIfTI files
in separate directories:

```text
project/
├── dicom/          # source data; do not modify
├── tables/         # CSV indexes and error reports
└── nifti/          # converted images and masks
```

## 1. Inspect the plan

Most pipeline commands support `--dry-run`. Use it to confirm resolved paths
and settings before processing a cohort:

```bash
imperandi ingest \
  --root_path ./project/dicom \
  --output_dir ./project/tables \
  --manifest generic \
  --dry-run
```

## 2. Ingest DICOM metadata

`ingest` runs `parse` and then `clean`:

```bash
imperandi ingest \
  --root_path ./project/dicom \
  --output_dir ./project/tables \
  --manifest generic
```

The main result is `project/tables/dicom_index_clean.csv`, with one curated row
per reconstructed volume. To inspect uncommon tags on a deterministic sample,
add `--snapshot_tags`.

## 3. Convert volumes to NIfTI

```bash
imperandi convert \
  --csv_path ./project/tables/dicom_index_clean.csv \
  --output_dir ./project/nifti \
  --csv_path_out ./project/tables/nifti_index.csv
```

Successful rows gain a `nifti_path`. Conversion failures are written to
`conv_errors.csv` without aborting all other rows.

## 4. Segment images

Install the `segment` extra first, then run the tasks from the selected
manifest:

```bash
imperandi segment \
  --csv_path ./project/tables/nifti_index.csv \
  --csv_path_out ./project/tables/nifti_index_segmented.csv \
  --manifest generic \
  --crop --crop-margin-mm 50
```

Mask columns are named `mask_<output>_path`, such as `mask_liver_path` and
`mask_liver_tumor_path`. The built-in manifest dispatches CT rows to CT tasks and
MR/MRI rows to the corresponding `_mr` tasks.

With `--crop`, the source NIfTI and every output mask are cropped in place to
one shared bounding box around all segmented foreground. The padding is 50 mm
by default; change it with `--crop-margin-mm`. If all masks are empty, the files
are left unchanged and the row receives a warning. Cropping occurs before mask
post-processing, so logical and morphological operations run on the cropped
grid. Crop bounds are clipped to the true image size and never introduce padded
voxels.

## 5. Resolve contrast phase

```bash
imperandi phase \
  --csv_path ./project/tables/nifti_index_segmented.csv \
  --csv_path_out ./project/tables/nifti_index_phased.csv \
  --manifest generic
```

The phase command writes canonical `phase` provenance and adds `totalseg_*`
fields only for rows that reach the prediction strategy. It also writes a selected
cohort and QC tables. The examples continue with the full cohort; see
[Outputs](outputs.md#curation-and-selection-exports) to choose representative
volumes instead.

## 6. Optionally register organs and masks

Install `imperandi[registration]` (see [Installation](installation.md)), then run:

```bash
imperandi register \
  --csv_path ./project/tables/nifti_index_phased.csv \
  --csv_path_out ./project/tables/nifti_index_registered.csv \
  --output_dir ./project/registration \
  --manifest generic
```

By default, groups are defined by patient, study, and modality. Images remain on
their native grids; canonical organ and tumor mask paths point to the consensus
results for successful rows. Review `register_qc.csv` and registration/consensus
statuses before using these masks. Original mask paths remain in `source_*`
columns. See [Registration](registration.md) for common-reference output and
options that preserve individual segmentations.

## 7. Extract radiomics

Install PyRadiomics separately using the [installation guide](installation.md).
For the workflow without registration:

```bash
imperandi radiomics \
  --csv_path ./project/tables/nifti_index_phased.csv \
  --csv_path_out ./project/tables/nifti_index_radiomics.csv \
  --manifest generic
```

If you ran registration, use `nifti_index_registered.csv` as `--csv_path` to
extract features using the registered masks. Radiomics discovers populated
`mask_*_path` columns and appends feature columns for the retained rows.

The `generic` manifest retains only `ARTERIAL` and `PORTAL_VENOUS` rows for
radiomics. Other phases are excluded from the resulting table. Edit
`radiomics.filters` in a custom manifest, or use `--skip_filter` to disable all
CLI and manifest filters. A manifest filter takes precedence over `--filter`
for the same column; adding a CLI phase filter will not override `generic`.

## 8. Check failures before analysis

Each heavy stage writes a command-specific error CSV. Review those reports and
row counts rather than treating a zero process exit alone as proof that every
volume succeeded. See [Outputs](outputs.md) for the complete file map.

Open the resulting image-and-mask table in a [QC viewer](viewers.md) to inspect
segmentations and alignment alongside the tabular reports.
