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
  --manifest generic
```

Mask columns are named `mask_<output>`, such as `mask_liver` and
`mask_liver_tumor`.

## 5. Extract phase and radiomics

```bash
imperandi phase \
  --csv_path ./project/tables/nifti_index_segmented.csv \
  --csv_path_out ./project/tables/nifti_index_phased.csv

imperandi radiomics \
  --csv_path ./project/tables/nifti_index_phased.csv \
  --csv_path_out ./project/tables/nifti_index_radiomics.csv \
  --manifest generic
```

The phase command adds `totalseg_*` fields. Radiomics discovers all populated
`mask_*` columns and appends feature columns to the table.

## 6. Check failures before analysis

Each heavy stage writes a command-specific error CSV. Review those reports and
row counts rather than treating a zero process exit alone as proof that every
volume succeeded. See [Outputs](outputs.md) for the complete file map.

