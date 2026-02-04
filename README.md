# IMPERANDI - **IM**aging **PRE**processing **A**nd **N**ormalization for **D**iagnostic **I**nteroperability

![Tests](https://github.com/dmandache/IMPERANDI/actions/workflows/tests.yml/badge.svg?branch=main)
[![codecov](https://codecov.io/gh/dmandache/IMPERANDI/branch/main/graph/badge.svg)](https://codecov.io/gh/dmandache/IMPERANDI)

IMPERANDI is a Python toolkit and CLI for building clean, analysis-ready indexes from DICOM datasets. It scans DICOM files, extracts header tags, standardizes IDs, and applies dataset-specific cleaning rules so downstream pipelines can rely on consistent metadata.

This project is a 🚧work in progress. A fuller data pipeline is coming soon, including ingestion, filtering, processing (conversion to NIfTI, segmentation with TotalSegmentator, registration), feature exctraction with PyRadiomics, quality control, and descriptive dashboards.

IMPERANDI targets multi-phasic, longitudinal CT imaging data and addresses the challenges of cleaning and harmonizing heterogeneous hospital datasets. Its dependency footprint is intentionally minimal, reflecting its intended use within closed, secure hospital data-warehouse environments.

**Highlights**
- Fast DICOM header parsing with optional parallelism and checkpointing.
- Flexible ID selection from tags or folder structure.
- Dataset manifests and hook functions for standardization and derived columns.
- Cleaning pipeline tailored to CT volumes and acquisition metadata.

**Install**
```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

**CLI Overview**
IMPERANDI ships a single CLI with three subcommands:
- `parse`: scan DICOMs and build a metadata index.
- `clean`: filter and normalize the index.
- `ingest`: run `parse` then `clean` in one step.

Get help:
```bash
imperandi --help
imperandi parse --help
imperandi clean --help
imperandi ingest --help
```

**Quickstart**
Parse a dataset:
```bash
imperandi parse \
  --root_path /path/to/dicom \
  --output_dir /path/to/output \
  --manifest generic
```

Clean the parsed CSV:
```bash
imperandi clean \
  --csv_path /path/to/output/dicom_index.csv \
  --csv_path_out /path/to/output/dicom_index_clean.csv \
  --manifest generic
```

Run both steps together:
```bash
imperandi ingest \
  --root_path /path/to/dicom \
  --output_dir /path/to/output \
  --manifest operandi
```

**Outputs**
`parse` writes two CSVs into `--output_dir`:
- `dicom_paths_with_tags.csv`: raw paths plus extracted tags.
- `dicom_index.csv`: finalized index with patient/study/series IDs.

`clean` writes the cleaned CSV to `--csv_path_out` (or `dicom_index_clean.csv` when using `ingest`).

**Manifests And Hooks**
Manifests define dataset-specific behavior and live in:
- `src/imperandi/datasets_config/manifests/*.json`

You can pass a manifest by name (`generic`, `operandi`) or a path. Manifests can specify:
- `id_standardization`: a hook to normalize patient keys.
- `derived_columns`: hook-driven columns derived from existing metadata.

Hook implementations live under:
- `src/imperandi/datasets_config/hooks/`

**Performance Notes**
- Use `--num_workers` to control parallelism.
- Use `--checkpoint_frequency` to write intermediate CSV chunks for very large datasets.
- Use `--flatten_all_tags` only when you need the full DICOM header; it can be very large.

**Data Expectations**
- By default, IMPERANDI looks for `*.dcm` files. If none are found, it scans all files and attempts a header read.
- ID selection can come from tags, path structure, or a hybrid (`--id_source auto`).

**Running From Source**
```bash
python -m imperandi --help
```

**License**
See `LICENSE`.
