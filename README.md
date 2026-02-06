# IMPERANDI - **IM**aging **PRE**processing **A**nd **N**ormalization for **D**iagnostic **I**nteroperability

![image](https://github.com/dmandache/IMPERANDI/static/imperandi-logo.png)

<!-- ![Python](https://img.shields.io/pypi/pyversions/YOUR_PACKAGE_NAME) -->
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Linting](https://img.shields.io/badge/lint-ruff-red)
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
IMPERANDI ships a single CLI with several subcommands:
- `parse`: scan DICOMs and build a metadata index.
- `clean`: filter and normalize the index.
- `ingest`: run `parse` then `clean` in one step.
- `convert`: converts DICOMs from index to NIfTI.
- `segment`: run configurable segmentation on NIfTI volumes.

Get help:
```bash
imperandi --help
imperandi parse --help
imperandi clean --help
imperandi ingest --help
imperandi convert --help
imperandi segment --help
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

Segment NIfTI volumes (default liver config):
```bash
imperandi segment \
  --csv_path /path/to/output/nifti_index.csv \
  --csv_path_out /path/to/output/nifti_index_segmented.csv
```

Example segmentation config (JSON):
```json
{
  "backend": "totalsegmentator",
  "tasks": [
    {
      "key": "liver",
      "task": "total",
      "output": "liver.nii.gz",
      "extra": { "roi_subset_robust": ["liver"] }
    },
    {
      "key": "vessels",
      "task": "liver_vessels",
      "output": "liver_tumor.nii.gz",
      "extra": {}
    }
  ],
  "postprocess": {
    "merge_keys": ["liver", "vessels"],
    "output": "liver_all.nii.gz",
    "radius_mm": 5.0,
    "largest_cc": true,
    "fill_holes": true,
    "close": true
  }
}
```

**Outputs**
`parse` writes two CSVs into `--output_dir`:
- `dicom_paths_with_tags.csv`: raw paths plus extracted tags.
- `dicom_index.csv`: finalized index with patient/study/series IDs.

`clean` writes the cleaned CSV to `--csv_path_out` (or `dicom_index_clean.csv` when using `ingest`).

**Philosophy**
IMPERANDI is designed to work end‑to‑end out of the box, while still being easy to personalize. You can run the full pipeline in one go or tailor behavior through dataset manifests and user‑defined hooks. At the same time, each stage is modular, so you can intervene between steps to edit or enrich CSVs before moving on. The choice of CSV as the interchange format is intentional: it is generic, lightweight, and easy to inspect, edit, and share.

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
- All DICOM tags are read recursively and flattened into columns by default, which can result in large output CSVs.

**Data Expectations**
- By default, IMPERANDI looks for `*.dcm` files. If none are found, it scans all files and attempts a header read.
- ID selection can come from tags, path structure, or a hybrid (`--id_source auto`).

**Running From Source**
```bash
python -m imperandi --help
```

**Testing (Slow Datasets)**
Slow integration tests for the IRCAD dataset are available and are skipped unless data is present.
- Place the DICOM dataset at `tests/data/IRCAD_DICOM` (gitignored) or set `IRCAD_ROOT` to the dataset path.
- Optional: place NIfTI outputs at `tests/data/IRCAD_nifti` or set `IRCAD_NIFTI_ROOT`.
- Run slow tests with:
```bash
python -m pytest -m slow
```
- Regenerate golden CSVs locally (from repo root):
```bash
python -m imperandi parse --root_path tests/data/IRCAD_DICOM --output_dir tests/data
python -m imperandi clean --csv_path tests/data/dicom_index.csv --csv_path_out tests/data/dicom_index_clean.csv
```
- Note: there is no auto-download due to licensing; datasets must be placed manually.

**License**
See `LICENSE`.
