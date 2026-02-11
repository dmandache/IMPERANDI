# **IM**aging **PRE**processing **A**nd **N**ormalization for **D**iagnostic **I**nteroperability

![image](https://raw.githubusercontent.com/dmandache/IMPERANDI/main/static/imperandi-logo.png)

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Linting](https://img.shields.io/badge/lint-ruff-red)
![Tests](https://github.com/dmandache/IMPERANDI/actions/workflows/tests.yml/badge.svg?branch=main)
[![codecov](https://codecov.io/gh/dmandache/IMPERANDI/branch/main/graph/badge.svg)](https://codecov.io/gh/dmandache/IMPERANDI)

IMPERANDI is a Python framework and CLI for building analysis-ready CT imaging cohorts from heterogeneous DICOM sources. It standardizes identifiers, curates volume-level metadata, converts volumes to NIfTI, and supports downstream segmentation, perfusion phase detection, radiomics extraction, and quality control in one coherent pipeline.

## Why IMPERANDI matters

- Reduces manual data wrangling by turning raw DICOM trees into structured cohort tables.
- Improves reproducibility with explicit CSV outputs at every stage and deterministic ID logic.
- Improves reliability on real hospital exports with archive support, failure tracking, and resumable workflows.
- Keeps adoption practical in secure environments with a lightweight Python-first toolchain.

## Current framework functionalities

### 1) Ingest and harmonize imaging metadata (`parse` + `clean` = `ingest`)

- Scans DICOM files from folders, globbed roots, and nested archives (`.zip`, `.tar`, `.tar.gz`, `.tgz`).
- Extracts DICOM headers recursively into a raw metadata table (`dicom_paths_with_tags.csv`).
- Builds stable patient/study/series identifiers from tags, folder structure, or hybrid fallback rules.
- Applies manifest-driven hooks for patient-key standardization and derived columns.
- Cleans and curates CT cohorts by filtering modality/noise patterns, localizers, non-target anatomy, non-axial acquisitions, and implausible scan geometry.
- Aggregates slices into robust volume-level records and computes exam/acquisition ordering.

Impact: turns fragmented acquisition data into a consistent cohort backbone that downstream models and analytics can trust.

### 2) Convert DICOM volumes to NIfTI (`convert`)

- Converts curated DICOM volume rows to NIfTI in parallel using `dicom2nifti`.
- Preserves source-to-output traceability in a CSV (`nifti_path` per row).
- Handles archive-backed DICOM paths transparently via on-demand materialization.
- Writes explicit conversion error tables without aborting the whole run.

Impact: creates a standardized imaging representation for model training, segmentation, and feature extraction at scale.

### 3) Configurable segmentation (`segment`)

- Runs configurable task pipelines (default backend: TotalSegmentator).
- Supports multi-task mask generation per volume through a JSON task config.
- Adds optional post-processing (mask merge, closing, hole filling, largest connected component).
- Uses multiprocessing with timeout controls and produces warning/error tracking CSVs.

Impact: converts raw CT volumes into ready-to-use anatomical/tumor masks with operational safeguards for large cohort processing.

### 4) Contrast phase extraction (`phase`)

- Extracts CT contrast phase metadata from NIfTI volumes using TotalSegmentator phase utilities.
- Appends normalized phase outputs to cohort CSVs (`totalseg_*` columns).
- Captures per-row failures into dedicated error outputs.

Impact: enables phase-aware stratification and analysis without manual review of every study.

### 5) Radiomics feature extraction (`radiomics`)

- Extracts PyRadiomics features for liver and tumor regions from CT + masks.
- Includes a liver-minus-tumor extraction path for cleaner parenchyma characterization.
- Supports optional cohort filtering controls and error-aware output generation.

Impact: accelerates feature engineering for prognostic and response modeling pipelines.

### 6) Interactive quality control viewer (Jupyter)

- Provides an interactive CT + mask viewer for cohort navigation and quick visual QA.
- Supports patient/date/phase exploration, mask overlays, window presets, and keyboard navigation.

Impact: shortens the feedback loop between pipeline outputs and clinical/imaging validation.

![image](https://raw.githubusercontent.com/dmandache/IMPERANDI/main/static/viewer-demo.png)

## CLI overview

IMPERANDI ships a single CLI with these subcommands:

- `parse`: scan DICOMs and build metadata index tables.
- `clean`: filter and normalize parsed metadata.
- `ingest`: run `parse` then `clean`.
- `convert`: convert indexed DICOM volumes to NIfTI.
- `segment`: run configurable segmentation on NIfTI volumes (requires `.[segment]`).
- `phase`: extract contrast phase metadata from NIfTI volumes (requires `.[segment]`).
- `radiomics`: extract radiomics features from NIfTI volumes and masks (requires radiomics dependencies).

Get help:

```bash
imperandi --help
imperandi parse --help
imperandi clean --help
imperandi ingest --help
imperandi convert --help
imperandi segment --help
imperandi phase --help
imperandi radiomics --help
```

## Install

Base install:

```bash
python -m pip install -e .
```

Segmentation dependencies:

```bash
python -m pip install -e ".[segment]"
```

Radiomics dependencies:

```bash
python -m pip install -e ".[radiomics]"
```

Development and test tooling:

```bash
python -m pip install -e ".[dev]"
```

Install everything:

```bash
python -m pip install -e ".[all]"
```

Optional Jupyter kernel setup:

```bash
python -m ipykernel install --user --name imperandi310 --display-name "IMPERANDI (Python 3.10)"
```

## Quickstart

Run ingest (parse + clean):

```bash
imperandi ingest \
  --root_path /path/to/dicom \
  --output_dir /path/to/output \
  --manifest generic
```

Convert to NIfTI:

```bash
imperandi convert \
  --csv_path /path/to/output/dicom_index_clean.csv \
  --output_dir /path/to/nifti_root \
  --csv_path_out /path/to/output/nifti_index.csv
```

Run segmentation:

```bash
imperandi segment \
  --csv_path /path/to/output/nifti_index.csv \
  --csv_path_out /path/to/output/nifti_index_segmented.csv
```

Extract contrast phase:

```bash
imperandi phase \
  --csv_path /path/to/output/nifti_index_segmented.csv \
  --csv_path_out /path/to/output/nifti_index_phased.csv
```

Extract radiomics:

```bash
imperandi radiomics \
  --csv_path /path/to/output/nifti_index_segmented.csv \
  --csv_path_out /path/to/output/nifti_index_radiomics.csv
```

## Core outputs

- `parse`:
  - `dicom_paths_with_tags.csv` (raw extracted tags)
  - `dicom_index.csv` (resolved IDs and normalized metadata)
- `clean`:
  - cleaned cohort table (default `<input>_clean.csv`)
- `convert`:
  - NIfTI-enriched cohort table (`nifti_index.csv` by default)
  - conversion failures (`conv_errors.csv` by default)
- `segment`, `phase`, `radiomics`:
  - enriched cohort table + command-specific error CSV

## Manifests and hooks

Manifests define dataset-specific behavior and live in:

- `src/imperandi/datasets_config/manifests/*.json`

Hook implementations live in:

- `src/imperandi/datasets_config/hooks/`

You can pass either a manifest name (`generic`, `operandi`) or a custom manifest path.

## Performance and reliability notes

- Parallel execution controls are available for heavy stages (`parse`, `convert`, `segment`).
- `parse` supports checkpointing for large datasets (`--checkpoint_frequency`).
- Archive workflows are bounded by depth and include path-safety protections.
- Most commands support `--dry-run` for pipeline planning and CI smoke checks.

## Testing (slow datasets)

Slow integration tests for the [IRCAD dataset](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/) are available and skipped unless data is present.

- Place DICOM data at `tests/data/IRCAD_DICOM` (gitignored) or set `IRCAD_ROOT`.
- Optional: place NIfTI outputs at `tests/data/IRCAD_nifti` or set `IRCAD_NIFTI_ROOT`.
- Run slow tests:

```bash
python -m pytest -m slow
```

- Regenerate reference CSVs:

```bash
python -m imperandi parse --root_path tests/data/IRCAD_DICOM --output_dir tests/data
python -m imperandi clean --csv_path tests/data/dicom_index.csv --csv_path_out tests/data/dicom_index_clean.csv
```

Data is not auto-downloaded due to dataset licensing constraints.
