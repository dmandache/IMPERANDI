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
- Extracts selected DICOM header tags into a raw metadata table (`dicom_index.csv`).
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

- Extracts PyRadiomics features for organ and tumor regions from CT + masks.
- Includes a organ-minus-tumor extraction path for cleaner parenchyma characterization.
- Supports optional cohort filtering controls and error-aware output generation.
- Supports PyRadiomics parameterization from either `--pyradiomics_settings /path/to/Params.yaml` or manifest `radiomics` settings.

Impact: accelerates feature exctraction for prognostic and response modeling pipelines.

### 6) Registration and cohort harmonization

- Aligns scans across patients with `register-population` using an organ mask as the registration target.
- Aligns scans within each patient with `register-intra-patient` for multiphasic or longitudinal studies.
- Builds per-visit tumor consensus masks and longitudinal tumor consistency audits with `register-tumor-consensus`.
- Persists transform metadata, row-level logs, and explicit error CSVs for traceability and resumable execution.
- Supports rigid-only execution where needed via `--disable_elastic` for intra-patient and tumor-consensus workflows.

Impact: creates anatomically comparable scan spaces for downstream segmentation review, tumor tracking, and cohort-level analysis.

### 7) Interactive quality control viewer (Jupyter)

- Provides an interactive CT + mask viewer for cohort navigation and quick visual QA.
- Supports patient/date/phase exploration, mask overlays, window presets, and keyboard navigation.

Impact: shortens the feedback loop between pipeline outputs and clinical/imaging validation.

![image](https://raw.githubusercontent.com/dmandache/IMPERANDI/main/static/viewer-demo.png)

## CLI overview 🛠️

IMPERANDI ships a single CLI with these subcommands:

- `parse`: scan DICOMs and build metadata index tables.
- `clean`: filter and normalize parsed metadata.
- `ingest`: run `parse` then `clean`.
- `convert`: convert indexed DICOM volumes to NIfTI.
- `register-population`: rigidly align a cohort to an organ-derived population template (requires _SimpleITK_, install with `.[register]`).
- `register-intra-patient`: align scans within each patient, with optional elastic refinement (requires _SimpleITK_, install with `.[register]`).
- `register-tumor-consensus`: build per-visit tumor consensus masks and longitudinal audit tables (requires _SimpleITK_, install with `.[register]`).
- `segment`: run configurable segmentation on NIfTI volumes (requires _TotalSegmentator_, install with `.[segment]`).
- `phase`: extract contrast phase metadata from NIfTI volumes (requires _TotalSegmentator_, install with `.[segment]`).
- `radiomics`: extract radiomics features from NIfTI volumes and masks (requires _pyRadiomics_, install with `.[radiomics]`).

Get help:

```bash
imperandi --help
imperandi parse --help
imperandi clean --help
imperandi ingest --help
imperandi convert --help
imperandi register-population --help
imperandi register-intra-patient --help
imperandi register-tumor-consensus --help
imperandi segment --help
imperandi phase --help
imperandi radiomics --help
```

## Install ⚙️

Base install:

```bash
python -m pip install -e .
```

Registration dependencies:

```bash
python -m pip install -e ".[register]"
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

Enable tracked git hooks (recommended):

```bash
git config core.hooksPath .githooks
```

With hooks enabled, `git push` strips output/execution state from changed `*.ipynb` files, stages those changes, and stops once so you can commit the cleaned notebooks.

Install everything:

```bash
python -m pip install -e ".[all]"
```

Optional Jupyter kernel setup:

```bash
python -m ipykernel install --user --name imperandi310 --display-name "IMPERANDI (Python 3.10)"
```

## Quickstart 🚀

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

Extract radiomics with explicit PyRadiomics YAML settings:

```bash
imperandi radiomics \
  --csv_path /path/to/output/nifti_index_segmented.csv \
  --pyradiomics_settings /path/to/Params.yaml \
  --csv_path_out /path/to/output/nifti_index_radiomics.csv
```

Use manifest-defined radiomics settings:

```bash
imperandi radiomics \
  --csv_path /path/to/output/nifti_index_segmented.csv \
  --manifest generic \
  --csv_path_out /path/to/output/nifti_index_radiomics.csv
```

If both `--manifest` and `--pyradiomics_settings` are provided, IMPERANDI warns and
prefers manifest `radiomics` settings when that section exists.

## Registration workflows 🧭

Registration commands operate on NIfTI-index CSVs and expect valid organ masks. `register-tumor-consensus` additionally expects per-scan tumor masks. All three commands write an enriched cohort CSV plus dedicated logs/errors; the transform-heavy workflows also save artifacts under `--output_dir`.

### Population registration (`register-population`)

Use this when you want inter-patient spatial comparability across a cohort.

- Input columns: `nifti_path` and an organ mask column such as `mask_liver`.
- Main modes (set with `--template_mode`): `single_sample`, `mean_shape`, `principal_vectors`.
- Useful flags: `--save_registered_outputs`, `--normalize_registered_outputs`.

```bash
imperandi register-population \
  --csv_path /path/to/output/nifti_index_segmented.csv \
  --output_dir /path/to/output/registered_population \
  --template_mode mean_shape \
  --save_registered_outputs \
  --csv_path_out /path/to/output/nifti_index_registered_population.csv
```

### Intra-patient registration (`register-intra-patient`)

Use this when you want to align scans within the same patient across phases and/or visits.

- Input columns: `patient_key`, `nifti_path`, and an organ mask column such as `mask_liver`.
- Modes (set with `--intra_mode`): `auto`, `multiphasic`, `longitudinal`.
- Useful flags: `--disable_elastic`, `--grouping_visit_column`, `--grouping_phase_column`.

```bash
imperandi register-intra-patient \
  --csv_path /path/to/output/nifti_index_segmented.csv \
  --output_dir /path/to/output/registered_intra_patient \
  --intra_mode auto \
  --disable_elastic \
  --csv_path_out /path/to/output/nifti_index_registered_intra_patient.csv
```

### Tumor consensus and audit (`register-tumor-consensus`)

Use this when you have per-phase tumor masks and want a single consensus tumor representation per visit plus longitudinal consistency checks.

- Input columns: patient/visit/phase grouping columns, `nifti_path`, organ mask, and tumor mask.
- Consensus rules (set with `--consensus_rule`): `union`, `intersection`, `majority`.
- Useful flags: `--majority_threshold`, `--disable_elastic`.

```bash
imperandi register-tumor-consensus \
  --csv_path /path/to/output/nifti_index_segmented.csv \
  --output_dir /path/to/output/tumor_consensus \
  --consensus_rule majority \
  --csv_path_out /path/to/output/nifti_index_tumor_consensus.csv
```

## Core outputs 📁

- `parse`:
  - `dicom_index.csv` (resolved IDs and selected DICOM tags)
  - optional `dicom_tags_snapshot.ndjson` (full recursive tags on a sampled subset, via `--snapshot_tags`)
- `clean`:
  - cleaned cohort table (default `<input>_clean.csv`)
- `convert`:
  - NIfTI-enriched cohort table (`nifti_index.csv` by default)
  - conversion failures (`conv_errors.csv` by default)
- `register-population`:
  - registered cohort table (`<input>_registered_population.csv` by default)
  - row-level registration log (`register_population_log.csv` by default)
  - registration failures (`register_population_errors.csv` by default)
  - template and optional registered-image artifacts under `--output_dir`
- `register-intra-patient`:
  - intra-patient registered cohort table (`<input>_registered_intra_patient.csv` by default)
  - row-level registration log (`register_intra_patient_log.csv` by default)
  - registration failures (`register_intra_patient_errors.csv` by default)
  - per-row transform and warped-output artifacts under `--output_dir`
- `register-tumor-consensus`:
  - per-visit consensus cohort table (`<input>_tumor_consensus.csv` by default)
  - tumor component summary (`tumor_consensus_components.csv` by default)
  - longitudinal audit table (`tumor_consistency_audit.csv` by default)
  - consensus failures (`register_tumor_consensus_errors.csv` by default)
  - consensus masks and metadata under `--output_dir`
- `segment`, `phase`, `radiomics`:
  - enriched cohort table + command-specific error CSV

## Manifests and hooks 🎚️

Manifests define dataset-specific behavior and live in:

- `src/imperandi/datasets_config/manifests/*.json`

Hook implementations live in:

- `src/imperandi/datasets_config/hooks/`

You can pass either a manifest name (`generic`, `operandi`) or a custom manifest path.

For radiomics, manifest key `radiomics` can directly contain a PyRadiomics-style
settings object (same structure as `Params.yaml` content).

## Performance and reliability notes 🛡️

- Parallel execution controls are available for heavy stages (`parse`, `convert`, `register-population`, `register-intra-patient`, `segment`).
- Long-running stages (`parse`, `convert`, `register-population`, `register-intra-patient`, `segment`, `phase`, `radiomics`) use a unified checkpoint interface:
  `--checkpoint_every_rows`, `--checkpoint_every_sec`, `--no_resume`, `--strict_resume`.
- Resume is enabled by default; pass `--no_resume` to disable it.
- `parse` reads tags from defaults (`DEFAULT_DICOM_TAGS`) plus `--tags`; use `--snapshot_tags` for full recursive tag snapshots on sampled data.
- `parse` auto-detects archive-heavy inputs from a deterministic root sample (`--archive_detect_sample_size`) and can switch to archive-aware mode at runtime when needed.
- Archive workflows are bounded by depth and include path-safety protections.
- Most commands support `--dry-run` for pipeline planning and CI smoke checks.

<!-- ## Testing (slow datasets)

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
``` -->

## Use Case on [IRCAD Dataset](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/)

Download the dataset (~800MB):

```bash
wget https://cloud.ircad.fr/index.php/s/JN3z7EynBiwYyjy/download -O ircad.zip
```

Unzip the archive:

```bash
unzip ircad.zip -d ircad_dicom
```

After extraction, your structure should look similar to:
```
ircad_dicom/
└── 3Dircadb1/
    ├── 3Dircadb1.1/
    │   ├── PATIENT_DICOM.zip/
    │   ├── MASKS_DICOM.zip/
    │   └── ...
```

Install package:

```bash
conda create -n imperandi310 python=3.10
conda activate imperandi310
pip install -e .[all]
```

Execute pipeline:
```bash
imperandi ingest "ircad_dicom/3Dircadb1/**/PATIENT_DICOM*" . --snapshot_tags
imperandi convert dicom_index_clean.csv ircad_nifti/
imperandi segment nifti_index.csv
imperandi phase nifti_index.csv
imperandi radiomics nifti_index.csv
```

Inspect results with dashboards:
- explore images & segmentations with the interactive viewer
- inspect DICOM tags
- basic radiomics statistics
