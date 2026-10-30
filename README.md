<p align="center">
  <img src="https://raw.githubusercontent.com/dmandache/IMPERANDI/main/static/imperandi-logo-with-name.png"
       alt="IMPERANDI"
       width="380">
</p>

<p align="center">
  <a href="https://pypi.org/project/imperandi/"><img src="https://img.shields.io/pypi/v/imperandi.svg" alt="PyPI"></a>
  <a href="https://pypistats.com/packages/imperandi"><img src="https://pypistats.com/api/badges/imperandi?period=month" alt="PyPI Stats"></a>
  <img src="https://img.shields.io/badge/Python-3.10–3.13-blue" alt="Python 3.10–3.13">
  <a href="https://imperandi.readthedocs.io/en/latest/"><img src="https://img.shields.io/readthedocs/imperandi" alt="Documentation"></a>
  <a href="https://github.com/dmandache/IMPERANDI/actions/workflows/tests.yml"><img src="https://github.com/dmandache/IMPERANDI/actions/workflows/tests.yml/badge.svg?branch=main" alt="Tests"></a>
  <a href="https://app.codecov.io/github/dmandache/IMPERANDI"><img src="https://codecov.io/github/dmandache/IMPERANDI/graph/badge.svg" alt="Coverage"></a>
  <img src="https://img.shields.io/badge/lint-ruff-red" alt="Ruff">
  <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black">
</p>

IMPERANDI is a Python framework and CLI for transforming heterogeneous DICOM collections into analysis-ready CT and MR imaging cohorts. It standardizes identifiers, curates imaging series and volume-level metadata, converts selected volumes to NIfTI, and supports downstream segmentation, contrast-phase identification, radiomics extraction, and quality control within a coherent, reproducible pipeline.

## Google Colab demos

Try IMPERANDI on three public TCGA-LIHC patients:

| Demo | Workflow | Launch |
| --- | --- | --- |
| [Light demo](examples/demos/demo_light.ipynb) | DICOM ingestion, metadata phase curation, and QC. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmandache/IMPERANDI/blob/main/examples/demos/demo_light.ipynb) |
| [Full demo](examples/demos/demo_full.ipynb) | Adds NIfTI conversion, liver segmentation, radiomics, the interactive viewer and TorchIO preprocessing into a PyTorch batch. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmandache/IMPERANDI/blob/main/examples/demos/demo_full.ipynb) |

## What IMPERANDI does

| Stage | Purpose |
| --- | --- |
| `parse`, `clean`, `ingest` | Discover DICOM folders and archives, harmonize identifiers, curate CT/MR metadata, and select representative volumes. |
| `convert` | Convert curated volumes to NIfTI with source traceability. |
| `segment` | Run modality-specific segmentation and configurable mask post-processing. |
| `phase` | Resolve contrast phase using ordered metadata, ontology, and optional model strategies. |
| `register` | Align organs within configured groups and optionally build consensus masks in native or reference space. |
| `radiomics` | Extract features from images and masks with modality-specific settings and cohort filters. |
| Lightweight viewer | Inspect cohort tables, images, and masks in notebook or web viewers. |

Stages exchange CSV tables with image and mask paths. Long-running stages support
checkpoints and resume; error and QC reports preserve per-volume outcomes.

## Install

Requires Python 3.10–3.13:

```bash
python -m pip install imperandi
```

Optional extras include `imperandi[segment]`, `imperandi[registration]`, and
`imperandi[viewer]`. PyRadiomics requires a separate installation, including with
`[all]`. See the [installation guide](docs/source/installation.md) for commands,
source installs, and dependency details.

## First cohort

```bash
imperandi ingest --root_path /path/to/dicom --output_dir ./tables --manifest generic
imperandi convert --csv_path ./tables/dicom_index_clean.csv --output_dir ./nifti
```

This produces `tables/dicom_index_clean.csv` and `tables/nifti_index.csv`, plus
curation and error reports. Review [outputs](docs/source/outputs.md) before
continuing. The [quickstart](docs/source/quickstart.md) adds segmentation, phase
resolution, optional registration, and radiomics with explicit output paths.
Use `imperandi COMMAND --help` for the options in your installed version.

## Guides and examples

- [Workflow](docs/source/workflow.md): stage dependencies, cohort selection, and resume.
- [CLI reference](docs/source/cli.md): commands and important options.
- [Configuration](docs/source/manifests.md): YAML manifests, hooks, curation, and processing settings.
- [Registration](docs/source/registration.md): grouping, alignment, consensus, output spaces, and QC.
- [Quality-control viewers](docs/source/viewers.md): notebook and browser launch examples.
- [Examples](examples/README.md): public-data demos and scientific benchmarks.
- [IRCAD workflow](tests/slow/ircad/README.md): dataset download and pipeline instructions.
- [Troubleshooting](docs/source/troubleshooting.md): common installation and processing issues.

Start custom configuration by copying the built-in `generic` YAML and passing its
path with `--manifest ./site-a.yaml`. Repository-specific OPERANDI settings live in
`dataset_configs/` and are loaded by explicit path; they are not installed with
the package.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, notebook cleanup,
checks, and pull requests. [Dataset-backed tests](tests/slow/README.md) document
the optional full-pipeline validation workflows.

## Disclaimers

### Research use only

IMPERANDI is research software and is not a medical device. Outputs are not intended for clinical diagnosis or treatment decisions without independent validation.

### DICOM privacy

IMPERANDI should not be assumed to de-identify input DICOM data. Users are responsible for appropriate anonymization and data-governance procedures before processing or sharing data.

### Funding

This work performed under the RHU OPERANDI project was supported in part by the French National Research Agency (Agence Nationale de la Recherche, ANR) as its 3rd PIA, integrated to France 2030 plan under reference _ANR-21-RHUS-0012._
