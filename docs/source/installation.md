# Installation

IMPERANDI requires Python 3.10–3.12. A virtual environment keeps its
scientific dependencies isolated from other projects.

```bash
git clone https://github.com/dmandache/IMPERANDI.git
cd IMPERANDI
python3 -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Confirm that the command is available:

```bash
imperandi --help
```

## Optional features

The base installation supports DICOM ingest and NIfTI conversion. Install only
the heavier feature sets needed by your workflow. PyRadiomics must be installed
separately from Git, including when using `[all]` or `[all-dev]`.
This keeps IMPERANDI's package metadata compatible with PyPI. The separate
installation requires Git and network access:

```bash
# TotalSegmentator-based segmentation and phase-prediction fallback
python -m pip install -e ".[segment]"

# PyRadiomics extraction
python -m pip install "pyradiomics @ git+https://github.com/AIM-Harvard/pyradiomics.git@master"

# Notebook and web quality-control viewers
python -m pip install -e ".[viewer]"

# Tests, linting, and formatting
python -m pip install -e ".[dev]"

# All runtime features
python -m pip install -e ".[all]"
python -m pip install "pyradiomics @ git+https://github.com/AIM-Harvard/pyradiomics.git@master"

# All runtime features, development tools, and slow-test dependencies
python -m pip install -e ".[all-dev]"
python -m pip install "pyradiomics @ git+https://github.com/AIM-Harvard/pyradiomics.git@master"
```

TotalSegmentator may download model weights the first time a segmentation task
runs. Plan for network access and sufficient local storage at installation or
model-prefetch time; cohort processing can then run in a controlled environment.

## Build these docs

Install the project and documentation dependencies, then invoke Sphinx from the
repository root:

```bash
python -m pip install -e .
python -m pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs/source docs/build/html
```

Open `docs/build/html/index.html` in a browser. Omitting `-W` is useful while
editing, but CI should treat documentation warnings as errors.

## Jupyter quality-control viewer

The classic viewer uses the widget backend. After installing the `viewer`
dependencies, an optional named kernel can be registered with:

```bash
python -m ipykernel install --user \
  --name imperandi \
  --display-name "IMPERANDI"
```

