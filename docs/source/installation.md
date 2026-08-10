# Installation

IMPERANDI requires Python 3.10 or newer. A virtual environment keeps its
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
the heavier feature sets needed by your workflow:

```bash
# TotalSegmentator-based segmentation and phase-prediction fallback
python -m pip install -e ".[segment]"

# PyRadiomics extraction
python -m pip install -e ".[radiomics]"

# Tests, linting, and Jupyter tools
python -m pip install -e ".[dev]"

# Every optional feature
python -m pip install -e ".[all]"
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

The classic viewer uses the widget backend. After installing the development
dependencies, an optional named kernel can be registered with:

```bash
python -m ipykernel install --user \
  --name imperandi \
  --display-name "IMPERANDI"
```

