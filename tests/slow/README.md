# Dataset-backed pipeline tests

The slow suite runs the complete IMPERANDI workflow on small public cohorts.
IRCAD and TCGA-LIHC use the same directory contract:

```text
<dataset>/
├── download.py
├── pipeline.sh
├── pipeline.ps1
├── test_pipeline.py
└── data/
    ├── input/   # downloaded source data; ignored by Git
    └── work/    # generated pipeline output; ignored by Git
```

Install all optional dependencies before running the full suite:

```bash
python -m pip install -e '.[all]'
python tests/slow/ircad/download.py
python tests/slow/tcga_lihc/download.py
python -m pytest tests/slow -m slow
```

The tests skip when their input folder is empty. `IMPERANDI_IRCAD_INPUT` and
`IMPERANDI_TCGA_LIHC_INPUT` can point to existing input folders instead.

Pipeline scripts retain resumable output in `data/work`. Pytest uses a fresh
temporary work directory for each run.
