# Contributing to IMPERANDI

Contributions are welcome from imaging researchers, clinicians, data engineers,
and software developers. You can help by reporting bugs, improving documentation,
sharing reproducible use cases, refining CT/MR curation rules, or adding features
and tests.

We particularly welcome contributions extending cohort curation and processing 
to other thoracoabdominal anatomies, supported by explicit imaging rationale 
and reproducible validation examples.

## Questions, bug reports, and feature requests

Check the [documentation](https://imperandi.readthedocs.io/en/latest/) and
[existing issues](https://github.com/dmandache/IMPERANDI/issues) first. If your
question or problem is not covered, [open an issue](https://github.com/dmandache/IMPERANDI/issues/new).

For a bug report, include:

- Your IMPERANDI version or Git commit, Python version, and operating system.
- The pipeline stage and exact command or Python call that fails.
- Relevant manifest settings and optional dependency versions.
- The expected behavior, actual behavior, and error traceback.
- The produced logs and any relevant stage-specific warning or error CSVs.
- A minimal reproducible example, preferably using synthetic data or a public
  dataset that others can access.

Try rerunning the failing command with `--log-level DEBUG` and save the logs
with `--log-file`. Both options go before the subcommand, for example:

```bash
imperandi --log-level DEBUG --log-file imperandi-debug.log ingest \
  --root_path /path/to/dicom \
  --output_dir /path/to/output \
  --manifest generic
```

Replace `ingest` and its arguments with the command that reproduces your issue.
Include the resulting log file in your report after removing sensitive
information (see the guidance below).

For feature requests, describe the research need, proposed behavior, and an
example of how it would be used. Discuss substantial changes in an issue before
starting a large implementation so the scope and approach can be agreed upon.
Small fixes and documentation improvements can go directly to a pull request.

## Sharing imaging examples safely

IMPERANDI does not automatically de-identify DICOM data. Do not include patient
information, credentials, or restricted hospital data in issues, pull requests,
logs, screenshots, or notebook outputs. Review metadata, file paths, identifiers,
dates, and image content before sharing an example.

Use small synthetic fixtures for unit tests where possible. For public datasets,
provide a source link and reproduction instructions rather than committing image
collections or generated pipeline outputs.

## Development setup

Use Python 3.10, 3.11, 3.12, or 3.13, matching the versions supported by the project and
tested in CI. Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/YOUR_USERNAME/IMPERANDI.git
cd IMPERANDI
git remote add upstream https://github.com/dmandache/IMPERANDI.git
git fetch upstream
git switch -c fix/short-description upstream/main

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all-dev]"
python -m pip install "pyradiomics @ git+https://github.com/AIM-Harvard/pyradiomics.git@master"
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`.

The `all-dev` extra installs all runtime features together with pytest, coverage,
Ruff, Black, and dataset-backed test dependencies, matching the CI environment.
For all runtime features without development tools, use `all` instead:

```bash
python -m pip install -e ".[all]"
python -m pip install "pyradiomics @ git+https://github.com/AIM-Harvard/pyradiomics.git@master"
```

PyRadiomics must be installed separately with either `all` or `all-dev`.
See the [installation guide](docs/source/installation.md) for details.
Dataset-backed tests and segmentation may require downloads and model weights.

Enable the repository's notebook cleanup hook (recommended):

```bash
git config core.hooksPath .githooks
```

On push, this hook strips outputs and execution state from changed notebooks.
If it cleans a notebook, it stages the changes and stops the push; review and
commit the cleaned notebook, then push again. Review notebook contents for
sensitive information before committing.

## Making changes

- Keep each pull request focused on one fix or feature.
- Follow the surrounding code style. Use Black for changed Python files and Ruff
  for linting; avoid unrelated formatting changes.
- Add regression tests for bug fixes and tests for new behavior under
  `tests/unit/`. Prefer small, deterministic fixtures without network access,
  model downloads, or private data.
- Update documentation and manifest examples when changing CLI options,
  configuration, or output columns.
- For curation or preprocessing changes, explain the imaging rationale and
  assumptions, include representative edge cases, and describe how cohort
  selection or measured features may change.
- Keep optional dependencies optional, and preserve traceable outputs, failure
  reporting, and resumable processing when changing pipeline stages.

Source code lives in `src/imperandi/`, documentation in `docs/source/`, and
dataset-specific repository configurations in `dataset_configs/`. Built-in
manifests and hooks shipped with the package live in
`src/imperandi/builtin_datasets_config/`.

## Checks before opening a pull request

Run linting and the fast test suite from the repository root:

```bash
python -m ruff check .
python -m pytest -m "not slow"
```

CI runs Ruff and the fast tests on Python 3.10, 3.11, 3.12, and 3.13 using the full
development environment above. Its test command includes a 70% coverage minimum:

```bash
python -m pytest -m "not slow" --cov=imperandi --cov-report=xml --cov-fail-under=70
```

Format the Python files you changed with `python -m black path/to/changed_file.py`.

For documentation changes, install the documentation dependencies and build:

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs/source docs/build/html
```

When a change affects the full imaging workflow, consider running the relevant
dataset-backed tests. These are separate from normal CI and require prepared
data and optional dependencies; see [tests/slow/README.md](tests/slow/README.md).

## Submitting a pull request

Push your branch to your fork and open a pull request against `main`:

```bash
git push -u origin fix/short-description
```

Use a descriptive title. In the description, explain the problem and resulting
behavior, link any related issue, and state which checks you ran and their
results. Mention any checks you could not run, compatibility changes, or
limitations relevant to review.

Draft pull requests are welcome for early feedback. Maintainers may request
changes before merging; keep discussion constructive and address review comments
with focused updates.

## License

IMPERANDI is distributed under the [Apache License 2.0](LICENSE). By submitting a
contribution, you agree that it can be distributed under the same license. Only
contribute code and other material that you have the right to share.
