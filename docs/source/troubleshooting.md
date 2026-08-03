# Troubleshooting

## Configuration is rejected

Run:

```bash
imperandi validate imperandi.yaml
imperandi config resolve imperandi.yaml
```

Project models reject unknown fields. Remove obsolete keys instead of trying to
silence validation. `csv_warning_threshold_files` is intentionally an internal
product constant and cannot appear in project YAML.

Relative paths are resolved from the directory containing the project file, not
the current shell directory.

## A heavy stage reports missing dependencies

Install optional features in the same environment that owns the `imperandi`
executable:

```bash
python -m pip install -e ".[segment]"
python -m pip install -e ".[radiomics]"
python -m imperandi --help
```

The segment extra supplies TotalSegmentator and SimpleITK. Radiomics currently
installs PyRadiomics from Git, so installation requires Git and network access.

## No DICOM files are found

- Confirm every resolved `input.sources` value from `config resolve`.
- Quote globs in YAML strings when they contain special characters.
- Confirm archives are within `input.archive_depth`.
- Test with a pre-indexed CSV/Parquet sample to separate discovery from curation.
- Inspect `01_index/index_errors.*` and the stage state.

## Patient IDs are empty or unexpected

- Inspect configured `patient_id_columns`, `namespace_columns`, and fallback.
- Review `patient_id_method` and `identity_confidence`.
- Store and inspect `identity_map` only in the appropriate secure location.
- For crosswalks, check normalized key columns and duplicate mappings.
- For HMAC, confirm the configured secret environment variable is populated in
  the running process.
- Do not derive clinical/site variables by parsing `patient_id`; use ontology or
  rule outputs.

## Too many series are excluded or slots are missing

Inspect `04_annotate/volumes_annotated.*`,
`04_annotate/volumes_rejected.*`, and
`07_resolve_select/selection_qc.*`. Relevant fields include:

- `eligible`, `exclusion_reason`, and `exclusion_rule_id`;
- CT/MRI rule features and selection scores;
- ontology/rule/image evidence columns;
- `phase_conflict` and `clinical_slot_conflict`;
- required-slot QC codes.

Validate site rules on a small representative table. Prefer a narrow explicit
rule or ontology correction to broad threshold changes.

## Conversion fails for archive-backed series

Ensure source archives remain at the paths indexed in `dicom_path`, that the run
filesystem has enough temporary space, and that the configured archive depth is
sufficient. Review `05_convert/convert_errors.*`; successful independent rows
remain usable.

## A resumed run does not rerun a stage

Resume is keyed by the effective configuration hash. A completed stage is
reused while its recorded artifacts exist. To create a distinct run, change
project configuration. To rerun within the same hash, set
`execution.resume: false` before running; this changes the effective hash and
records the choice.

Do not manually edit `run.json`, `stage.json`, or backend checkpoint files while
a run is active.

## Multiprocessing exhausts memory

Reduce `execution.workers` and, if necessary, the stage-specific
`conversion.workers`. GPU segmentation workers multiply model and image memory.
Validate one representative volume before increasing parallelism.

## CSV is slow or very large

Set:

```yaml
output:
  table_format: parquet
  publish_formats: [parquet, csv]
```

This keeps intermediate artifacts efficient while still publishing a CSV copy.
The pipeline never changes an explicitly selected format automatically.

## The documentation build fails during API imports

Build from the repository root after installing the package and
`docs/requirements.txt`:

```bash
python -m pip install -e .
python -m pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs/source docs/build/html
```

Sphinx mocks heavy optional modules, but base runtime dependencies must still be
installed.
