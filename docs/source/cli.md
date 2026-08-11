# CLI Commands

The installed entry point is `imperandi`; `python -m imperandi` is equivalent.

```bash
imperandi [--log-level LEVEL] [--log-file PATH] [--quiet] COMMAND [OPTIONS]
```

Global options must appear before the subcommand. `--log-level` accepts normal
Python logging levels such as `DEBUG`, `INFO`, and `WARNING`.

Run `imperandi COMMAND --help` for the authoritative option list in your
installed version.

## `parse`

```bash
imperandi parse [ROOT_PATH] [OUTPUT_DIR] [OPTIONS]
```

Reads selected DICOM headers and writes `dicom_index.csv`. Important options:

- `--root_path`, `--output_dir`: named alternatives to positional paths; named
  values win when both forms are supplied.
- `--manifest NAME_OR_YAML`: dataset configuration.
- `--id_source {auto,tags,path}` and `--patient_key_from`, `--study_id_from`,
  `--series_id_from`: ID derivation.
- `--tags A,B,C`: additional DICOM keywords.
- `--force_dicom_read`: tolerate non-conformant DICOM files.
- `--snapshot_tags`, `--snapshot_sample_size`, `--snapshot_seed`: recursive tag
  snapshot controls.
- `--num_workers`: header-reading process count.
- `--archive_max_depth`, `--archive_cache_dir`, `--keep_archive_cache`: archive
  controls.

## `clean`

```bash
imperandi clean [CSV_PATH] [CSV_PATH_OUT] [OPTIONS]
```

Curates parsed instance metadata into a volume-level table. `--csv_path`
accepts one or more CSVs. Cleaning filters and phase-curation precedence come
from the YAML manifest.

## `ingest`

```bash
imperandi ingest [ROOT_PATH] [OUTPUT_DIR] [OPTIONS]
```

Combines `parse` and `clean`. `--csv_path_out` selects the final cleaned table;
it defaults to `<output_dir>/dicom_index_clean.csv`.

## `convert`

```bash
imperandi convert [CSV_PATH] [OUTPUT_DIR] [OPTIONS]
```

Converts series listed in one or more CSV files. The default final table is
`nifti_index.csv` and the default failure table is `conv_errors.csv`, both next
to the input CSV. If `OUTPUT_DIR`/`--output_dir` is omitted, converted files are
written to `<project_root>/NIFTI` (the input CSV's directory) and a warning is
logged. Use `--num_workers` to control conversion parallelism.

## `segment`

```bash
imperandi segment [CSV_PATH] [CSV_PATH_OUT] [OPTIONS]
```

Requires `imperandi[segment]`. It reads `nifti_path` and `Modality`, then runs
the selected manifest's matching `segmentation.modalities` configuration. CT
and MR/MRI rows are dispatched to separate model/task lists; unconfigured
modalities are retained without segmentation.

- `--manifest NAME_OR_YAML`: task and post-processing configuration.
- `--num_workers`: process count.
- `--start_method {spawn,fork,forkserver}`: multiprocessing strategy; `spawn`
  is the robust default.
- `--timeout_sec`: per-volume timeout.
- `--force`: rerun when output masks already exist.
- `--error_csv_path`: defaults to `seg_errors.csv` beside the input.

The output table overwrites the input unless `--csv_path_out` is supplied.

## `phase`

```bash
imperandi phase [CSV_PATH] [CSV_PATH_OUT] [OPTIONS]
```

Uses `--manifest NAME_OR_YAML` to resolve the ordered `phase_curation`
strategies. Rows matched by ontology or metadata rules do not invoke
TotalSegmentator when prediction is a later fallback. Prediction requires
`imperandi[segment]` and `nifti_path`; it defaults to CT rows only, and
`--force` recomputes existing predictions. The command writes canonical
`phase` and provenance columns. Failures default to `phase_errors.csv`.

## `radiomics`

```bash
imperandi radiomics [CSV_PATH] [CSV_PATH_OUT] [OPTIONS]
```

Requires `imperandi[radiomics]`, `nifti_path`, and `mask_*` columns.

- `--manifest NAME_OR_YAML`: load settings and filters from `radiomics`.
- `--pyradiomics_settings PARAMS.yaml`: use explicit PyRadiomics settings.
- `--filter column=value1,value2`: filter rows; repeat for more columns.
- `--skip_filter`: ignore both CLI and manifest filters.
- `--error_csv_path`: defaults to `radiomics_errors.csv`.

When a manifest contains PyRadiomics settings, they take precedence over an
explicit YAML path and a warning is emitted.

## Shared long-running options

`parse`, `convert`, `segment`, `phase`, and `radiomics` accept checkpoint and
resume options described in [Workflow](workflow.md). All commands support
`--dry-run`.
