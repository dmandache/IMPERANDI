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

## `postprocess`

```bash
imperandi postprocess [CSV_PATH] [OUTPUT_DIR] [OPTIONS]
imperandi postprocess nifti_index.csv --normalization zscore --mask nonzero
imperandi postprocess nifti_index.csv --manifest site-a.yaml
```

Processes scalar 3-D NIfTI images listed in `nifti_path`. Contrast and
normalization use the base installation. N4 bias correction requires
`pip install 'imperandi[postprocess]'`. The standalone module command is
`python -m imperandi.process.postprocess` with the same arguments.

- `--bias-correction {none,n4}`: optional bias correction, applied first.
- `--contrast {none,percentile,window,gamma,histogram}`: contrast adjustment.
- `--normalization {none,zscore,robust_zscore,minmax,percentile}`: intensity
  normalization. `z-score` and `z_score` are accepted aliases.
- `--percentiles LOW HIGH`: clipping/scaling percentiles (default 1, 99).
- `--window LOW HIGH`: required bounds for `--contrast window`.
- `--gamma VALUE`: positive exponent for `--contrast gamma` (default 1).
- `--mask {all,nonzero,positive}`: foreground selection (default `nonzero`).
- `--mask-column COLUMN`: use an existing NIfTI mask from this CSV column.
- `--outside-mask {preserve,zero}`: treatment of unselected voxels.
- `--csv_path`, `--output_dir`: named paths override positional paths.
- `--csv_path_out`: defaults to `nifti_index_postprocessed.csv` beside the input.
- `--error_csv_path`: defaults to `postprocess_errors.csv` beside the input.
- `--force`: recompute even when a matching derived image exists.

Supply a manifest `image_postprocessing` section or at least one method flag.
Any method flag replaces the entire manifest postprocessing configuration with
a global bias → contrast → normalization sequence. Mask flags can override
manifest profiles without replacing steps. Algorithm parameters beyond the
listed flags, including N4 iterations and normalization output ranges, belong
in the manifest. See [configuration](manifests.md#image-intensity-postprocessing).

The default image root is `<csv_dir>/POSTPROCESSED`. The new CSV points
`nifti_path` to successful derivatives and retains the input path as
`source_nifti_path`. Inputs are preserved; input, output, and error CSV paths
must differ. Each derivative has a JSON provenance sidecar. Unconfigured
modalities keep their source images with status `skipped`; failed rows have an
empty `nifti_path` and an error message. Exit codes: 0 success, 1 per-volume
failures, 2 invalid configuration or missing dependencies.

`--dry-run` validates settings and prints the resolved configuration without
loading images or optional backends. Shared checkpoint options apply. Resume
checks each derivative against its source, mask, parameters, library versions,
and output fingerprint; failures are retried and changed/missing outputs are
rebuilt. `--strict_resume` hashes image and mask contents as well as the CSV.
`--no_resume` recomputes all selected volumes.

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

Requires a separate PyRadiomics installation (see
[installation](installation.md)), `nifti_path`, and `mask_*` columns.

- `--manifest NAME_OR_YAML`: load settings and filters from `radiomics`.
- `--pyradiomics_settings PARAMS.yaml`: use explicit PyRadiomics settings.
- `--filter column=value1,value2`: filter rows; repeat for more columns.
- `--skip_filter`: ignore both CLI and manifest filters.
- `--error_csv_path`: defaults to `radiomics_errors.csv`.

When a manifest contains PyRadiomics settings, they take precedence over an
explicit YAML path and a warning is emitted.

## Shared long-running options

`parse`, `convert`, `postprocess`, `segment`, `phase`, and `radiomics` accept checkpoint and
resume options described in [Workflow](workflow.md). All commands support
`--dry-run`.
