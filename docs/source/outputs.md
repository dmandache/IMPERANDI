# Outputs

CSV tables are the pipeline's audit spine. Each stage preserves existing
columns and adds or normalizes the fields needed downstream.

| Stage | Main output | Error/auxiliary output | Key additions |
|---|---|---|---|
| `parse` | `dicom_index.csv` | `dicom_index_errors.csv`; optional `dicom_tags_snapshot.ndjson` | IDs, `dicom_path`, selected DICOM tags |
| `clean` | `<input>_clean.csv` | selection and QC exports when phase curation is enabled | volume grouping, normalized date/time, ordering and geometry fields |
| `ingest` | `dicom_index_clean.csv` | parse artifacts; selection and QC exports when phase curation is enabled | parsed and curated volume rows |
| `convert` | `nifti_index.csv` | `conv_errors.csv` | `nifti_path` |
| `segment` | input CSV in place, or `--csv_path_out` | `seg_errors.csv` and warning report when applicable | `mask_<output>_path` paths |
| `phase` | input CSV in place, or `--csv_path_out` | `phase_errors.csv`; selection and QC exports | canonical `phase`, provenance columns, and `totalseg_*` predictions when used |
| `register` | `<input>_registered.csv` | `register_errors.csv`, `register_qc.csv`, one JSONL log per group, checkpoint/state files | Canonical image/mask paths in the configured moving or common-reference output space, original `source_*` paths, registration status/QC fields, and selected-space artifact paths |
| `radiomics` | `<input>_radiomics.csv` | `radiomics_errors.csv` | ROI-prefixed PyRadiomics features |

Defaults are relative to the input CSV or selected output directory. Explicit
paths are recommended in scheduled pipelines.

## Curation and selection exports

A `phase_curation` cleaning step (including the built-in `generic` ingest flow)
and the `phase` command write these additional tables:

| File | Contents | Intended use |
| --- | --- | --- |
| `<input_stem>_curated.csv` | Best eligible CT volume per phase or MR volume per sequence/phase slot, per exam (`selected_long`) | Optional input to subsequent processing stages |
| `qc_<input_stem>_curated.csv` | One row per exam and modality, with selection slots, candidate descriptions, scores, and provenance (`selected_wide`) | Review selections; do not use as a processing input |
| `qc_unresolved_phase.csv` | Rows unresolved after the configured strategies, before fallback, with raw evidence retained | Investigate unresolved phases, including rows excluded from selection |

The full main output retains every surviving row; selection does not reduce it.
Use it to process all candidates, or use the selected CSV for representative
volumes. Both must contain the image or mask paths needed by the next stage.

These filenames and their directory derive from the **input** CSV, even when
`--csv_path_out` names a different location. For `ingest`, the input is its
intermediate `dicom_index.csv`, producing `dicom_index_curated.csv` and
`qc_dicom_index_curated.csv`. With multiple cleaning inputs, the first supplies
the location and stem. Cleaning without a `phase_curation` step skips these exports.

Only `phase` supports `--selected_csv_path /path/to/selected.csv`; its wide QC
output then becomes `/path/to/qc_selected.csv`, with `qc_unresolved_phase.csv`
in that directory. Selection and QC files are replaced on subsequent exports,
including completed phase resumes; keep separate directories for runs whose
reports you want to preserve. See [Best series per exam](manifests.md#best-series-per-exam)
for grouping, ranking, and fallback behavior.

## Identity and traceability

The core identifiers are `patient_key`, `study_id`, and `series_id`. Parse also
retains `_patient_key_raw` when standardization is applied. `dicom_path` may
serialize multiple files for a volume or contain archive-aware locations;
consumers should not assume it is a single ordinary filesystem path.

Long-running stages use an internal `_source_idx` for stable resume and merge
behavior. It is removed from finalized user-facing tables.

## Image and mask paths

`nifti_path` identifies the converted image. Segmentation outputs are stored in
columns matching `mask_<output>_path`; their exact set is the union of the configured
CT and MR task outputs. A row only populates outputs from its matching
`segmentation.modalities` block, and rows whose modality is not configured keep
those columns empty. Paths may be absolute depending on the supplied output
directory, so moving a dataset can invalidate a table. If portability matters,
move artifacts and rewrite paths as one controlled operation.

Mask filepath columns must use the `mask_<output>_path` convention. Metrics and
registration status columns such as `mask_liver_volume` and `mask_affine_status`
are not mask paths and are excluded from automatic mask discovery. Registration
preserves original paths in `source_mask_<output>_path` columns; these backups are
also excluded from automatic discovery.

For existing cohort CSVs, rename only mask filepath columns, for example
`mask_liver` to `mask_liver_path` and `source_mask_liver` to
`source_mask_liver_path`, and update the registration mask-column settings in
custom manifests. Leave metric and status columns unchanged. Segmentation uses
the new names on subsequent runs; checkpoints from the old naming scheme are
invalidated.

When `segment --crop` is enabled, the files referenced by `nifti_path` and the
row's populated `mask_*_path` columns are cropped in place to the same voxel bounds.
Their affines are translated to preserve world coordinates; their CSV paths do
not change. Cropping occurs before segmentation post-processing. Requested
margins are clipped to the true image extent; no synthetic padding is added.

## Error tables

Error CSVs contain the failed source row plus an error message. A command can
complete while some rows fail, making these tables part of the expected output
rather than disposable logs. Check all of the following before downstream use:

1. expected input and output row counts;
2. missing `nifti_path` or `mask_*_path` values;
3. command-specific error and warning tables;
4. whether filtering intentionally reduced radiomics rows.

Checkpoint files and JSON state may appear beside the configured main/error
outputs during resumable runs. They are implementation artifacts, not cohort
tables, and should not be passed to the next stage.
