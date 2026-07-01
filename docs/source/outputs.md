# Outputs

CSV tables are the pipeline's audit spine. Each stage preserves existing
columns and adds or normalizes the fields needed downstream.

| Stage | Main output | Error/auxiliary output | Key additions |
|---|---|---|---|
| `parse` | `dicom_index.csv` | `dicom_index_errors.csv`; optional `dicom_tags_snapshot.ndjson` | IDs, `dicom_path`, selected DICOM tags |
| `clean` | `<input>_clean.csv` | — | volume grouping, normalized date/time, ordering and geometry fields |
| `ingest` | `dicom_index_clean.csv` | parse artifacts | parsed and curated volume rows |
| `convert` | `nifti_index.csv` | `conv_errors.csv` | `nifti_path` |
| `segment` | input CSV in place, or `--csv_path_out` | `seg_errors.csv` and warning report when applicable | `mask_<output>` paths |
| `phase` | input CSV in place, or `--csv_path_out` | `phase_errors.csv` | `totalseg_*`, including `totalseg_phase` |
| `radiomics` | `<input>_radiomics.csv` | `radiomics_errors.csv` | ROI-prefixed PyRadiomics features |

Defaults are relative to the input CSV or selected output directory. Explicit
paths are recommended in scheduled pipelines.

## Identity and traceability

The core identifiers are `patient_key`, `study_id`, and `series_id`. Parse also
retains `_patient_key_raw` when standardization is applied. `dicom_path` may
serialize multiple files for a volume or contain archive-aware locations;
consumers should not assume it is a single ordinary filesystem path.

Long-running stages use an internal `_source_idx` for stable resume and merge
behavior. It is removed from finalized user-facing tables.

## Image and mask paths

`nifti_path` identifies the converted CT image. Segmentation outputs are stored
in columns beginning with `mask_`; their exact set follows the manifest tasks.
Paths may be absolute depending on the supplied output directory, so moving a
dataset can invalidate a table. If portability matters, move artifacts and
rewrite paths as one controlled operation.

## Error tables

Error CSVs contain the failed source row plus an error message. A command can
complete while some rows fail, making these tables part of the expected output
rather than disposable logs. Check all of the following before downstream use:

1. expected input and output row counts;
2. missing `nifti_path` or `mask_*` values;
3. command-specific error and warning tables;
4. whether filtering intentionally reduced radiomics rows.

Checkpoint files and JSON state may appear beside the configured main/error
outputs during resumable runs. They are implementation artifacts, not cohort
tables, and should not be passed to the next stage.

