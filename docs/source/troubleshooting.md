# Troubleshooting

## The `segment` or `phase` command reports missing dependencies

Install the TotalSegmentator feature set in the same environment that owns the
`imperandi` executable:

```bash
python -m pip install -e ".[segment]"
python -m imperandi segment --help
```

Using `python -m imperandi` is a quick way to detect when `pip` and the shell
entry point refer to different environments.

## Radiomics cannot import PyRadiomics

Install `.[radiomics]`. The project currently sources PyRadiomics from its Git
repository, so installation requires Git and network access. Confirm the input
has `nifti_path` and at least one populated `mask_*` column.

## No DICOM files are found

- Check glob quoting so the shell does not expand it unexpectedly.
- Try `--force_dicom_read` for non-conformant exports.
- Raise `--archive_max_depth` only when archives are intentionally nested.
- Add `--snapshot_tags` to inspect what is readable in a representative sample.
- Use `--verbose` and reduce to a small root while diagnosing.

## IDs are empty or unexpected

Use `--id_source tags` to require tag-based identity or `--id_source path` for
a stable directory hierarchy. Override the tag keywords with
`--patient_key_from`, `--study_id_from`, and `--series_id_from`. When a manifest
standardizes patient keys, compare `patient_key` with `_patient_key_raw`.

## Cleaning removes too many volumes

Inspect `Modality`, `ImageType`, `SeriesDescription`, orientation, spacing, and
computed volume length in the parsed CSV, then inspect the selected manifest's
`cleaning.steps` in order. Most unexpected losses come from one of these:

- a row-scope keep filter that is too strict;
- a row-scope discard filter that matches too broadly;
- a geometry-derived column such as `PixelSpacingXY` or `acquisition_axis`;
- a volume-scope filter on `volume_length`.

For legitimately short or long protocols, edit the relevant volume-scope filter
inside the manifest rather than looking for CLI threshold flags. Keep changes
narrow until you know which step caused the loss.

## Conversion fails for archive-backed series

Ensure the archive remains at the same location recorded during parse. Set
`--archive_cache_dir` to a filesystem with enough free space and permissions.
`--keep_archive_cache` is useful for inspection but can consume substantial
storage. Review `conv_errors.csv` for the affected rows.

## A resumed run uses stale results

Resume only occurs for compatible state, but lightweight fingerprints cannot
detect every in-place content change. Use `--strict_resume` when inputs may have
changed without a path or metadata change, or `--no_resume` to deliberately
start fresh. Do not combine outputs from manifests with different task
definitions.

## Multiprocessing hangs or exhausts memory

Reduce `--num_workers` and, for segmentation, keep the default
`--start_method spawn`. Increase `--timeout_sec` only after confirming a volume
is making progress. Parallel GPU workers also multiply model and image memory
requirements.

## The documentation build fails during API imports

Build from the repository root after installing the package and
`docs/requirements.txt`. The Sphinx configuration mocks heavy optional modules,
but base runtime dependencies must still be installed through `pip install -e
.`. Re-run with `-W --keep-going` to see all warnings in one build.
