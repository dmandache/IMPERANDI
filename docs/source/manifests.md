# Manifests

A manifest is a JSON document that captures dataset-specific identity rules,
hooks, segmentation tasks, post-processing, and radiomics configuration. Pass a
built-in name (`generic` or `operandi`) or a path to a JSON file:

```bash
imperandi ingest --root_path ./dicom --manifest generic
imperandi segment ./nifti_index.csv --manifest ./site-a.json
```

Built-ins live under `src/imperandi/datasets_config/manifests/`. Treat them as
examples; keep institution-specific configuration in a reviewed, versioned
file rather than editing package defaults in place.

## Structure

```json
{
  "dataset_name": "site-a",
  "id_extraction": {
    "source": "auto",
    "force_dicom_read": false,
    "patient_key": {"from_tag": "PatientID", "fallback": "path"},
    "study_id": {"from_tag": "StudyInstanceUID", "fallback": "path"},
    "series_id": {"from_tag": "SeriesInstanceUID", "fallback": "path"}
  },
  "id_standardization": {
    "hook_module": "datasets_config.hooks.generic",
    "function": "standardize_patient_key"
  },
  "segmentation": {
    "backend": "totalsegmentator",
    "tasks": [
      {"task": "total", "extra": {"roi_subset": ["liver"]}},
      {
        "task": "liver_lesions",
        "output": "liver_tumor",
        "fetch_output": "liver_lesions"
      }
    ],
    "postprocess": {
      "merge_keys": ["liver", "liver_tumor"],
      "output": "liver",
      "radius_mm": 5.0,
      "largest_cc": true,
      "fill_holes": true,
      "close": true
    }
  },
  "radiomics": {
    "pyradiomics": {
      "setting": {"binWidth": 25},
      "imageType": {"Original": {}}
    },
    "filters": {"totalseg_phase": ["portal_venous", "arterial_late"]}
  }
}
```

## Hooks

`id_standardization` resolves a callable below the `imperandi` package and
applies it to the raw patient key. A `derived_columns` list can similarly call
functions that return mappings of extra fields. Each derived entry names a
`from_column` and may set `join_mode` to `missing_only` (default) or
`overwrite`.

Custom hooks are executable Python, not passive configuration. Review them as
code, test them against malformed and missing identifiers, and never load an
untrusted manifest that points to untrusted modules.

## Precedence

Explicit parse CLI values override manifest-derived parse defaults. For
radiomics, CLI `--filter` values take precedence per column, while manifest
filters fill columns not specified on the CLI. `--skip_filter` disables all of
them. Manifest PyRadiomics settings take precedence when both a manifest
settings object and `--pyradiomics_settings` are supplied.

## Validation advice

Before a full run:

1. run each command with `--dry-run`;
2. parse a small representative sample;
3. compare raw and standardized identifiers;
4. verify expected mask column names;
5. confirm radiomics filters retain the intended phases.

