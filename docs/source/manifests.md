# Configuration

IMPERANDI configuration is expressed through dataset manifests and optional
Python hooks.

- A manifest is a JSON document that defines dataset-specific behavior for
  parsing, cleaning, segmentation, and radiomics.
- A hook is a Python callable referenced by the manifest when declarative JSON
  is not enough.

Pass either a built-in manifest name (`generic` or `operandi`) or a path to a
custom JSON file:

```bash
imperandi ingest --root_path ./dicom --manifest generic
imperandi segment ./nifti_index.csv --manifest ./site-a.json
```

Built-ins live under `src/imperandi/datasets_config/manifests/`. Treat them as
examples; keep project-specific configuration in a reviewed, versioned file
outside the package defaults.

## How configuration works

In practice, configuration usually follows this flow:

1. start from `generic.json` or `operandi.json`;
2. edit parse settings under `id_extraction`;
3. add parse hooks with `id_standardization` and `derived_columns` if needed;
4. define clean behavior under `cleaning.steps`;
5. configure optional `segmentation` and `radiomics`;
6. run IMPERANDI with `--manifest your_config.json`.

Use manifests for declarative settings that belong in JSON. Use hooks for
site-specific logic such as identifier normalization or metadata extraction
that would be awkward to encode as static rules.

## Manifest structure

```json
{
  "dataset_name": "site-a",
  "id_extraction": {
    "source": "auto",
    "force_dicom_read": false,
    "patient_key": { "from_tag": "PatientID", "fallback": "path" },
    "study_id": { "from_tag": "StudyInstanceUID", "fallback": "path" },
    "series_id": { "from_tag": "SeriesInstanceUID", "fallback": "path" }
  },
  "id_standardization": {
    "hook_module": "datasets_config.hooks.site_a",
    "function": "standardize_patient_key"
  },
  "derived_columns": [
    {
      "hook_module": "datasets_config.hooks.site_a",
      "function": "extract_from_patient_key",
      "from_column": "patient_key",
      "join_mode": "missing_only"
    }
  ],
  "cleaning": {
    "version": 1,
    "steps": [
      {
        "type": "hook",
        "function": "datasets_config.hooks.site_a:extract_from_patient_key",
        "source_columns": ["patient_key"]
      },
      { "type": "coalesce_date" },
      { "type": "coalesce_time" },
      { "type": "sop_class" },
      {
        "type": "filter",
        "kind": "keep",
        "scope": "row",
        "logic": "and",
        "rules": [
          { "column": "Modality", "op": "eq", "value": "CT" },
          { "column": "sop_class", "op": "eq", "value": "CTImageStorage" }
        ]
      },
      {
        "type": "build_volume_id",
        "preferred_columns": [
          "patient_key",
          "study_id",
          "series_id",
          "ImageType",
          "AcquisitionNumber"
        ],
        "fallback_columns": ["patient_key", "study_id", "series_id"]
      },
      { "type": "group_volumes" },
      { "type": "compute_volume_length" },
      {
        "type": "filter",
        "kind": "keep",
        "scope": "volume",
        "logic": "and",
        "rules": [
          { "column": "volume_length", "op": "gte", "value": 30.0 },
          { "column": "volume_length", "op": "lte", "value": 1700.0 }
        ]
      },
      { "type": "finalize" }
    ]
  },
  "segmentation": {
    "backend": "totalsegmentator",
    "tasks": [
      {
        "task": "total",
        "extra": { "roi_subset": ["liver"] }
      }
    ]
  },
  "radiomics": {
    "filters": {
      "totalseg_phase": ["portal_venous"]
    }
  }
}
```

## Top-level sections

`dataset_name`
Use a short stable label such as `site-a`, `operandi_v2`, or
`external_validation`.

`id_extraction`
Controls how `patient_key`, `study_id`, and `series_id` are derived during
`parse` and `ingest`.

Typical fields:

- `source`: `auto`, `tags`, or `path`
- `force_dicom_read`
- per-ID blocks with `from_tag` and `fallback`

Keep this aligned with the parse or ingest CLI options you review:
`--id_source`, `--force_dicom_read`, `--patient_key_from`,
`--study_id_from`, and `--series_id_from`.

`id_standardization`
Optional parse-time hook that rewrites `patient_key` after extraction.

`derived_columns`
Optional parse-time hooks that derive extra cohort columns from an existing
column such as `patient_key`.

`cleaning`
Manifest-driven cleaning pipeline. This is where clean-stage transforms and
filters now live.

`segmentation`
TotalSegmentator task and post-processing configuration.

`radiomics`
PyRadiomics settings plus optional cohort filters.

## Parse hooks

`id_standardization` and `derived_columns` are still parse/ingest-time hooks.
They use the same import style as before:

```json
{
  "hook_module": "datasets_config.hooks.site_a",
  "function": "standardize_patient_key"
}
```

At runtime this resolves:

```text
imperandi.datasets_config.hooks.site_a.standardize_patient_key
```

### `id_standardization`

An `id_standardization` hook receives one raw `patient_key` value and returns
the normalized value to store in `patient_key`.

- IMPERANDI preserves the original value in `_patient_key_raw`.
- If a non-empty raw key becomes empty or null, the row is flagged with
  `patient_key_std_failed`.

### `derived_columns`

A `derived_columns` hook receives the value from `from_column` for each row and
returns a mapping or `pandas.Series` of extra fields to join back into the
table.

- `join_mode: "missing_only"` adds only missing columns.
- `join_mode: "overwrite"` replaces existing columns.

If the same custom logic is needed during `clean`, add an explicit clean hook
step as well. Parse hooks do not automatically become clean steps.

## Cleaning schema

`cleaning` must define:

```json
{
  "version": 1,
  "steps": []
}
```

`clean` and the clean phase of `ingest` validate the step list and then run the
steps in order. The pipeline is intentionally flat and explicit: built-in
transforms are named directly by `type`, not wrapped in handler objects.

### Built-in step types

The current built-in step types are:

- `coalesce_date`
- `coalesce_time`
- `sop_class`
- `parse_image_type`
- `clean_scan_size`
- `normalize_string`
- `pixel_spacing_xy`
- `standardize_iop`
- `classify_acquisition_plane`
- `build_volume_id`
- `merge_volume_ids`
- `group_volumes`
- `compute_volume_length`
- `compute_visit_order`
- `compute_acquisition_order`
- `finalize`

Most built-ins accept either no extra fields or a small, step-specific set of
options such as `column`, `preferred_columns`, `fallback_columns`,
`group_columns`, `z_sources`, or `z_tolerance`. The built-in manifests are the
best reference examples.

### Hook steps

Use a clean hook step when the clean pipeline needs custom Python logic:

```json
{
  "type": "hook",
  "function": "datasets_config.hooks.site_a:extract_from_patient_key",
  "source_columns": ["patient_key"]
}
```

Rules:

- `function` must be a string in `module:function` form.
- Relative modules are resolved under `imperandi`, so
  `datasets_config.hooks.site_a:extract_from_patient_key` is valid.
- `source_columns` is required for hook steps.
- There is no `target_columns` field in the manifest.

Hook outputs are declared in Python with `@clean_hook(outputs=[...])`:

```python
from imperandi.ingest.hooks import clean_hook


@clean_hook(outputs=["center", "source", "tumor_type"])
def extract_from_patient_key(value):
    ...
```

For a single-output hook that rewrites `patient_key`, use:

```python
@clean_hook(outputs=["patient_key"])
```

When a clean hook rewrites `patient_key` from `source_columns: ["patient_key"]`,
IMPERANDI preserves `_patient_key_raw` and flags failed standardizations in
`patient_key_std_failed`.

### Filter steps

Filters are explicit objects with a step `type` of `filter`:

```json
{
  "type": "filter",
  "kind": "discard",
  "scope": "row",
  "logic": "or",
  "rules": [
    { "column": "ImageType", "op": "contains", "value": "LOCALIZER" },
    { "column": "SeriesDescription", "op": "icontains", "value": "scout" }
  ]
}
```

Required fields:

- `kind`: `keep` or `discard`
- `scope`: `row` or `volume`
- `logic`: `and` or `or`
- `rules`: non-empty list

Optional fields:

- `keep_null`: when `true`, rows with nulls in any referenced rule column are
  kept even if the rule expression would otherwise fail
- `name`: custom label used in logs

Supported rule operators:

- `eq`
- `ne`
- `in`
- `not_in`
- `contains`
- `icontains`
- `regex`
- `lt`
- `lte`
- `gt`
- `gte`
- `is_null`
- `not_null`

Use `scope: "row"` for per-instance filtering and `scope: "volume"` after
volume-level columns such as `volume_length` have been computed.

### Common clean patterns

Keep only CT rows:

```json
{
  "type": "filter",
  "kind": "keep",
  "scope": "row",
  "logic": "and",
  "rules": [
    { "column": "Modality", "op": "eq", "value": "CT" },
    { "column": "sop_class", "op": "eq", "value": "CTImageStorage" }
  ]
}
```

Discard multiple unwanted descriptions:

```json
{
  "type": "filter",
  "kind": "discard",
  "scope": "row",
  "logic": "or",
  "rules": [
    { "column": "SeriesDescription", "op": "contains", "value": "pelvis" },
    { "column": "SeriesDescription", "op": "contains", "value": "femur" }
  ]
}
```

Keep only acceptable volume lengths:

```json
{
  "type": "filter",
  "kind": "keep",
  "scope": "volume",
  "logic": "and",
  "keep_null": true,
  "rules": [
    { "column": "volume_length", "op": "gte", "value": 30.0 },
    { "column": "volume_length", "op": "lte", "value": 1700.0 }
  ]
}
```

## Segmentation

`segmentation.backend`
This selects the segmentation engine. The current supported value is
`totalsegmentator`.

`segmentation.tasks`
Defines which TotalSegmentator runs happen and how their outputs should be
named in IMPERANDI.

Useful keys include:

- `task`
- `extra`
- `output` or `outputs`
- `fetch_output` or `fetch_outputs`

Official references:
[TotalSegmentator subtasks guide](https://github.com/wasserth/TotalSegmentator#subtasks)
and [class details](https://github.com/wasserth/TotalSegmentator/blob/master/resources/class_details.md).

`segmentation.postprocess`
Optional mask merge and cleanup settings. Common keys include `merge_keys`,
`output`, `radius_mm`, `close`, `fill_holes`, `largest_cc`, and `on_failure`.

## Radiomics

`radiomics.pyradiomics`
Optional in-manifest PyRadiomics settings object. This is equivalent in spirit
to a PyRadiomics parameter YAML file.

Official guide:
[PyRadiomics customization docs](https://pyradiomics.readthedocs.io/en/latest/customization.html)

`radiomics.filters`
Optional row filters expressed as `column -> [allowed_value, ...]`.

## Precedence

- Parse and ingest CLI flags remain the authoritative runtime switches for
  `id_extraction` behavior.
- `radiomics.filters` in the manifest override conflicting CLI `--filter`
  values.
- `--skip_filter` disables both CLI and manifest radiomics filters.
- When a manifest contains `radiomics.pyradiomics`, it takes precedence over an
  explicit `--pyradiomics_settings` file.

## Validation advice

Before a full run:

1. run commands with `--dry-run`;
2. parse a representative sample first;
3. compare `patient_key` with `_patient_key_raw` when using normalization;
4. inspect the clean output after each major manifest change;
5. test custom hooks against nulls and malformed values.
