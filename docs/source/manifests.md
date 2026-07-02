# Configuration

IMPERANDI configuration is done through manifests and hooks.

- A manifest is a JSON document that declares dataset-specific settings such as
  identifier extraction, segmentation tasks, post-processing, and radiomics
  options.
- A hook is a Python callable referenced by the manifest when declarative JSON
  is not enough, for example to normalize patient identifiers or derive cohort
  columns from existing values.

Pass a built-in manifest name (`generic` or `operandi`) or a path to a JSON
file:

```bash
imperandi ingest --root_path ./dicom --manifest generic
imperandi segment ./nifti_index.csv --manifest ./site-a.json
```

Built-ins live under `src/imperandi/datasets_config/manifests/`. Treat them as
examples; keep institution-specific configuration in a reviewed, versioned
file rather than editing package defaults in place. This Markdown page is the
reference for how configuration is expressed through manifest JSON plus hook
functions.

## How configuration works

In practice, configuration usually follows this flow:

1. start from a built-in manifest or a project-owned copy;
2. edit manifest keys for parsing, segmentation, and radiomics behavior;
3. point manifest hook entries at Python functions when you need custom logic;
4. run IMPERANDI with `--manifest your_config.json`.

Use manifests for declarative settings that can live safely in versioned JSON.
Use hooks for executable logic such as parsing institution-specific identifiers,
mapping cohort metadata, or other transformations that need Python code.

## Manifest structure

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
  "derived_columns": [
    {
      "hook_module": "datasets_config.hooks.operandi",
      "function": "extract_from_patient_key",
      "from_column": "patient_key",
      "join_mode": "missing_only"
    }
  ],
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

## Customize a manifest file

The usual workflow is:

1. copy `src/imperandi/datasets_config/manifests/generic.json` or
   `src/imperandi/datasets_config/manifests/operandi.json`;
2. save it as a project-owned file such as `site-a.json`;
3. edit the JSON keys that match your dataset;
4. pass the file path with `--manifest ./site-a.json`.

The easiest way to read the manifest is by configuration area:

### Identity and cohort metadata

`dataset_name`
Use this as a short, stable label for the configuration. It is mainly for
readability and review, so names such as `site-a`, `operandi_v2`, or
`external_validation` are usually better than vague labels.

`id_extraction`
This controls how raw patient, study, and series identifiers are obtained.
The usual fields are `source` (`auto`, `tags`, or `path`),
`force_dicom_read`, and the per-ID blocks `patient_key`, `study_id`, and
`series_id` with `from_tag` and `fallback`.
Typical `from_tag` values are DICOM keywords such as `PatientID`,
`PatientName`, `StudyInstanceUID`, and `SeriesInstanceUID`.
Typical `fallback` usage is `"path"` when a tag is missing or unreliable.
Keep this section aligned with the parse or ingest CLI options you actually run:
`--id_source`, `--force_dicom_read`, `--patient_key_from`,
`--study_id_from`, and `--series_id_from`.

`id_standardization`
Use this when raw identifiers need cleanup after extraction.
It points to a hook that rewrites `patient_key` into your canonical cohort ID,
for example by removing prefixes, normalizing zero-padding, or extracting the
parts of the string that matter for downstream grouping.

`derived_columns`
Use this when you want to compute extra cohort columns from an existing value.
Each entry can define `hook_module`, `function`, `from_column`, and
`join_mode` (`missing_only` or `overwrite`).
This is a good place to derive fields such as `center`, `source`,
`tumor_type`, or other site-specific metadata from a patient key or similar
identifier.

### Segmentation

`segmentation.backend`
This selects the segmentation engine. In the current implementation, the
supported value is `totalsegmentator`.

`segmentation.tasks`
This section defines which TotalSegmentator runs happen and how IMPERANDI names
their outputs.
Each task entry needs at least `task`.
Common optional keys are `extra` for backend kwargs,
`output` or `outputs` for the logical mask names you want in IMPERANDI, and
`fetch_output` or `fetch_outputs` when the backend-produced filenames differ
from your preferred output names.
Inside `extra`, common runtime options are `roi_subset`, `roi_subset_robust`,
`fast`, and `fastest`, but other supported TotalSegmentator kwargs can also be
passed through.
If you use `roi_subset` or `roi_subset_robust`, IMPERANDI can infer mask names
from those class names, but explicit `output` names are usually clearer and
more stable for downstream CSV columns.
Official references:
[TotalSegmentator subtasks guide](https://github.com/wasserth/TotalSegmentator#subtasks)
and [class details](https://github.com/wasserth/TotalSegmentator/blob/master/resources/class_details.md).
To inspect the exact capabilities of your installed version, run
`totalseg_info --list-tasks` and `totalseg_info --classes -ta total`.

### Segmentation Mask Postprocessing

`segmentation.postprocess`
Use this section when several masks should be merged or cleaned after
segmentation.
`merge_keys` is required when post-processing is enabled and should name the
logical outputs to combine, such as `liver` and `liver_tumor`
or their `mask_*` column equivalents.
Common options are `output` for the merged mask name, `radius_mm` for
morphological closing, `close`, `fill_holes`, and `largest_cc` for cleanup, and
`on_failure` with `warn_only` or `fail`.

### Radiomics

`radiomics.pyradiomics`
Use this when you want the manifest itself to carry your feature extraction
settings instead of referencing a separate YAML file.
Common top-level sections are `setting`, `imageType`, and `featureClass`,
plus filter-specific keys such as LoG `sigma`.
This is the manifest equivalent of a PyRadiomics parameter file and is useful
when you want one versioned configuration file to describe both cohort logic
and radiomics behavior.
Official guide:
[PyRadiomics customization and parameter file docs](https://pyradiomics.readthedocs.io/en/latest/customization.html).

`radiomics.filters`
Use this when radiomics should run only on a subset of rows.
The format is `column_name -> [allowed_value, ...]`.
Typical columns include `totalseg_phase`, `phase`, `center`, `source`, or any
other cohort column already present in the CSV.
This is the manifest equivalent of repeated CLI filters such as
`--filter totalseg_phase=portal_venous,arterial_late`.

Example:

```json
{
  "dataset_name": "site-a",
  "id_extraction": {
    "source": "auto",
    "patient_key": {"from_tag": "PatientID", "fallback": "path"},
    "study_id": {"from_tag": "StudyInstanceUID", "fallback": "path"},
    "series_id": {"from_tag": "SeriesInstanceUID", "fallback": "path"}
  },
  "id_standardization": {
    "hook_module": "datasets_config.hooks.site_a",
    "function": "standardize_patient_key"
  },
  "radiomics": {
    "filters": {"totalseg_phase": ["portal_venous"]}
  }
}
```

Keep custom manifests outside package defaults so they can be reviewed,
versioned, and reused across runs.

## Hooks

`id_standardization` resolves a callable below the `imperandi` package and
applies it to the raw patient key. A `derived_columns` list can similarly call
functions that return mappings of extra fields. Each derived entry names a
`from_column` and may set `join_mode` to `missing_only` (default) or
`overwrite`.

At runtime, IMPERANDI imports hooks as:

```text
imperandi.<hook_module>.<function>
```

That means this manifest block:

```json
{
  "hook_module": "datasets_config.hooks.generic",
  "function": "standardize_patient_key"
}
```

loads `imperandi.datasets_config.hooks.generic.standardize_patient_key`.

### `id_standardization` hooks

An `id_standardization` hook receives one raw `patient_key` value and should
return the normalized value to write back into `patient_key`.

- IMPERANDI preserves the original value in `_patient_key_raw`.
- If a non-empty raw key becomes empty or `null`, the row is flagged with
  `patient_key_std_failed`.
- The built-in `generic` hook extracts numeric groups and joins them with `-`.
- The built-in `operandi` hook applies project-specific parsing rules for that
  dataset.

### `derived_columns` hooks

A `derived_columns` hook receives the value from `from_column` for each row and
returns a mapping or `pandas.Series` of new fields to join back into the table.

- `join_mode: "missing_only"` adds only columns that do not already exist.
- `join_mode: "overwrite"` replaces existing columns with the derived values.
- The built-in `operandi` hook `extract_from_patient_key` derives
  `center`, `source`, and `tumor_type` from the standardized patient key.

Example:

```json
{
  "derived_columns": [
    {
      "hook_module": "datasets_config.hooks.operandi",
      "function": "extract_from_patient_key",
      "from_column": "patient_key",
      "join_mode": "missing_only"
    }
  ]
}
```

### Writing a custom hook

Create an importable module under `src/imperandi/`, for example
`src/imperandi/datasets_config/hooks/site_a.py`:

```python
def standardize_patient_key(value):
    if value is None:
        return None
    return str(value).strip().upper()
```

Then reference it from your manifest:

```json
{
  "id_standardization": {
    "hook_module": "datasets_config.hooks.site_a",
    "function": "standardize_patient_key"
  }
}
```

Custom hooks are executable Python, not passive configuration. Review them as
code, test them against malformed and missing identifiers, and never load an
untrusted manifest that points to untrusted modules.

## Precedence

For radiomics, manifest `radiomics.filters` are merged with CLI `--filter`
values; when both specify the same column, the manifest values win.
`--skip_filter` disables both CLI and manifest filters. Manifest
`radiomics.pyradiomics` settings take precedence when both a manifest settings
object and `--pyradiomics_settings` are supplied.

For parse and ingest identity controls, the runtime CLI flags remain the
authoritative switches to review: `--id_source`, `--force_dicom_read`,
`--patient_key_from`, `--study_id_from`, and `--series_id_from`.

## Validation advice

Before a full run:

1. run each command with `--dry-run`;
2. parse a small representative sample;
3. compare raw and standardized identifiers;
4. verify expected mask column names;
5. confirm radiomics filters retain the intended phases.
