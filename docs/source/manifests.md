# Configuration

IMPERANDI configuration is defined in YAML manifests. JSON manifests are not
accepted. A manifest can reference Python hooks when declarative configuration
is insufficient for institution-specific identifiers or derived metadata.

Pass a built-in name (`generic` or `operandi`) or a YAML path:

```bash
imperandi ingest --root_path ./dicom --manifest generic
imperandi phase ./nifti_index.csv --manifest ./site-a.yaml
```

Built-ins live under `src/imperandi/datasets_config/manifests/`. Keep
institution-specific manifests in a reviewed, versioned project file.

## Main library stages

One manifest configures the main data path:

1. `id_extraction` controls DICOM identifier extraction during `parse`.
2. `id_standardization` and `derived_columns` define parse-time hooks.
3. `cleaning.steps` defines the ordered metadata-cleaning pipeline.
4. `phase_curation` defines canonical phase resolution during `clean` and
   after optional TotalSegmentator prediction during `phase`.
5. `segmentation` defines TotalSegmentator mask tasks and post-processing.
6. `radiomics` defines PyRadiomics settings and cohort filters.

The `cleaning.steps` order is executable configuration. In particular,
`modality_curation` should run after volume grouping and acquisition ordering,
and `finalize` should remain last.

## YAML manifest skeleton

```yaml
dataset_name: site-a

id_extraction:
  source: auto
  force_dicom_read: false
  patient_key: {from_tag: PatientID, fallback: path}
  study_id: {from_tag: StudyInstanceUID, fallback: path}
  series_id: {from_tag: SeriesInstanceUID, fallback: path}

id_standardization:
  hook_module: datasets_config.hooks.generic
  function: standardize_patient_key

phase_curation:
  strategies:
    - type: ontology
      name: site_ontology
      columns: [site_phase]
      mapping:
        pre: NATIVE
        art: ARTERIAL
        pv: PORTAL_VENOUS
    - type: rules
      name: metadata_rules
    - type: totalsegmentator
      name: totalsegmentator_prediction
      column: totalseg_phase
      modalities: [CT]
      confidence_columns: [totalseg_probability, totalseg_confidence]
      mapping:
        native: NATIVE
        arterial_early: ARTERIAL
        arterial_late: ARTERIAL
        portal_venous: PORTAL_VENOUS
        delayed: DELAYED
  unresolved_labels: ["", OTHER, UNKNOWN, UNCLASSIFIED, NONE]
  fallback: OTHER

cleaning:
  version: 1
  steps:
    - type: hook
      function: "datasets_config.hooks.generic:standardize_patient_key"
      source_columns: [patient_key]
    - type: coalesce_date
    - type: coalesce_time
    - type: build_volume_id
    - type: group_volumes
    - type: compute_volume_length
    - type: compute_visit_order
    - type: compute_acquisition_order
    - type: modality_curation
    - type: finalize

segmentation:
  backend: totalsegmentator
  modalities:
    CT:
      tasks:
        - task: total
          extra:
            roi_subset: [liver]
        - task: liver_lesions
          output: liver_tumor
          fetch_output: liver_lesions
      postprocess:
        merge_keys: [liver, liver_tumor]
        output: liver
    MR:
      tasks:
        - task: total_mr
          extra:
            roi_subset: [liver]
        - task: liver_lesions_mr
          output: liver_tumor
          fetch_output: liver_lesions
      postprocess:
        merge_keys: [liver, liver_tumor]
        output: liver

radiomics:
  pyradiomics:
    setting:
      binWidth: 25
    imageType:
      Original: {}
  filters:
    phase: [ARTERIAL, PORTAL_VENOUS]
```

Copy a built-in YAML manifest as the starting point because the full built-in
cleaning pipeline contains the geometry, modality, volume, and quality-control
steps omitted from this abbreviated skeleton.

## Phase curation and fallback

`phase_curation.strategies` is an ordered fallback chain. Each strategy is
optional, but the list must contain at least one strategy. The first strategy
that produces a value not listed in `unresolved_labels` wins.

The resolver writes:

- `phase`: canonical uppercase phase, such as `ARTERIAL` or `PORTAL_VENOUS`;
- `phase_source`: configured strategy name or `fallback`;
- `phase_confidence`: rule, ontology, or predictor confidence when available;
- `phase_reason`: concise provenance for the decision.

The metadata engines also retain their unmodified result in `rule_phase`,
`rule_phase_confidence`, and `rule_phase_reason`. This lets the post-conversion
`phase` command apply the same manifest without losing clean-time evidence.

### Explicit ontology

Use `type: ontology` when the input already carries a controlled site label.
`columns` is checked in order. `mapping` performs exact, case-insensitive value
mapping; it does not run regexes or substring matching.

```yaml
phase_curation:
  strategies:
    - type: ontology
      columns: [site_phase, reviewed_phase]
      confidence: high
      mapping:
        sans injection: NATIVE
        arteriel: ARTERIAL
        portal: PORTAL_VENOUS
  fallback: OTHER
```

### Metadata rules

Use `type: rules` to consume IMPERANDI's CT and MRI metadata rule engines.
These rules use sequence descriptions, timing, acquisition order, and other
DICOM-derived features. An optional `mapping` can rename their canonical
outputs, although the built-in labels normally need no mapping.

```yaml
phase_curation:
  strategies:
    - type: rules
  fallback: OTHER
```

### TotalSegmentator prediction

Use `type: totalsegmentator` to consume or generate a prediction. The `phase`
command invokes TotalSegmentator only for rows that reach this strategy. If it
appears after ontology and rules, rows already resolved by either earlier
strategy skip model inference.

```yaml
phase_curation:
  strategies:
    - type: rules
    - type: totalsegmentator
      column: totalseg_phase
      modalities: [CT]
      confidence_columns: [totalseg_probability, totalseg_confidence]
      mapping:
        native: NATIVE
        arterial_early: ARTERIAL
        arterial_late: ARTERIAL
        portal_venous: PORTAL_VENOUS
  fallback: OTHER
```

Put TotalSegmentator first when its prediction should override metadata rules.
Set `fallback: null` to leave unresolved rows empty instead of assigning a
sentinel phase. The bundled predictor is CT-specific, so the strategy defaults
to `modalities: [CT]`; MRI rows continue to later strategies without invoking
the model.

## Identity and hooks

`id_extraction` controls raw patient, study, and series identifiers. Typical
`from_tag` values are `PatientID`, `PatientName`, `StudyInstanceUID`, and
`SeriesInstanceUID`; `fallback: path` uses the source path when a tag is absent.

`id_standardization` references a hook that rewrites `patient_key`.
`derived_columns` can derive fields such as `center`, `source`, or `tumor_type`.
Clean-time hook steps use `module:function` paths and the callable must declare
its outputs with `@clean_hook`:

```python
from imperandi.ingest.hooks import clean_hook


@clean_hook(outputs=["center", "source"])
def extract_site_fields(patient_key):
    return {"center": "SITE_A", "source": "clinical"}
```

```yaml
cleaning:
  version: 1
  steps:
    - type: hook
      function: "datasets_config.hooks.site_a:extract_site_fields"
      source_columns: [patient_key]
```

Only use manifests and hook modules from trusted sources; hook references load
and execute Python code.

## Segmentation and radiomics

`segmentation.modalities` is required and maps `CT` and/or `MR` to independent
TotalSegmentator task lists. `MRI` is accepted as an alias for `MR` in input
tables and manifest keys. CT task names must not end in `_mr`; MR task names
must resolve to an `_mr` task. Rows without a configured modality are retained
but skipped, and only models needed by modalities present in the cohort are
prefetched.

Optional task keys include `extra`, `output`, `outputs`, `fetch_output`, and
`fetch_outputs`. Each modality may define its own `postprocess` block to merge
logical masks and apply closing, hole filling, and largest-component cleanup.
The same logical outputs may be used across modalities, for example mapping
both `liver_lesions` and `liver_lesions_mr` to `mask_liver_tumor`.

`radiomics.pyradiomics` follows the normal PyRadiomics parameter structure.
`radiomics.filters` maps an existing cohort column to its accepted values. Use
the canonical `phase` output for phase-based filtering.

## Validation checklist

Before a full cohort run:

1. load the YAML manifest by path and by built-in name where applicable;
2. run `ingest --dry-run` and `phase --dry-run`;
3. process a small cohort through parse, clean, convert, segment, and phase;
4. inspect `phase`, `phase_source`, and `phase_reason` distributions;
5. confirm ontology values are exact and TotalSegmentator runs only on expected
   fallback rows;
6. confirm CT and MR rows select their respective TotalSegmentator models;
7. verify radiomics filters use canonical phase values.
