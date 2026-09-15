# Configuration

IMPERANDI configuration is defined in YAML manifests. JSON manifests are not
accepted. A manifest can reference Python hooks when declarative configuration
is insufficient for institution-specific identifiers or derived metadata.

Pass a built-in name (`generic`) or a YAML path:

```bash
imperandi ingest --root_path ./dicom --manifest generic
imperandi phase ./nifti_index.csv --manifest ./site-a.yaml
```

Built-ins live under `src/imperandi/builtin_datasets_config/manifests/`.
Project-local OPERANDI configuration lives under `dataset_configs/` and is not
installed with the package. Load it only by its explicit path, for example
`--manifest ./dataset_configs/manifests/operandi.yaml`. Keep other
institution-specific manifests in a reviewed, versioned project directory.

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
  hook_module: imperandi.builtin_datasets_config.hooks.generic
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
  steps:
    - type: hook
      function: "imperandi.builtin_datasets_config.hooks.generic:standardize_patient_key"
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
        on_failure: warn_only
        operations:
          - op: union
            inputs: [liver, liver_tumor]
            output: liver
          - op: close
            input: liver
            output: liver
            radius_mm: 5.0
          - op: fill_holes
            input: liver
            output: liver
          - op: largest_cc
            input: liver
            output: liver
          - op: intersection
            inputs: [liver_tumor, liver]
            output: liver_tumor
    MR:
      tasks:
        - task: total_mr
          extra:
            roi_subset: [liver]
        - task: liver_lesions_mr
          output: liver_tumor
          fetch_output: liver_lesions
      postprocess:
        on_failure: warn_only
        operations:
          - op: union
            inputs: [liver, liver_tumor]
            output: liver
          - op: close
            input: liver
            output: liver
            radius_mm: 5.0
          - op: fill_holes
            input: liver
            output: liver
          - op: largest_cc
            input: liver
            output: liver
          - op: intersection
            inputs: [liver_tumor, liver]
            output: liver_tumor

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
  steps:
    - type: hook
      function: "site_config.hooks.site_a:extract_site_fields"
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
`fetch_outputs`. Each modality may define its own `postprocess` block to combine
logical masks and apply morphological operations in a configurable order.
The same logical outputs may be used across modalities, for example mapping
both `liver_lesions` and `liver_lesions_mr` to `mask_liver_tumor`.

Task output declarations need not enumerate every mask a backend produces.
For example, `task: total` may supply `liver` without an explicit `output` or
ROI subset. The worker resolves undeclared input names against masks produced
by the tasks or their actual filenames, including existing masks reused without
rewriting them. Explicit `fetch_output` aliases are respected.

### Sequential mask operations

Set `postprocess.operations` to an ordered list under each modality. Every step
names an `op`, its `input` (one mask) or `inputs` (a list), and an `output`.
Cleanup settings belong to the individual operations that use them.
Inputs can refer to task outputs or results of earlier steps. Bare logical names,
`mask_*` column names, and `.nii.gz` filenames are accepted and normalized to
logical names.

For example, clean the liver mask, combine it with the tumor mask, clean the
combined result, and subtract the tumor to create a separate tissue mask:

```yaml
# Under segmentation.modalities.CT (or MR), alongside tasks:
postprocess:
  operations:
    - op: largest_cc
      input: liver
      output: liver_clean
    - op: union
      inputs: [liver_clean, liver_tumor]
      output: liver_all
    - op: close
      input: liver_all
      output: liver_all
      radius_mm: 3.0
    - op: fill_holes
      input: liver_all
      output: liver_all
    - op: difference
      inputs: [liver_all, liver_tumor]
      output: liver_without_tumor
```

| `op` | Inputs | Behavior |
| --- | --- | --- |
| `union` (alias `or`) | One or more | Foreground in any input; one input copies the mask |
| `intersection` (alias `and`) | One or more | Foreground in every input |
| `difference` (alias `subtract`) | Two or more | First input minus all remaining inputs |
| `xor` | One or more | Foreground in an odd number of inputs |
| `not` | One | Invert foreground/background within the image volume |
| `dilate` / `erode` | One | Expand / shrink foreground |
| `open` / `close` | One | Erosion then dilation / dilation then erosion |
| `fill_holes` | One | Fill enclosed background regions |
| `largest_cc` | One | Keep the largest connected foreground component |

Morphology also accepts the names `dilation`, `erosion`, `opening`, and `closing`;
`largest_component` aliases `largest_cc`.

For `dilate`, `erode`, `open`, and `close`, `radius_mm` defaults to `1.0` and must
be finite and non-negative. The spherical neighborhood uses each axis's voxel
spacing, so thick slices are handled independently of in-plane resolution.
Alternatively, set a finite, non-negative `radius_vox` to use a sphere measured
in voxels, independent of spacing. Supply only one of `radius_mm` and `radius_vox`.
Radius zero is an identity operation. `iterations` defaults to `1` and must be a
positive integer; for opening/closing it repeats each erosion/dilation phase.
Outside the image volume is treated as background.

`fill_holes` and `largest_cc` accept `connectivity: 1`, `2`, or `3` (face,
face-and-edge, or face-edge-and-corner neighbors). Defaults are `1` for hole
filling and `3` for largest-component selection. Empty masks remain empty under
morphology and component selection.

Masks must be 3-D; logical combinations require matching shapes and affines.
Positive voxels are treated as foreground. Each distinct output is saved as a
binary `uint8` NIfTI and appears in the corresponding `mask_<output>` CSV column,
including intermediate outputs. New names create `<output>.nii.gz`; existing
logical task names write to their configured `fetch_output` filename.
Reusing an output name replaces its current value for subsequent steps and saves
its final value.
Only explicitly named outputs are written. To clip an original mask to a merged
result, add an `intersection` step with that original mask as the output.
For example, an output named `liver_tumor` mapped to `fetch_output: liver_lesions`
updates `liver_lesions.nii.gz`, and `mask_liver_tumor` still points to that file.

The full sequence is computed before any output is written. Missing or
unreadable input files always fail the row. Geometry errors also fail by default;
`postprocess.on_failure: warn_only` records a warning and returns the input masks
without reporting derived outputs when geometry prevents the sequence completing.
Unknown operations, inappropriate parameters, and references to later results
are rejected during manifest validation. If an initial backend mask is also the
output of a later step, declare it in the task's `output`/`outputs` or ROI subset
so validation can distinguish it from a forward reference.
`operations: []` disables postprocessing.

Completed sequences are reused when their operation list and input/output file
signatures match the local completion record. If a sequence overwrites an input,
use `--force` to regenerate backend masks before applying a changed sequence;
otherwise the changed sequence starts from the current files.

### Merge, clean, and clip masks

This sequence combines the masks, applies closing, fills holes, keeps the largest
component, and clips each source mask to the cleaned merge:

```yaml
postprocess:
  on_failure: warn_only
  operations:
    - op: union
      inputs: [liver, liver_tumor]
      output: liver_all
    - op: close
      input: liver_all
      output: liver_all
      radius_mm: 5.0
    - op: fill_holes
      input: liver_all
      output: liver_all
    - op: largest_cc
      input: liver_all
      output: liver_all
    - op: intersection
      inputs: [liver, liver_all]
      output: liver
    - op: intersection
      inputs: [liver_tumor, liver_all]
      output: liver_tumor
```

Include the cleanup operations needed for the dataset. A `union` with one input
copies the mask before cleanup. If the merged output
replaces a source (for example, `output: liver`), write the union and cleanup
steps to that name and omit its final intersection so the merge stays intact.

### Radiomics

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
