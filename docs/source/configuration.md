# Project configuration

IMPERANDI uses strict YAML project files validated by Pydantic. Configuration
describes intended cohort policy and backend settings; stage order and artifact
wiring are owned by the pipeline.

## Profiles and overrides

Set `project.profile` to load a built-in baseline:

```yaml
project:
  name: site-a-liver
  profile: liver_ct_mri
```

The loader recursively merges mappings. Scalars and lists in project YAML
replace profile values. Relative paths are resolved from the project file's
directory. Use `imperandi config resolve` to inspect the exact result.

## Reference project

```yaml
version: 2

project:
  name: site-a-liver
  profile: liver_ct_mri

input:
  sources: [/data/site-a/dicom]
  archive_depth: 3

output:
  root: ./results
  table_format: parquet
  publish_formats: [parquet, csv]

identity:
  source:
    patient_id_columns: [PatientID]
    namespace_columns: [site_id, IssuerOfPatientID]
    fallback:
      columns: []
      on_missing: error
  normalization:
    strip: true
    case: upper
    collapse_whitespace: true
  canonical:
    strategy: hmac
    hmac:
      secret_env: IMPERANDI_ID_SECRET
      namespace: site-a/liver-v1
      prefix: P
      length: 20
  validation:
    source_collision: error
    multiple_source_ids_per_patient: allow
  sensitive_fields:
    persist_raw_identifiers: separate_table

annotations:
  ontologies: []
  rule_packs:
    - builtin:liver_ct
    - builtin:liver_mri
  contextual_strategies:
    - art_port
    - mask_multiart
    - generic_dynamic_volume_order

phase_prediction:
  enabled: false
  backend: totalsegmentator
  modalities: [CT]
  scope: unresolved
  minimum_confidence: 0.6
  resolution:
    precedence:
      - phase_ontology
      - phase_rules_explicit
      - phase_rules_inferred
      - phase_image
    disagreement: flag

selection:
  required_slots:
    CT: [CT_NATIVE, CT_ARTERIAL, CT_PORTAL_VENOUS]
    MR: [MR_T2, MR_DWI, MR_T1_NATIVE, MR_T1_ARTERIAL]
  precedence:
    - slot_ontology
    - slot_rules_explicit
    - slot_rules_inferred
    - slot_image
  disagreement: flag

conversion:
  enabled: true

segmentation:
  enabled: true
  tasks:
    - id: liver_ct
      backend: totalsegmentator
      modality: CT
      task: total
      output: liver
    - id: liver_mr
      backend: totalsegmentator
      modality: MR
      task: total_mr
      output: liver

registration:
  enabled: false
  transform: rigid_affine
  pairs: ./registration_pairs.parquet

radiomics:
  enabled: false
  settings: ./Params.yaml
  slots: [CT_PORTAL_VENOUS, MR_T2]
  masks: [liver]

execution:
  workers: 4
  resume: true
  checkpoint_every_rows: 100
  checkpoint_every_seconds: 300
  log_level: INFO
```

## Input and output

`input.sources` accepts DICOM roots/globs and pre-indexed `.csv` or `.parquet`
tables. Archives are discovered through the existing bounded archive reader.

`output.table_format` accepts exactly `csv` or `parquet` and controls stage
tables. `output.publish_formats` controls final cohort copies and can contain
either or both formats; when omitted, it defaults to `output.table_format`. A
large explicit CSV inventory triggers a warning and still runs as CSV.

The CSV warning threshold is deliberately not configurable per project. Adding
`csv_warning_threshold_files` or any other unknown field causes validation to
fail.

## Identity

Identity configuration separates four decisions:

| Area | Purpose |
|---|---|
| `source` | Ordered raw ID columns, optional namespace columns, missing-ID fallback |
| `normalization` | Deterministic whitespace/case normalization before lookup or hashing |
| `canonical` | `source`, `crosswalk`, `hmac`, or `crosswalk_then_hmac` resolution |
| `sensitive_fields` | Whether raw values are absent, isolated in `identity_map`, or retained in cohort data |

`validation.source_collision` is `error` or `flag`.
`validation.multiple_source_ids_per_patient` is `allow`, `flag`, or `error`;
many-to-one mappings are often legitimate when a crosswalk reconciles IDs
across systems, so the default is `allow`. `separate_table` keeps raw mapping
fields out of normal cohort artifacts, but filesystem permissions for the run
directory remain the operator's responsibility.

For a crosswalk:

```yaml
identity:
  canonical:
    strategy: crosswalk_then_hmac
    crosswalk: ./identity_crosswalk.parquet
    crosswalk_keys: [site_id, dicom_patient_id]
    crosswalk_value: patient_id
    hmac:
      secret_env: IMPERANDI_ID_SECRET
      namespace: site-a/liver-v1
      length: 20
```

The HMAC secret is an environment variable value, never YAML. Crosswalk misses
fall through to HMAC only for `crosswalk_then_hmac`. Collision policies create
QC evidence and can fail the run.

## Ontologies

An ontology is a CSV or Parquet lookup table. It can match one or more columns
and produce a canonical evidence field or any new column.

```yaml
annotations:
  ontologies:
    - id: site_protocol_slots
      source: ./protocol_slots.csv
      keys:
        SeriesDescription: {match: normalized_exact}
        AcquisitionNumber: {match: numeric_exact}
      output:
        value_column: clinical_slot
        target_column: slot_ontology
        vocabulary: clinical_slot
      unmatched: keep
      conflicts: error
```

`value_column` is the column read from the ontology table after its keys match;
`target_column` is the new evidence or project column written to cohort rows.

Supported key matching:

- `exact`: string equality;
- `normalized_exact`: Unicode/whitespace/case normalized equality;
- `numeric_exact`: numeric equality after coercion.

`unmatched` is `keep` or `error`. `conflicts` is `error`, `flag`, or `first`.
Use vocabulary `clinical_slot` or `contrast_phase` for controlled outputs. Omit
the vocabulary to populate a free project column such as `protocol_family`.

Ontology application records the ontology ID, lookup row, and conflict state.

## Rule packs

Rules live in separate, reviewable YAML files and are listed after built-in
packs:

```yaml
annotations:
  rule_packs:
    - builtin:liver_ct
    - builtin:liver_mri
    - ./site_rules.yaml
```

The two `builtin:` entries enable their modality curators. Because project lists
replace profile lists, retain the built-in entries when adding a site rule pack;
omit one deliberately to route that modality using ontology/custom rules only.

```yaml
version: 1
rules:
  - id: phase.portal
    action: set
    target: phase_rules_explicit
    value: PORTAL_VENOUS
    evidence: explicit
    priority: 100
    when:
      all:
        - column: Modality
          operator: eq
          value: CT
      any:
        - column: SeriesDescription
          operator: regex
          value: "portal|veineux"

  - id: exclude.localizer
    action: exclude
    reason: localizer
    priority: 200
    when:
      any:
        - column: SeriesDescription
          operator: contains
          value: scout
```

Rule packs retain their independent `version: 1` schema; project files require
`version: 2`.

Actions are `set`, `exclude`, and `qc`. Operators are `eq`, `normalized_eq`,
`contains`, `regex`, `in`, `exists`, `lt`, `lte`, `gt`, and `gte`. Higher
priority wins; different values at equal priority are an error.

Use the standard phase/slot evidence targets to participate in final
resolution. Rules may also populate arbitrary columns.

## Image evidence and resolution

`phase_prediction.enabled` controls TotalSegmentator-based contrast evidence.
The current TotalSegmentator contrast backend accepts `modalities: [CT]`; MRI
phase/slot evidence comes from ontology and metadata rules. `scope` is one of:

- `unresolved`: no ontology or explicit-rule phase exists;
- `all_eligible`: all eligible configured modalities;
- `selected_and_unresolved`: ontology/rule-selected candidates plus unresolved
  rows.

Predictions below `minimum_confidence` keep their confidence but do not produce
a usable `phase_image` value.

`resolution.precedence` is an ordered evidence-column list. `disagreement` is
`flag`, `error`, or `ignore`. The chosen value, its source, and a conflict flag
remain in the output.

## Selection and modality routing

`selection.required_slots` describes expected clinical slots per modality.
Missing slots produce QC rows; they are not fabricated. Within an exam and
slot, selection ranks evidence provenance, then modality-specific quality, then
a stable volume identifier for deterministic ties.

CT and MR/MRI rows are routed independently. Unknown modalities are excluded
with provenance. The built-in liver profile supplies CT and MRI curation rules
and contextual dynamic-series strategies.

## Heavy stages

- `conversion`: DICOM-to-NIfTI settings;
- `segmentation.tasks`: backend, modality, explicit backend task, output name,
  and backend `parameters`;
- `registration`: explicit pair table, transform type (`rigid`,
  `rigid_affine`, or `deformable`), and saved forward transforms;
- `radiomics`: PyRadiomics settings path plus clinical-slot and mask filters.

Disabled heavy stages pass their input artifact forward so downstream contracts
and publication remain stable.

Without a profile, conversion and segmentation default to disabled. Enabling
segmentation requires at least one task. The `liver_ct_mri` profile explicitly
enables conversion and supplies its modality-routed segmentation tasks; project
overrides can disable either stage for a metadata-only validation run.

TotalSegmentator task names are never rewritten implicitly. Every task has one
explicit `modality`; configure CT and MR as separate entries. For example, CT
liver segmentation uses `task: total`,
whereas MR liver segmentation uses `task: total_mr`. The shared logical
`output: liver` still produces a consistent `mask_liver` cohort column.

An intra-patient pair table uses `fixed_volume_id` and `moving_volume_id`. A
template pair can use `fixed_nifti_path` plus `moving_volume_id`; an optional
`moving_nifti_path` can override the cohort lookup. `pair_id` is optional and is
generated deterministically when absent. When both images are cohort volumes,
their `patient_id` values must match; an external fixed template is exempt.

## Validation workflow

Before a full run:

1. run `imperandi validate`;
2. review `imperandi config resolve`;
3. inspect `imperandi plan`;
4. run metadata-only on a representative sample;
5. verify identity mappings, exclusions, evidence conflicts, required-slot QC,
   and selected CT/MR volumes;
6. enable image stages incrementally.
