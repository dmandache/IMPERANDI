# **IM**aging **PRE**processing **A**nd **N**ormalization for **D**iagnostic **I**nteroperability

![IMPERANDI logo](https://raw.githubusercontent.com/dmandache/IMPERANDI/main/static/imperandi-logo.png)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![Documentation](https://readthedocs.org/projects/imperandi/badge/?version=latest)](https://imperandi.readthedocs.io/en/latest/)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Linting](https://img.shields.io/badge/lint-ruff-red)
![Tests](https://github.com/dmandache/IMPERANDI/actions/workflows/tests.yml/badge.svg?branch=main)

IMPERANDI builds traceable, analysis-ready CT and MRI cohorts from heterogeneous
DICOM exports. One typed project YAML describes the intended cohort; the runner
handles indexing, identity resolution, modality-aware curation, conversion,
optional image-based phase prediction, selection, segmentation, registration,
radiomics, and publication.

The public workflow is deliberately small:

```bash
imperandi init imperandi.yaml
imperandi validate imperandi.yaml
imperandi plan imperandi.yaml
imperandi run imperandi.yaml
```

IMPERANDI is research software, not a certified medical device. Validate its
outputs for the intended research setting before use.

## Design principles

- Project configuration expresses cohort intent, not a hand-written sequence of
  internal commands.
- Every run is content-addressed by the resolved configuration and records its
  configuration, environment, stage state, artifacts, errors, and QC flags.
- Ontology, metadata rules, and image prediction remain separate evidence
  columns. Resolution is explicit and disagreements are retained.
- Patient identity extraction, canonicalization, and sensitive mappings are
  separate concerns.
- CSV and Parquet are explicit user choices. Large CSV inventories produce a
  non-blocking product warning; the warning threshold is not project YAML.

## Pipeline architecture

| Stage | Responsibility | Main artifact |
|---|---|---|
| `01_index` | Discover DICOM/archive inputs or load a pre-indexed table | `instances_raw` |
| `02_identity` | Resolve canonical `patient_id`; isolate sensitive mappings | `instances`, `identity_map` |
| `03_assemble` | Normalize metadata and aggregate instances into volumes | `volumes` |
| `04_annotate` | Apply ontologies, custom rules, and CT/MRI curation; exclude ineligible series | `volumes_annotated`, `volumes_shortlist` |
| `05_convert` | Convert shortlisted volumes to NIfTI | `volumes_converted` |
| `06_predict_phase` | Optionally add TotalSegmentator image-contrast evidence | `volumes_predicted` |
| `07_resolve_select` | Resolve evidence and deterministically select clinical slots | `volumes_resolved`, `selected_volumes` |
| `08_segment` | Route configured segmentation tasks by modality | `volumes_segmented` |
| `09_register` | Execute an explicit pair table and save transforms | `volumes_registered` |
| `10_radiomics` | Extract configured features from selected slots/masks | `radiomics_table` |
| `11_publish` | Publish the final cohort in requested formats | `cohort_index` |

Disabled heavy stages are traceable pass-through stages, so the artifact graph
and downstream rules stay consistent. Metadata annotation occurs before image
conversion; optional image prediction occurs only after conversion and before
final selection.

## Install

```bash
python -m pip install -e .
```

Install optional image-processing features as needed:

```bash
python -m pip install -e ".[segment]"
python -m pip install -e ".[radiomics]"
python -m pip install -e ".[all]"  # development plus all optional features
```

## Quickstart

Create `imperandi.yaml` with `imperandi init`, then replace the input source:

```yaml
version: 1

project:
  name: liver-cohort
  profile: liver_ct_mri

input:
  sources:
    - /data/site-a/dicom

output:
  root: ./imperandi-results
  table_format: parquet
  publish_formats: [parquet, csv]

identity:
  source:
    patient_id_columns: [PatientID]
    namespace_columns: [site_id, IssuerOfPatientID]
    fallback:
      columns: []
      on_missing: error
  canonical:
    strategy: source

phase_prediction:
  enabled: false

conversion: {enabled: false}
segmentation: {enabled: false}
registration: {enabled: false}
radiomics: {enabled: false}

execution:
  workers: 4
  resume: true
```

Then validate, inspect, and execute:

```bash
imperandi validate imperandi.yaml
imperandi config resolve imperandi.yaml
imperandi plan imperandi.yaml
imperandi run imperandi.yaml
```

The built-in `liver_ct_mri` profile supplies CT/MRI rules, required clinical
slots, and modality-aware liver segmentation tasks. Project values override
profile values; mappings merge recursively and lists replace profile lists.

## Phase and clinical-slot evidence

Resolution uses separate evidence fields in descending default precedence:

1. `phase_ontology`
2. `phase_rules_explicit`
3. `phase_rules_inferred`
4. `phase_image`

Clinical slots use the analogous `slot_ontology`, `slot_rules_explicit`,
`slot_rules_inferred`, and `slot_image` fields. The resolved value, chosen
source, and conflict flag are stored separately. A higher-precedence source
does not erase lower-precedence evidence.

### Explicit ontology mapping

An ontology can match one or more metadata columns and populate either a
canonical evidence field or any project-specific derived column:

```yaml
annotations:
  ontologies:
    - id: site_protocol_slots
      source: ./protocol_slots.csv
      keys:
        SeriesDescription: {match: normalized_exact}
        AcquisitionNumber: {match: numeric_exact}
      output:
        source_column: clinical_slot
        target_column: slot_ontology
        vocabulary: clinical_slot
      unmatched: keep
      conflicts: error
```

For a generic new column, use any `target_column`, for example
`protocol_family`, and omit `vocabulary`.

### Extensible rules and exclusions

Project YAML references reviewed rule-pack YAML files:

```yaml
annotations:
  rule_packs:
    - builtin:liver_ct
    - builtin:liver_mri
    - ./site_rules.yaml
```

```yaml
# site_rules.yaml
version: 1
rules:
  - id: phase.portal.fr
    target: phase_rules_explicit
    value: PORTAL_VENOUS
    priority: 100
    when:
      any:
        - column: SeriesDescription
          operator: regex
          value: "portal|veineux"

  - id: exclude.scout
    action: exclude
    reason: localizer
    priority: 200
    when:
      any:
        - column: SeriesDescription
          operator: contains
          value: scout
```

Rules support `set`, `exclude`, and `qc` actions; exact, normalized, text,
regular-expression, membership, existence, and numeric conditions; and stable
priority handling.

### Optional image prediction

```yaml
phase_prediction:
  enabled: true
  backend: totalsegmentator
  modalities: [CT]
  scope: unresolved
  minimum_confidence: 0.70
  resolution:
    precedence:
      - phase_ontology
      - phase_rules_explicit
      - phase_rules_inferred
      - phase_image
    disagreement: flag
```

Image prediction is a fallback by default. It runs after conversion and can be
scoped to unresolved, all eligible, or selected-and-unresolved volumes.

## Patient identity

`patient_id` is the only canonical public identifier. Its lifecycle is:

1. extract source columns and optional namespace columns;
2. normalize deterministically;
3. resolve with `source`, `crosswalk`, `hmac`, or `crosswalk_then_hmac`;
4. validate source/canonical collisions;
5. keep raw identifiers only according to `sensitive_fields` policy.

For HMAC pseudonymization, the secret is read from an environment variable and
is never stored in YAML:

```yaml
identity:
  canonical:
    strategy: hmac
    hmac:
      secret_env: IMPERANDI_ID_SECRET
      namespace: site-a/liver-v1
      prefix: P
      length: 20
  sensitive_fields:
    persist_raw_identifiers: secure_table_only
```

Use ontology mappings—not identifier parsing hooks—to derive clinical/project
columns such as center, protocol family, or tumor group.

## CSV and Parquet

`output.table_format` controls intermediate tables and accepts exactly `csv` or
`parquet`. `output.publish_formats` controls final cohort copies. Structured
DICOM cells round-trip in either format through schema sidecars.

When an explicitly selected CSV inventory exceeds the built-in product
heuristic, IMPERANDI recommends Parquet but continues with CSV. There is no
`csv_warning_threshold_files` configuration key; unknown configuration fields
are rejected so misspellings and obsolete settings fail early.

## Run outputs and resume

Runs are stored under:

```text
<output.root>/runs/<config-hash-prefix>/
  run.json
  resolved_config.yaml
  environment.json
  01_index/stage.json
  ...
  11_publish/cohort_index.parquet
```

The effective hash includes external crosswalk, ontology, rule, registration,
and radiomics-settings file contents. Each completed stage records artifact
paths and metrics. A matching completed stage is reused only when resume is
enabled, its source fingerprint still matches, and all table/schema artifacts
still exist. If one stage reruns, every downstream stage reruns. Failures update
both the stage state and overall run state.

Inspect a run with:

```bash
imperandi status ./imperandi-results/runs/<config-hash-prefix>
```

## Python modality router

```python
from imperandi.curation import curate_by_modality

result = curate_by_modality(
    volumes,
    patient_col="patient_id",
    study_col="study_id",
    date_col="date",
)
```

The result contains `ct`, `mri`, `other`, `curated_all`, and
`selected_long_all`. CT and MR rows are routed independently; unsupported or
ambiguous modalities are returned in `other`, never silently coerced.

## Development

```bash
python -m pip install -e ".[dev]"
pytest -q
ruff check .
black --check .
```

_This work performed under the RHU OPERANDI project was supported in part by
the French National Research Agency (ANR) as its 3rd PIA, integrated into the
France 2030 plan under reference ANR-21-RHUS-0012._
