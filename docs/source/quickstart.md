# Quickstart

IMPERANDI runs from one project YAML. Raw DICOM inputs remain read-only and
every run is written below the configured output root.

## 1. Create a project

```bash
imperandi init imperandi.yaml
```

Edit the generated file:

```yaml
version: 2

project:
  name: first-cohort
  profile: liver_ct_mri

input:
  sources:
    - /data/dicom

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

Relative paths are resolved from the directory containing `imperandi.yaml`.

## 2. Validate and inspect

```bash
imperandi validate imperandi.yaml
imperandi config resolve imperandi.yaml
imperandi plan imperandi.yaml
```

Validation rejects unknown fields. The resolved configuration shows all profile
defaults and absolute paths; the plan shows stage dependencies without reading
or writing cohort data.

## 3. Start safely with metadata only

For a first site validation, disable expensive image stages:

```yaml
conversion: {enabled: false}
phase_prediction: {enabled: false}
segmentation: {enabled: false}
registration: {enabled: false}
radiomics: {enabled: false}
```

Then run:

```bash
imperandi run imperandi.yaml
```

Inspect these artifacts first:

- `04_annotate/volumes_annotated.*`: all evidence and exclusion decisions;
- `04_annotate/volumes_shortlist.*`: rows eligible for image processing;
- `07_resolve_select/volumes_resolved.*`: resolved phase/slot plus provenance;
- `07_resolve_select/selected_volumes.*`: deterministic final choices;
- `11_publish/cohort_index.*`: published table.

This is the smallest safe implementation and validation slice: it covers input,
identity, CT/MRI routing, ontology/rules, evidence resolution, and output formats
without requiring TotalSegmentator, SimpleITK, or PyRadiomics.

## 4. Add site-specific ontology and rules

Reference mapping tables and rule packs from project YAML:

```yaml
annotations:
  ontologies:
    - id: site_slots
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
  rule_packs:
    - builtin:liver_ct
    - builtin:liver_mri
    - ./site_rules.yaml
```

Re-run `validate`, `config resolve`, and the metadata-only pipeline on a small,
representative sample. Review conflicts and exclusions before enabling image
processing.

## 5. Enable image stages

Install the required extras and enable only the needed features:

```bash
python -m pip install -e ".[imaging,radiomics]"
```

```yaml
conversion:
  enabled: true

phase_prediction:
  enabled: true
  modalities: [CT]
  scope: unresolved
  minimum_confidence: 0.70

segmentation:
  enabled: true

registration:
  enabled: false

radiomics:
  enabled: true
  settings: ./Params.yaml
  slots: [CT_PORTAL_VENOUS, MR_T2]
  masks: [liver]
```

Image prediction adds fallback evidence before final selection. Segmentation,
registration, and radiomics process selected volumes, limiting unnecessary
compute.

## 6. Check run status

The run directory is named with the first 12 characters of the effective
configuration hash:

```bash
imperandi status ./results/runs/<config-hash-prefix>
```

Do not treat a completed process alone as cohort validation. Review `run.json`,
stage metrics, error tables, QC tables, missing required-slot flags, and final
row counts before analysis.
