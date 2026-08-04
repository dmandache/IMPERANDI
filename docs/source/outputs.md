# Outputs

Tables, logs, checkpoints, and provenance receive a deterministic run directory:

```text
<output.root>/runs/<first-12-config-hash>/
├── run.json
├── resolved_config.yaml
├── environment.json
├── 01_index/
├── 02_identity/
├── 03_assemble/
├── 04_annotate/
├── 05_convert/
├── 06_predict_phase/
├── 07_resolve_select/
├── 08_segment/
├── 09_register/
├── 10_radiomics/
└── 11_publish/
```

Cohort imaging is stored separately. By default its root is
`<output.root>/<project.name>/`; set `output.imaging_root` to place it elsewhere:

```text
<output.imaging_root>/
├── <patient_id>/<study_id>/<series_id>/
│   ├── scan.nii.gz
│   └── <mask>.nii.gz
└── registrations/
    ├── <pair_id>.tfm
    └── <pair_id>_registered.nii.gz
```

`run.json` records overall status and the final artifact registry.
`resolved_config.yaml` is the complete profile-plus-project configuration.
`environment.json` records IMPERANDI, Python, platform, pipeline-contract
version, and configuration hash.
Each stage directory contains `stage.json` with status, artifact paths, and
metrics.

## Artifact map

| Stage | Primary tables | Optional tables/files |
|---|---|---|
| `01_index` | `instances_raw` | `index_errors`, per-source parser checkpoints |
| `02_identity` | `instances`, `identity_map` | `identity_qc`, `instances_unresolved_identity` |
| `03_assemble` | `volumes` | — |
| `04_annotate` | `volumes_annotated`, `volumes_shortlist` | `volumes_rejected`, `annotation_qc` |
| `05_convert` | `volumes_converted` | `convert_errors`, NIfTI images |
| `06_predict_phase` | `volumes_predicted` | `phase_prediction_errors` |
| `07_resolve_select` | `volumes_resolved`, `selected_volumes` | `selection_qc` |
| `08_segment` | `volumes_segmented` | `segment_errors`, masks |
| `09_register` | `volumes_registered` | `registration_pairs`, `registration_errors`, transforms, registered images |
| `10_radiomics` | `radiomics_table` | `radiomics_errors` |
| `11_publish` | `cohort_index.<format>` | additional requested publication formats |

File extensions follow `output.table_format` for intermediate tables and
`output.publish_formats` for final tables.

## Identity and provenance

`patient_id` is the canonical cohort identifier. Companion columns include
`patient_id_method`, `identity_confidence`, and `identity_algorithm_version`.

`identity_map` is intentionally separate from `instances`. Under
`separate_table` it contains raw-to-canonical mapping fields and must be
handled according to the project's data-protection controls. Under `never` it
contains only canonical IDs. Under `cohort`, raw identity fields are also kept
in cohort tables. A `source` canonical strategy may itself expose a source ID;
choose HMAC or a pseudonymized crosswalk when canonical IDs must be opaque.
`separate_table` describes logical separation, not automatic access control;
protect the configured output root with suitable filesystem permissions.

Annotation provenance remains explicit:

- `<target>_ontology_id`, `<target>_ontology_row`, and `<target>_conflict` for
  ontology results;
- `<target>_rule_id` and `<target>_evidence` for custom rules;
- `phase_source`/`phase_conflict` and
  `clinical_slot_source`/`clinical_slot_conflict` for final resolution;
- `exclusion_reason` and `exclusion_rule_id` for rule-based exclusions.

## CSV and Parquet schemas

IMPERANDI writes `<table>.<format>.schema.json` beside each table. The sidecar
lists original dtypes and JSON-encoded structured columns. This permits DICOM
metadata columns containing lists, mappings, or mixed scalar/list values to
round-trip consistently in CSV and Parquet.

Use `imperandi.io.read_table` when consuming intermediate artifacts in Python;
plain Pandas readers do not apply the sidecar decoding.

## Errors and QC

Errors and QC are different contracts:

- error tables record a processing failure for a source row or pair;
- QC tables record a completed decision that still requires review, such as a
  missing required slot or identity collision;
- conflict columns retain disagreement between evidence sources even when a
  value is resolved.

A pipeline can finish with isolated row errors. Before analysis, review:

1. `run.json` and every stage status;
2. input/output/error row counts;
3. identity and selection QC tables;
4. exclusion reasons and evidence conflicts;
5. missing image, mask, transform, or feature paths;
6. expected CT and MRI clinical-slot coverage.

Checkpoint and temporary bridge files inside heavy-stage directories are
implementation artifacts. Use the named tables recorded by `stage.json` as the
public artifacts.
