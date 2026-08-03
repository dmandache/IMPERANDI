# Workflow

IMPERANDI maintains a named artifact graph. Tables carry metadata and paths;
image data remains on disk. Each stage consumes named artifacts and publishes
new named artifacts rather than relying on manually coordinated filenames.

## Two-pass curation

The pipeline separates inexpensive metadata decisions from expensive image
processing:

1. index DICOM headers or load pre-indexed instances;
2. resolve canonical patient identity;
3. assemble instances into volume rows;
4. apply explicit ontology, custom rules, and built-in CT/MRI rules;
5. exclude unsuitable series and shortlist eligible volumes;
6. convert shortlisted volumes to NIfTI;
7. optionally predict contrast from image content;
8. resolve evidence and choose one volume per exam/clinical slot;
9. segment, register, and extract radiomics from selected volumes;
10. publish the cohort.

This ordering lets explicit site knowledge avoid unnecessary conversions while
still allowing image evidence to fill unresolved metadata cases.

## Metadata pass

### Index

`01_index` accepts DICOM roots/globs, bounded nested archives, and pre-indexed
CSV/Parquet tables. Multiple inputs are concatenated into `instances_raw`.
Parser failures are isolated in `index_errors`.

When CSV is explicitly selected and the inventory exceeds an internal product
heuristic, the pipeline recommends Parquet and continues with CSV. This warning
does not change output format or project configuration.

### Identity

`02_identity` extracts and normalizes source IDs, resolves `patient_id`, checks
collisions, and removes raw ID columns from normal cohort data unless policy
permits them. Sensitive mapping is written separately as `identity_map`.

Clinical or site-derived variables do not belong in `patient_id`. Populate them
through ontologies or rules so identity remains stable when cohort logic
changes.

### Assemble

`03_assemble` normalizes DICOM dates, times, pixel spacing, and orientation;
constructs stable volume groupings; and computes visit/acquisition order.
`patient_id` remains the public grouping identifier.

### Annotate and shortlist

`04_annotate` applies evidence in this order:

1. project ontologies;
2. project rule packs;
3. built-in CT and MRI curation;
4. modality exclusions and shortlist construction.

This is application order, not resolution precedence. Evidence remains in
separate columns until `07_resolve_select`.

## Image pass

### Convert

`05_convert` converts only `volumes_shortlist`. Successful rows gain
`nifti_path`; per-row failures are recorded without discarding successful rows.
When conversion is disabled, the artifact passes through unchanged.

### Predict phase

`06_predict_phase` optionally runs the configured TotalSegmentator backend on
eligible modalities and scope. It writes `phase_image` and confidence without
overwriting ontology/rule evidence. Low-confidence values are left unresolved.

### Resolve and select

`07_resolve_select` applies configured evidence precedence and disagreement
policy, derives image-based slots where possible, and deterministically ranks
candidate volumes by:

1. evidence source priority;
2. modality-specific quality score;
3. stable volume identifier.

Required-but-missing CT/MR slots produce `selection_qc` rows. They do not abort
unrelated exams.

### Segment

`08_segment` routes each selected volume by CT or MR and runs only tasks whose
explicit `modality` matches that route. Outputs are stored as mask-path columns.
Backend task names remain exactly as configured: use separate task entries such
as `total` for CT liver and `total_mr` for MR liver. Failures are isolated per
source row.

### Register

`09_register` consumes an explicit registration pair table. Each pair names a
fixed and moving volume. Rigid, rigid-plus-affine, and deformable transforms are
supported. The forward transform and resampled image are always persisted for
successful pairs; failures do not stop independent pairs. Cohort-to-cohort
pairs are rejected when their canonical `patient_id` values differ.

### Radiomics and publish

`10_radiomics` optionally filters selected rows by clinical slot and mask name,
then applies the configured PyRadiomics settings. `11_publish` writes the final
cohort in each requested format.

## Checkpoint and resume behavior

The effective configuration is SHA-256 hashed together with the contents of
external crosswalk, ontology, rule, registration-pair, and radiomics-settings
files that affect enabled stages. Its first 12 characters name the run
directory. Each stage transitions through `running`, `completed`, or `failed`
in `stage.json`.

With `execution.resume: true`, a completed stage is skipped only if its resume
token matches and every recorded table plus schema sidecar still exists. Input
inventory paths, sizes, and modification times form the index-stage token.
Deleting an artifact or changing an input forces that stage and every downstream
stage to run again. Changing configuration or a policy-file content creates a
different run directory. The installed IMPERANDI version and pipeline contract
are also part of stage resume tokens, preventing artifacts created by an older
implementation from being silently reused after an upgrade.

Heavy adapters also receive `checkpoint_every_rows` and
`checkpoint_every_seconds`, allowing their backend algorithms to recover within
a stage. Do not edit state or checkpoint files while a run is active.

## Batch execution

A scheduler runs the same project command used interactively:

```bash
#!/bin/bash
#SBATCH --job-name=imperandi-pipeline
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
source /path/to/venv/bin/activate
export IMPERANDI_ID_SECRET="${SITE_MANAGED_SECRET}"
imperandi validate /path/to/project/imperandi.yaml
imperandi run /path/to/project/imperandi.yaml
```

Keep secrets in scheduler/environment secret management, not project YAML or
logs. Keep raw DICOM roots read-only, configuration/rule/ontology files under
reviewed version control, and run artifacts with their QC/error tables.
