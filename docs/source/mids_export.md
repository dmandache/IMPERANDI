# MIDS-style CT/MR export

`imperandi export mids` copies a curated IMPERANDI cohort into a separate,
self-contained dataset. It implements a strict volumetric CT/MR profile based
on the hierarchy and naming concepts in the [MIDS
proposal](https://arxiv.org/abs/2010.00434) and its [2022
publication](https://pubmed.ncbi.nlm.nih.gov/35612110/).

This is a **MIDS-style organization profile**, not a claim of full MIDS
conformance. The proposal does not define a versioned validation target used
by this exporter. IMPERANDI validates its own narrow profile: identifiers,
labels, unique paths, sources, TSV references, mask/image links, and the
privacy allowlist.

## CLI

Use a secret of at least 16 bytes from a protected environment variable (the
bundled profile uses `IMPERANDI_MIDS_KEY`) or a protected key file:

```bash
export IMPERANDI_MIDS_KEY='replace-with-a-project-secret'

imperandi export mids \
  --csv_path ./tables/cohort_curated.csv \
  --output_dir ./exports/study-mids \
  --dry-run

imperandi export mids \
  --csv_path ./tables/cohort_curated.csv \
  --output_dir ./exports/study-mids
```

The dry run writes nothing. Its JSON report lists all planned dataset files,
missing required fields, unresolved collisions, excluded rows, and counts.
Unsupported modalities, missing/failed NIfTI paths, and explicitly failed or
low-confidence derivatives are excluded with reasons.

Files are copied by default; source images, masks, and the source CSV are never
modified. An existing output directory is rejected. `--overwrite replace`
builds and validates a complete sibling staging dataset before swapping it
into place.

## Identity boundary

Raw `patient_key`, study/series identifiers, DICOM UIDs, dates, and arbitrary
source strings never become exported path components. The default profile
uses HMAC-SHA256 to derive stable `sub-`, `ses-`, and safe source-image IDs.
The secret is neither logged nor copied. Session identity defaults to
`study_id`; configure the columns that identify an examination at your site.
`exam_stage` cannot be the only session identity because one clinical stage
can contain repeated examinations.

Instead of an HMAC secret, pass `--id-map /protected/location/ids.csv`. The
map stays outside the dataset and must contain the configured raw identity
columns plus:

```text
patient_key,study_id,volume_id,participant_label,session_label,image_label
```

Labels must contain only ASCII letters and digits. Every source volume needs
an entry, examination labels must be consistent, and participant/image labels
cannot be reused. Treat this file as a reverse map and protect it accordingly.

To run the generator automatically during export:

```bash
imperandi export mids \
  --csv_path ./tables/cohort_curated.csv \
  --output_dir ./exports/study-mids \
  --id-map auto
```

This mode does not require an HMAC key. The protected map defaults to
`<cohort_stem>_mids_id_map.csv` beside the cohort; select another protected
location with `--id-map-output /protected/location/ids.csv`. Dry-run generation
stays in memory and writes neither the map nor the dataset. An identical map is
reused, while a differing existing map requires
`--id-map-overwrite replace`. The automatic mode also accepts
`--patient-key-mode`, `--sort-columns`, and `--minimum-digits`.

### Generate an ID map

Generate a directly usable map from a curated cohort with:

```bash
imperandi export mids-id-map \
  --csv_path ./tables/cohort_curated.csv \
  --output_path /protected/location/cohort_mids_id_map.csv \
  --dry-run

imperandi export mids-id-map \
  --csv_path ./tables/cohort_curated.csv \
  --output_path /protected/location/cohort_mids_id_map.csv
```

The generator requires `patient_key`, `study_id`, `series_id`, and `volume_id`.
It retains these raw columns for lookup and adds:

```text
mapped_patient_key,mapped_study_id,mapped_series_id,mapped_volume_id,
participant_label,session_label,image_label
```

The three `*_label` fields alias the corresponding mapped patient, study, and
volume IDs, making the result directly consumable by the exporter:

```bash
imperandi export mids \
  --csv_path ./tables/cohort_curated.csv \
  --output_dir ./exports/study-mids \
  --id-map /protected/location/cohort_mids_id_map.csv
```

In the default `--patient-key-mode map`, patients, studies, series, and volumes
receive globally unique numeric labels. Width is fixed independently for each
entity type: at least four digits (`0001`) and automatically wider when its
entity count requires it. Change the floor with `--minimum-digits N`.

Numbering follows the available default ordering columns: `date`, DICOM study/
acquisition/series dates, `time` and related DICOM times, `visit_order`,
`acquisition_order`, `volume_ordinal_in_series`, `SeriesNumber`, and
`AcquisitionNumber`. Missing columns are ignored and raw identifiers are used
only as deterministic final tie-breakers. Override precedence with:

```bash
--sort-columns date time visit_order acquisition_order volume_ordinal_in_series
```

Patient keys are mapped using the same fixed-width rule by default.
`--patient-key-mode keep` is the exception: it preserves their case and value
without adding numeric padding, but only accepts alphanumeric labels. Use it
only when those keys are already approved pseudonyms; otherwise it would place
the original key in `sub-` paths and weaken the export privacy boundary.

The generated file contains the original identifiers and must remain in
protected storage outside the exported dataset. Sequential labels are stable
for an unchanged cohort and ordering configuration. Adding an earlier entity
can renumber later entities, so preserve the generated map used for a released
dataset rather than regenerating it independently.

Only fields explicitly listed under `metadata` cross the tabular privacy
boundary. Common identifying fields and all raw path/UID columns are rejected
even if accidentally allowlisted. The complete IMPERANDI CSV is never copied.
Unknown or uncertain allowlisted values remain in TSV metadata, but only
values explicitly mapped in the filename configuration can enter a filename.

## Manifest schema

Pass `--manifest export.yaml`. A normal IMPERANDI manifest may contain a
`mids_export` block; a dedicated YAML may consist of that block alone. The
bundled `mids` manifest supplies the default strict profile.

```yaml
mids_export:
  profile: strict-ct-mr-volumetric-v1
  dataset:
    name: Example CT/MR cohort
    license: CC-BY-4.0

  identity:
    subject_column: patient_key
    session: {source_columns: [study_id]}
    image_source_columns: [volume_id]
    digest_length: 12
    key_env: IMPERANDI_MIDS_KEY

  modality:
    aliases: {MRI: MR}
    mapping:
      CT: {directory: mim-ct, suffix: ct}
      MR: {directory: mim-mr, suffix: mr}

  filename:
    body_part:
      columns: [curated_body_part, BodyPartExamined]
      mapping: {abdomen: abdomen, liver: liver, chest: chest}
    acquisition:
      columns: [mri_sequence, mri_sequence_classification]
      mapping: {T1: t1, T1W: t1w, T2: t2, T2W: t2w}
    phase:
      columns: [phase]
      mapping: {NATIVE: native, ARTERIAL: arterial, PORTAL_VENOUS: portal, DELAYED: delayed}
    reconstruction:
      columns: [curated_reconstruction]
      mapping: {standard: standard, sharp: sharp}
    repeat:
      columns: [repeat, volume_ordinal_in_series]

  metadata:
    participants:
      sex: [PatientSex]
    sessions:
      clinical_stage: [exam_stage]
    scans:
      phase: [phase]
      phase_source: [phase_source]
      phase_confidence: [phase_confidence]
      mri_sequence_classification: [mri_sequence, mri_sequence_classification]
      mri_sequence_confidence: [mri_sequence_confidence]
      mri_sequence_reason: [mri_sequence_reason]

  derivatives:
    masks:
      source:
        pipeline: imperandi-segmentation
        column_patterns: ["mask_*"]
        exclude_patterns: ["mask_consensus_*", "consensus_mask_*"]
        qc_status_columns: [segmentation_qc_status]
        rejected_statuses: [failed, rejected, low_confidence]
      consensus:
        pipeline: imperandi-registration-consensus
        column_patterns: ["consensus_mask_*", "mask_consensus_*"]
        qc_status_columns: [registration_qc_status]
        confidence_columns: [registration_confidence]
        minimum_confidence: 0.5
    radiomics:
      pipeline: imperandi-radiomics
      column_patterns: ["*_original_*", "*_wavelet_*"]
      qc_status_columns: [radiomics_qc_status]
      rejected_statuses: [failed, rejected, low_confidence]
```

CT/MR mappings are mandatory and explicit. `bp-`, `acq-`, `pc-`, and `rec-`
entities are emitted only for mapped curated values. A positive configured
repeat is a `run-` entity; otherwise volumes with identical descriptive labels
receive deterministic `run-01`, `run-02`, … suffixes ordered by safe image ID.

For MR, the bundled profile encodes IMPERANDI's curated `mri_sequence` in the
acquisition entity (`acq-t1`, `acq-t2`, `acq-dwi`, or `acq-adc`). The contrast
phase is an independent `pc-` entity for both CT and MR: for example,
`pc-arterial`, `pc-portal`, or `pc-delayed`. A portal T1 MR volume can therefore
be named with both `acq-t1_pc-portal`. Sequence confidence and classification
reason remain in the scan table. Unknown sequence or phase values are not
guessed into filenames; their allowlisted values remain metadata.

Mask groups declare separate pipelines. Missing QC status is recorded as
`not_provided`, never as success. Explicitly rejected/unaccepted statuses and
confidence below the configured threshold are excluded. Radiomics export is
limited to numeric columns matched by its allowlist and always includes a safe
image ID, relative source-image link, and available QC status.

## Example tree

```text
study-mids/
├── README
├── dataset_description.json
├── participants.tsv
├── export_report.json
├── code/
│   └── imperandi_mids_export.yaml
├── sub-a31f.../
│   ├── sub-a31f..._sessions.tsv
│   └── ses-19b2.../
│       ├── sub-a31f..._ses-19b2..._scans.tsv
│       ├── mim-ct/
│       │   └── sub-a31f..._ses-19b2..._bp-liver_pc-arterial_ct.nii.gz
│       └── mim-mr/
│           └── sub-a31f..._ses-19b2..._bp-liver_acq-t2w_pc-portal_mr.nii.gz
└── derivatives/
    ├── imperandi-segmentation/
    │   ├── dataset_description.json
    │   ├── masks.tsv
    │   └── sub-a31f.../ses-19b2.../mim-ct/..._desc-source_label-liver_seg.nii.gz
    ├── imperandi-registration-consensus/
    │   ├── dataset_description.json
    │   └── masks.tsv
    └── imperandi-radiomics/
        ├── dataset_description.json
        └── radiomics.tsv
```

The exporter intentionally does not support radiography, ultrasound, PET,
two-dimensional formats, DICOM export, or general MIDS/BIDS validation. It
does not de-identify voxel data (for example, burned-in annotations or facial
anatomy); data owners remain responsible for image-level de-identification and
governance before sharing.
