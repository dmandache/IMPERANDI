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
followed by `phase_curation`, and `finalize` should remain last.
`modality_curation` generates CT/MR features, sequence labels, and raw rule
evidence. It can run without a top-level `phase_curation` block and does not
resolve canonical phases or write selection exports. The separate
`phase_curation` step requires that block, resolves the ordered strategies from
existing evidence, and refreshes phase-dependent selection scores. It can also
run alone on a table that already contains rule, ontology, or model evidence.
Cleaning consumes existing model predictions; use `imperandi phase` to run
TotalSegmentator when prediction is needed.

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
    - type: phase_curation
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
  modalities:
    CT:
      pyradiomics:
        setting:
          binWidth: 25
          resegmentRange: [-150, 250]
        imageType:
          Original: {}
    MR:
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

## Date and time coalescing

`coalesce_date` and `coalesce_time` accept an ordered `candidates` list. Each
row uses the first candidate that parses successfully; blank, malformed, and
absent columns fall through to the next candidate. An explicit list replaces
the defaults completely and may name site-specific CSV columns:

```yaml
cleaning:
  steps:
    - type: coalesce_date
      candidates:
        - StudyDate
        - AcquisitionDate
        - SeriesDate

    - type: coalesce_time
      candidates:
        - FrameAcquisitionDateTime
        - AcquisitionDateTime
        - AcquisitionTime
```

Without `candidates`, the date order is `StudyDate`, `AcquisitionDate`, then
`SeriesDate`. The time order is `FrameReferenceDateTime`,
`FrameAcquisitionDateTime`, `AcquisitionDateTime`, then `AcquisitionTime`.
The resulting `date_source` and `time_source` columns contain the exact source
column name and survive volume aggregation and CSV export. Candidate lists must
be nonempty lists of nonempty column names. DICOM DA, TM, and DT values,
including fractional seconds and DT UTC offsets, are supported. A configured
datetime contributes its represented date or time component while the raw
datetime remains available.

The canonical `date` is the examination date by default because `StudyDate`
has first priority and is used for visit grouping. Ordering and phase inference
use the canonical `time`, `acquisition_order`, and series-local
`volume_order_in_series` / `n_volumes_in_series` columns. Raw DICOM datetime
and bolus fields remain metadata; they do not calibrate contrast delays or
override this ordering. Phase curation compares canonical time-of-day values
without reconstructing cross-midnight timestamps.

`ContentTime` and `InstanceCreationTime` describe content or object creation,
and `StudyTime` cannot order acquisitions within a study. `SeriesTime` can also
be too coarse for individual acquisitions. All remain available as raw tags
and may be named explicitly for datasets that need them, but they are excluded
from the default acquisition-time candidates. Their literal column names are
retained in `time_source` for provenance. Once coalesced, `time` is the
canonical clock used by curation regardless of the selected source.

Enhanced DICOM values found under shared or per-frame functional groups are
promoted to their canonical keyword columns during selected-tag parsing.
Multiple per-frame values remain lists until volume aggregation. Private tags,
vendor-specific timing sequences, and conflicting frame clock domains are not
interpreted automatically.

## Cleaning filters

Add `type: filter` steps to `cleaning.steps` to keep or discard records that
match one or more rules. Set `kind` to `keep` or `discard`, set `scope` to
`row` or `volume`, and combine multiple rules with `logic: and` or `logic: or`.
A single rule defaults to `logic: and`. Place row filters before
`group_volumes` and volume filters after it so that rules run at the declared
granularity.

Each rule names a `column`, an `op`, and, except for the null checks, a `value`:

| `op` | Expected `value` | Behavior |
| --- | --- | --- |
| `eq` | Scalar | Exact equality |
| `ne` | Scalar | Exact inequality |
| `in` | List | Value occurs in the list |
| `not_in` | List | Value does not occur in the list |
| `contains` | Scalar | Case-sensitive literal substring match |
| `icontains` | Scalar | Case-insensitive literal substring match |
| `regex` | Regular-expression string | Case-sensitive regular-expression match; use an inline flag such as `(?i)` for case-insensitive matching |
| `lt` / `lte` | Number | Numeric value is less than / less than or equal to the threshold |
| `gt` / `gte` | Number | Numeric value is greater than / greater than or equal to the threshold |
| `is_null` | Omit | Value is missing |
| `not_null` | Omit | Value is present |

Missing values do not match operators other than `is_null`. Set
`keep_null: true` on the filter step to preserve records with a missing value
in any referenced column, whether the step keeps matches or discards them.
If a referenced column is absent entirely, the step is skipped with a warning.
For list-valued cells, a positive rule matches any populated item; `ne` and
`not_in` require every populated item to pass. Empty/all-null lists count as null,
including for `keep_null`. Explicit phase assignments use the same matching code.

For example, retain CT and MR volumes whose reconstructed length is at least
30 mm, while preserving volumes whose length is unknown:

```yaml
cleaning:
  steps:
    - type: filter
      name: volume_length_quality
      kind: keep
      scope: volume
      logic: and
      keep_null: true
      rules:
        - {column: Modality, op: in, value: [CT, MR, MRI]}
        - {column: volume_length, op: gte, value: 30.0}
```

## Phase curation and fallback

`phase_curation.strategies` is an ordered fallback chain. Each strategy is
optional, but the list must contain at least one strategy. The first strategy
that produces a value not listed in `unresolved_labels` wins.

The resolver writes:

- `phase`: canonical uppercase phase such as `NATIVE`, `ARTERIAL`, `PORTAL_VENOUS`, or `DELAYED`; genuinely unresolved diagnostic acquisitions use `OTHER`, while non-phaseable acquisitions remain empty;
- `phase_status`: `RESOLVED`, `UNRESOLVED`, or `NOT_APPLICABLE`;
- `phase_source`: configured strategy name, `fallback`/`unresolved`, or `not_applicable`;
- `phase_strategy`: stable strategy type (`explicit`, `ontology`, `rules`, or `totalsegmentator`), `fallback`/`unresolved`, or `not_applicable`;
- `phase_confidence`: explicit-assignment, rule, ontology, or predictor confidence when available;
- `phase_reason`: concise provenance for the decision (for example `LOCALIZER` or `NON_T1_SEQUENCE` for not-applicable rows).

`qc_unresolved_phase.csv` contains only rows with `phase_status=UNRESOLVED`. Acquisitions explicitly classified as `NOT_APPLICABLE` are excluded from phase QC.

Perfusion phase and diagnostic selection eligibility are independent. Derived
products, including subtraction, MIP/MPR, secondary/key images, and quantitative
maps, can retain an evidenced acquisition phase. These product markers do not
produce `NOT_APPLICABLE`; without phase evidence the row remains `UNRESOLVED`.
MRI MIP/MPR reconstructions remain eligible for diagnostic selection, with score
penalties that prefer source images when comparable candidates exist. They can
fill a phase slot when its only resolved candidate is a reconstruction.
Within eligible sequence/phase slots, MRI selection excludes only rows flagged
as subtraction or quantitative/report products. Dixon component labels and
secondary/key-product markers do not independently exclude a candidate.
`ImageType=DERIVED` alone does not exclude
diagnostic CT reconstructions or MRI Dixon/GRE images from selection.
Legacy `phase_applicability_reason=DERIVED` values are ignored by the resolver;
rerun `clean` to regenerate metadata phase evidence suppressed by older rules.

The metadata engines also retain their unmodified result in `rule_phase`,
`rule_phase_confidence`, and `rule_phase_reason`. This lets the post-conversion
`phase` command apply the same manifest without losing clean-time evidence.

### Override metadata text columns

Use `phase_curation.text_columns` to replace the ordered DICOM text fields used
for CT or MR metadata classification and phase rules:

```yaml
phase_curation:
  text_columns:
    CT: [ProtocolName, SeriesDescription]
    MR: [SequenceName, ProtocolName, SeriesDescription]
  strategies:
    - type: rules
  fallback: OTHER
```

Each modality's list replaces its defaults rather than extending them. Fields
are checked in the listed order, independently for each decision (sequence,
phase, plane, Dixon component, product type, and acquisition features). The
first field with relevant evidence owns that decision throughout classification,
group inference, and scoring; later fields are used only when earlier fields
have no relevant evidence. A generic sequence name can therefore fall through
to a later phase description without losing its sequence classification.
Missing and empty fields are skipped. Lists must be non-empty and contain unique
column names. `MRI` is an alias for `MR`; define only one of these keys.
You can include custom CSV columns as well as DICOM keyword columns. Custom tags
must already have been parsed or added to the CSV before curation.

Omitting a modality retains its current defaults: CT uses `SeriesDescription`,
`ProtocolName`, `StudyDescription`, `ImageType`, `ScanningSequence`,
`SequenceVariant`, `ScanOptions`, and `SequenceName`; MR uses
`SeriesDescription`. Overrides also apply to candidate display text. Structured
evidence such as `ImageType` product markers, dimensions, spacing, and acquisition
order continues to be evaluated separately. These lists do not change routing
by `Modality`, ontology columns, or TotalSegmentator prediction columns.

### Best series per exam

`clean` saves all curated rows and `phase` saves all input rows with resolved
phases. Both also save the best-series dataframe and its QC table automatically
beside the input CSV as `<input_stem>_curated.csv` and
`qc_<input_stem>_curated.csv`. For example, `cohort.csv` produces
`cohort_curated.csv` and `qc_cohort_curated.csv`, even when the main output
has a different name or directory. For multiple cleaning inputs, the first
input supplies the default directory and stem. `ingest` uses its intermediate
`dicom_index.csv` as the cleaning input.

`clean` and `ingest` always use these default paths. For `phase`, override
them with the CLI option `--selected_csv_path`:

```bash
imperandi phase nifti_index.csv --manifest site.yaml --selected_csv_path selected_phase.csv
```

Configure the columns that uniquely identify each best-candidate group in the
manifest:

```yaml
phase_curation:
  best_candidate_group_columns: [patient_key, study_id, date]
  strategies:
    - type: rules
  fallback: OTHER
```

`--selected_csv_path` applies only to `phase`. Relative paths resolve from the working
directory. Omitting the option uses the input-derived defaults; it is not a
manifest setting. Cleaning pipelines without a `phase_curation` step do not
produce selection exports.
The main output retains all its rows.
The selected CSV saves `selected_long`: one best eligible CT volume per phase
and one best eligible MR volume per sequence/phase slot, per exam. It retains identifiers,
image paths, resolved phase provenance, and selection scores.
`selected_wide` is saved beside it as `qc_<selected_stem>.csv` (for example,
`qc_selected_phase.csv`). This QC table has one row per exam and modality,
with selection slots as columns and candidate descriptions and scores as
values; MR slots also retain their alternative-candidate columns.
`qc_unresolved_phase.csv` is also saved in the same directory. It includes
all rows still unresolved after the configured phase strategies, before
fallback, including rows excluded from best-series selection. The `phase`
and phase provenance fields are empty; raw rule/model evidence and other
metadata are retained. Resolved rows with the same label as the configured
fallback are excluded. The file is written even when there are no unresolved
rows, replacing previous contents; it is also regenerated on resume.
Unsupported modalities and exams without eligible candidates contribute no selected rows.
Both tables use the final cohort and phases, including TotalSegmentator results
when running `phase`. Existing selected and QC exports are replaced, and a
completed resumed `phase` run regenerates both exports from its saved main output (resume
is enabled by default).

`best_candidate_group_columns` defines the columns that together identify each
group for which one best candidate is selected per modality and selection slot;
the same grouping is also used for MR phase inference. An explicit list must be
nonempty and unique. If any configured columns are missing, the command logs a
warning and falls back to the available `patient_key` and `date` columns. `null`
or omission preserves the default: use whichever of `patient_key`, `study_id`,
and `date` are available. If none are available, the command logs a warning and
writes empty selected and QC tables because it cannot identify candidate groups.
For example, `[patient_key, visit_id]` selects one best candidate per patient
visit rather than per study. Custom columns are loaded by the cleaning pipeline.

### Explicit phase assignments

Use `type: explicit` to assign a phase when one or more metadata conditions
match. Put it before `rules`, ontology, or prediction strategies when the manual
assignment should take precedence:

```yaml
phase_curation:
  strategies:
    - type: explicit
      name: reviewed_assignments
      assignments:
        - name: reviewed_ph1
          phase: ARTERIAL
          logic: and
          rules:
            - {column: patient_key, op: in, value: [patient_001, patient_002]}
            - {column: SeriesDescription, op: icontains, value: ph1}
    - type: rules
    - type: totalsegmentator
  fallback: OTHER
```

In this example, both rules must match the same row. Each assignment uses the
same `logic` and `rules` structure as a cleaning filter. Assignments default to
`logic: and`; set `logic: or` to accept any rule. A single rule is also accepted.
The first matching assignment wins, and later strategies only process unresolved
rows. Assignment names are optional and default to `rule_1`,
`rule_2`, and so on. Strategy `confidence` defaults to `high`. Assigned phase
labels are normalized like other phase labels.

Rules share operator validation and matching code with cleaning filters:

| Operators | Matching |
| --- | --- |
| `eq`, `ne` | Exact, case-sensitive scalar equality or inequality |
| `in`, `not_in` | Membership in a YAML list |
| `contains`, `icontains` | Literal substring, case-sensitive or case-insensitive |
| `regex` | Regular expression; use `(?i)` for case-insensitive patterns |
| `lt`, `lte`, `gt`, `gte` | Numeric comparison; `value` must be a number |
| `is_null`, `not_null` | Missing/populated cell; omit `value` |

Missing columns never satisfy a condition. Null cells only match `is_null`.
List-valued cells follow the [cleaning-filter matching rules](#cleaning-filters).
Custom condition columns are loaded by `clean` and included
in phase-command resume fingerprints. Quote identifier and date strings when
YAML could otherwise interpret them as numbers or dates.

Explicit matches record the strategy name in `phase_source`, `explicit` in
`phase_strategy`, and the rule name and input values in `phase_reason`. CT and MRI T1
scoring fetches the shared default bonus (1000 points) for resolved explicit
assignments, so a reviewed assignment takes priority over an
automatically assigned candidate in the same exam and phase slot. Unmatched and
inferred rows receive no bonus. Phase applicability and diagnostic-product
eligibility still apply. The numeric bonus is calculated during scoring and is
not stored as a CSV column; persisted strategy provenance supports later scoring
with the current default value.

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

Use `type: rules` to infer CT and MRI phases from descriptions, timing,
acquisition order, and other DICOM-derived features:

```yaml
phase_curation:
  strategies:
    - type: rules
  fallback: OTHER
```

An optional `mapping` can rename the canonical outputs. Timing windows and MRI
profiles are built in; use `type: ontology` for site-specific controlled labels.

#### Evidence and precedence

Text matching is case-insensitive and uses whole terms, including accented
letters and common separators. Fields are checked in the configured
`phase_curation.text_columns` order. The first field with phase
evidence controls phase matching and context, even when it only says `post` or
`dynamic`. A later field cannot override it with a contradictory phase.

Standard `ContrastBolusAgentPhase` values provide direct phase evidence.
Conflicting values within a volume remain unresolved. Explicit phases and
resolved strategies before `rules` also provide context for MRI dynamics;
order-based inference preserves their labels.

Mixed descriptions such as `w/wo` or `pre/post` describe a group containing both
conditions. Both conditions must occur in the same scalar value. They can make
the first ordered acquisition native when it has no conflicting phase evidence;
ambiguous order leaves it unresolved. A single `w/o` or a Dixon water suffix
does not establish mixed context.

#### Timing

The built-in duration windows are:

| Phase | Duration |
| --- | --- |
| `ARTERIAL` | 20–35 seconds |
| `PORTAL_VENOUS` | 60–90 seconds |
| `DELAYED` | 3–15 minutes |
| `HEPATOBILIARY` (MRI only) | 20 minutes, or 2 hours through 2h59m59s |

Complete durations are parsed together: `1 min 31 sec` is 91 seconds, outside
the portal window. Conflicting durations without a phase label remain unresolved.
For CT, a duration attached to a named delayed phase must fit the delayed window.
MRI resolves named phases and combined profiles before durations, except that an
unambiguous hepatobiliary duration overrides generic delayed wording: `tardif 2h` is
hepatobiliary. These windows are curation heuristics.

#### MRI dynamic acquisitions

Dynamics use canonical `time`, then `acquisition_order` to refine ties or replace
missing times. Supplied volume ranks apply within one series only. Incomplete or
ambiguous chronology leaves affected acquisitions unresolved.

Generic order inference requires at least three distinct acquisitions, including
compatible explicit phases. Full dynamics progress through `NATIVE`, `ARTERIAL`,
`PORTAL_VENOUS`, then `DELAYED`; post-only groups start at arterial. A compatible
native anchors the progression and counts once. Explicit phases keep their
positions, and earlier generic acquisitions cannot consume a succeeding named
phase. A matching `time_delta` window takes precedence over the generic order
template within these bounds, and can resolve a phase with fewer than three
acquisitions.

Named profiles use these mappings:

| Profile | Phase progression |
| --- | --- |
| ART-PORT | First arterial, last portal; intermediate positions unresolved |
| ART/PORT/LATE | First arterial, last delayed; middle portal only with three acquisitions |
| Multiart | All arterial |
| Mask+Multiart | First native, then arterial |

Vendor `PhN` numbering alone does not identify a phase: `Ph1` is not automatically
arterial. A compatible unnumbered native establishes postcontrast context for
numbered acquisitions. An unlabelled acquisition matching the same protocol
after removing `PhN` can supply that native context. Bare contrast-agent names
such as `gado` alone do not establish postcontrast status.

#### Protocol and Dixon grouping

Groups preserve patient, date, study, sequence-family, plane, MIP/MPR, and
subtraction boundaries. Dixon components can share compatible native context,
but each component has its own acquisition ranks and timing reference.
Reconstruction volumes therefore do not consume extra phase positions. Sharing
authoritative phase labels between components requires a compatible, validated
acquisition identity.

Raw `ImageType` differences do not split protocol groups. Structured tokens still
identify Dixon components and image products. Component names such as `water` and
Dixon suffixes such as `mDIXON-W` also identify reconstructions; contrast shorthand
such as `w/wo` does not identify water.

`time_delta` is the volume's canonical `time` minus the first volume's `time` in
its component group, in seconds. Compatible native and explicitly labelled
acquisitions of that component participate in the reference. Missing or invalid
times leave the value empty; acquisition ranks do not supply elapsed seconds.
Raw injection clocks do not calibrate these deltas.

#### Native fallback

Compatible perfusion context can identify a separate unlabelled native from any
diagnostic T1 sequence. This requires two distinct acquisitions when a
postcontrast phase is resolved, or three when only unresolved dynamic context
is available.

After contextual inference, an eligible diagnostic T1 acquisition without
perfusion or postcontrast evidence falls back to `NATIVE`. One acquisition is
sufficient; repeated descriptions and chronology are not required. This records
`mri_perfusion_source=native_no_perfusion_evidence` and confidence `fallback`.
It does not anchor dynamic progression and ranks below context-supported native
evidence during selection. Phase conflicts, unresolved dynamics, numbered phases,
mixed descriptions, and unsupported durations retain their dedicated inference
routes; they do not receive this final fallback.

Inspect `phase_text_column`, `mri_perfusion_source`, `mri_perfusion_reason`, and
`mri_dynamic_order_source` to trace the selected evidence and acquisition order.

### TotalSegmentator prediction

Use `type: totalsegmentator` to consume or generate a prediction. The `phase`
command invokes TotalSegmentator only for rows that reach this strategy. If it
appears after ontology and rules, rows already resolved by either earlier
strategy skip model inference.

Prediction uses an existing `nifti_path` when available. Otherwise, the `phase`
command can convert the row's `dicom_path` into a temporary NIfTI using the same
conversion code as `convert`, including archive inputs. Enable
`force_nifti_conversion: true` in this strategy or pass `--force-nifti-conversion`
to `phase`; conversion is disabled by default. Temporary files are
removed after prediction; their paths are not saved to the output CSV.
Conversion failures are recorded in the phase error CSV.

```yaml
phase_curation:
  strategies:
    - type: rules
    - type: totalsegmentator
      column: totalseg_phase
      force_nifti_conversion: false
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

Cropping can be configured for the complete segmentation stage:

```yaml
segmentation:
  backend: totalsegmentator
  crop: true
  crop_margin_mm: 50.0
  modalities:
    CT:
      tasks:
        - task: total
          extra:
            roi_subset: [liver]
```

Cropping runs after backend segmentation and before modality-specific
post-processing. It uses one foreground bounding box for the image and masks,
retains up to `crop_margin_mm` of existing image data on each side, and never
adds padded voxels beyond the true image extent. The defaults are `crop: false`
and `crop_margin_mm: 50.0`. CLI `--crop` or `--no-crop` overrides the manifest;
`--crop-margin-mm` overrides its margin for that run.

Optional task keys include `extra`, `output`, `outputs`, `fetch_output`, and
`fetch_outputs`. Each modality may define its own `postprocess` block to combine
logical masks and apply morphological operations in a configurable order.
The same logical outputs may be used across modalities, for example mapping
both `liver_lesions` and `liver_lesions_mr` to `mask_liver_tumor_path`.

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
`mask_*_path` column names, and `.nii.gz` filenames are accepted and normalized to
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
Morphology uses scikit-image's distance-transform-based `isotropic_dilation`,
`isotropic_erosion`, `isotropic_opening`, and `isotropic_closing`, including their
image-boundary behavior.

`fill_holes` and `largest_cc` accept `connectivity: 1`, `2`, or `3` (face,
face-and-edge, or face-edge-and-corner neighbors). Defaults are `1` for hole
filling and `3` for largest-component selection. Empty masks remain empty under
morphology and component selection.

Masks must be 3-D; logical combinations require matching shapes and affines.
Positive voxels are treated as foreground. Each distinct output is saved as a
binary `uint8` NIfTI and appears in the corresponding `mask_<output>_path` CSV column,
including intermediate outputs. New names create `<output>.nii.gz`; existing
logical task names write to their configured `fetch_output` filename.
Reusing an output name replaces its current value for subsequent steps and saves
its final value.
Only explicitly named outputs are written. To clip an original mask to a merged
result, add an `intersection` step with that original mask as the output.
For example, an output named `liver_tumor` mapped to `fetch_output: liver_lesions`
updates `liver_lesions.nii.gz`, and `mask_liver_tumor_path` still points to that file.

The full sequence is computed before any output is written. Postprocessing errors
fail the row by default (`postprocess.on_failure: fail`). Setting
`postprocess.on_failure: warn_only` records a warning and retains successfully
segmented mask paths in the CSV without reporting derived outputs from the failed
sequence. This includes missing or unreadable inputs, geometry errors, and
operation or output-writing errors. It does not roll back files already written
before an output-writing error. Segmentation and cropping errors still fail the row.
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

### Registration stages

`registration.stages` is an ordered list of `mask_rigid`, `mask_affine`,
`mi_affine`, `mask_elastic`, and `mi_elastic` steps. It replaces the default
`mask_rigid -> mask_affine` chain. An empty list runs only baseline evaluation
and automatic initialization. All linear steps must precede elastic steps,
and each stage type may appear once. The old enable flags, global stopping
thresholds, and stage-acceptance maps are no longer supported.

```yaml
registration:
  stages:
    - type: mask_rigid
    - type: mi_affine
      metric: {roi: organ_band, sampling_seed: 42}
      acceptance: {minimum_mi_improvement: 0.0, maximum_dice_decrease: 0.002}
    - type: mi_elastic
      run_if: always
      metric: {roi: organ}
      optimizer: {iterations: 25}
```

See [the registration workflow](workflow.md) for objective selection,
per-stage conditions, deformation limits, and rejection behavior.

### Individual tumor masks

Disable tumor fusion while retaining organ registration with:

```yaml
registration:
  tumor_consensus_method: none
  output_space: reference  # Resample each scan's own tumor to the reference grid.
```

Use `output_space: moving` to retain each original tumor path instead. This mode
preserves empty masks, leaves missing tumors missing, and bypasses organ clipping.
Original paths remain in `source_<tumor_mask_column>` in either output space.

### Registration pyramid levels

Shared `linear_shrink_factors` / `linear_smoothing_sigmas_mm` and
`elastic_shrink_factors` / `elastic_smoothing_sigmas_mm` default to `[4, 2, 1]`
and `[2.0, 1.0, 0.0]`. Each stage can override them under `pyramid`:

```yaml
registration:
  stages:
    - type: mask_affine
      pyramid:
        shrink_factors: [2, 1]
        smoothing_sigmas_mm: [1.0, 0.0]
```

The lists must be nonempty and have equal lengths. Shrink factors are strictly
decreasing positive integers; smoothing widths are finite nonnegative values
in millimeters, in nonincreasing order. `[1]` / `[0]` uses one unsmoothed level.
Omitting a final factor of 1 uses a coarser final optimization grid. Acceptance
still uses full-resolution source data. Pyramid settings do not enable stages.

### Radiomics

`radiomics.modalities.CT.pyradiomics` and `radiomics.modalities.MR.pyradiomics`
each contain a complete PyRadiomics parameter mapping. The CSV's `Modality`
column selects the configuration; names are case-insensitive and `MRI` aliases
`MR`. Every modality retained after filtering must be configured, and missing
or unknown modality values produce an error before extraction.

The bundled CT settings retain the HU resegmentation range; MR omits it.
Adjust each mapping's intensity settings for your dataset. Without explicit
settings, the defaults also omit this range for MR. Older tables without a
`Modality` column use CT defaults.

For a shared configuration, the existing `radiomics.pyradiomics` mapping and
`--pyradiomics_settings` YAML file remain supported. A manifest must use either
`modalities` or the shared `pyradiomics` mapping; configurations are not merged.
Manifest settings take precedence over the CLI settings file.

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
