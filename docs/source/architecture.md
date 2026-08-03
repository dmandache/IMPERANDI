# Architecture

IMPERANDI uses a fixed, dependency-checked stage graph and a typed project
configuration. The project file selects policy and backends; it does not expose
the internal command sequence as a customization surface.

## Module breakdown

| Module | Responsibility |
|---|---|
| `imperandi.config` | Strict Pydantic models, built-in profiles, path resolution, effective-config hashing |
| `imperandi.io` | CSV/Parquet artifact round-tripping and product-level CSV size warning |
| `imperandi.identity` | Source identity extraction, normalization, crosswalk/HMAC resolution, collision QC, sensitive mapping |
| `imperandi.annotations` | Composite-key ontologies, declarative rule packs, evidence resolution |
| `imperandi.curation` | CT/MRI routing and built-in metadata curation |
| `imperandi.pipeline` | Stage contracts, artifact dependencies, run state, checkpoint/resume orchestration |
| `imperandi.pipeline.stages.core` | Index, identity, volume assembly, annotation, resolution, selection |
| `imperandi.pipeline.stages.imaging` | Conversion, image prediction, segmentation, registration, radiomics, publication |
| `imperandi.process.registration` | Pair-table registration and transform persistence |

The primary source layout is:

```text
src/imperandi/
├── config/
│   ├── models.py
│   ├── loader.py
│   └── profiles/liver_ct_mri.yaml
├── io/tables.py
├── identity/resolver.py
├── annotations/{ontology,rules,resolver}.py
├── curation/
│   ├── ct/{curate,rules}.py
│   └── mri/{curate,rules}.py
├── pipeline/
│   ├── base.py
│   ├── runner.py
│   ├── defaults.py
│   └── stages/{core,imaging}.py
└── process/registration.py
```

## Stage graph

| Order | Stage | Requires | Produces |
|---:|---|---|---|
| 1 | `01_index` | input sources | `instances_raw` |
| 2 | `02_identity` | `instances_raw` | `instances`, `identity_map` |
| 3 | `03_assemble` | `instances` | `volumes` |
| 4 | `04_annotate` | `volumes` | `volumes_annotated`, `volumes_shortlist` |
| 5 | `05_convert` | `volumes_shortlist` | `volumes_converted` |
| 6 | `06_predict_phase` | `volumes_converted` | `volumes_predicted` |
| 7 | `07_resolve_select` | `volumes_predicted` | `volumes_resolved`, `selected_volumes` |
| 8 | `08_segment` | `selected_volumes` | `volumes_segmented` |
| 9 | `09_register` | `volumes_segmented` | `volumes_registered` |
| 10 | `10_radiomics` | `volumes_registered` | `radiomics_table` |
| 11 | `11_publish` | `radiomics_table` | `cohort_index` |

This is a two-pass curation design:

1. metadata evidence annotates, excludes, and shortlists volumes;
2. shortlisted images are converted;
3. optional image evidence is added;
4. evidence is resolved and clinical slots are selected;
5. expensive downstream processing runs only on selected volumes.

## Evidence contract

Each source writes to its own column:

| Concept | Ontology | Explicit rule | Inferred rule | Image | Resolution outputs |
|---|---|---|---|---|---|
| Phase | `phase_ontology` | `phase_rules_explicit` | `phase_rules_inferred` | `phase_image` | `phase_resolved`, `phase_source`, `phase_conflict` |
| Slot | `slot_ontology` | `slot_rules_explicit` | `slot_rules_inferred` | `slot_image` | `clinical_slot`, `clinical_slot_source`, `clinical_slot_conflict` |

Precedence and disagreement behavior are configuration. Resolution selects a
value; it never deletes competing evidence.

## `curate_by_modality` interface

```python
def curate_by_modality(
    df: pandas.DataFrame,
    patient_col: str = "patient_key",
    study_col: str | None = "study_id",
    date_col: str = "date",
    contextual_strategies: Collection[str] | None = None,
    curators: Collection[str] | None = None,
) -> dict[str, object]: ...
```

Return keys:

| Key | Value |
|---|---|
| `ct` | CT curation result, or `None` when no CT rows exist |
| `mri` | MRI curation result, or `None` when no MR/MRI rows exist |
| `other` | Unrouted rows, preserving their source fields |
| `curated_all` | Concatenated CT/MR annotated rows with `curation_modality` |
| `selected_long_all` | Concatenated built-in selections for inspection |

The pipeline invokes this interface with `patient_col="patient_id"`. Routing is
case-normalized; `MR` and `MRI` use the MRI path; mixed/unknown labels remain in
`other`.

## Test matrix

| Route | Metadata cases | Selection cases | Failure/safety cases |
|---|---|---|---|
| CT | localizer/derived detection, phase text, geometry and quality features | native, arterial, portal venous, delayed; deterministic ties | missing columns, invalid/ambiguous metadata, excluded series |
| MRI | sequence family, Dixon, DWI/ADC, dynamic T1, explicit and inferred perfusion | T2, DWI, T1 phase slots; contextual acquisition ordering | localizer/key images, quantitative Dixon labels, missing timing/context |
| Mixed | one DataFrame containing CT and MR exams | independent modality selection with merged result | unsupported modality isolation and no CT/MR cross-contamination |
| Evidence | ontology, explicit rules, inferred rules, image fallback | precedence and provenance | conflict flag/error policy, equal-priority rule conflicts |
| Storage | CSV and Parquet | structured DICOM cells | scalar/list mixed cells, internal-only CSV warning threshold |
| Identity | source, crosswalk, HMAC, crosswalk-then-HMAC | stable canonical `patient_id` | missing ID, collision QC, sensitive-field policy |
| Pipeline | metadata-only CT/MR run | final CT and MR clinical slots | resume state, artifact existence, disabled-stage pass-through |

The smallest safe unit remains modality routing plus annotation only: load a
pre-indexed table, disable conversion and all heavy stages, and verify
`04_annotate`/`07_resolve_select` artifacts. This exercises the public config,
identity, CT/MRI router, evidence resolver, selection, and publication without
requiring image backends.
