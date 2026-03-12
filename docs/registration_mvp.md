# Registration MVP (March 2026)

## Implemented now
- Shared registration components in `imperandi.process.registration`:
  - intra-patient grouping/task builder (`GroupingKeys`, `build_intra_patient_tasks`)
  - transform artifact persistence (`save_transform_artifacts`)
  - tumor consensus per visit (`build_visit_consensus`)
  - longitudinal tumor consistency audit (`build_longitudinal_audit`)
  - organ extraction/normalization helpers (`normalize_image_and_masks`)
- `register-intra-patient` refactor:
  - explicit intra mode: `auto`, `multiphasic`, `longitudinal`
  - task-based patient execution using shared grouping utilities
  - transform `.tfm` + JSON metadata persisted per row
  - extra metadata columns for task kind, reference row, mode, and transform artifacts
- New command `register-tumor-consensus`:
  - per-visit consensus mask generation across phases
  - component-level tumor descriptors (centroid, volume, bbox)
  - longitudinal audit flags across visits
- `register-population` extension:
  - optional organ-centric normalization after rigid registration
  - configurable crop mode, margin, spacing, orientation, background retention, centering
  - normalized artifacts written with metadata

## Intentionally left for later
- Full lesion identity tracking across complex split/merge trajectories.
- Multi-backend registration plugin system (currently SimpleITK path only).
- Advanced consensus post-processing beyond rule-based union/intersection/majority.
- Dedicated cross-visit common-space registration cache reuse for consensus audit.

## Extension points
- Add alternative consensus strategies inside `imperandi.process.registration.consensus`.
- Add stricter/organ-specific audit heuristics in `imperandi.process.registration.audit`.
- Add cohort- or modality-specific normalization presets through CLI profile wrappers.
- Add additional transform serialization backends in `imperandi.process.registration.transforms`.
