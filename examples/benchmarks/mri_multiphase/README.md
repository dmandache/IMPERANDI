# MRI multiphase cohort

[Paper: Gross et al., Data in Brief (2023)](https://doi.org/10.1016/j.dib.2023.109662)
· [Data v1.1](https://zenodo.org/records/8179129)
· [Release file inventory](https://zenodo.org/records/8179129/preview/nifti_and_segms.zip?include_deleted=0)

## Inclusion and reference

The paper starts from 97 TCGA-LIHC patients: excluding CT-only and PET-only
patients leaves 40 MRI patients with 73 studies; excluding 23 without a complete multiphasic study
leaves **17 liver patients**. Excluding 3 without visible tumor/residual tumor
leaves **14 tumor patients**. Required sequences are T1 native, arterial, portal
venous and delayed. T2/DWI are not required. For multiple eligible studies the
authors prefer pretreatment imaging, then visually highest quality.

`reference_cohort.csv` transcribes the 17 patient/date directories in the v1.1
NIfTI archive; `include_tumor` records the paper's tumor subgroup (14 patients).
Dates in the archive are MM-DD-YYYY. `cohort.py` expands each reference patient
to the four required phases. No DICOM series UID is claimed.

## IMPERANDI protocol

The runner uses metadata-only phase curation. If `TOTALSEG_PHASE` is set, it
explains that the image-contrast phase-prediction backend currently supports CT
only, then continues with `TOTALSEG_PHASE=0` (metadata only).

`manifest.yaml` retains MR volumes and performs native sequence/phase annotation
and candidate ranking. `cohort.py` accepts a date containing **at least one resolved canonical T1 phase**.
For each patient it selects the date with the most distinct resolved phases,
falling back to the earliest date when phase counts tie. Same-day study UIDs
contribute to that date's phase coverage. Repeated reconstructions count once;
the highest-scoring candidate is retained per phase, with study and volume UIDs
breaking score ties. Source UIDs are preserved, and phases from different dates
are never combined.
This relaxed recovery benchmark is **not equivalent** to the paper's strict
four-phase inclusion criterion. `selected_cohort.csv` records
`n_resolved_phases` and `complete_multiphase` to distinguish partial dates. Treatment status and tumor visibility require
clinical/image review and are not inferred from metadata. The benchmark reports permissive 17-patient recovery and coverage of the reference
tumor subgroup, alongside exact phase-level recall (which penalizes missing phases).
`summary.json` includes complete/partial selected-date counts and the phase-count
distribution. Comparison also writes `mri_phase_matches.csv`: each reference
patient/date has `n_matched_phases` (0-4), matched and missing phase names, and
`full_phase_match`. Full phase matches require all four canonical phases on the
reference date; a complete selected exam on another date contributes zero.
Repeated phases count once, including across same-day study UIDs.
`summary.json` additionally reports `matched_mri_phases`,
`matched_complete_mri_exams`, `matched_partial_mri_exams`, and
`matched_mri_phase_count_distribution` (including exams with zero matches).
The `patient_date_full_phase` metric measures agreement between complete
four-phase dates. Console output lists each reference exam's matched count and
highlights full recovery with `full_phase_match` and a full-match total.

The runners also perform alternate matching with the complete candidate inventory.
For each published patient/date, `mri_reference_phase_matches.csv` counts how many
required phases were detected across all eligible T1 MRI candidates, before
cohort date selection. Thus a phase on the reference date counts even if another
date was selected for that patient. CT, T2, unresolved and noncanonical phases
are excluded. Repeated phases count once; phases from different dates are never
combined. `n_detected_phases`, `n_matched_phases`, matched/missing phase names and
`full_phase_match` make each result inspectable. `summary.json` records aggregate
counts under `mri_reference_phase_detection`.

To recompute both comparisons from existing CSVs in PowerShell:

```powershell
$work = "D:\work\lihc-mr"
python examples/benchmarks/cohort.py compare mri_multiphase `
  "$work/selected_cohort.csv" examples/benchmarks/mri_multiphase/reference_cohort.csv `
  "$work" --inventory "$work/dicom_index_curated.csv"
```

Geometry tolerances in the manifest support volume assembly; they are not paper
inclusion thresholds. No extra thickness or matrix-size filter is imposed.

## Preprocessing boundary

The paper converts DICOM to NIfTI and registers native/portal/delayed images to
arterial using BioImage Suite 3.5: nonrigid B-spline FFD, normalized mutual
information, gradient descent, three pyramid levels, final 80 mm control-point
spacing, followed by visual review. This metadata-only benchmark does not
reproduce that preprocessing.
