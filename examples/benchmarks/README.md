# TCGA-LIHC cohort ingestions benchmarks

Interactive notebooks: [demos](../demos/).

Compare independently built IMPERANDI cohorts with two published cohorts:

| Benchmark | Reference paper | Published target | Public reference |
| --- | --- | --- | --- |
| [MRI multiphase](mri_multiphase/) | [Gross et al., Data in Brief (2023)](https://doi.org/10.1016/j.dib.2023.109662) | 17 four-phase T1w MRI patients (benchmark permits incomplete phase sets) | Patient/date list and tumor-subgroup membership |
| [CT portal venous](ct_portal_venous/) | [Sarfati et al., MICCAI (2023)](https://arxiv.org/abs/2307.04617) | 49 portal venous CT patients with Ishak scores | Author patient/date/label CSV; processed scans |

Use the **full TCGA-LIHC DICOM collection** from
[TCIA](https://doi.org/10.7937/K9/TCIA.2016.IMMQW8UQ), preserving PatientID and
study/series UIDs. The authors' smaller, already selected image releases cannot
measure recovery from the original collection.

## Download DICOMs with idc-index

From the repository root, install the optional
[IDC downloader](https://idc-index.readthedocs.io/en/stable/api/idc_index.html):

```bash
python -m pip install --upgrade idc-index

# Download all available CT/MR series.
python examples/benchmarks/download.py /data/TCGA-LIHC
```

The downloader selects CT/MR series from `tcga_lihc`
and calls IDC's downloader directly. It uses all available patients by default,
without filtering to either paper's cohort. DICOMs go under
`PatientID/StudyInstanceUID/Modality/SeriesInstanceUID`; pass `/data/TCGA-LIHC`
directly to the benchmark runners below. After downloading, `idc_selection.csv`
records the selected patients, modalities, study/series UIDs, and series sizes in MB
for the demos' inventory overview. The output directory is a required
positional argument. Local downloads under `examples/benchmarks/data` are
ignored by Git.
IDC's available data may differ from the historical TCIA snapshot: check reference
patient/series availability before interpreting missing selections.

For a small smoke test, use a separate directory and repeat `--patient` as needed:

```bash
python examples/benchmarks/download.py /data/lihc-smoke \
  --patient TCGA-BC-A10Y --patient TCGA-DD-A113
```

The full cohort comparison uses the default download of all available patients.

## Run

Activate an environment with IMPERANDI installed (`python -m pip install -e .`).
From the repository root, choose a study. Each study's runner sits beside its
manifest, reference CSV, and inclusion notes:

```bash
# Gross et al.: metadata-only MRI phase curation and cohort comparison.
bash examples/benchmarks/mri_multiphase/run.sh /data/TCGA-LIHC /work/lihc-mr

# Sarfati et al.: metadata-only CT phase curation and cohort comparison.
bash examples/benchmarks/ct_portal_venous/run.sh /data/TCGA-LIHC /work/lihc-ct

# CT with TotalSegmentator phase prediction, including NIfTI conversion.
TOTALSEG_PHASE=1 bash examples/benchmarks/ct_portal_venous/run.sh /data/TCGA-LIHC /work/lihc-ct-ts
```

The study runners take **two positional arguments**: `DICOM_DIR WORK_DIR`.
Both use metadata-only phase curation by default. For CT, `TOTALSEG_PHASE=1`
converts the full cleaned inventory and runs metadata rules followed by
TotalSegmentator prediction before cohort selection; `TOTALSEG_PHASE=0` skips
conversion and prediction. Install `imperandi[segment]` for the CT predictor.
For MRI, setting `TOTALSEG_PHASE` prints the CT-only backend explanation and
continues with metadata only. Use separate work directories for the two CT modes.

From PowerShell, use the study's `run.ps1` with the same two positional arguments:

```powershell
.\examples\benchmarks\mri_multiphase\run.ps1 "D:\data\TCGA-LIHC" "D:\work\lihc-mr"
.\examples\benchmarks\ct_portal_venous\run.ps1 "D:\data\TCGA-LIHC" "D:\work\lihc-ct"

# Enable CT conversion and phase prediction.
$env:TOTALSEG_PHASE = "1"
.\examples\benchmarks\ct_portal_venous\run.ps1 "D:\data\TCGA-LIHC" "D:\work\lihc-ct-ts"
Remove-Item Env:TOTALSEG_PHASE
```

All runners support the existing environment overrides: `PYTHON` chooses the
interpreter, `SLURM_CPUS_PER_TASK` sets the worker count (default 4),
`REFERENCE_CSV` supplies an alternate reference, and `TOTALSEG_PHASE` controls
CT image-based phase prediction.

### Shared entry point and Slurm

The shared runners dispatch to the study scripts and preserve the existing
**three-argument syntax**: `TARGET DICOM_DIR WORK_DIR`. Existing commands continue
to work:

```bash
bash examples/benchmarks/run.sh mri_multiphase /data/TCGA-LIHC /work/lihc-mr
TOTALSEG_PHASE=1 bash examples/benchmarks/run.sh ct_portal_venous /data/TCGA-LIHC /work/lihc-ct-ts
```

```powershell
.\examples\benchmarks\run.ps1 mri_multiphase "D:\data\TCGA-LIHC" "D:\work\lihc-mr"
.\examples\benchmarks\run.ps1 ct_portal_venous "D:\data\TCGA-LIHC" "D:\work\lihc-ct"
```

The shared Slurm wrapper still forwards all arguments to `run.sh`. Submit from
the repository root and add your cluster's account/partition options:

```bash
sbatch examples/benchmarks/run.sbatch mri_multiphase /data/TCGA-LIHC /work/lihc-mr
sbatch examples/benchmarks/run.sbatch ct_portal_venous /data/TCGA-LIHC /work/lihc-ct
TOTALSEG_PHASE=1 sbatch examples/benchmarks/run.sbatch ct_portal_venous /data/TCGA-LIHC /work/lihc-ct-ts
```

## Main comparison: cohort building

For MRI, `cohort.py build` uses `dicom_index_curated.csv` and retains partially
resolved T1w dates (at least one canonical phase), prioritizing complete dates,
then greater phase coverage. This permissive recovery measure must not be
interpreted as reproduction of the paper's strict four-phase eligibility.
`selected_cohort.csv` records the number of resolved phases and completeness;
`summary.json` reports complete and partial MRI exam counts. CT portal-venous
also uses `dicom_index_curated.csv` by default. With `TOTALSEG_PHASE=1`, it uses
`nifti_index_phased_curated.csv`, produced after conversion and the
metadata→TotalSegmentator phase chain. Neither path reads
reference IDs. The builder writes `selected_cohort.csv` and
`selection_audit.csv`; `dicom_index_clean.csv` retains the cleaned volume
inventory and, when prediction is enabled, `nifti_index_phased.csv` retains the
full CT phase output. Use separate work directories for the two CT modes.
`cohort.py compare` writes:

- `cohort_metrics.csv`: counts, TP/FP/FN, precision, recall, F1 and Jaccard at
  patient, patient/date and patient/date/phase levels. MRI additionally reports
  `patient_date_full_phase`, comparing dates containing all four canonical phases.
- `cohort_mismatches.csv`: every extra or missing entry at each level.
- `mri_phase_matches.csv` (MRI only): each reference patient/date's selected,
  reference and matched phase counts, matched/missing phase names, and
  `full_phase_match` highlighting recovery of all four phases on the reference date.
  Missing exams have zero matched phases; repeated phases count once.
- `mri_reference_phase_matches.csv` (MRI with `--inventory`): alternate matching
  across all eligible inventory dates. Each reference patient/date reports the
  required phases detected before cohort date selection, including full four-phase
  matches. Both MRI runners pass `dicom_index_curated.csv` automatically.
- `summary.json`: exact-series matching availability, MRI complete/partial
  selected phase coverage, matched phase totals and the 0-4 matched-phase
  distribution across reference exams, and MRI tumor-subgroup patient recovery.
  This is subgroup coverage, not a tumor-presence classifier.
  `mri_reference_phase_detection` contains alternate inventory phase counts and
  the 0-4 distribution when `--inventory` is supplied.

The supplied references do not contain SeriesInstanceUIDs, so date/phase matches
do not establish exact series identity. An extended reference with nonempty
`patient_key,date,phase,series_id` enables that additional metric. Same-day studies
remain indistinguishable without those identifiers. Empty predictions are scored
as zero recovery; undefined ratios are left empty.

Review discrepancies against source images and eligibility records. If a
reviewed cohort is compared separately, keep the automatic outputs and label the
reviewed results explicitly; do not filter predictions to published membership.

## Results

| Cohort selection measure | CT portal-venous<br>Metadata only | CT portal-venous<br>Metadata + image | MRI multiphasic<br>Metadata only |
| :--- | :---: | :---: | :---: |
| Patients in reference |  49 | 49 | 17 |
| **Patients selected** | **40** | **65** | **36** |
| ↳ *matched to reference* | 33 | 49 | 16 |
| ↳ *outside reference* | 7 | 16 | 20 |
| Patient-level recall | 67.3% | 100% | 94.1% |
| Patient–date recall | 53.1% | 89.8% | 76.5% |
| Patient–date–phase recall | 53.1% | 89.8% | 25.0% |

These benchmarks assess automated recovery of two published cohorts from the broader TCGA-LIHC collection. Selection does not use reference patient identifiers, and recorded decisions make discrepancies auditable.

For CT, adding image-based phase prediction recovers all reference patients, although the selected acquisition date does not always match. For MRI, patient recovery is substantially higher than recovery of the required date–phase combinations: identifying a patient does not establish recovery of a complete multiphasic examination. The MRI benchmark permits incomplete phase sets, unlike the published cohort.

Patients outside the references are not necessarily incorrect selections: the original studies also applied criteria such as histological-score availability, tumor visibility or visual image quality that this benchmark does not fully reproduce.

These results measure cohort agreement, not standalone phase-classification accuracy. Without reference series identifiers, exact image recovery cannot be established.
