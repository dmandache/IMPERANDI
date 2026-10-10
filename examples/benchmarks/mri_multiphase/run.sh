#!/usr/bin/env bash
set -euo pipefail

# Usage: bash examples/benchmarks/mri_multiphase/run.sh DICOM_DIR WORK_DIR
# This study uses metadata-only MRI phase curation.
STUDY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd -- "${STUDY_DIR}/.." && pwd)"
INPUT_DIR="${1:?Usage: mri_multiphase/run.sh DICOM_DIR WORK_DIR}"
WORK_DIR="${2:?Supply a separate work directory}"
PYTHON_BIN="${PYTHON:-python}"
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-4}"
if [[ "${TOTALSEG_PHASE+x}" == "x" ]]; then
  echo 'The backend used for phase prediction from image contrast does not currently support MRI, only CT. Running with `TOTALSEG_PHASE=0` (metadata only).' >&2
fi
MANIFEST="${STUDY_DIR}/manifest.yaml"
REFERENCE="${REFERENCE_CSV:-${STUDY_DIR}/reference_cohort.csv}"
mkdir -p "${WORK_DIR}"

# 1. Curate the full MR export using this study's metadata rules.
"${PYTHON_BIN}" -m imperandi --log-file "${WORK_DIR}/ingest.log" ingest \
  "${INPUT_DIR}" "${WORK_DIR}" --manifest "${MANIFEST}" \
  --id_source tags --patient_key_from PatientID \
  --study_id_from StudyInstanceUID --series_id_from SeriesInstanceUID \
  --snapshot_tags --num_workers "${NUM_WORKERS}" \
  --tags ImageComments
COHORT_INVENTORY="${WORK_DIR}/dicom_index_curated.csv"

# 2. Select each patient's date with the greatest resolved phase coverage.
"${PYTHON_BIN}" "${BENCHMARK_DIR}/cohort.py" build mri_multiphase \
  "${COHORT_INVENTORY}" "${WORK_DIR}"

# 3. Compare with Gross et al., including phases detected before date selection.
"${PYTHON_BIN}" "${BENCHMARK_DIR}/cohort.py" compare mri_multiphase \
  "${WORK_DIR}/selected_cohort.csv" "${REFERENCE}" "${WORK_DIR}" \
  --inventory "${COHORT_INVENTORY}"
