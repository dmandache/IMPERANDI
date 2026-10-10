#!/usr/bin/env bash
set -euo pipefail

# Usage: bash examples/benchmarks/ct_portal_venous/run.sh DICOM_DIR WORK_DIR
# Set TOTALSEG_PHASE=1 to add NIfTI conversion and CT phase prediction.
STUDY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd -- "${STUDY_DIR}/.." && pwd)"
INPUT_DIR="${1:?Usage: ct_portal_venous/run.sh DICOM_DIR WORK_DIR}"
WORK_DIR="${2:?Supply a separate work directory}"
PYTHON_BIN="${PYTHON:-python}"
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-4}"
USE_TOTALSEG_PHASE="${TOTALSEG_PHASE-0}"
case "${USE_TOTALSEG_PHASE}" in
  0|1) ;;
  *) echo "TOTALSEG_PHASE must be 0 or 1" >&2; exit 2 ;;
esac
MANIFEST="${STUDY_DIR}/manifest.yaml"
REFERENCE="${REFERENCE_CSV:-${STUDY_DIR}/reference_cohort.csv}"
mkdir -p "${WORK_DIR}"

# 1. Curate the full CT export using this study's metadata rules.
"${PYTHON_BIN}" -m imperandi --log-file "${WORK_DIR}/ingest.log" ingest \
  "${INPUT_DIR}" "${WORK_DIR}" --manifest "${MANIFEST}" \
  --id_source tags --patient_key_from PatientID \
  --study_id_from StudyInstanceUID --series_id_from SeriesInstanceUID \
  --snapshot_tags --num_workers "${NUM_WORKERS}" \
  --tags ImageComments
COHORT_INVENTORY="${WORK_DIR}/dicom_index_curated.csv"

# 2. Optionally predict unresolved CT phases from image contrast.
if [[ "${USE_TOTALSEG_PHASE}" == "1" ]]; then
  # Convert all cleaned CT candidates so metadata and prediction results
  # compete in the same best-per-exam selection.
  "${PYTHON_BIN}" -m imperandi --log-file "${WORK_DIR}/convert.log" convert \
    "${WORK_DIR}/dicom_index_clean.csv" "${WORK_DIR}/NIFTI" \
    --csv_path_out "${WORK_DIR}/nifti_index.csv" --manifest "${MANIFEST}" \
    --num_workers "${NUM_WORKERS}"

  "${PYTHON_BIN}" -m imperandi --log-file "${WORK_DIR}/phase.log" phase \
    "${WORK_DIR}/nifti_index.csv" "${WORK_DIR}/nifti_index_phased.csv" \
    --manifest "${MANIFEST}" \
    --selected_csv_path "${WORK_DIR}/nifti_index_phased_curated.csv"

  COHORT_INVENTORY="${WORK_DIR}/nifti_index_phased_curated.csv"
fi

# 3. Select the earliest portal-venous exam and compare with Sarfati et al.
"${PYTHON_BIN}" "${BENCHMARK_DIR}/cohort.py" build ct_portal_venous \
  "${COHORT_INVENTORY}" "${WORK_DIR}"
"${PYTHON_BIN}" "${BENCHMARK_DIR}/cohort.py" compare ct_portal_venous \
  "${WORK_DIR}/selected_cohort.csv" "${REFERENCE}" "${WORK_DIR}"
