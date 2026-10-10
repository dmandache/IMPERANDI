#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:?Usage: run.sh TARGET DICOM_DIR WORK_DIR}"
INPUT_DIR="${2:?Supply the full TCGA-LIHC DICOM export}"
WORK_DIR="${3:?Supply a separate work directory}"
PYTHON_BIN="${PYTHON:-python}"
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-4}"
case "${TARGET}" in
  mri_multiphase|ct_portal_venous) ;;
  *) echo "Unknown benchmark: ${TARGET}" >&2; exit 2 ;;
esac
USE_TOTALSEG_PHASE="${TOTALSEG_PHASE-0}"
if [[ "${TARGET}" == "mri_multiphase" && "${TOTALSEG_PHASE+x}" == "x" ]]; then
  echo 'The backend used for phase prediction from image contrast does not currently support MRI, only CT. Running with `TOTALSEG_PHASE=0` (metadata only).' >&2
  USE_TOTALSEG_PHASE=0
fi
case "${USE_TOTALSEG_PHASE}" in
  0|1) ;;
  *) echo "TOTALSEG_PHASE must be 0 or 1" >&2; exit 2 ;;
esac
MANIFEST="${BENCHMARK_DIR}/${TARGET}/manifest.yaml"
REFERENCE="${REFERENCE_CSV:-${BENCHMARK_DIR}/${TARGET}/reference_cohort.csv}"
mkdir -p "${WORK_DIR}"

"${PYTHON_BIN}" -m imperandi --log-file "${WORK_DIR}/ingest.log" ingest \
  "${INPUT_DIR}" "${WORK_DIR}" --manifest "${MANIFEST}" \
  --id_source tags --patient_key_from PatientID \
  --study_id_from StudyInstanceUID --series_id_from SeriesInstanceUID \
  --snapshot_tags --num_workers "${NUM_WORKERS}" \
  --tags ImageComments
COHORT_INVENTORY="${WORK_DIR}/dicom_index_curated.csv"

if [[ "${TARGET}" == "ct_portal_venous" && "${USE_TOTALSEG_PHASE}" == "1" ]]; then
  # TotalSegmentator's CT phase predictor needs NIfTI input. Convert the full
  # cleaned CT inventory so metadata-resolved and predictor-resolved candidates
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

"${PYTHON_BIN}" "${BENCHMARK_DIR}/cohort.py" build "${TARGET}" \
  "${COHORT_INVENTORY}" "${WORK_DIR}"
COMPARE_ARGS=()
if [[ "${TARGET}" == "mri_multiphase" ]]; then
  COMPARE_ARGS=(--inventory "${COHORT_INVENTORY}")
fi
"${PYTHON_BIN}" "${BENCHMARK_DIR}/cohort.py" compare "${TARGET}" \
  "${WORK_DIR}/selected_cohort.csv" "${REFERENCE}" "${WORK_DIR}" "${COMPARE_ARGS[@]}"
