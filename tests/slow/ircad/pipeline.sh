#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${1:-${SCRIPT_DIR}/data/input}"
WORK_DIR="${2:-${SCRIPT_DIR}/data/work}"
MANIFEST="${3:-generic}"
NUM_WORKERS="${4:-1}"
PYTHON_BIN="${PYTHON:-python3}"

mkdir -p "${WORK_DIR}"

echo "Running IRCAD pipeline"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "INPUT_DIR=${INPUT_DIR}"
echo "WORK_DIR=${WORK_DIR}"
echo "MANIFEST=${MANIFEST}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "PYTHON_BIN=${PYTHON_BIN}"

# 1. Discover DICOM files, extract metadata, and curate the cohort table.
"${PYTHON_BIN}" -m imperandi ingest \
  "${INPUT_DIR}" \
  "${WORK_DIR}" \
  --manifest "${MANIFEST}" \
  --snapshot_tags \
  --num_workers "${NUM_WORKERS}"

# 2. Convert each retained DICOM volume to NIfTI.
"${PYTHON_BIN}" -m imperandi convert \
  "${WORK_DIR}/dicom_index_clean.csv" \
  "${WORK_DIR}/NIFTI" \
  --csv_path_out "${WORK_DIR}/nifti_index.csv" \
  --manifest "${MANIFEST}" \
  --num_workers "${NUM_WORKERS}"

# 3. Create manifest-configured organ and lesion segmentations.
"${PYTHON_BIN}" -m imperandi segment \
  "${WORK_DIR}/nifti_index.csv" \
  --manifest "${MANIFEST}" \
  --num_workers "${NUM_WORKERS}"

# 4. Resolve contrast phases using metadata and model fallbacks.
"${PYTHON_BIN}" -m imperandi phase \
  "${WORK_DIR}/nifti_index.csv" \
  --manifest "${MANIFEST}"

# 5. Extract radiomics for every retained row in this demonstration cohort.
"${PYTHON_BIN}" -m imperandi radiomics \
  "${WORK_DIR}/nifti_index.csv" \
  "${WORK_DIR}/nifti_index_radiomics.csv" \
  --manifest "${MANIFEST}" \
  --skip_filter

echo "IRCAD pipeline completed. Results: ${WORK_DIR}"
