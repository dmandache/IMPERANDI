#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${1:-${SCRIPT_DIR}/data/input}"
WORK_DIR="${2:-${SCRIPT_DIR}/data/work}"
MANIFEST="${3:-generic}"
PYTHON_BIN="${PYTHON:-python3}"

mkdir -p "${WORK_DIR}"

# 1. Discover DICOM files, extract metadata, and curate the cohort table.
"${PYTHON_BIN}" -m imperandi ingest \
  "${INPUT_DIR}" \
  "${WORK_DIR}" \
  --manifest "${MANIFEST}" \
  --snapshot_tags \
  --num_workers 1

# 2. Convert each retained DICOM volume to NIfTI.
"${PYTHON_BIN}" -m imperandi convert \
  "${WORK_DIR}/dicom_index_clean.csv" \
  "${WORK_DIR}/NIFTI" \
  --csv_path_out "${WORK_DIR}/nifti_index.csv" \
  --manifest "${MANIFEST}" \
  --num_workers 1

# 3. Create manifest-configured organ and lesion segmentations.
"${PYTHON_BIN}" -m imperandi segment \
  "${WORK_DIR}/nifti_index.csv" \
  --manifest "${MANIFEST}" \
  --num_workers 1

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

echo "TCGA-LIHC pipeline completed. Results: ${WORK_DIR}"
