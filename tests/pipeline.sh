#!/usr/bin/env bash
set -euo pipefail

# Default value
DIR_PATH="./tests/data"

# If first argument exists, override
if [ -n "$1" ]; then
  DIR_PATH="$1"
fi

imperandi ingest "${DIR_PATH}/IRCAD_DICOM" --snapshot_tags
imperandi convert "${DIR_PATH}/dicom_index_clean.csv" "${DIR_PATH}/IRCAD_nifti"
imperandi segment "${DIR_PATH}/nifti_index.csv"
imperandi phase "${DIR_PATH}/nifti_index.csv"
imperandi radiomics "${DIR_PATH}/nifti_index.csv"

echo "Pipeline executed successfully!"