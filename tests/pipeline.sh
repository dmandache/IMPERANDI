#!/usr/bin/env bash
set -euo pipefail

# Default value for DIR_PATH is "./tests/data" if not provided as an argument
DIR_PATH="${1:-"./tests/data"}"

imperandi ingest "${DIR_PATH}/IRCAD_DICOM" --snapshot_tags --manifest generic
imperandi convert "${DIR_PATH}/dicom_index_clean.csv" "${DIR_PATH}/IRCAD_nifti"
imperandi segment "${DIR_PATH}/nifti_index.csv"
imperandi phase "${DIR_PATH}/nifti_index.csv"
imperandi radiomics "${DIR_PATH}/nifti_index.csv"

echo "Pipeline executed successfully!"