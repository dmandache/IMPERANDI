#!/usr/bin/env bash
set -euo pipefail

# Default value for DIR_PATH is "./tests/data" if not provided as an argument
DIR_PATH="${1:-"./tests/data/IRCAD"}"

imperandi ingest "${DIR_PATH}/DICOM" --snapshot_tags --manifest generic
imperandi convert "${DIR_PATH}/dicom_index_clean.csv"
imperandi segment "${DIR_PATH}/nifti_index.csv" --manifest generic
imperandi phase "${DIR_PATH}/nifti_index.csv"
imperandi radiomics "${DIR_PATH}/nifti_index.csv" --manifest generic

echo "Pipeline executed successfully!"