#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:?Usage: run.sh TARGET DICOM_DIR WORK_DIR}"
INPUT_DIR="${2:?Supply the full TCGA-LIHC DICOM export}"
WORK_DIR="${3:?Supply a separate work directory}"
case "${TARGET}" in
  mri_multiphase|ct_portal_venous) ;;
  *) echo "Unknown benchmark: ${TARGET}" >&2; exit 2 ;;
esac

# Keep the shared entry point; each study owns its workflow and defaults.
exec bash "${BENCHMARK_DIR}/${TARGET}/run.sh" "${INPUT_DIR}" "${WORK_DIR}"
