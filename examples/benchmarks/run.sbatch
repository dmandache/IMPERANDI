#!/usr/bin/env bash
#SBATCH --job-name=tcga-lihc-benchmark
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

# Usage:
# sbatch examples/benchmarks/run.sbatch TARGET DICOM_DIR WORK_DIR
# CT metadata + prediction: TOTALSEG_PHASE=1 sbatch examples/benchmarks/run.sbatch ...
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}}"
bash "${REPO_ROOT}/examples/benchmarks/run.sh" "${@}"
