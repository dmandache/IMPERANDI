# Usage: .\examples\benchmarks\mri_multiphase\run.ps1 DICOM_DIR WORK_DIR
# This study uses metadata-only MRI phase curation.
param(
    [Parameter(Position = 0)]
    [string]$InputDir,
    [Parameter(Position = 1)]
    [string]$WorkDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (-not $InputDir -or -not $WorkDir) {
    [Console]::Error.WriteLine("Usage: mri_multiphase/run.ps1 DICOM_DIR WORK_DIR")
    exit 2
}

$BenchmarkDir = Split-Path -Parent $PSScriptRoot
$PythonBin = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$NumWorkers = if ($env:SLURM_CPUS_PER_TASK) { $env:SLURM_CPUS_PER_TASK } else { "4" }
if ($null -ne $env:TOTALSEG_PHASE) {
    [Console]::Error.WriteLine('The backend used for phase prediction from image contrast does not currently support MRI, only CT. Running with `TOTALSEG_PHASE=0` (metadata only).')
}
$Manifest = Join-Path $PSScriptRoot "manifest.yaml"
$Reference = if ($env:REFERENCE_CSV) {
    $env:REFERENCE_CSV
} else {
    Join-Path $PSScriptRoot "reference_cohort.csv"
}
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

# 1. Curate the full MR export using this study's metadata rules.
& $PythonBin -m imperandi --log-file (Join-Path $WorkDir "ingest.log") ingest `
    $InputDir $WorkDir --manifest $Manifest `
    --id_source tags --patient_key_from PatientID `
    --study_id_from StudyInstanceUID --series_id_from SeriesInstanceUID `
    --snapshot_tags --num_workers $NumWorkers `
    --tags ImageComments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$CohortInventory = Join-Path $WorkDir "dicom_index_curated.csv"

# 2. Select each patient's date with the greatest resolved phase coverage.
& $PythonBin (Join-Path $BenchmarkDir "cohort.py") build mri_multiphase `
    $CohortInventory $WorkDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 3. Compare with Gross et al., including phases detected before date selection.
& $PythonBin (Join-Path $BenchmarkDir "cohort.py") compare mri_multiphase `
    (Join-Path $WorkDir "selected_cohort.csv") $Reference $WorkDir `
    --inventory $CohortInventory
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
