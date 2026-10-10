# Usage: .\examples\benchmarks\ct_portal_venous\run.ps1 DICOM_DIR WORK_DIR
# Set $env:TOTALSEG_PHASE = "1" to add NIfTI conversion and CT phase prediction.
param(
    [Parameter(Position = 0)]
    [string]$InputDir,
    [Parameter(Position = 1)]
    [string]$WorkDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (-not $InputDir -or -not $WorkDir) {
    [Console]::Error.WriteLine("Usage: ct_portal_venous/run.ps1 DICOM_DIR WORK_DIR")
    exit 2
}

$BenchmarkDir = Split-Path -Parent $PSScriptRoot
$PythonBin = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$NumWorkers = if ($env:SLURM_CPUS_PER_TASK) { $env:SLURM_CPUS_PER_TASK } else { "4" }
$UseTotalSegPhase = if ($null -ne $env:TOTALSEG_PHASE) { $env:TOTALSEG_PHASE } else { "0" }
if ($UseTotalSegPhase -cnotin @("0", "1")) {
    [Console]::Error.WriteLine("TOTALSEG_PHASE must be 0 or 1")
    exit 2
}
$Manifest = Join-Path $PSScriptRoot "manifest.yaml"
$Reference = if ($env:REFERENCE_CSV) {
    $env:REFERENCE_CSV
} else {
    Join-Path $PSScriptRoot "reference_cohort.csv"
}
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

# 1. Curate the full CT export using this study's metadata rules.
& $PythonBin -m imperandi --log-file (Join-Path $WorkDir "ingest.log") ingest `
    $InputDir $WorkDir --manifest $Manifest `
    --id_source tags --patient_key_from PatientID `
    --study_id_from StudyInstanceUID --series_id_from SeriesInstanceUID `
    --snapshot_tags --num_workers $NumWorkers `
    --tags ImageComments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$CohortInventory = Join-Path $WorkDir "dicom_index_curated.csv"

# 2. Optionally predict unresolved CT phases from image contrast.
if ($UseTotalSegPhase -ceq "1") {
    # Convert all cleaned CT candidates so metadata and prediction results
    # compete in the same best-per-exam selection.
    & $PythonBin -m imperandi --log-file (Join-Path $WorkDir "convert.log") convert `
        (Join-Path $WorkDir "dicom_index_clean.csv") (Join-Path $WorkDir "NIFTI") `
        --csv_path_out (Join-Path $WorkDir "nifti_index.csv") --manifest $Manifest `
        --num_workers $NumWorkers
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $PythonBin -m imperandi --log-file (Join-Path $WorkDir "phase.log") phase `
        (Join-Path $WorkDir "nifti_index.csv") (Join-Path $WorkDir "nifti_index_phased.csv") `
        --manifest $Manifest `
        --selected_csv_path (Join-Path $WorkDir "nifti_index_phased_curated.csv")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $CohortInventory = Join-Path $WorkDir "nifti_index_phased_curated.csv"
}

# 3. Select the earliest portal-venous exam and compare with Sarfati et al.
& $PythonBin (Join-Path $BenchmarkDir "cohort.py") build ct_portal_venous `
    $CohortInventory $WorkDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $PythonBin (Join-Path $BenchmarkDir "cohort.py") compare ct_portal_venous `
    (Join-Path $WorkDir "selected_cohort.csv") $Reference $WorkDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
