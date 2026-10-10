# Usage: .\examples\benchmarks\run.ps1 TARGET DICOM_DIR WORK_DIR
# Set $env:TOTALSEG_PHASE = "1" to enable CT phase prediction.
# PYTHON, SLURM_CPUS_PER_TASK, and REFERENCE_CSV override their defaults.
param(
    [Parameter(Position = 0)]
    [string]$Target,
    [Parameter(Position = 1)]
    [string]$InputDir,
    [Parameter(Position = 2)]
    [string]$WorkDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $Target -or -not $InputDir -or -not $WorkDir) {
    [Console]::Error.WriteLine("Usage: run.ps1 TARGET DICOM_DIR WORK_DIR")
    exit 2
}
if ($Target -cnotin @("mri_multiphase", "ct_portal_venous")) {
    [Console]::Error.WriteLine("Unknown benchmark: $Target")
    exit 2
}

$PythonBin = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$NumWorkers = if ($env:SLURM_CPUS_PER_TASK) { $env:SLURM_CPUS_PER_TASK } else { "4" }
$UseTotalSegPhase = if ($null -ne $env:TOTALSEG_PHASE) { $env:TOTALSEG_PHASE } else { "0" }
if ($Target -ceq "mri_multiphase" -and $null -ne $env:TOTALSEG_PHASE) {
    [Console]::Error.WriteLine('The backend used for phase prediction from image contrast does not currently support MRI, only CT. Running with `TOTALSEG_PHASE=0` (metadata only).')
    $UseTotalSegPhase = "0"
}
if ($UseTotalSegPhase -cnotin @("0", "1")) {
    [Console]::Error.WriteLine("TOTALSEG_PHASE must be 0 or 1")
    exit 2
}

$TargetDir = Join-Path $PSScriptRoot $Target
$Manifest = Join-Path $TargetDir "manifest.yaml"
$Reference = if ($env:REFERENCE_CSV) {
    $env:REFERENCE_CSV
} else {
    Join-Path $TargetDir "reference_cohort.csv"
}
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

& $PythonBin -m imperandi --log-file (Join-Path $WorkDir "ingest.log") ingest `
    $InputDir $WorkDir --manifest $Manifest `
    --id_source tags --patient_key_from PatientID `
    --study_id_from StudyInstanceUID --series_id_from SeriesInstanceUID `
    --snapshot_tags --num_workers $NumWorkers `
    --tags ImageComments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$CohortInventory = Join-Path $WorkDir "dicom_index_curated.csv"

if ($Target -ceq "ct_portal_venous" -and $UseTotalSegPhase -ceq "1") {
    # Convert the full cleaned CT inventory so metadata-resolved and
    # predictor-resolved candidates compete in the same best-per-exam selection.
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

& $PythonBin (Join-Path $PSScriptRoot "cohort.py") build $Target `
    $CohortInventory $WorkDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$CompareArgs = @()
if ($Target -ceq "mri_multiphase") {
    $CompareArgs = @("--inventory", $CohortInventory)
}
& $PythonBin (Join-Path $PSScriptRoot "cohort.py") compare $Target `
    (Join-Path $WorkDir "selected_cohort.csv") $Reference $WorkDir @CompareArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
