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

# Keep the shared entry point; each study owns its workflow and defaults.
$StudyRunner = Join-Path (Join-Path $PSScriptRoot $Target) "run.ps1"
& $StudyRunner -InputDir $InputDir -WorkDir $WorkDir
exit $LASTEXITCODE
