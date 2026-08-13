param(
    [string]$InputDir = (Join-Path $PSScriptRoot "data\input"),
    [string]$WorkDir = (Join-Path $PSScriptRoot "data\work"),
    [string]$Manifest = "generic",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

# 1. Discover DICOM files, extract metadata, and curate the cohort table.
& $Python -m imperandi ingest `
    $InputDir `
    $WorkDir `
    --manifest $Manifest `
    --snapshot_tags `
    --num_workers 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 2. Convert each retained DICOM volume to NIfTI.
& $Python -m imperandi convert `
    (Join-Path $WorkDir "dicom_index_clean.csv") `
    (Join-Path $WorkDir "NIFTI") `
    --csv_path_out (Join-Path $WorkDir "nifti_index.csv") `
    --manifest $Manifest `
    --num_workers 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 3. Create manifest-configured organ and lesion segmentations.
& $Python -m imperandi segment `
    (Join-Path $WorkDir "nifti_index.csv") `
    --manifest $Manifest `
    --num_workers 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 4. Resolve contrast phases using metadata and model fallbacks.
& $Python -m imperandi phase `
    (Join-Path $WorkDir "nifti_index.csv") `
    --manifest $Manifest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 5. Extract radiomics for every retained row in this demonstration cohort.
& $Python -m imperandi radiomics `
    (Join-Path $WorkDir "nifti_index.csv") `
    (Join-Path $WorkDir "nifti_index_radiomics.csv") `
    --manifest $Manifest `
    --skip_filter
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "TCGA-LIHC pipeline completed. Results: $WorkDir"
