param(
    [string]$ConfigPath = (Join-Path $PSScriptRoot "ircad.yaml")
)

$ErrorActionPreference = "Stop"

# End-to-end IRCAD usage example for pipeline v2. Pass another project file to
# run the same validate/plan/run workflow against a different real dataset.
if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    throw "Missing project configuration: $ConfigPath"
}

imperandi validate $ConfigPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

imperandi plan $ConfigPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

imperandi run $ConfigPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "IRCAD v2 pipeline completed successfully."
