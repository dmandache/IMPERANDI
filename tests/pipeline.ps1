param(
    [string]$ConfigPath = ".\tests\data\imperandi.yaml"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -Path $ConfigPath -PathType Leaf)) {
    throw "Missing project configuration: $ConfigPath. Create one with: imperandi init $ConfigPath"
}

imperandi validate $ConfigPath
imperandi plan $ConfigPath
imperandi run $ConfigPath

Write-Output "Pipeline test completed successfully."
