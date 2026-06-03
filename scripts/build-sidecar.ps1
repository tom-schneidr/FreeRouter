param(
    [string]$OutputDir = "dist-sidecar"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TargetDir = Join-Path $ProjectRoot $OutputDir

if (-not (Test-Path $VenvPython)) {
    throw "Missing .venv. Run .\run.ps1 -InstallOnly -RuntimeOnly first."
}

New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
Write-Host "Sidecar packaging is not implemented yet."
Write-Host "Target output will be: $TargetDir\freerouterd.exe"
Write-Host "The next milestone should replace this placeholder with PyInstaller or Nuitka packaging."
