param(
    [string]$OutputDir = "dist-sidecar",
    [string]$TargetTriple = "x86_64-pc-windows-msvc"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TargetDir = Join-Path $ProjectRoot $OutputDir
$WorkDir = Join-Path $ProjectRoot "build\sidecar"
$SpecDir = Join-Path $WorkDir "spec"
$PyInstallerName = "freerouterd-$TargetTriple"
$ExpectedExe = Join-Path $TargetDir "$PyInstallerName.exe"

if (-not (Test-Path $VenvPython)) {
    throw "Missing .venv. Run .\run.ps1 -InstallOnly -RuntimeOnly first."
}

New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
New-Item -ItemType Directory -Force -Path $SpecDir | Out-Null

Write-Host "Installing packaging dependencies..."
& $VenvPython -m pip install -e ".[packaging]"
if ($LASTEXITCODE -ne 0) {
    throw "Could not install packaging dependencies."
}

Write-Host "Building FreeRouter sidecar..."
& $VenvPython -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --name $PyInstallerName `
    --distpath $TargetDir `
    --workpath $WorkDir `
    --specpath $SpecDir `
    --hidden-import app.main `
    --hidden-import app.sidecar `
    --collect-submodules app `
    (Join-Path $ProjectRoot "app\sidecar.py")
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller sidecar build failed."
}

if (-not (Test-Path $ExpectedExe)) {
    throw "Expected sidecar executable was not produced: $ExpectedExe"
}

Write-Host "Built sidecar: $ExpectedExe"
