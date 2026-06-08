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
$ReactDist = Join-Path $ProjectRoot "apps\ui\dist"
$InvalidModuleFiles = Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "app") -Recurse -File -Filter "*.py" |
    Where-Object { $_.BaseName -ne "__init__" -and $_.BaseName -notmatch '^[A-Za-z_][A-Za-z0-9_]*$' }

if (-not (Test-Path $VenvPython)) {
    throw "Missing .venv. Run .\run.ps1 -InstallOnly -RuntimeOnly first."
}
if (-not (Test-Path (Join-Path $ReactDist "index.html"))) {
    throw "Missing React build output. Run npm run build:web before building the sidecar."
}
if ($InvalidModuleFiles) {
    $InvalidPaths = ($InvalidModuleFiles | ForEach-Object { $_.FullName }) -join [Environment]::NewLine
    Write-Warning "Excluding non-importable Python filenames from the sidecar build:`n$InvalidPaths"
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
& $VenvPython -m app.desktop_icon --sync-all
if ($LASTEXITCODE -ne 0) {
    throw "Could not sync FreeRouter desktop icons."
}

$PyInstallerArgs = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onefile",
    "--name", $PyInstallerName,
    "--distpath", $TargetDir,
    "--workpath", $WorkDir,
    "--specpath", $SpecDir,
    "--hidden-import", "app.main",
    "--hidden-import", "app.sidecar",
    "--collect-submodules", "app",
    "--add-data", "$ReactDist;apps\ui\dist"
)
foreach ($InvalidModuleFile in $InvalidModuleFiles) {
    $RelativeModulePath = [IO.Path]::GetRelativePath($ProjectRoot, $InvalidModuleFile.FullName)
    $InvalidModuleName = [IO.Path]::ChangeExtension($RelativeModulePath, $null).Replace([IO.Path]::DirectorySeparatorChar, ".")
    $PyInstallerArgs += @("--exclude-module", $InvalidModuleName)
}
$PyInstallerArgs += (Join-Path $ProjectRoot "app\sidecar.py")

& $VenvPython @PyInstallerArgs
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller sidecar build failed."
}

if (-not (Test-Path $ExpectedExe)) {
    throw "Expected sidecar executable was not produced: $ExpectedExe"
}

Write-Host "Built sidecar: $ExpectedExe"
