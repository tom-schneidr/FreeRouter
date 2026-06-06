param(
    [switch]$NoLint,
    [switch]$NoFrontend,
    [switch]$NoDesktop
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Python = if (Get-Command "python" -ErrorAction SilentlyContinue) { "python" } else { $VenvPython }

& $Python -m pytest -p no:cacheprovider
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if (-not $NoLint) {
    & $Python -m ruff check .
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

if (-not $NoFrontend) {
    & npm.cmd run typecheck:web
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & npm.cmd run test:web
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & npm.cmd run build:web
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not $NoDesktop) {
    & npm.cmd run check:desktop
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
