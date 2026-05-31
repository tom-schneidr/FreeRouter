param(
    [string]$Output
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    $Python = "python"
}

$args = @("-m", "app.local_backup", "export")
if ($Output) {
    $args += @("--output", $Output)
}

& $Python @args
