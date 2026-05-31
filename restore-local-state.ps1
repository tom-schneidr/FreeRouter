param(
    [Parameter(Mandatory = $true)]
    [string]$Backup,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    $Python = "python"
}

$args = @("-m", "app.local_backup", "import", $Backup)
if ($Overwrite) {
    $args += "--overwrite"
}

& $Python @args
