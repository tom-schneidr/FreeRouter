param(
    [string]$HostName = $(if ($env:GATEWAY_HOST) { $env:GATEWAY_HOST } else { "127.0.0.1" }),
    [int]$Port = $(if ($env:GATEWAY_PORT) { [int]$env:GATEWAY_PORT } else { 8000 }),
    [switch]$Reload,
    [switch]$InstallOnly,
    [switch]$RuntimeOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunScript = Join-Path $ProjectRoot "run.ps1"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

& $RunScript -HostName $HostName -Port $Port -InstallOnly -RuntimeOnly:$RuntimeOnly

if ($InstallOnly) {
    exit 0
}

$launcherArgs = @("-m", "app.tray_launcher", "--host", $HostName, "--port", $Port)
if ($Reload) {
    $launcherArgs += "--reload"
}

Write-Host "Starting FreeRouter tray console. Close the window to keep it running in the tray."
& $VenvPython @launcherArgs
