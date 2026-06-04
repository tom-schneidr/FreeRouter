param(
    [switch]$NoDesktopShortcut
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

Set-Location $ProjectRoot

Write-Host "Building production FreeRouter desktop app..."
Invoke-Checked "npm.cmd" @("run", "build:desktop")

Write-Host "Refreshing desktop shortcuts..."
$installArgs = @()
if ($NoDesktopShortcut) {
    $installArgs += "-NoDesktopShortcut"
}
$installCommandArgs = @(
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    (Join-Path $ProjectRoot "install-desktop.ps1")
)
$installCommandArgs += $installArgs
Invoke-Checked "powershell.exe" $installCommandArgs

Write-Host "FreeRouter desktop rebuild complete."
