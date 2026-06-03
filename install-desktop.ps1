param(
    [switch]$NoDesktopShortcut
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunScript = Join-Path $ProjectRoot "run.ps1"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Pythonw = Join-Path $ProjectRoot ".venv\Scripts\pythonw.exe"
$IconPath = Join-Path $ProjectRoot "data\freerouter.ico"

function New-FreeRouterShortcut {
    param(
        [string]$ShortcutPath
    )

    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($ShortcutPath)
    $shortcut.TargetPath = $Pythonw
    $shortcut.Arguments = "-m app.desktop_app"
    $shortcut.WorkingDirectory = $ProjectRoot
    $shortcut.Description = "FreeRouter local desktop app"
    if (Test-Path $IconPath) {
        $shortcut.IconLocation = $IconPath
    }
    $shortcut.Save()
}

& $RunScript -InstallOnly -RuntimeOnly

if (-not (Test-Path $Pythonw)) {
    throw "Could not find pythonw.exe in the local virtual environment."
}

& $Python -m app.desktop_icon --output $IconPath
if ($LASTEXITCODE -ne 0) {
    throw "Could not create FreeRouter desktop icon."
}

$ProgramsDir = [Environment]::GetFolderPath("Programs")
$StartMenuDir = Join-Path $ProgramsDir "FreeRouter"
New-Item -ItemType Directory -Force -Path $StartMenuDir | Out-Null
New-FreeRouterShortcut -ShortcutPath (Join-Path $StartMenuDir "FreeRouter.lnk")

if (-not $NoDesktopShortcut) {
    $DesktopDir = [Environment]::GetFolderPath("Desktop")
    New-FreeRouterShortcut -ShortcutPath (Join-Path $DesktopDir "FreeRouter.lnk")
}

Write-Host "FreeRouter desktop shortcuts installed."
Write-Host "Start Menu: $StartMenuDir\FreeRouter.lnk"
if (-not $NoDesktopShortcut) {
    Write-Host "Desktop: $DesktopDir\FreeRouter.lnk"
}
