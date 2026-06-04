param(
    [switch]$NoDesktopShortcut
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunScript = Join-Path $ProjectRoot "run.ps1"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TauriExe = Join-Path $ProjectRoot "apps\desktop\src-tauri\target\release\freerouter_desktop.exe"
$IconSource = Join-Path $ProjectRoot "apps\desktop\src-tauri\icons\icon.ico"
$IconPath = Join-Path $ProjectRoot "data\freerouter.ico"

function New-FreeRouterShortcut {
    param(
        [string]$ShortcutPath,
        [string]$TargetPath
    )

    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($ShortcutPath)
    $shortcut.TargetPath = $TargetPath
    $shortcut.WorkingDirectory = $ProjectRoot
    $shortcut.Description = "FreeRouter local desktop app"
    if (Test-Path $IconPath) {
        $shortcut.IconLocation = "$IconPath,0"
    }
    $shortcut.Save()
}

& $RunScript -InstallOnly -RuntimeOnly

if (-not (Test-Path $Python)) {
    throw "Could not find python.exe in the local virtual environment."
}

& $Python -m app.desktop_icon --sync-all
if ($LASTEXITCODE -ne 0) {
    throw "Could not sync FreeRouter desktop icons."
}

$needsBuild = -not (Test-Path $TauriExe)
if (-not $needsBuild -and (Test-Path $IconSource)) {
    $needsBuild = (Get-Item $IconSource).LastWriteTimeUtc -gt (Get-Item $TauriExe).LastWriteTimeUtc
}

if ($needsBuild) {
    Write-Host "FreeRouter desktop executable is missing or older than the app icon. Rebuilding..."
    Push-Location $ProjectRoot
    try {
        if (Test-Path $TauriExe) {
            & npm.cmd --workspace @freerouter/desktop run build
        }
        else {
            & npm.cmd run build:desktop
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Desktop build failed."
        }
    }
    finally {
        Pop-Location
    }
}

if (-not (Test-Path $TauriExe)) {
    throw "Expected desktop executable was not produced: $TauriExe"
}

$ProgramsDir = [Environment]::GetFolderPath("Programs")
$StartMenuDir = Join-Path $ProgramsDir "FreeRouter"
New-Item -ItemType Directory -Force -Path $StartMenuDir | Out-Null
New-FreeRouterShortcut -ShortcutPath (Join-Path $StartMenuDir "FreeRouter.lnk") -TargetPath $TauriExe

if (-not $NoDesktopShortcut) {
    $DesktopDir = [Environment]::GetFolderPath("Desktop")
    New-FreeRouterShortcut -ShortcutPath (Join-Path $DesktopDir "FreeRouter.lnk") -TargetPath $TauriExe
}

Write-Host "FreeRouter desktop shortcuts installed."
Write-Host "Target: $TauriExe"
Write-Host "Start Menu: $StartMenuDir\FreeRouter.lnk"
if (-not $NoDesktopShortcut) {
    Write-Host "Desktop: $DesktopDir\FreeRouter.lnk"
}
