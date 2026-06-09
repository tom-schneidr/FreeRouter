param(
    [int]$ApiPort = 8000,
    [switch]$NoInstall
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
. (Join-Path $ProjectRoot "scripts\lib\process-utils.ps1")

$script:DevCleanupPort = $null
$script:DevCleanupProcess = $null
$script:DevShuttingDown = $false
$script:DevBackendEnv = @{}
$script:DevBackendJob = $null

function Invoke-DevCleanup {
    $script:DevShuttingDown = $true
    $job = Get-Job -Name "FreeRouterDevBackend" -ErrorAction SilentlyContinue
    if ($job) {
        Write-Host "Stopping dev backend supervisor..."
        Stop-Job -Job $job -ErrorAction SilentlyContinue
        Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
    }
    $script:DevBackendJob = $null
    if ($script:DevCleanupProcess -and -not $script:DevCleanupProcess.HasExited) {
        Write-Host "Stopping dev backend process tree (PID $($script:DevCleanupProcess.Id))..."
        Stop-ProcessTree -ProcessId $script:DevCleanupProcess.Id | Out-Null
    }
    if ($null -ne $script:DevCleanupPort) {
        Write-Host "Clearing gateway listeners on ports $($script:DevCleanupPort)-$($script:DevCleanupPort + 20)..."
        Stop-FreeRouterGatewayPorts -MinPort $script:DevCleanupPort -MaxPort ($script:DevCleanupPort + 20)
    }
}

function Start-DevBackendSupervisorJob {
    param(
        [int]$Port,
        [string]$ProjectRoot,
        [string]$VenvPython,
        [string]$DesktopToken
    )

    $script:DevBackendJob = Start-Job -Name "FreeRouterDevBackend" -ScriptBlock {
        param($Root, $Python, $Port, $Token, $AppDataDir)
        Set-Location $Root
        $env:PYTHONUNBUFFERED = "1"
        $env:FREEROUTER_DESKTOP_TOKEN = $Token
        $env:FREEROUTER_DESKTOP_PROJECT_ROOT = $Root
        $env:FREEROUTER_APP_DATA_DIR = $AppDataDir
        $env:FREEROUTER_DEV_BACKEND = "1"

        while ($true) {
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = $Python
            $psi.Arguments = "-m uvicorn app.main:app --host 127.0.0.1 --port $Port"
            $psi.WorkingDirectory = $Root
            $psi.UseShellExecute = $false
            $psi.CreateNoWindow = $true

            $proc = [System.Diagnostics.Process]::Start($psi)
            if (-not $proc) {
                break
            }
            $proc.WaitForExit()
            if ($proc.ExitCode -ne 42) {
                break
            }
            Start-Sleep -Milliseconds 500
        }
    } -ArgumentList $ProjectRoot, $VenvPython, $Port, $DesktopToken, $ProjectRoot
}

Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action { Invoke-DevCleanup } | Out-Null
trap {
    Invoke-DevCleanup
    break
}

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

function Start-ChildProcess {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory,
        [hashtable]$Environment = @{}
    )

    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $FilePath
    $escapedArguments = @()
    foreach ($argument in $Arguments) {
        $escapedArguments += '"' + ($argument -replace '"', '\"') + '"'
    }
    $startInfo.Arguments = $escapedArguments -join " "
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    foreach ($key in $Environment.Keys) {
        $startInfo.Environment[$key] = [string]$Environment[$key]
    }

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw "Could not start $FilePath"
    }
    return $process
}

function Test-FreeRouterHealth {
    param([int]$Port)

    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/v1/gateway/health.json" -TimeoutSec 2
        if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
            return $true
        }
    }
    catch {
        return $false
    }
    return $false
}

function Test-BackendIsCurrent {
    param([int]$Port)

    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/gateway/health.json" -TimeoutSec 3
        if ($null -eq $health.status) {
            return $false
        }
        $snapshot = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/gateway/live/snapshot" -TimeoutSec 3
        return $null -ne $snapshot.data
    }
    catch {
        return $false
    }
}

function Resolve-DevBackendPort {
    param([int]$PreferredPort)

    if (Test-PortOpen -Port $PreferredPort) {
        Write-Host "Freeing 127.0.0.1:$PreferredPort..."
        Stop-FreeRouterGatewayPorts -MinPort $PreferredPort -MaxPort ($PreferredPort + 20)
        Start-Sleep -Seconds 2
    }

    if (Test-FreeRouterHealth -Port $PreferredPort -and (Test-BackendIsCurrent -Port $PreferredPort)) {
        Write-Host "Reusing current dev backend on 127.0.0.1:$PreferredPort."
        return @{
            Port = $PreferredPort
            ReuseExisting = $true
        }
    }

    if (Test-PortOpen -Port $PreferredPort) {
        $alternate = Find-FreePort -StartingPort $PreferredPort
        if (-not $alternate) {
            throw "127.0.0.1:$PreferredPort is held by an old FreeRouter backend and no free port was found in range. Run scripts\stop-all-freerouter.ps1 as Administrator, then retry."
        }
        Write-Host ""
        Write-Host "WARNING: 127.0.0.1:$PreferredPort is still serving an older bundled backend" -ForegroundColor Yellow
        Write-Host "         (could not stop it without admin). Using 127.0.0.1:$alternate for this dev session." -ForegroundColor Yellow
        Write-Host "         To reclaim port ${PreferredPort}: run scripts\stop-all-freerouter.ps1 as Administrator." -ForegroundColor Yellow
        Write-Host ""
        return @{
            Port = $alternate
            ReuseExisting = $false
        }
    }

    return @{
        Port = $PreferredPort
        ReuseExisting = $false
    }
}

function Wait-ForBackendCurrent {
    param(
        [int]$Port,
        [int]$Seconds = 35
    )

    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-BackendIsCurrent -Port $Port) {
            return
        }
        Start-Sleep -Milliseconds 500
    }
    throw "Timed out waiting for current FreeRouter backend on 127.0.0.1:$Port"
}

function Test-PortOpen {
    param([int]$Port)

    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $connect = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        if (-not $connect.AsyncWaitHandle.WaitOne(500)) {
            return $false
        }
        $client.EndConnect($connect)
        return $true
    }
    catch {
        return $false
    }
    finally {
        $client.Close()
    }
}

function Get-PortOwnerSummary {
    param([int]$Port)

    $rows = netstat -ano | Select-String ":$Port" | ForEach-Object {
        $parts = $_.Line.Trim() -split "\s+"
        if ($parts.Count -ge 5 -and $parts[1] -like "*:$Port") {
            [pscustomobject]@{
                State = $parts[3]
                Pid = $parts[4]
            }
        }
    }
    $owners = $rows |
        Where-Object { $_.State -eq "LISTENING" } |
        Select-Object -ExpandProperty Pid -Unique

    if (-not $owners) {
        return "No listening process was reported by netstat."
    }

    $details = foreach ($owner in $owners) {
        $process = Get-Process -Id ([int]$owner) -ErrorAction SilentlyContinue
        if ($process) {
            "$owner ($($process.ProcessName))"
        }
        else {
            "$owner (process details unavailable)"
        }
    }
    return $details -join ", "
}

function Find-FreePort {
    param(
        [int]$StartingPort,
        [int]$Limit = 20
    )

    for ($offset = 1; $offset -le $Limit; $offset += 1) {
        $candidate = $StartingPort + $offset
        if (-not (Test-PortOpen -Port $candidate)) {
            return $candidate
        }
    }
    return $null
}

function Wait-ForUrl {
    param(
        [string]$Url,
        [int]$Seconds = 35
    )

    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                return
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }
    throw "Timed out waiting for $Url"
}

Set-Location $ProjectRoot

if (-not $NoInstall) {
    Invoke-Checked "powershell.exe" @(
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        (Join-Path $ProjectRoot "run.ps1"),
        "-InstallOnly",
        "-RuntimeOnly"
    )
}

if (-not (Test-Path $VenvPython)) {
    throw "Missing .venv. Run .\run.ps1 -InstallOnly -RuntimeOnly first, or rerun .\dev-desktop.ps1 without -NoInstall."
}

if (-not (Test-Path (Join-Path $ProjectRoot "node_modules"))) {
    if ($NoInstall) {
        throw "Missing node_modules. Run npm install first, or rerun .\dev-desktop.ps1 without -NoInstall."
    }
    Invoke-Checked "npm.cmd" @("install")
}

$backendPlan = Resolve-DevBackendPort -PreferredPort $ApiPort
$ApiPort = $backendPlan.Port
$reuseBackend = [bool]$backendPlan.ReuseExisting
$script:DevCleanupPort = $ApiPort

$desktopToken = [guid]::NewGuid().ToString()
$script:DevBackendEnv = @{
    "PYTHONUNBUFFERED" = "1"
    "FREEROUTER_DESKTOP_TOKEN" = $desktopToken
    "FREEROUTER_DESKTOP_PROJECT_ROOT" = $ProjectRoot
    "FREEROUTER_APP_DATA_DIR" = $ProjectRoot
    "FREEROUTER_DEV_BACKEND" = "1"
}

Write-Host "Building React UI from current source..."
Invoke-Checked "npm.cmd" @("run", "build:web")

$backendProcess = $null

try {
    if (-not $reuseBackend) {
        Write-Host "Starting source backend on http://127.0.0.1:$ApiPort/v1 (auto-restart enabled)..."
        Start-DevBackendSupervisorJob -Port $ApiPort -ProjectRoot $ProjectRoot -VenvPython $VenvPython -DesktopToken $desktopToken

        Wait-ForUrl -Url "http://127.0.0.1:$ApiPort/v1/gateway/health.json"
        Wait-ForBackendCurrent -Port $ApiPort
    }

    $env:FREEROUTER_DEV_BACKEND = "1"
    $env:FREEROUTER_DESKTOP_TOKEN = $desktopToken
    $env:GATEWAY_PORT = "$ApiPort"
    Write-Host "Desktop will use dev backend at http://127.0.0.1:$ApiPort/app"
    Write-Host 'Launching Tauri desktop shell (close window hides to tray; tray Quit stops the backend)...'
    Invoke-Checked "npm.cmd" @("--workspace", "@freerouter/desktop", "run", "dev")
}
finally {
    Invoke-DevCleanup
    Remove-Item Env:\FREEROUTER_DEV_BACKEND -ErrorAction SilentlyContinue
    Remove-Item Env:\FREEROUTER_DEV_BACKEND_PID -ErrorAction SilentlyContinue
    Remove-Item Env:\FREEROUTER_DESKTOP_TOKEN -ErrorAction SilentlyContinue
    Remove-Item Env:\GATEWAY_PORT -ErrorAction SilentlyContinue
}
