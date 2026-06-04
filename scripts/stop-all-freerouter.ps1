# Stop orphaned FreeRouter desktop backends, dev servers, and Tauri shells.
# Run in Windows Terminal (outside Cursor) — use "Run as administrator" if listeners remain.
param(
    [int]$MinPort = 8000,
    [int]$MaxPort = 8020,
    [switch]$Aggressive
)

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$IsAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $IsAdmin) {
    Write-Host "Note: not running as Administrator. Elevated orphans may survive." -ForegroundColor Yellow
}

function Test-FreeRouterHealth {
    param([int]$Port)
    try {
        $response = Invoke-WebRequest -UseBasicParsing `
            -Uri "http://127.0.0.1:$Port/v1/gateway/health.json" `
            -TimeoutSec 2
        return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500)
    }
    catch {
        return $false
    }
}

function Stop-ProcessTree {
    param([int]$ProcessId)
    if ($ProcessId -le 0) {
        return $false
    }
    if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
        return $false
    }

    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    try {
        & taskkill.exe /PID $ProcessId /T /F 2>$null | Out-Null
    }
    finally {
        $ErrorActionPreference = $previousErrorAction
    }
    return $true
}

function Get-ListenerPids {
    param([int]$Port)
    $pids = @()
    $rows = netstat -ano | Select-String "127\.0\.0\.1:$Port\s+.*LISTENING"
    foreach ($row in $rows) {
        $parts = $row.Line.Trim() -split "\s+"
        if ($parts.Count -ge 5) {
            $pids += [int]$parts[-1]
        }
    }
    if (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue) {
        $pids += Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $Port -State Listen `
            -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess
    }
    return $pids | Where-Object { $_ -gt 0 } | Sort-Object -Unique
}

Write-Host "Stopping FreeRouter processes (project: $ProjectRoot)" -ForegroundColor Cyan

$matched = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and (
        ($_.CommandLine -like "*FreeRouter*" -and $_.CommandLine -match "uvicorn|freerouterd|app\.desktop_app|dev-desktop|@freerouter/desktop") -or
        $_.CommandLine -match "freerouterd" -or
        $_.Name -match "^freerouterd"
    )
}

foreach ($proc in $matched) {
    Write-Host "  Process $($proc.ProcessId) $($proc.Name)"
    Write-Host "    $($proc.CommandLine)"
    Stop-ProcessTree -ProcessId $proc.ProcessId
}

Get-Process -Name "freerouterd*" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  freerouterd $($_.Id)"
    Stop-ProcessTree -ProcessId $_.Id
}

if ($Aggressive) {
    $gatewayPids = Get-NetTCPConnection -LocalAddress 127.0.0.1 -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalPort -ge $MinPort -and $_.LocalPort -le $MaxPort } |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($listenerPid in $gatewayPids) {
        $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$listenerPid" -ErrorAction SilentlyContinue).CommandLine
        Write-Host "  Aggressive: PID $listenerPid"
        if ($cmd) { Write-Host "    $cmd" }
        Stop-ProcessTree -ProcessId $listenerPid
    }
}

for ($port = $MinPort; $port -le $MaxPort; $port += 1) {
    foreach ($listenerPid in Get-ListenerPids -Port $port) {
        $proc = Get-Process -Id $listenerPid -ErrorAction SilentlyContinue
        $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$listenerPid" -ErrorAction SilentlyContinue).CommandLine
        if ($proc -or $cmd) {
            Write-Host "  Port $port listener PID $listenerPid ($($proc.ProcessName))"
            if ($cmd) { Write-Host "    $cmd" }
        }
        else {
            Write-Host "  Port $port listener PID $listenerPid (stale or elevated)"
        }
        Stop-ProcessTree -ProcessId $listenerPid
    }
}

Start-Sleep -Seconds 1

Write-Host ""
Write-Host "Gateway health check:" -ForegroundColor Cyan
$alive = @()
for ($port = $MinPort; $port -le $MaxPort; $port += 1) {
    if (Test-FreeRouterHealth -Port $port) {
        $alive += $port
        Write-Host "  http://127.0.0.1:$port/v1 still responding" -ForegroundColor Yellow
    }
}

if ($alive.Count -eq 0) {
    Write-Host "  All FreeRouter gateway ports in ${MinPort}-${MaxPort} are stopped." -ForegroundColor Green
}
else {
    Write-Host ""
    Write-Host "Some servers are still running on port(s): $($alive -join ', ')." -ForegroundColor Yellow
    Write-Host "Try: Right-click PowerShell -> Run as administrator, then:"
    Write-Host "  .\scripts\stop-all-freerouter.ps1 -Aggressive"
    Write-Host "Or reboot if PIDs are still stuck."
    Write-Host ""
    Write-Host "Task Manager tip: Details tab -> add 'Command line' column -> look for"
    Write-Host "  python / python3.13 running uvicorn app.main:app (not named FreeRouter)."
}
