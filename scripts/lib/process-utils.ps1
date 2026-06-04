# Shared process cleanup helpers for FreeRouter dev/stop scripts.

function Stop-ProcessTree {
    param([int]$ProcessId)

    if ($ProcessId -le 0) {
        return $false
    }

    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    try {
        & taskkill.exe /PID $ProcessId /T /F 2>$null | Out-Null
        return $LASTEXITCODE -eq 0
    }
    finally {
        $ErrorActionPreference = $previousErrorAction
    }
}

function Get-ListenerPids {
    param([int]$Port)

    $pids = @()
    if (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue) {
        $pids += Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $Port -State Listen `
            -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess
    }
    $rows = netstat -ano | Select-String "127\.0\.0\.1:$Port\s+.*LISTENING"
    foreach ($row in $rows) {
        $parts = $row.Line.Trim() -split "\s+"
        if ($parts.Count -ge 5) {
            $pids += [int]$parts[-1]
        }
    }
    return $pids | Where-Object { $_ -gt 0 } | Sort-Object -Unique
}

function Stop-OrphanGatewayWorkers {
    # Uvicorn --reload and some exits leave multiprocessing spawn_main children alive
    # while netstat still references a dead parent PID.
    $patterns = @(
        "uvicorn.*app\.main:app",
        "multiprocessing\.spawn import spawn_main",
        "freerouterd"
    )
    $filter = "Name='python.exe' OR Name='python3.13.exe' OR Name='python3.exe'"
    $procs = Get-CimInstance Win32_Process -Filter $filter -ErrorAction SilentlyContinue
    foreach ($proc in $procs) {
        $cmd = $proc.CommandLine
        if (-not $cmd) { continue }
        $match = $false
        foreach ($pattern in $patterns) {
            if ($cmd -match $pattern) { $match = $true; break }
        }
        if (-not $match) { continue }
        if ($cmd -notlike "*FreeRouter*" -and $cmd -notmatch "app\.main:app|freerouterd|spawn_main") { continue }
        Write-Host "  Stopping gateway worker PID $($proc.ProcessId)"
        Stop-ProcessTree -ProcessId $proc.ProcessId | Out-Null
    }
}

function Stop-PortListeners {
    param(
        [int]$Port,
        [switch]$Quiet
    )

    $stopped = $false
    foreach ($listenerPid in Get-ListenerPids -Port $Port) {
        if (-not $Quiet) {
            $proc = Get-Process -Id $listenerPid -ErrorAction SilentlyContinue
            $name = if ($proc) { $proc.ProcessName } else { "orphan/stale pid" }
            Write-Host "  Stopping listener on port ${Port}: PID $listenerPid ($name)"
        }
        if (Stop-ProcessTree -ProcessId $listenerPid) {
            $stopped = $true
        }
    }
    return $stopped
}

function Stop-FreeRouterGatewayPorts {
    param(
        [int]$MinPort = 8000,
        [int]$MaxPort = 8020
    )

    Get-Process -Name "freerouterd*" -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "  Stopping $($_.ProcessName) (PID $($_.Id))"
        Stop-ProcessTree -ProcessId $_.Id | Out-Null
    }

    Stop-OrphanGatewayWorkers

    for ($port = $MinPort; $port -le $MaxPort; $port += 1) {
        Stop-PortListeners -Port $port -Quiet | Out-Null
    }

    # Second pass after workers drop stale parent ownership.
    Stop-OrphanGatewayWorkers
    for ($port = $MinPort; $port -le $MaxPort; $port += 1) {
        Stop-PortListeners -Port $port -Quiet | Out-Null
    }
}
