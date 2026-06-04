# Stop orphaned FreeRouter desktop backends, dev servers, and Tauri shells.
param(
    [int]$MinPort = 8000,
    [int]$MaxPort = 8020,
    [switch]$Aggressive
)

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
. (Join-Path $ProjectRoot "scripts\lib\process-utils.ps1")

$IsAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $IsAdmin) {
    Write-Host "Note: not running as Administrator. Elevated orphans may survive." -ForegroundColor Yellow
}

Write-Host "Stopping FreeRouter gateway listeners on ports ${MinPort}-${MaxPort}..." -ForegroundColor Cyan
Stop-FreeRouterGatewayPorts -MinPort $MinPort -MaxPort $MaxPort

if ($Aggressive) {
    Write-Host "Aggressive: scanning process command lines..." -ForegroundColor Cyan
    Get-Process -Name "freerouterd*" -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "  freerouterd PID $($_.Id)"
        Stop-ProcessTree -ProcessId $_.Id | Out-Null
    }
    Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='python3.13.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match "uvicorn.*app\.main:app|freerouterd" } |
        ForEach-Object {
            Write-Host "  $($_.Name) PID $($_.ProcessId)"
            Stop-ProcessTree -ProcessId $_.ProcessId | Out-Null
        }
}

Start-Sleep -Seconds 1

Write-Host ""
Write-Host "Gateway health check:" -ForegroundColor Cyan
$alive = @()
for ($port = $MinPort; $port -le $MaxPort; $port += 1) {
    $healthUrl = "http://127.0.0.1:$port/v1/gateway/health.json"
    try {
        $response = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
            $alive += $port
            Write-Host "  http://127.0.0.1:$port/v1 still responding" -ForegroundColor Yellow
        }
    }
    catch {
        # not serving
    }
}

if ($alive.Count -eq 0) {
    Write-Host "  All FreeRouter gateway ports in ${MinPort}-${MaxPort} are stopped." -ForegroundColor Green
}
else {
    Write-Host ""
    Write-Host "Some servers are still running on port(s): $($alive -join ', ')." -ForegroundColor Yellow
    Write-Host "Run PowerShell as Administrator, then:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\stop-all-freerouter.ps1 -Aggressive"
    Write-Host "Or reboot if processes are elevated and cannot be killed."
}
