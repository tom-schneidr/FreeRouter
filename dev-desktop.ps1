param(
    [int]$ApiPort = 8000,
    [switch]$NoInstall,
    [switch]$ReplaceExisting
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

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

function Stop-ChildProcess {
    param([System.Diagnostics.Process]$Process)
    if ($null -ne $Process -and -not $Process.HasExited) {
        Stop-ProcessTree -ProcessId $Process.Id
    }
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

function Stop-PortOwners {
    param([int]$Port)

    $rows = netstat -ano | Select-String ":$Port" | ForEach-Object {
        $parts = $_.Line.Trim() -split "\s+"
        if ($parts.Count -ge 5 -and $parts[1] -like "*:$Port" -and $parts[3] -eq "LISTENING") {
            $parts[4]
        }
    }
    $owners = $rows | Select-Object -Unique
    foreach ($owner in $owners) {
        if (Stop-ProcessTree -ProcessId ([int]$owner)) {
            Write-Host "Stopped existing listener on 127.0.0.1:$Port (PID $owner)."
        }
    }
}

function Stop-FreeRouterGatewayBackends {
    param([int]$Port)

    Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and $_.CommandLine -match "uvicorn.*app\.main:app|freerouterd"
    } | ForEach-Object {
        if ($_.CommandLine -like "*$Port*" -or $_.CommandLine -like "*FreeRouter*") {
            Stop-ProcessTree -ProcessId $_.ProcessId
        }
    }
    Stop-PortOwners -Port $Port
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

if (Test-FreeRouterHealth -Port $ApiPort -or (Test-PortOpen -Port $ApiPort)) {
    if ($ReplaceExisting -or -not $PSBoundParameters.ContainsKey("ApiPort")) {
        if (Test-FreeRouterHealth -Port $ApiPort) {
            Write-Host "Stopping existing FreeRouter backend on 127.0.0.1:$ApiPort..."
        }
        else {
            $owners = Get-PortOwnerSummary -Port $ApiPort
            Write-Host "Port 127.0.0.1:$ApiPort is in use by: $owners. Attempting to free it..."
        }
        Stop-FreeRouterGatewayBackends -Port $ApiPort
        Start-Sleep -Seconds 1
    }
    elseif (Test-FreeRouterHealth -Port $ApiPort) {
        throw "FreeRouter is already running on 127.0.0.1:$ApiPort. Quit it first or rerun with -ReplaceExisting."
    }
    else {
        $owners = Get-PortOwnerSummary -Port $ApiPort
        throw "Port 127.0.0.1:$ApiPort is already in use by: $owners. Quit those processes or choose a different -ApiPort."
    }
}

Write-Host "Building React UI from current source..."
Invoke-Checked "npm.cmd" @("run", "build:web")

$desktopToken = [guid]::NewGuid().ToString()
$backendEnv = @{
    "PYTHONUNBUFFERED" = "1"
    "FREEROUTER_DESKTOP_TOKEN" = $desktopToken
    "FREEROUTER_DESKTOP_PROJECT_ROOT" = $ProjectRoot
    "FREEROUTER_APP_DATA_DIR" = $ProjectRoot
}
$backendProcess = $null

try {
    Write-Host "Starting source backend on http://127.0.0.1:$ApiPort/v1 with reload enabled..."
    $backendProcess = Start-ChildProcess `
        -FilePath $VenvPython `
        -Arguments @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$ApiPort", "--reload") `
        -WorkingDirectory $ProjectRoot `
        -Environment $backendEnv

    Wait-ForUrl -Url "http://127.0.0.1:$ApiPort/v1/gateway/health.json"

    $env:FREEROUTER_DEV_BACKEND = "1"
    $env:FREEROUTER_DESKTOP_TOKEN = $desktopToken
    $env:GATEWAY_PORT = "$ApiPort"
    Write-Host "Launching Tauri desktop shell against the source backend..."
    Invoke-Checked "npm.cmd" @("--workspace", "@freerouter/desktop", "run", "dev")
}
finally {
    Write-Host "Stopping FreeRouter dev backend on 127.0.0.1:$ApiPort..."
    Stop-ChildProcess $backendProcess
    Stop-FreeRouterGatewayBackends -Port $ApiPort
    Remove-Item Env:\FREEROUTER_DEV_BACKEND -ErrorAction SilentlyContinue
    Remove-Item Env:\FREEROUTER_DESKTOP_TOKEN -ErrorAction SilentlyContinue
    Remove-Item Env:\GATEWAY_PORT -ErrorAction SilentlyContinue
}
