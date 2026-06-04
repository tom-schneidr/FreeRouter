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

function Stop-ChildProcess {
    param([System.Diagnostics.Process]$Process)
    if ($null -ne $Process -and -not $Process.HasExited) {
        Stop-Process -Id $Process.Id -Force
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
        try {
            Stop-Process -Id ([int]$owner) -Force -ErrorAction Stop
            Write-Host "Stopped existing listener on 127.0.0.1:$Port (PID $owner)."
        }
        catch {
            Write-Host "Could not stop listener PID $owner; it may have already exited."
        }
    }
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

if (Test-FreeRouterHealth -Port $ApiPort) {
    if ($ReplaceExisting) {
        Stop-PortOwners -Port $ApiPort
        Start-Sleep -Seconds 1
    }
    elseif ($PSBoundParameters.ContainsKey("ApiPort")) {
        throw "FreeRouter is already running on 127.0.0.1:$ApiPort. Quit it first or choose a different -ApiPort."
    }
    else {
        $fallbackPort = Find-FreePort -StartingPort $ApiPort
        if ($null -eq $fallbackPort) {
            throw "FreeRouter is already running on 127.0.0.1:$ApiPort, and no nearby free port was found."
        }
        Write-Host "FreeRouter is already running on 127.0.0.1:$ApiPort."
        Write-Host "Starting this dev desktop session on http://127.0.0.1:$fallbackPort/v1 instead."
        $ApiPort = $fallbackPort
    }
}
if (Test-PortOpen -Port $ApiPort) {
    $owners = Get-PortOwnerSummary -Port $ApiPort
    if ($ReplaceExisting) {
        Stop-PortOwners -Port $ApiPort
        Start-Sleep -Seconds 1
    }
    elseif ($PSBoundParameters.ContainsKey("ApiPort")) {
        throw "Port 127.0.0.1:$ApiPort is already in use by: $owners. Quit those processes or choose a different -ApiPort."
    }
    $fallbackPort = Find-FreePort -StartingPort $ApiPort
    if ($null -eq $fallbackPort) {
        throw "Port 127.0.0.1:$ApiPort is already in use by: $owners, and no nearby free port was found."
    }
    Write-Host "Port 127.0.0.1:$ApiPort is already in use by: $owners."
    Write-Host "Starting this dev desktop session on http://127.0.0.1:$fallbackPort/v1 instead."
    $ApiPort = $fallbackPort
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
    Remove-Item Env:\FREEROUTER_DEV_BACKEND -ErrorAction SilentlyContinue
    Remove-Item Env:\FREEROUTER_DESKTOP_TOKEN -ErrorAction SilentlyContinue
    Stop-ChildProcess $backendProcess
}
