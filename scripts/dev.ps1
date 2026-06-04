param(
    [int]$ApiPort = 8000,
    [int]$WebPort = 5173,
    [switch]$NoInstall
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

function Start-ChildProcess {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory
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

if (-not $NoInstall -and -not (Test-Path $VenvPython)) {
    Write-Host "Preparing Python environment..."
    & (Join-Path $ProjectRoot "run.ps1") -InstallOnly
}

if (-not (Test-Path $VenvPython)) {
    throw "Missing .venv. Run .\run.ps1 -InstallOnly first, or rerun npm run dev without -NoInstall."
}

if (-not (Test-Path (Join-Path $ProjectRoot "node_modules"))) {
    if ($NoInstall) {
        throw "Missing node_modules. Run npm install first, or rerun npm run dev without -NoInstall."
    }
    Write-Host "Installing npm dependencies..."
    & npm.cmd install
    if ($LASTEXITCODE -ne 0) {
        throw "npm install failed."
    }
}

$apiProcess = $null

try {
    $env:GATEWAY_HOST = "127.0.0.1"
    $env:GATEWAY_PORT = "$ApiPort"
    Write-Host "Starting FreeRouter backend on http://127.0.0.1:$ApiPort/v1"
    $apiProcess = Start-ChildProcess `
        -FilePath $VenvPython `
        -Arguments @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$ApiPort") `
        -WorkingDirectory $ProjectRoot

    Wait-ForUrl -Url "http://127.0.0.1:$ApiPort/v1/gateway/health.json"

    $env:GATEWAY_PORT = "$ApiPort"
    $env:FREEROUTER_API_TARGET = "http://127.0.0.1:$ApiPort"
    Write-Host ""
    Write-Host "FreeRouter development workspace is running."
    Write-Host "  App: http://127.0.0.1:$WebPort/app/"
    Write-Host "  API: http://127.0.0.1:$ApiPort/v1"
    Write-Host ""
    Write-Host "Starting React UI. Press Ctrl+C to stop frontend and backend."

    & npm.cmd --workspace "@freerouter/ui" run dev -- --port "$WebPort"
    if ($LASTEXITCODE -ne 0) {
        throw "React dev server exited with code $LASTEXITCODE."
    }
}
finally {
    Stop-ChildProcess $apiProcess
}
