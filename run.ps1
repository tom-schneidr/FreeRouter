param(
    [string]$HostName = $(if ($env:GATEWAY_HOST) { $env:GATEWAY_HOST } else { "127.0.0.1" }),
    [int]$Port = $(if ($env:GATEWAY_PORT) { [int]$env:GATEWAY_PORT } else { 8000 }),
    [switch]$NoReload,
    [switch]$InstallOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

function Resolve-Python {
    $candidates = @(
        @{ Command = "py"; Args = @("-3.13") },
        @{ Command = "py"; Args = @("-3.12") },
        @{ Command = "python"; Args = @() },
        @{ Command = "python3"; Args = @() }
    )

    foreach ($candidate in $candidates) {
        $command = Get-Command $candidate.Command -ErrorAction SilentlyContinue
        if (-not $command) {
            continue
        }

        try {
            $versionOutput = & $candidate.Command @($candidate.Args + @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"))
            $version = [version]$versionOutput
            if ($version.Major -eq 3 -and $version.Minor -ge 12) {
                return @{ Command = $candidate.Command; Args = $candidate.Args }
            }
        }
        catch {
            continue
        }
    }

    throw "Python 3.12+ is required. Install Python from https://www.python.org/downloads/ and rerun this script."
}

function Sync-EnvTemplate {
    param(
        [string]$EnvPath,
        [string]$EnvExamplePath
    )

    if (-not (Test-Path $EnvExamplePath)) {
        return
    }

    if (-not (Test-Path $EnvPath)) {
        Copy-Item $EnvExamplePath $EnvPath
        Write-Host "Created .env from .env.example. Add your provider API keys before expecting live completions."
        return
    }

    $existingKeys = @{}
    foreach ($line in Get-Content $EnvPath) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=') {
            $existingKeys[$Matches[1]] = $true
        }
    }

    $missingLines = @()
    foreach ($line in Get-Content $EnvExamplePath) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=') {
            $key = $Matches[1]
            if (-not $existingKeys.ContainsKey($key)) {
                $missingLines += $line
            }
        }
    }

    if ($missingLines.Count -gt 0) {
        Add-Content -Path $EnvPath -Value ""
        Add-Content -Path $EnvPath -Value "# Added from .env.example"
        Add-Content -Path $EnvPath -Value $missingLines
        Write-Host "Added missing settings to .env: $($missingLines -replace '=.*$', '' -join ', ')"
    }
}

Set-Location $ProjectRoot

if (-not (Test-Path $VenvPython)) {
    $python = Resolve-Python
    Write-Host "Creating local virtual environment at .venv..."
    & $python.Command @($python.Args + @("-m", "venv", $VenvDir))
}

Write-Host "Installing/updating gateway dependencies..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -e .

$EnvPath = Join-Path $ProjectRoot ".env"
$EnvExamplePath = Join-Path $ProjectRoot ".env.example"
Sync-EnvTemplate -EnvPath $EnvPath -EnvExamplePath $EnvExamplePath

if ($InstallOnly) {
    Write-Host "Install complete. Start later with: .\run.ps1"
    exit 0
}

$uvicornArgs = @("app.main:app", "--host", $HostName, "--port", $Port)
if (-not $NoReload) {
    $uvicornArgs += "--reload"
}

Write-Host "Starting FreeRouter at http://${HostName}:$Port/v1"
& $VenvPython -m uvicorn @uvicornArgs
