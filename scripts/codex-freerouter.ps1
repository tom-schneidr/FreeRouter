$ErrorActionPreference = "Stop"

function Get-CodexExecutable {
    $command = Get-Command codex -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        throw "Codex CLI was not found on PATH. Install it with: npm install -g @openai/codex"
    }
    if (-not [string]::IsNullOrWhiteSpace($command.Source)) {
        return $command.Source
    }
    return $command.Definition
}

function ConvertTo-TomlString([string]$Value) {
    $normalized = $Value.Replace('\', '/')
    $escaped = $normalized.Replace('"', '\"')
    return '"' + $escaped + '"'
}

function Get-FreeRouterBaseUrl {
    $baseUrl = [Environment]::GetEnvironmentVariable("FREEROUTER_BASE_URL", "Process")
    if ([string]::IsNullOrWhiteSpace($baseUrl)) {
        $baseUrl = "http://127.0.0.1:8000/v1"
    }
    $baseUrl = $baseUrl.TrimEnd('/')
    if ($baseUrl -notmatch '/v1$') {
        $baseUrl = "$baseUrl/v1"
    }
    return $baseUrl
}

function Get-FreeRouterProjectRoot {
    return (Split-Path -Parent $PSScriptRoot)
}

function Get-FreeRouterDesktopExecutable {
    $desktopExecutable = Join-Path (Get-FreeRouterProjectRoot) "apps\desktop\src-tauri\target\release\freerouter_desktop.exe"
    if (-not (Test-Path -LiteralPath $desktopExecutable -PathType Leaf)) {
        throw "The latest FreeRouter desktop executable was not found at '$desktopExecutable'. Build the release desktop app first."
    }
    return (Get-Item -LiteralPath $desktopExecutable).FullName
}

function Test-FreeRouterHealth([string]$BaseUrl) {
    try {
        $health = Invoke-RestMethod -Uri "$($BaseUrl.TrimEnd('/'))/gateway/health.json" -TimeoutSec 2
        return $health.status -eq "ok" -and $health.service -eq "freerouter"
    } catch {
        return $false
    }
}

function Test-LocalFreeRouterBaseUrl([string]$BaseUrl) {
    try {
        $uri = [Uri]$BaseUrl
        return $uri.Scheme -eq "http" -and $uri.Host -in @("127.0.0.1", "localhost") -and $uri.Port -eq 8000
    } catch {
        return $false
    }
}

function Ensure-FreeRouterDesktop([string]$BaseUrl) {
    if (-not (Test-LocalFreeRouterBaseUrl $BaseUrl)) {
        Write-Output "Using configured FreeRouter endpoint $BaseUrl; local desktop startup skipped."
        return
    }

    if (Test-FreeRouterHealth $BaseUrl) {
        Write-Output "FreeRouter desktop is already running at $BaseUrl."
        return
    }

    $desktopExecutable = Get-FreeRouterDesktopExecutable
    Write-Output "Starting the latest FreeRouter desktop release: $desktopExecutable"
    Start-Process -FilePath $desktopExecutable -WorkingDirectory (Get-FreeRouterProjectRoot) | Out-Null

    $deadline = (Get-Date).AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 500
        if (Test-FreeRouterHealth $BaseUrl) {
            Write-Output "FreeRouter desktop is ready at $BaseUrl."
            return
        }
    } while ((Get-Date) -lt $deadline)

    throw "FreeRouter desktop did not become healthy at $BaseUrl within 30 seconds. Start '$desktopExecutable' directly to inspect the desktop runtime."
}

function Get-FreeRouterOverrides {
    $baseUrl = Get-FreeRouterBaseUrl
    return @(
        "-c", 'model="auto"',
        "-c", 'model_provider="freerouter"',
        "-c", "model_providers.freerouter.name=$(ConvertTo-TomlString 'FreeRouter')",
        "-c", "model_providers.freerouter.base_url=$(ConvertTo-TomlString $baseUrl)",
        "-c", 'model_providers.freerouter.env_key="FREEROUTER_API_KEY"',
        "-c", 'model_providers.freerouter.wire_api="responses"',
        # These temporary values keep older Codex CLI builds compatible with the
        # current desktop-generated root config. They never touch config.toml.
        "-c", 'model_reasoning_effort="high"',
        "-c", 'service_tier="flex"'
    )
}

function Get-ConfigKey([string]$Value) {
    $equals = $Value.IndexOf('=')
    if ($equals -lt 0) {
        return $null
    }
    $key = $Value.Substring(0, $equals).Trim()
    return $key.Trim([char[]]@([char]39, [char]34))
}

function Test-ManagedConfigKey([string]$Key) {
    if ([string]::IsNullOrWhiteSpace($Key)) {
        return $false
    }
    return $Key -in @(
        "model",
        "model_provider",
        "model_catalog_json",
        "model_reasoning_effort",
        "profile",
        "service_tier"
    ) -or $Key.StartsWith("model_providers.")
}

function Assert-NoConflictingArguments([string[]]$ExtraArguments) {
    for ($index = 0; $index -lt $ExtraArguments.Count; $index++) {
        $argument = [string]$ExtraArguments[$index]

        if ($argument -in @("-p", "--profile") -or $argument.StartsWith("--profile=")) {
            throw "FreeRouter mode manages the Codex profile/provider; remove conflicting argument '$argument'."
        }
        if ($argument -eq "-m" -or $argument -eq "--model" -or $argument.StartsWith("--model=") -or ($argument.StartsWith("-m") -and -not $argument.StartsWith("--"))) {
            throw "FreeRouter mode manages model=auto; remove conflicting argument '$argument'."
        }
        if ($argument -eq "--oss" -or $argument -eq "--local-provider" -or $argument.StartsWith("--local-provider=")) {
            throw "FreeRouter mode manages the custom provider; remove conflicting argument '$argument'."
        }

        $configOverride = $null
        if ($argument -in @("-c", "--config")) {
            if ($index + 1 -ge $ExtraArguments.Count) {
                throw "'$argument' requires a key=value argument."
            }
            $index++
            $configOverride = [string]$ExtraArguments[$index]
        } elseif ($argument.StartsWith("--config=")) {
            $configOverride = $argument.Substring("--config=".Length)
        } elseif ($argument.StartsWith("-c") -and $argument.Length -gt 2) {
            $configOverride = $argument.Substring(2)
        }

        if ($null -ne $configOverride) {
            $key = Get-ConfigKey $configOverride
            if (Test-ManagedConfigKey $key) {
                throw "FreeRouter mode manages '$key'; remove conflicting config override '$configOverride'."
            }
        }
    }
}

function Get-CodexVersionText([string]$CodexExecutable) {
    $result = Invoke-CodexCapture $CodexExecutable @("--version")
    if ($result.ExitCode -ne 0) {
        throw "Codex did not return a version. $($result.Output -join ' ')"
    }
    return ($result.Output -join ' ').Trim()
}

function Test-FreeRouterConfiguration([string]$CodexExecutable) {
    $probeArguments = @("debug", "models") + @(Get-FreeRouterOverrides)
    $result = Invoke-CodexCapture $CodexExecutable $probeArguments
    if ($result.ExitCode -ne 0) {
        $details = (($result.Output | Select-Object -Last 8) -join [Environment]::NewLine).Trim()
        throw "Codex rejected the FreeRouter launch configuration. No Codex file was changed.`n$details"
    }
}

function Invoke-CodexCapture([string]$CodexExecutable, [string[]]$CodexArguments) {
    # Native stderr becomes an ErrorRecord when ErrorActionPreference is Stop.
    # Capture probes with Continue so a rejected config can be reported without
    # aborting the launcher before the exit code is inspected.
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = @(& $CodexExecutable @CodexArguments 2>&1)
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    return [pscustomobject]@{
        ExitCode = $exitCode
        Output = $output
    }
}

function Get-ExtraArguments([string[]]$RawArguments) {
    $extraArguments = @($RawArguments)
    if ($extraArguments.Count -gt 0 -and $extraArguments[0] -eq "--") {
        if ($extraArguments.Count -eq 1) {
            return @()
        }
        return @($extraArguments[1..($extraArguments.Count - 1)])
    }
    return $extraArguments
}

function Write-Usage {
    @"
FreeRouter Codex launcher

Commands:
  start, on       Launch Codex with FreeRouter model=auto (default)
  normal, off     Launch Codex with its existing configuration unchanged
  status          Validate both launch paths without starting a model turn
  help            Show this help

Examples:
  .\codex-freerouter.bat start
  .\codex-freerouter.bat normal
  .\codex-freerouter.bat status
  .\codex-freerouter.bat start -- --no-alt-screen

The launcher uses per-process Codex -c overrides. It never edits config.toml,
creates a Codex profile, changes authentication, or persists a mode switch.
"@
}

try {
    $codexExecutable = Get-CodexExecutable
    # Keep the script parameter-free so powershell.exe -File forwards every
    # token to $args, including Codex switches such as --model and -c.
    $rawArguments = @($args)
    $Mode = "start"
    $rawExtraArguments = @()
    if ($rawArguments.Count -gt 0) {
        $Mode = [string]$rawArguments[0]
        if ($rawArguments.Count -gt 1) {
            $rawExtraArguments = @($rawArguments[1..($rawArguments.Count - 1)])
        }
    }
    $extraArguments = @(Get-ExtraArguments $rawExtraArguments)
    $normalizedMode = $Mode.ToLowerInvariant()

    switch ($normalizedMode) {
        "help" {
            Write-Usage
            exit 0
        }
        "status" {
            $version = Get-CodexVersionText $codexExecutable
            $statusExitCode = 0
            Write-Output "Codex: $version"
            $baseUrl = Get-FreeRouterBaseUrl
            Write-Output "FreeRouter base URL: $baseUrl"
            if (Test-FreeRouterHealth $baseUrl) {
                Write-Output "FreeRouter desktop runtime: healthy"
            } elseif (Test-LocalFreeRouterBaseUrl $baseUrl) {
                Write-Output "FreeRouter desktop runtime: not running"
            } else {
                Write-Output "FreeRouter desktop runtime: external endpoint (not probed by launcher)"
            }
            Write-Output "Root Codex config: unchanged by this launcher"

            $normalProbe = Invoke-CodexCapture $codexExecutable @("debug", "models")
            if ($normalProbe.ExitCode -eq 0) {
                Write-Output "Normal Codex configuration: loadable"
            } else {
                $statusExitCode = 1
                $normalDetails = (($normalProbe.Output | Select-Object -Last 3) -join ' ').Trim()
                Write-Output "Normal Codex configuration: rejected by the installed CLI"
                if (-not [string]::IsNullOrWhiteSpace($normalDetails)) {
                    Write-Output "Normal probe: $normalDetails"
                }
            }

            try {
                Test-FreeRouterConfiguration $codexExecutable
                Write-Output "FreeRouter launch configuration: valid"
            } catch {
                $statusExitCode = 1
                Write-Output "FreeRouter launch configuration: invalid"
                Write-Output $_.Exception.Message
            }
            exit $statusExitCode
        }
        { $_ -in @("start", "on", "freerouter") } {
            Assert-NoConflictingArguments $extraArguments
            $baseUrl = Get-FreeRouterBaseUrl
            Ensure-FreeRouterDesktop $baseUrl
            Test-FreeRouterConfiguration $codexExecutable

            $managedArguments = @(Get-FreeRouterOverrides)
            $launchArguments = @($managedArguments) + @($extraArguments)
            $hadApiKey = Test-Path Env:FREEROUTER_API_KEY
            $previousApiKey = [Environment]::GetEnvironmentVariable("FREEROUTER_API_KEY", "Process")
            $temporaryApiKey = [string]::IsNullOrWhiteSpace($previousApiKey)
            if ($temporaryApiKey) {
                $env:FREEROUTER_API_KEY = "sk-local"
            }

            try {
                Write-Output "Starting Codex with FreeRouter model=auto at $baseUrl"
                & $codexExecutable @launchArguments
                $exitCode = $LASTEXITCODE
            } finally {
                if ($temporaryApiKey) {
                    if ($hadApiKey) {
                        $env:FREEROUTER_API_KEY = $previousApiKey
                    } else {
                        Remove-Item Env:FREEROUTER_API_KEY -ErrorAction SilentlyContinue
                    }
                }
            }
            exit $exitCode
        }
        { $_ -in @("normal", "off", "restore") } {
            Write-Output "Starting Codex with its existing configuration"
            & $codexExecutable @extraArguments
            exit $LASTEXITCODE
        }
        default {
            throw "Unknown command '$Mode'. Use start, normal, status, or help."
        }
    }
} catch {
    Write-Error $_.Exception.Message
    exit 1
}
