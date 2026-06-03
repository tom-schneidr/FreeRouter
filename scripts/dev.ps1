param(
    [int]$ApiPort = 8000,
    [int]$WebPort = 5173
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "FreeRouter development workspace"
Write-Host "API: http://127.0.0.1:$ApiPort/v1"
Write-Host "Web: http://127.0.0.1:$WebPort/app-next"
Write-Host ""
Write-Host "Run these in separate terminals for now:"
Write-Host "  .\run.ps1 -Port $ApiPort"
Write-Host "  npm run dev:web -- --port $WebPort"

Set-Location $ProjectRoot
