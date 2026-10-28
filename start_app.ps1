$ErrorActionPreference = "Stop"
$python = "C:\Users\ThinamaWijesekara\Envs\chronos\Scripts\python.exe"
$root = $PSScriptRoot
Set-Location $root
if (-not (Test-Path .env)) { throw "Missing .env. Copy .env.example and set DATABASE_URL and API_KEY." }
$listener = Get-NetTCPConnection -State Listen -LocalPort 8000 -ErrorAction SilentlyContinue
if ($listener) { throw "Port 8000 is in use by process $($listener.OwnendingPid -join ', ')." }
& $python -m db.check
if ($LASTEXITCODE -ne 0) { throw "Database check failed." }
& $python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
