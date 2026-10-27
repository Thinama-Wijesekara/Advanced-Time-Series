$ErrorActionPreference = "Stop"

$listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
  Select-Object -First 1
if ($listener) {
  $process = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)"
  $command = if ($process) { $process.CommandLine } else { "command line unavailable" }
  Write-Error "Port 8000 is already used by PID $($listener.OwningProcess): $command"
  exit 1
}

Set-Location $PSScriptRoot
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
