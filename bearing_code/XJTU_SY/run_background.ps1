param(
    [string]$Python = 'D:/XJTU_SY_runtime/venv/Scripts/python.exe',
    [string]$Config = (Join-Path $PSScriptRoot 'config.json')
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
$logs = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$stdout = Join-Path $logs ($stamp + '.log')
$stderr = Join-Path $logs ($stamp + '.err')
$arguments = '-u "{0}" all --config "{1}" --resume' -f (Join-Path $PSScriptRoot 'experiment.py'), $Config
$process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$info = [PSCustomObject]@{pid=$process.Id; stdout=$stdout; stderr=$stderr; started=(Get-Date -Format o); config=$Config}
$info | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $logs 'background_process.json')
$info | Format-List
