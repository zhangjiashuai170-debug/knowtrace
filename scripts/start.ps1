param([int]$Port = 8766)
$ErrorActionPreference = 'Stop'
$ProjectDirectory = Split-Path -Parent $PSScriptRoot
$PythonExecutable = Join-Path $ProjectDirectory '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw '请先运行 scripts/setup.ps1 安装本地依赖。'
}
Set-Location -LiteralPath $ProjectDirectory
& $PythonExecutable -m knowtrace.cli serve --port $Port
