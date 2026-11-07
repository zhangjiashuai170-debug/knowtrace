param([string]$PythonExecutable = 'python')
$ErrorActionPreference = 'Stop'
$ProjectDirectory = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectDirectory
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    & $PythonExecutable -m venv '.venv'
    if ($LASTEXITCODE -ne 0) { throw '创建虚拟环境失败' }
}
$VenvPython = Join-Path $ProjectDirectory '.venv/Scripts/python.exe'
if (Test-Path -LiteralPath 'requirements-lock.txt') {
    & $VenvPython -m pip install -r requirements-lock.txt
    if ($LASTEXITCODE -ne 0) { throw '安装锁定依赖失败' }
    & $VenvPython -m pip install --no-deps -e .
} else {
    & $VenvPython -m pip install -e '.[dev]'
}
if ($LASTEXITCODE -ne 0) { throw '安装项目失败' }
& $VenvPython -m knowtrace.cli seed
if ($LASTEXITCODE -ne 0) { throw '演示语料导入失败' }
Write-Host '准备完成。运行 scripts/start.ps1，浏览器打开 http://127.0.0.1:8766'
