param([string]$ModelDirectory = '', [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$ProjectDirectory = Split-Path -Parent $PSScriptRoot
$PythonExecutable = Join-Path $ProjectDirectory '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $PythonExecutable)) { throw '请先运行 scripts/setup.ps1。' }
if (-not $ModelDirectory) { $ModelDirectory = Join-Path $ProjectDirectory '.models/multilingual-e5-small-614241f6' }
if ($CheckOnly) {
    @{installs_dependencies=$false; downloads_model=$false; model_id='intfloat/multilingual-e5-small'; expected_model_bytes=135429554} | ConvertTo-Json
    exit 0
}
& $PythonExecutable -m pip install --disable-pip-version-check -r (Join-Path $ProjectDirectory 'requirements-semantic-lock.txt')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $PythonExecutable (Join-Path $PSScriptRoot 'prepare_semantic_model.py') --destination $ModelDirectory --activate
exit $LASTEXITCODE
