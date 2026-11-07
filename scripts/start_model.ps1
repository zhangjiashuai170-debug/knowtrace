param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('ollama','openai_compatible')][string]$Provider,
    [Parameter(Mandatory=$true)][string]$Model,
    [string]$BaseUrl,
    [ValidateRange(1024,65535)][int]$Port = 8766,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$ProjectDirectory = Split-Path -Parent $PSScriptRoot
$PythonExecutable = Join-Path $ProjectDirectory '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $PythonExecutable)) { throw '请先运行 scripts/setup.ps1。' }
if (-not $BaseUrl -and $Provider -eq 'ollama') { $BaseUrl = 'http://127.0.0.1:11434' }
$ModelUri = $null
if (-not [Uri]::TryCreate($BaseUrl, [UriKind]::Absolute, [ref]$ModelUri)) { throw '模型地址无效。' }
if ($ModelUri.Scheme -notin @('http','https') -or $ModelUri.UserInfo -or $ModelUri.Query -or $ModelUri.Fragment) {
    throw '模型地址须为HTTP(S)，且不能含账号密码、查询参数或片段。'
}
if ($ModelUri.Scheme -eq 'http' -and -not $ModelUri.IsLoopback) { throw 'HTTP仅限本机；远程服务须使用HTTPS。' }
if (-not $Model.Trim() -or $Model.Length -gt 200 -or $Model -match '[\x00-\x1f]') { throw '模型名称无效。' }
if ($CheckOnly) {
    @{provider=$Provider; model=$Model; port=$Port; starts_service=$false; sends_model_request=$false; api_key_prompt_on_start=($Provider -eq 'openai_compatible')} | ConvertTo-Json
    exit 0
}

# No secret in command-line arguments, files, logs, or the parent environment.
$StartInfo = [System.Diagnostics.ProcessStartInfo]::new()
$StartInfo.FileName = $PythonExecutable
$StartInfo.WorkingDirectory = $ProjectDirectory
$StartInfo.UseShellExecute = $false
$StartInfo.CreateNoWindow = $true
$StartInfo.Arguments = "-m knowtrace.cli serve --port $Port"
$StartInfo.EnvironmentVariables['KNOWTRACE_MODEL_PROVIDER'] = $Provider
$StartInfo.EnvironmentVariables['KNOWTRACE_MODEL_BASE_URL'] = $BaseUrl.TrimEnd('/')
$StartInfo.EnvironmentVariables['KNOWTRACE_MODEL_NAME'] = $Model.Trim()
$StartInfo.EnvironmentVariables['PYTHONUTF8'] = '1'
$StartInfo.EnvironmentVariables.Remove('KNOWTRACE_MODEL_API_KEY')
$SecureKey = $null
$SecretPointer = [IntPtr]::Zero
$ModelProcess = $null
try {
    if ($Provider -eq 'openai_compatible') {
        $SecureKey = Read-Host '在本机输入API密钥（隐藏输入；无需认证的兼容服务可留空）' -AsSecureString
        $SecretPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureKey)
        $StartInfo.EnvironmentVariables['KNOWTRACE_MODEL_API_KEY'] = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($SecretPointer)
    }
    $ModelProcess = [System.Diagnostics.Process]::Start($StartInfo)
    Write-Host '已启动服务进程；就绪情况请看日志，模型可用性以实际请求为准。按Ctrl+C结束当前服务。'
    $StartInfo.EnvironmentVariables.Remove('KNOWTRACE_MODEL_API_KEY')
    if ($SecretPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($SecretPointer)
        $SecretPointer = [IntPtr]::Zero
    }
    $ModelProcess.WaitForExit()
    exit $ModelProcess.ExitCode
} finally {
    $StartInfo.EnvironmentVariables.Remove('KNOWTRACE_MODEL_API_KEY')
    if ($SecretPointer -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($SecretPointer) }
    if ($SecureKey) { $SecureKey.Dispose() }
    if ($ModelProcess) { $ModelProcess.Dispose() }
}
