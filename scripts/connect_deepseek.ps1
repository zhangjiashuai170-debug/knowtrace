param([switch]$CheckOnly, [switch]$Upgrade)
$ErrorActionPreference = 'Stop'
$ProjectDirectory = Split-Path -Parent $PSScriptRoot
$PythonExecutable = Join-Path $ProjectDirectory '.venv/Scripts/python.exe'
$WorkerScript = Join-Path $PSScriptRoot 'connect_deepseek.py'
$StateDirectory = Join-Path $ProjectDirectory 'artifacts/deepseek-connection'
$StatePath = Join-Path $StateDirectory 'prompt-status.json'
if ($Upgrade) { $StatePath = Join-Path $StateDirectory 'upgrade-prompt-status.json' }
if (-not (Test-Path -LiteralPath $PythonExecutable) -or -not (Test-Path -LiteralPath $WorkerScript)) {
    throw '请先完成 KnowTrace 安装。'
}
if ($CheckOnly) {
    @{provider='openai_compatible'; model='deepseek-flash'; base_url='https://api.deepseek.com'; port=8766; upgrade=[bool]$Upgrade; starts_service=$false; sends_model_request=$false; prompts_for_key=$false} | ConvertTo-Json
    exit 0
}
function Write-InputStatus([string]$Status, [string]$ErrorClass = '') {
    [IO.Directory]::CreateDirectory($StateDirectory) | Out-Null
    $Value = @{status=$Status; updated_at=[DateTimeOffset]::UtcNow.ToString('o'); pid=$PID; provider='openai_compatible'; model='deepseek-flash'}
    if ($Upgrade) { $Value.upgrade = $true }
    if ($ErrorClass) { $Value.error_class = $ErrorClass }
    [IO.File]::WriteAllText($StatePath, ($Value | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
}

# Do not replace a running worker's status or ask for a second key.
# A live worker keeps this lock file open for its whole lifetime.
$PromptMutex = [Threading.Mutex]::new($false, 'Local\KnowTraceDeepSeekKeyPrompt8766')
$PromptLockHeld = $false
try { $PromptLockHeld = $PromptMutex.WaitOne(0) }
catch [Threading.AbandonedMutexException] { $PromptLockHeld = $true }
if (-not $PromptLockHeld) {
    $PromptMutex.Dispose()
    Write-Host '已经打开了一个本机密钥输入窗口。'
    exit 3
}
$WorkerLockPath = Join-Path $StateDirectory 'worker.lock'
if ($Upgrade) { $WorkerLockPath = Join-Path $StateDirectory 'upgrade.lock' }
if (Test-Path -LiteralPath $WorkerLockPath) {
    $LockProbe = $null
    try {
        $LockProbe = [IO.File]::Open($WorkerLockPath, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    } catch [IO.IOException] {
        $PromptMutex.ReleaseMutex()
        $PromptMutex.Dispose()
        if ($Upgrade) { Write-Host 'DeepSeek升级检查已在运行，请查看本机状态。' }
        else { Write-Host 'DeepSeek连接进程已在运行，请查看本机状态。' }
        exit 3
    } finally {
        if ($LockProbe) { $LockProbe.Dispose() }
    }
}

# The key goes only into the new worker's environment. No key file or CLI argument.
$StartInfo = [System.Diagnostics.ProcessStartInfo]::new()
$StartInfo.FileName = $PythonExecutable
$StartInfo.Arguments = '-u scripts/connect_deepseek.py'
if ($Upgrade) { $StartInfo.Arguments += ' --upgrade' }
$StartInfo.WorkingDirectory = $ProjectDirectory
$StartInfo.UseShellExecute = $false
$StartInfo.CreateNoWindow = $true
$StartInfo.EnvironmentVariables['KNOWTRACE_MODEL_PROVIDER'] = 'openai_compatible'
$StartInfo.EnvironmentVariables['KNOWTRACE_MODEL_BASE_URL'] = 'https://api.deepseek.com'
$StartInfo.EnvironmentVariables['KNOWTRACE_MODEL_NAME'] = 'deepseek-flash'
$StartInfo.EnvironmentVariables['PYTHONUTF8'] = '1'
$StartInfo.EnvironmentVariables.Remove('KNOWTRACE_MODEL_API_KEY')
$SecureKey = $null
$SecretPointer = [IntPtr]::Zero
$WorkerProcess = $null
$WorkerStarted = $false
try {
    $Host.UI.RawUI.WindowTitle = 'KnowTrace - DeepSeek 本机密钥输入'
    if ($Upgrade) { $Host.UI.RawUI.WindowTitle = 'KnowTrace - DeepSeek 升级连接' }
    Write-Host '在此窗口输入 DeepSeek API Key 后按回车。输入隐藏，请勿发到聊天。'
    Write-Host '将用一小段合成资料检查回答、引用、拒答和权限，通过后切换本机演示。'
    if ($Upgrade) { Write-Host '升级使用本次新输入的密钥；检查失败时保留现有服务。' }
    Write-Host '密钥仅供本次进程使用；关闭连接进程后需重新输入。'
    Write-InputStatus 'awaiting_key'
    $SecureKey = Read-Host 'DeepSeek API Key' -AsSecureString
    if ($null -eq $SecureKey -or $SecureKey.Length -eq 0) {
        Write-InputStatus 'cancelled'
        exit 2
    }
    if ($SecureKey.Length -gt 4096) {
        Write-InputStatus 'input_invalid' 'InvalidKeyLength'
        exit 2
    }
    $SecretPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureKey)
    $StartInfo.EnvironmentVariables['KNOWTRACE_MODEL_API_KEY'] = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($SecretPointer)
    Write-InputStatus 'starting_worker'
    $WorkerProcess = [System.Diagnostics.Process]::Start($StartInfo)
    $WorkerStarted = $true
    Write-Host '已启动连接检查，窗口即将关闭。连接结果保存在本机状态文件。'
} catch {
    if (-not $WorkerStarted) { Write-InputStatus 'error' 'PromptOrWorkerLaunchError' }
    Write-Host '本机输入或启动未完成；请重新运行此程序。'
    exit 1
} finally {
    $StartInfo.EnvironmentVariables.Remove('KNOWTRACE_MODEL_API_KEY')
    if ($SecretPointer -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($SecretPointer) }
    if ($SecureKey) { $SecureKey.Dispose() }
    if ($WorkerProcess) { $WorkerProcess.Dispose() }
    if ($PromptLockHeld) { $PromptMutex.ReleaseMutex() }
    $PromptMutex.Dispose()
}
