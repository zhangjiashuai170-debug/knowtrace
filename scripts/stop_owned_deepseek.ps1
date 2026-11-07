param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$ProjectDirectory = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$ProjectPython = [IO.Path]::GetFullPath((Join-Path $ProjectDirectory '.venv/Scripts/python.exe'))
$WorkerScript = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'connect_deepseek.py'))
$StatusPath = Join-Path $ProjectDirectory 'artifacts/deepseek-connection/status.json'
$ExpectedArguments = '^\s*(?:"[^"]+"|\S+)\s+-u\s+(?<Script>"[^"]+"|\S+)(?:\s+--upgrade)?\s*$'

function Write-SafeResult([string]$State, [int[]]$ProcessIds = @()) {
    @{status=$State; port=8766; check_only=[bool]$CheckOnly; owned_pids=@($ProcessIds)} | ConvertTo-Json -Compress
}

function Get-Listeners {
    @(Get-NetTCPConnection -State Listen -ErrorAction Stop | Where-Object {$_.LocalPort -eq 8766})
}

function Read-WorkerStatus {
    $File = Get-Item -LiteralPath $StatusPath -ErrorAction Stop
    if ($File.PSIsContainer -or $File.Length -gt 8192 -or
        ($File.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Invalid status file' }
    $State = Get-Content -LiteralPath $StatusPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($State.status -ne 'serving' -or $State.host -ne '127.0.0.1' -or
        $State.port -ne 8766 -or -not ($State.pid -is [int] -or $State.pid -is [long]) -or
        $State.pid -le 0) { throw 'Invalid worker status' }
    return $State
}

function Is-ExpectedCommand($Process) {
    if (-not $Process.CommandLine -or $Process.CommandLine -cnotmatch $ExpectedArguments) { return $false }
    $ScriptArgument = $Matches.Script.Trim('"')
    if (-not [IO.Path]::IsPathRooted($ScriptArgument)) {
        if ($ScriptArgument -cnotin @('scripts/connect_deepseek.py','scripts\connect_deepseek.py')) { return $false }
        $ScriptArgument = Join-Path $ProjectDirectory $ScriptArgument
    }
    return [string]::Equals([IO.Path]::GetFullPath($ScriptArgument), $WorkerScript, [StringComparison]::OrdinalIgnoreCase)
}

function Same-Identity($Current, $Original) {
    return ($Current -and $Current.CreationDate -eq $Original.CreationDate -and
        $Current.ParentProcessId -eq $Original.ParentProcessId -and
        $Current.ExecutablePath -eq $Original.ExecutablePath -and
        $Current.CommandLine -ceq $Original.CommandLine)
}

try {
    $State = Read-WorkerStatus
    $RecordedPid = [int]$State.pid
    $Listeners = @(Get-Listeners)
    if ($Listeners.Count -eq 0 -or @($Listeners | Where-Object {[int]$_.OwningProcess -ne $RecordedPid}).Count -gt 0) {
        Write-SafeResult 'blocked_unknown_owner'; exit 2
    }
    $Processes = @(Get-CimInstance Win32_Process -ErrorAction Stop)
    $Recorded = @($Processes | Where-Object {[int]$_.ProcessId -eq $RecordedPid})
    if ($Recorded.Count -ne 1 -or -not (Is-ExpectedCommand $Recorded[0])) {
        Write-SafeResult 'blocked_unknown_owner'; exit 2
    }
    $Worker = $Recorded[0]
    if ($Worker.ExecutablePath -and [string]::Equals($Worker.ExecutablePath, $ProjectPython, [StringComparison]::OrdinalIgnoreCase)) {
        $Owned = @($Worker)
    } else {
        if (-not $Worker.ExecutablePath -or [IO.Path]::GetFileName($Worker.ExecutablePath) -ine 'python.exe') {
            Write-SafeResult 'blocked_unknown_owner'; exit 2
        }
        $WorkerArguments = $Worker.CommandLine -creplace '^\s*(?:"[^"]+"|\S+)', ''
        $Parent = @($Processes | Where-Object {
            $_.ProcessId -eq $Worker.ParentProcessId -and $_.ExecutablePath -and
            [string]::Equals($_.ExecutablePath, $ProjectPython, [StringComparison]::OrdinalIgnoreCase) -and
            (Is-ExpectedCommand $_) -and
            ($_.CommandLine -creplace '^\s*(?:"[^"]+"|\S+)', '') -ceq $WorkerArguments
        })
        if ($Parent.Count -ne 1) { Write-SafeResult 'blocked_unknown_owner'; exit 2 }
        $Owned = @($Worker, $Parent[0])
    }
    $OwnedIds = @($Owned | ForEach-Object {[int]$_.ProcessId})
    # Before the first stop, verify every identity and the current status owner.
    foreach ($Candidate in $Owned) {
        $Current = Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$Candidate.ProcessId)" -ErrorAction Stop
        if (-not (Same-Identity $Current $Candidate)) { Write-SafeResult 'blocked_unknown_owner'; exit 2 }
    }
    $FreshState = Read-WorkerStatus
    $FreshListeners = @(Get-Listeners)
    if ($FreshState.pid -ne $RecordedPid -or $FreshListeners.Count -eq 0 -or
        @($FreshListeners | Where-Object {[int]$_.OwningProcess -ne $RecordedPid}).Count -gt 0) {
        Write-SafeResult 'blocked_unknown_owner'; exit 2
    }
    if ($CheckOnly) { Write-SafeResult 'owned_worker_found' $OwnedIds; exit 0 }
    foreach ($Candidate in $Owned) {
        $Current = Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$Candidate.ProcessId)" -ErrorAction Stop
        if (-not $Current) { continue }
        if (-not (Same-Identity $Current $Candidate)) { Write-SafeResult 'blocked_unknown_owner'; exit 2 }
        Stop-Process -Id ([int]$Candidate.ProcessId) -ErrorAction Stop
    }
    for ($Try = 0; $Try -lt 20; $Try++) {
        if (@(Get-Listeners).Count -eq 0) { Write-SafeResult 'stopped' $OwnedIds; exit 0 }
        Start-Sleep -Milliseconds 100
    }
    Write-SafeResult 'port_still_in_use'; exit 2
} catch {
    Write-SafeResult 'inspection_or_stop_failed'; exit 2
}
