param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$ProjectDirectory = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$ProjectPython = [IO.Path]::GetFullPath((Join-Path $ProjectDirectory '.venv/Scripts/python.exe'))
$ExpectedArguments = '^\s*(?:"[^"]+"|\S+)\s+-m\s+knowtrace\.cli\s+serve\s+--port\s+8766\s*$'

function Write-SafeResult([string]$State, [int[]]$ProcessIds = @()) {
    @{status=$State; port=8766; check_only=[bool]$CheckOnly; owned_pids=@($ProcessIds)} | ConvertTo-Json -Compress
}

function Get-Listeners {
    # Enumerate once rather than treating an unknown query failure as no owner.
    @(Get-NetTCPConnection -State Listen -ErrorAction Stop | Where-Object {$_.LocalPort -eq 8766})
}

function Is-ExpectedCommand($Process) {
    $Process.CommandLine -and $Process.CommandLine -cmatch $ExpectedArguments
}

try {
    $Processes = @(Get-CimInstance Win32_Process -ErrorAction Stop)
    $Roots = @($Processes | Where-Object {
        $_.ExecutablePath -and [string]::Equals($_.ExecutablePath, $ProjectPython, [StringComparison]::OrdinalIgnoreCase) -and (Is-ExpectedCommand $_)
    })
    $RootIds = @($Roots | ForEach-Object {[int]$_.ProcessId})
    # Windows venv's python.exe launches a base-interpreter child. Trust only a
    # direct child of the exact project launcher, with the same exact CLI form.
    $Children = @($Processes | Where-Object {
        $_.ParentProcessId -in $RootIds -and $_.ExecutablePath -and
        [IO.Path]::GetFileName($_.ExecutablePath) -ieq 'python.exe' -and (Is-ExpectedCommand $_)
    })
    $Owned = @($Children) + @($Roots)
    $OwnedIds = @($Owned | ForEach-Object {[int]$_.ProcessId})
    $Listeners = @(Get-Listeners)
    $Unknown = @($Listeners | Where-Object {[int]$_.OwningProcess -notin $OwnedIds})
    if ($Unknown.Count -gt 0) { Write-SafeResult 'blocked_unknown_owner'; exit 2 }
    if ($CheckOnly) {
        if ($Owned.Count -gt 0) { Write-SafeResult 'owned_service_found' $OwnedIds }
        else { Write-SafeResult 'no_service' }
        exit 0
    }
    if ($Owned.Count -eq 0) { Write-SafeResult 'no_service'; exit 0 }
    # Recheck every identity before the first mutation, including creation time,
    # to avoid acting on a reused PID or a changed command line.
    foreach ($Candidate in $Owned) {
        $Current = Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$Candidate.ProcessId)" -ErrorAction Stop
        if ($Current -and ($Current.CreationDate -ne $Candidate.CreationDate -or
            $Current.ExecutablePath -ne $Candidate.ExecutablePath -or $Current.CommandLine -cne $Candidate.CommandLine)) {
            Write-SafeResult 'blocked_unknown_owner'; exit 2
        }
    }
    $CurrentListeners = @(Get-Listeners)
    if (@($CurrentListeners | Where-Object {[int]$_.OwningProcess -notin $OwnedIds}).Count -gt 0) {
        Write-SafeResult 'blocked_unknown_owner'; exit 2
    }
    foreach ($Candidate in $Owned) {
        $Current = Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$Candidate.ProcessId)" -ErrorAction Stop
        if (-not $Current) { continue }
        if ($Current.CreationDate -ne $Candidate.CreationDate -or $Current.ExecutablePath -ne $Candidate.ExecutablePath -or
            $Current.CommandLine -cne $Candidate.CommandLine) { Write-SafeResult 'blocked_unknown_owner'; exit 2 }
        Stop-Process -Id ([int]$Candidate.ProcessId) -ErrorAction Stop
    }
    # No second kill pass: only wait briefly for those already stopped to leave.
    for ($Try = 0; $Try -lt 10; $Try++) {
        if (@(Get-Listeners).Count -eq 0) { Write-SafeResult 'stopped' $OwnedIds; exit 0 }
        Start-Sleep -Milliseconds 100
    }
    Write-SafeResult 'port_still_in_use'; exit 2
} catch {
    # Never print raw process command lines, provider details, or exception text.
    Write-SafeResult 'inspection_or_stop_failed'; exit 2
}
