# Host-only policy: changing this does not change the saved trainer identity.
# These are operational observations, not a guarantee against driver/OS failure.
#
# v3 deliberately separates observation from intervention. The default launcher
# is MonitorOnly because repeated training stops were losing useful progress
# without proving that low host RAM was the cause of the instability. Protective
# retains the former v2 sustained thresholds as an explicit opt-in.
function Get-TrainingMemoryPolicy {
    @{
        version = 3
        warning_gib = 2.5

        # Extreme thresholds used only by CriticalOnly/Protective. MonitorOnly
        # records these states but never stops training automatically.
        critical_ram_gib = 0.25
        critical_commit_gib = 0.75

        # Legacy v2 sustained protection, retained behind -MemoryPolicy Protective.
        protective_ram_gib = 1.0
        protective_ram_seconds = 30.0
        protective_commit_gib = 2.5
        protective_commit_seconds = 15.0
    }
}

function Get-TrainingMemoryDecision {
    param(
        [double]$AvailableGiB,
        [double]$CommitHeadroomGiB,
        [double]$ElapsedSeconds,
        [hashtable]$History,
        [ValidateSet('MonitorOnly', 'CriticalOnly', 'Protective')]
        [string]$Mode = 'MonitorOnly',
        [hashtable]$Policy = (Get-TrainingMemoryPolicy)
    )
    foreach ($value in @($AvailableGiB, $CommitHeadroomGiB, $ElapsedSeconds)) {
        if ([double]::IsNaN($value) -or [double]::IsInfinity($value)) {
            throw 'Invalid memory sample.'
        }
    }
    if ($AvailableGiB -lt 0 -or $CommitHeadroomGiB -lt 0 -or $ElapsedSeconds -lt 0) {
        throw 'Invalid memory sample.'
    }
    if ($History.ContainsKey('last_elapsed') -and $ElapsedSeconds -lt $History.last_elapsed) {
        throw 'Memory sample clock moved backwards.'
    }
    $History.last_elapsed = $ElapsedSeconds

    foreach ($entry in @(
        @{ key = 'ram_since'; low = ($AvailableGiB -lt $Policy.protective_ram_gib) },
        @{ key = 'commit_since'; low = ($CommitHeadroomGiB -lt $Policy.protective_commit_gib) }
    )) {
        if (-not $entry.low) { $History.Remove($entry.key) }
        elseif (-not $History.ContainsKey($entry.key)) { $History[$entry.key] = $ElapsedSeconds }
    }

    $ramSeconds = if ($History.ContainsKey('ram_since')) {
        $ElapsedSeconds - $History.ram_since
    } else { 0.0 }
    $commitSeconds = if ($History.ContainsKey('commit_since')) {
        $ElapsedSeconds - $History.commit_since
    } else { 0.0 }

    $pressureReason = $null
    $severity = 'normal'
    if ($CommitHeadroomGiB -lt $Policy.critical_commit_gib) {
        $pressureReason = 'critical_commit'
        $severity = 'critical'
    }
    elseif ($AvailableGiB -lt $Policy.critical_ram_gib) {
        $pressureReason = 'critical_ram'
        $severity = 'critical'
    }
    elseif ($ramSeconds -ge $Policy.protective_ram_seconds) {
        $pressureReason = 'sustained_low_ram'
        $severity = 'pressure'
    }
    elseif ($commitSeconds -ge $Policy.protective_commit_seconds) {
        $pressureReason = 'sustained_low_commit'
        $severity = 'pressure'
    }
    elseif (
        $AvailableGiB -lt $Policy.warning_gib -or
        $CommitHeadroomGiB -lt $Policy.warning_gib
    ) {
        $pressureReason = 'low_headroom_warning'
        $severity = 'warning'
    }

    $stopReason = $null
    if ($Mode -eq 'CriticalOnly' -and $severity -eq 'critical') {
        $stopReason = $pressureReason
    }
    elseif ($Mode -eq 'Protective' -and $severity -in @('critical', 'pressure')) {
        $stopReason = $pressureReason
    }

    [pscustomobject]@{
        stop = ($null -ne $stopReason)
        reason = $stopReason
        pressure_reason = $pressureReason
        severity = $severity
        mode = $Mode
        warning = ($severity -ne 'normal')
        low_ram_seconds = $ramSeconds
        low_commit_seconds = $commitSeconds
    }
}
