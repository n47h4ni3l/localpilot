# Host-only policy: changing this does not change the saved trainer identity.
# These are operational margins, not a guarantee against driver/OS failure.
function Get-TrainingMemoryPolicy {
    @{
        version = 2
        warning_gib = 2.5
        emergency_ram_gib = 0.25
        emergency_commit_gib = 1.0
        sustained_ram_gib = 1.0
        sustained_ram_seconds = 30.0
        sustained_commit_gib = 2.5
        sustained_commit_seconds = 15.0
    }
}

function Get-TrainingMemoryDecision {
    param(
        [double]$AvailableGiB,
        [double]$CommitHeadroomGiB,
        [double]$ElapsedSeconds,
        [hashtable]$History,
        [hashtable]$Policy = (Get-TrainingMemoryPolicy)
    )
    foreach ($value in @($AvailableGiB, $CommitHeadroomGiB, $ElapsedSeconds)) {
        if ([double]::IsNaN($value) -or [double]::IsInfinity($value)) {
            throw 'Invalid memory sample.'
        }
    }
    if ($AvailableGiB -lt 0 -or $ElapsedSeconds -lt 0) { throw 'Invalid memory sample.' }
    if ($History.ContainsKey('last_elapsed') -and $ElapsedSeconds -lt $History.last_elapsed) {
        throw 'Memory sample clock moved backwards.'
    }
    $History.last_elapsed = $ElapsedSeconds
    foreach ($entry in @(
        @{ key = 'ram_since'; low = ($AvailableGiB -lt $Policy.sustained_ram_gib) },
        @{ key = 'commit_since'; low = ($CommitHeadroomGiB -lt $Policy.sustained_commit_gib) }
    )) {
        if (-not $entry.low) { $History.Remove($entry.key) }
        elseif (-not $History.ContainsKey($entry.key)) { $History[$entry.key] = $ElapsedSeconds }
    }
    $ramSeconds = if ($History.ContainsKey('ram_since')) { $ElapsedSeconds - $History.ram_since } else { 0.0 }
    $commitSeconds = if ($History.ContainsKey('commit_since')) { $ElapsedSeconds - $History.commit_since } else { 0.0 }
    $reason = $null
    if ($CommitHeadroomGiB -lt $Policy.emergency_commit_gib) { $reason = 'emergency_commit' }
    elseif ($AvailableGiB -lt $Policy.emergency_ram_gib) { $reason = 'emergency_ram' }
    elseif ($ramSeconds -ge $Policy.sustained_ram_seconds) { $reason = 'sustained_low_ram' }
    elseif ($commitSeconds -ge $Policy.sustained_commit_seconds) { $reason = 'sustained_low_commit' }
    [pscustomobject]@{
        stop = ($null -ne $reason)
        reason = $reason
        warning = ($AvailableGiB -lt $Policy.warning_gib -or $CommitHeadroomGiB -lt $Policy.warning_gib)
        low_ram_seconds = $ramSeconds
        low_commit_seconds = $commitSeconds
    }
}
