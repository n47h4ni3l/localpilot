param(
    [ValidateRange(2, 60)]
    [int]$RefreshSeconds = 10,
    [switch]$Once
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$statusPath = Join-Path $repoRoot 'training\reports\adapter_v1_guard_status.json'
$checkpointRoot = Join-Path $repoRoot 'training\outputs\adapter_v1_eager_20g_20260922\checkpoints'
$totalSteps = 11112
$Host.UI.RawUI.WindowTitle = 'LocalPilot training monitor'

function Show-TrainingStatus {
    if (-not $Once) { Clear-Host }
    Write-Host 'LocalPilot training monitor' -ForegroundColor Cyan
    Write-Host 'Closing this window does not stop training.'
    Write-Host "Display refreshed: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host ''

    if (-not (Test-Path -LiteralPath $statusPath)) {
        Write-Host 'Training guard status is not available yet.' -ForegroundColor Yellow
        return
    }

    $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    $displayState = $status.state
    if ($displayState -eq 'running' -and -not (Get-Process -Id $status.windows_process_id -ErrorAction SilentlyContinue)) {
        $displayState = 'interrupted (launcher process is gone)'
    }
    $color = if ($displayState -eq 'running') { 'Green' } else { 'Yellow' }
    Write-Host "Guard: $displayState" -ForegroundColor $color
    Write-Host "Last guard update: $(([datetime]$status.updated_at_utc).ToLocalTime())"

    $step = 0
    if ($status.stderr_log -and (Test-Path -LiteralPath $status.stderr_log)) {
        $progressTail = (Get-Content -LiteralPath $status.stderr_log -Tail 30 -ErrorAction SilentlyContinue) -join "`n"
        foreach ($match in [regex]::Matches($progressTail, '(\d+)/11112')) {
            $step = [math]::Max($step, [int]$match.Groups[1].Value)
        }
    }
    $lastLoss = $null
    if ($status.stdout_log -and (Test-Path -LiteralPath $status.stdout_log)) {
        $recentOutput = @(Get-Content -LiteralPath $status.stdout_log -Tail 12 -ErrorAction SilentlyContinue)
        $lastLoss = $recentOutput | Where-Object { $_ -match "'loss':" } | Select-Object -Last 1
        $resumeLine = $recentOutput | Where-Object { $_ -match 'Resuming LocalPilot training from .*checkpoint-(\d+)' } | Select-Object -Last 1
        if ($resumeLine -match 'checkpoint-(\d+)') {
            $step = [math]::Max($step, [int]$Matches[1])
        }
    }

    $completeSteps = @(
        Get-ChildItem -LiteralPath $checkpointRoot -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^checkpoint-(\d+)$' -and (Test-Path -LiteralPath (Join-Path $_.FullName 'localpilot_checkpoint_complete.json')) } |
            ForEach-Object { [int]($_.Name -replace '^checkpoint-', '') }
    )
    $lastCheckpoint = if ($completeSteps.Count) { ($completeSteps | Measure-Object -Maximum).Maximum } else { 0 }
    $step = [math]::Max($step, $lastCheckpoint)
    $percent = [math]::Round(100 * $step / $totalSteps, 2)
    Write-Host "Progress: step $step / $totalSteps ($percent%)"
    Write-Host "Last complete checkpoint: step $lastCheckpoint"
    if ($lastLoss) { Write-Host "Latest training reading: $lastLoss" }

    try {
        $os = Get-CimInstance Win32_OperatingSystem
        $counters = Get-Counter '\Memory\Committed Bytes', '\Memory\Commit Limit'
        $committed = ($counters.CounterSamples | Where-Object Path -Like '*committed bytes').CookedValue
        $limit = ($counters.CounterSamples | Where-Object Path -Like '*commit limit').CookedValue
        $freeRam = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
        $commitHeadroom = [math]::Round(($limit - $committed) / 1GB, 2)
        Write-Host "Windows free RAM: $freeRam GiB (guard stops below 2.5)"
        Write-Host "Commit headroom: $commitHeadroom GiB (guard stops below 2.5)"
    } catch {
        Write-Host 'Windows memory reading temporarily unavailable.' -ForegroundColor Yellow
    }
    Write-Host ''
    Write-Host "This display refreshes every $RefreshSeconds seconds. It does not control the run."
    if ($displayState -ne 'running') {
        Write-Host 'Training is not running. Do not repeatedly restart after a safety stop.' -ForegroundColor Yellow
    }
}

do {
    Show-TrainingStatus
    if ($Once) { break }
    Start-Sleep -Seconds $RefreshSeconds
} while ($true)
