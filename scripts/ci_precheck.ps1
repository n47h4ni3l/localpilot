# scripts/ci_precheck.ps1
# This script will perform pre-CI checks on the candidate changes.

# Function to check if the candidate diff is clean
function Check-CleanDiff {
    $diff = git diff --cached
    if ($diff) {
        Write-Host "There are uncommitted changes in the candidate diff."
        return $false
    } else {
        Write-Host "The candidate diff is clean."
        return $true
    }
}

# Function to run static checks
function Run-StaticChecks {
    $result = run_candidate_static_checks
    if ($result -eq "pass") {
        Write-Host "Static checks passed."
        return $true
    } else {
        Write-Host "Static checks failed."
        return $false
    }
}

# Main script execution
if (Check-CleanDiff) {
    if (Run-StaticChecks) {
        Write-Host "Pre-CI checks passed. Candidate is ready for CI."
    } else {
        Write-Host "Pre-CI checks failed. Candidate is not ready for CI."
    }
} else {
    Write-Host "Pre-CI checks failed. Candidate is not ready for CI."
}