param(
    [Parameter(Mandatory=$true)]
    [string]$RepoUrl
)

$ErrorActionPreference = "Stop"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is not installed or not on PATH."
}

if (-not (Test-Path ".git")) {
    git init -b main
}

$existing = git remote 2>$null
if ($existing -contains "origin") {
    git remote set-url origin $RepoUrl
} else {
    git remote add origin $RepoUrl
}

Write-Host "Connected origin to $RepoUrl" -ForegroundColor Green
Write-Host "No push was performed automatically. Review the project, then run:"
Write-Host "  git add ."
Write-Host "  git commit -m 'Initial LocalPilot v0.1'"
Write-Host "  git push -u origin main"
