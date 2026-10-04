# Public releases need an externally issued, publicly trusted code-signing
# identity. This helper neither provisions certificates nor changes trust stores.
# Verification assumes a clean release runner with its normal Windows trust roots;
# a private root imported into that runner must not be used to claim public trust.
# ExpectedPublisher is the complete, exact Subject of the approved leaf certificate.

function Get-WindowsSignTool {
    [CmdletBinding()]
    [OutputType([string])]
    param([string]$SignToolPath = '')

    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
        throw 'Windows code signing requires a Windows release runner.'
    }
    if ($SignToolPath) {
        $selected = Get-Item -LiteralPath $SignToolPath -ErrorAction Stop
        if ($selected.PSIsContainer -or $selected.Extension -ine '.exe' -or
            $selected.PSProvider.Name -ne 'FileSystem') {
            throw 'SignToolPath must identify a Windows executable.'
        }
        return $selected.FullName
    }

    $architecture = switch ($env:PROCESSOR_ARCHITECTURE) {
        'ARM64' { 'arm64' }
        'AMD64' { 'x64' }
        default { 'x86' }
    }
    $architectures = @($architecture, 'x64', 'x86') | Select-Object -Unique
    $programDirectories = @(${env:ProgramFiles(x86)}, $env:ProgramFiles) |
        Where-Object { $_ } | Select-Object -Unique
    foreach ($programDirectory in $programDirectories) {
        $sdkBin = Join-Path $programDirectory 'Windows Kits\10\bin'
        if (-not (Test-Path -LiteralPath $sdkBin -PathType Container)) { continue }
        $versions = Get-ChildItem -LiteralPath $sdkBin -Directory -ErrorAction Stop |
            Where-Object { $_.Name -match '^\d+\.\d+\.\d+\.\d+$' } |
            Sort-Object { [version]$_.Name } -Descending
        foreach ($directory in @($versions) + @((Get-Item -LiteralPath $sdkBin))) {
            foreach ($candidateArchitecture in $architectures) {
                $candidate = Join-Path $directory.FullName "$candidateArchitecture\signtool.exe"
                if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                    return (Get-Item -LiteralPath $candidate).FullName
                }
            }
        }
    }
    $command = Get-Command -Name 'signtool.exe' -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($command) { return $command.Source }
    throw 'SignTool is unavailable. Install the Windows SDK signing tools or supply SignToolPath.'
}

function Assert-WindowsCodeSignature {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][ValidateNotNullOrEmpty()][string]$Path,
        [Parameter(Mandatory)][ValidateNotNullOrEmpty()][string]$ExpectedPublisher,
        [string]$SignToolPath = ''
    )

    if ([string]::IsNullOrWhiteSpace($ExpectedPublisher)) {
        throw 'ExpectedPublisher must identify the approved certificate Subject.'
    }
    $artifact = Get-Item -LiteralPath $Path -ErrorAction Stop
    if ($artifact.PSIsContainer -or $artifact.PSProvider.Name -ne 'FileSystem') {
        throw 'The signed artifact must be a file.'
    }
    $signature = Get-AuthenticodeSignature -LiteralPath $artifact.FullName -ErrorAction Stop
    if ([string]$signature.Status -ne 'Valid') {
        throw "Authenticode signature is not valid for $($artifact.FullName): $($signature.Status)."
    }
    if ([string]$signature.SignatureType -ne 'Authenticode') {
        throw 'A valid embedded Authenticode signature is required; a catalog signature is insufficient.'
    }
    $certificate = $signature.SignerCertificate
    if ($null -eq $certificate -or
        -not [string]::Equals($certificate.Subject, $ExpectedPublisher, [StringComparison]::Ordinal)) {
        throw 'The signing certificate does not match the approved publisher Subject.'
    }
    if ([string]::Equals($certificate.Subject, $certificate.Issuer, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'A self-signed signing certificate cannot establish public publisher trust.'
    }
    if ($null -eq $certificate.PublicKey -or $certificate.PublicKey.Oid.Value -ne '1.2.840.113549.1.1.1') {
        throw 'The signing certificate must use RSA for Smart App Control compatibility.'
    }
    $codeSigningUsage = $false
    foreach ($extension in @($certificate.Extensions)) {
        if ($extension.Oid.Value -eq '2.5.29.37') {
            foreach ($usage in @($extension.EnhancedKeyUsages)) {
                if ($usage.Value -eq '1.3.6.1.5.5.7.3.3') { $codeSigningUsage = $true }
            }
        }
    }
    if (-not $codeSigningUsage) {
        throw 'The signing certificate must explicitly allow the Code Signing enhanced key usage.'
    }
    if ($null -eq $signature.TimeStamperCertificate) {
        throw 'The embedded signature must have a trusted timestamp.'
    }

    $signTool = Get-WindowsSignTool -SignToolPath $SignToolPath
    $global:LASTEXITCODE = 0
    $verification = @(& $signTool verify /pa /all /tw /v $artifact.FullName 2>&1)
    $verificationSucceeded = $?
    $verificationExit = $LASTEXITCODE
    if (-not $verificationSucceeded -or $verificationExit -ne 0) {
        $diagnostic = ($verification | Out-String).Trim()
        throw "SignTool embedded-signature verification failed (exit $verificationExit) for $($artifact.FullName). $diagnostic"
    }
    Write-Host "Verified signed artifact: $($artifact.FullName)"
}

function Invoke-WindowsArtifactSigning {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][ValidateNotNullOrEmpty()][string[]]$Paths,
        [Parameter(Mandatory)][ValidateNotNullOrEmpty()][string]$SigningScriptPath,
        [Parameter(Mandatory)][ValidateNotNullOrEmpty()][string]$ExpectedPublisher,
        [string]$SignToolPath = ''
    )

    if (-not $Paths -or [string]::IsNullOrWhiteSpace($ExpectedPublisher)) {
        throw 'Artifacts and an approved publisher Subject are required before signing.'
    }
    $signer = Get-Item -LiteralPath $SigningScriptPath -ErrorAction Stop
    if ($signer.PSIsContainer -or $signer.PSProvider.Name -ne 'FileSystem' -or $signer.Extension -ine '.ps1') {
        throw 'SigningScriptPath must identify the authorized provider adapter PowerShell script.'
    }
    $signTool = Get-WindowsSignTool -SignToolPath $SignToolPath
    $resolvedPaths = @()
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($path in $Paths) {
        if ([string]::IsNullOrWhiteSpace($path)) { throw 'A signing artifact path is empty.' }
        $artifact = Get-Item -LiteralPath $path -ErrorAction Stop
        if ($artifact.PSIsContainer -or $artifact.PSProvider.Name -ne 'FileSystem') {
            throw 'Each signing artifact must be a file.'
        }
        if ($seen.Add($artifact.FullName)) { $resolvedPaths += $artifact.FullName }
    }
    # The provider adapter owns its authorized credentials and SHA-256/RFC3161
    # signing operation. Never pass PFX passwords or create a local trust shortcut.
    $global:LASTEXITCODE = 0
    $callbackOutput = @(& $signer.FullName -Paths $resolvedPaths)
    $callbackSucceeded = $?
    $callbackExit = $LASTEXITCODE
    if (-not $callbackSucceeded -or $callbackExit -ne 0) {
        throw "The authorized signing provider failed (exit $callbackExit)."
    }
    foreach ($path in $resolvedPaths) {
        Assert-WindowsCodeSignature -Path $path -ExpectedPublisher $ExpectedPublisher -SignToolPath $signTool
    }
}
