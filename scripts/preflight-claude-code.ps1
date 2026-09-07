param(
    [string]$ClaudePath = "claude",
    [string]$Model = "gpt-oss:20b",
    [int]$RequiredContext = 65536
)

$ErrorActionPreference = "Stop"

if ($Model -ne "gpt-oss:20b") {
    throw "LocalPilot's one-model contract requires gpt-oss:20b."
}
if ($RequiredContext -lt 65536) {
    throw "Claude Code with Ollama requires at least 65536 context tokens."
}

$claude = Get-Command -Name $ClaudePath -ErrorAction SilentlyContinue
if (-not $claude -and (Test-Path -LiteralPath $ClaudePath -PathType Leaf)) {
    $claude = Get-Item -LiteralPath $ClaudePath
}
if (-not $claude) {
    throw "Claude Code was not found. Install it or set selfdev.implementation_executable to its absolute path."
}

Write-Verbose "Checking Claude Code version and required flags."
$version = & $ClaudePath --version
$helpText = (& $ClaudePath --help) -join "`n"
$requiredFlags = @(
    "--allowedTools", "--disallowedTools", "--model",
    "--output-format", "--restricted", "--permission-prompts"
)
$missing = @($requiredFlags | Where-Object { -not $helpText.Contains($_) })
if ($missing.Count -gt 0) {
    throw "Claude Code is missing required CLI flags: $($missing -join ', ')"
}

Write-Verbose "Checking the configured Ollama model."
$models = ollama list
if (($models -join "`n") -notmatch [regex]::Escape($Model)) {
    throw "Ollama model $Model is not installed. Run: ollama pull $Model"
}

$loadBody = @{
    model = $Model
    prompt = ""
    stream = $false
    keep_alive = "10m"
    options = @{ num_ctx = $RequiredContext; num_predict = 1 }
} | ConvertTo-Json -Depth 4 -Compress
Add-Type -AssemblyName System.Net.Http
$handler = [System.Net.Http.HttpClientHandler]::new()
$handler.UseProxy = $false
$client = [System.Net.Http.HttpClient]::new($handler)
$client.Timeout = [TimeSpan]::FromSeconds(300)
try {
    Write-Verbose "Loading $Model with a $RequiredContext-token context for a live allocation check."
    $content = [System.Net.Http.StringContent]::new(
        $loadBody, [System.Text.Encoding]::UTF8, "application/json"
    )
    $loadResponse = $client.PostAsync(
        "http://127.0.0.1:11434/api/generate", $content
    ).GetAwaiter().GetResult()
    $loadResponse.EnsureSuccessStatusCode() | Out-Null
    Write-Verbose "Reading the live Ollama allocation."
    $psResponse = $client.GetAsync(
        "http://127.0.0.1:11434/api/ps"
    ).GetAwaiter().GetResult()
    $psResponse.EnsureSuccessStatusCode() | Out-Null
    $processes = $psResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult() | ConvertFrom-Json
} finally {
    $client.Dispose()
    $handler.Dispose()
}
$activeModel = $processes.models | Where-Object {
    $candidateName = if ($_.name) { $_.name } else { $_.model }
    $candidateName.Split(':')[0] -eq $Model.Split(':')[0]
} | Select-Object -First 1
$allocated = if ($activeModel) { [int]$activeModel.context_length } else { 0 }
if ($allocated -lt $RequiredContext) {
    throw "Ollama allocated $allocated context tokens after loading $Model; at least $RequiredContext are required."
}

[pscustomobject]@{
    Healthy = $true
    Claude = $version
    Model = $Model
    RequiredContext = $RequiredContext
    OllamaAllocation = $allocated
    AnthropicLoginRequired = $false
}
