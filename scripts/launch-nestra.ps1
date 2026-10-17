param(
    [Parameter(Mandatory = $true)][ValidateSet('Desktop', 'Broker', 'Worker')][string]$Role,
    [Parameter(Mandatory = $true)][string]$InstallDirectory,
    [switch]$VerifyOnly
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Assert-NestraLaunchPath {
    param([string]$Path)
    $absolute = [IO.Path]::GetFullPath($Path)
    $current = $absolute
    while ($current) {
        if (Test-Path -LiteralPath $current) {
            if (((Get-Item -LiteralPath $current -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Nestra launch refuses a linked path or junction: $current"
            }
        }
        $parent = [IO.Path]::GetDirectoryName($current)
        if ($parent -eq $current) { break }
        $current = $parent
    }
    return $absolute.TrimEnd('\', '/')
}

function Read-NestraLaunchJson {
    param([string]$Path)
    $verified = Assert-NestraLaunchPath $Path
    if (-not (Test-Path -LiteralPath $verified -PathType Leaf) -or (Get-Item -LiteralPath $verified).Length -gt 1048576) {
        throw "Missing or oversized Nestra launch metadata: $verified"
    }
    return (Get-Content -LiteralPath $verified -Raw | ConvertFrom-Json)
}

function Get-NestraTomlLaunchValue {
    param([string]$Text, [string]$Section, [string]$Key)
    $current = ''; $values = @()
    foreach ($line in ($Text -split '\r?\n')) {
        if ($line -match '^\s*\[([A-Za-z0-9_.]+)\]\s*(?:#.*)?$') { $current = $Matches[1]; continue }
        if ($current -ceq $Section -and $line -match ('^\s*' + [regex]::Escape($Key) + '\s*=\s*(.*?)\s*$')) {
            $value = $Matches[1].Trim()
            if ($value -match '^("(?:[^"\\]|\\.)*")\s*(?:#.*)?$') {
                # The generated TOML quotes Windows paths with JSON-compatible
                # escaping. Decode it before checking the private executable.
                $value = ConvertFrom-Json -InputObject $Matches[1]
                if ($value -isnot [string]) { throw 'Nestra launch value must be a string.' }
            } elseif ($value -match "^'([^']*)'\s*(?:#.*)?$") { $value = $Matches[1] }
            else { $value = ($value -split '#', 2)[0].Trim() }
            $values += $value
        }
    }
    if ($values.Count -ne 1) { throw "Nestra configuration requires exactly one [$Section] $Key value." }
    return $values[0]
}

function Get-NestraLaunchConfiguration {
    param([string]$Root)
    $rootPath = Assert-NestraLaunchPath $Root
    $app = Join-Path $rootPath 'app'
    $runtimeRoot = Join-Path $rootPath 'runtime'
    $models = Join-Path $rootPath 'models'
    $configPath = Join-Path $app 'localpilot.toml'
    $runtime = Read-NestraLaunchJson (Join-Path $app '.nestra-runtime.json')
    $tasks = Read-NestraLaunchJson (Join-Path $rootPath 'tasks.json')
    if ($runtime.schema_version -ne 1 -or $tasks.format_version -ne 1) { throw 'Unsupported Nestra launch metadata version.' }
    foreach ($pair in @(@('install_directory', $rootPath), @('app_root', $app), @('runtime_root', $runtimeRoot), @('models_root', $models), @('config_path', $configPath))) {
        if ((Assert-NestraLaunchPath ([string]$runtime.($pair[0]))) -ine (Assert-NestraLaunchPath $pair[1])) {
            throw "Nestra runtime path does not match this installation: $($pair[0])"
        }
    }
    if ((Assert-NestraLaunchPath ([string]$tasks.install_directory)) -ine $rootPath) { throw 'Nestra task metadata belongs to another installation.' }
    if ($runtime.ollama_host -cne 'http://127.0.0.1:11435' -or $runtime.ollama_port -ne 11435 -or
        $runtime.broker_host -cne 'http://127.0.0.1:8766' -or $runtime.broker_port -ne 8766) {
        throw 'Nestra bundle endpoints must remain on their private loopback ports 11435 and 8766.'
    }
    if ([string]$tasks.owner_sid -notmatch '^S-1-5-[0-9-]+$') { throw 'Nestra task owner SID is invalid.' }
    $taskNames = @()
    foreach ($taskRole in @('Desktop', 'Broker', 'Worker')) {
        $name = [string]$tasks.tasks.$taskRole
        if (-not $name -or $name.Length -gt 238 -or $name -match '[\\/\x00-\x1f]') { throw 'Nestra task names are invalid.' }
        $taskNames += $name
    }
    if (@($taskNames | Select-Object -Unique).Count -ne 3) { throw 'Nestra task names must be distinct.' }
    $files = [ordered]@{
        Python = (Join-Path $app '.venv/Scripts/python.exe')
        Pythonw = (Join-Path $app '.venv/Scripts/pythonw.exe')
        BasePython = (Join-Path $runtimeRoot 'python/python/python.exe')
        Ollama = (Join-Path $runtimeRoot 'ollama/ollama.exe')
        Git = (Join-Path $runtimeRoot 'git/cmd/git.exe')
        Bash = (Join-Path $runtimeRoot 'git/bin/bash.exe')
        GitHub = (Join-Path $runtimeRoot 'gh/bin/gh.exe')
        Claude = (Join-Path $runtimeRoot 'claude/claude.exe')
        Launcher = (Join-Path $app 'scripts/launch-nestra.ps1')
        Config = $configPath
    }
    foreach ($entry in $files.GetEnumerator()) {
        $entry.Value = Assert-NestraLaunchPath $entry.Value
        if (-not (Test-Path -LiteralPath $entry.Value -PathType Leaf)) { throw "Nestra runtime file is missing: $($entry.Value)" }
    }
    if (-not (Test-Path -LiteralPath $models -PathType Container)) { throw 'The private Nestra model store is missing.' }
    if ((Get-Item -LiteralPath $configPath).Length -gt 1048576) { throw 'Nestra configuration is oversized.' }
    $text = Get-Content -LiteralPath $configPath -Raw
    foreach ($expected in @(@('agent', 'scaffold_mode', 'guide_first'), @('agent', 'data_dir', 'localpilot-data'),
                            @('model', 'provider', 'ollama'), @('model', 'name', 'nestra:p1.2'),
                            @('selfdev', 'developer_model', 'nestra:p1.2'), @('selfdev', 'implementation_model', 'nestra:p1.2'),
                            @('selfdev', 'implementation_backend', 'local_tools'),
                            @('desktop', 'host', '127.0.0.1'), @('desktop', 'port', '8766'))) {
        if ((Get-NestraTomlLaunchValue $text $expected[0] $expected[1]) -cne $expected[2]) {
            throw "Nestra configuration differs from its accepted bundle identity: [$($expected[0])] $($expected[1])"
        }
    }
    $configuredClaude = Get-NestraTomlLaunchValue $text 'selfdev' 'implementation_executable'
    if ((Assert-NestraLaunchPath $configuredClaude) -ine $files.Claude) {
        throw 'Nestra Claude Code tool must use this installation''s private executable.'
    }
    return [pscustomobject]@{ Root = $rootPath; App = $app; RuntimeRoot = $runtimeRoot; Models = $models;
        Config = $configPath; Runtime = $runtime; Tasks = $tasks; Files = $files;
        Data = (Join-Path $app 'localpilot-data'); Logs = (Join-Path $rootPath 'state/logs') }
}

function ConvertTo-NestraLaunchArgument {
    param([string]$Value)
    if ($Value -and $Value -notmatch '[\s"]') { return $Value }
    return '"' + [regex]::Replace([regex]::Replace($Value, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
}

function Join-NestraLaunchArguments {
    param([string[]]$Arguments)
    return (@($Arguments | ForEach-Object { ConvertTo-NestraLaunchArgument $_ }) -join ' ')
}

function Get-NestraAccountSid {
    param([string]$Account)
    if ($Account -match '^S-1-') { return $Account }
    return (New-Object Security.Principal.NTAccount($Account)).Translate([Security.Principal.SecurityIdentifier]).Value
}

function Assert-NestraLaunchTasks {
    param($Configuration)
    $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    if ($sid -cne $Configuration.Tasks.owner_sid) { throw 'Nestra must run under the account that owns this installation.' }
    foreach ($taskRole in @('Desktop', 'Broker', 'Worker')) {
        $task = Get-ScheduledTask -TaskName $Configuration.Tasks.tasks.$taskRole -ErrorAction Stop
        $actions = @($task.Actions)
        $expectedArguments = '-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File ' +
            (ConvertTo-NestraLaunchArgument $Configuration.Files.Launcher) + ' -Role ' + $taskRole + ' -InstallDirectory ' +
            (ConvertTo-NestraLaunchArgument $Configuration.Root)
        $level = if ($taskRole -eq 'Desktop') { 'Limited' } else { 'Highest' }
        if ((Get-NestraAccountSid ([string]$task.Principal.UserId)) -cne $sid -or [string]$task.Principal.RunLevel -cne $level -or
            [string]$task.Principal.LogonType -cne 'Interactive' -or $actions.Count -ne 1 -or
            [string]$actions[0].Execute -ine (Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0/powershell.exe') -or
            [string]$actions[0].Arguments -cne $expectedArguments -or [string]$actions[0].WorkingDirectory -ine $Configuration.App) {
            throw "Nestra $taskRole task no longer matches this installation's owner, privilege, or launcher. Review that task; no process was stopped."
        }
    }
}

function Set-NestraPrivateEnvironment {
    param($Configuration)
    $env:PATH = @((Split-Path -Parent $Configuration.Files.Python), (Split-Path -Parent $Configuration.Files.BasePython),
        (Join-Path $Configuration.RuntimeRoot 'git/cmd'), (Join-Path $Configuration.RuntimeRoot 'git/bin'),
        (Join-Path $Configuration.RuntimeRoot 'gh/bin'), (Join-Path $Configuration.RuntimeRoot 'claude'),
        (Join-Path $Configuration.RuntimeRoot 'ollama'), $env:PATH) -join ';'
    $env:OLLAMA_HOST = [string]$Configuration.Runtime.ollama_host
    $env:OLLAMA_MODELS = $Configuration.Models
    $env:OLLAMA_NO_CLOUD = '1'
    $env:CLAUDE_CODE_GIT_BASH_PATH = $Configuration.Files.Bash
    $env:CLAUDE_CONFIG_DIR = Join-Path $Configuration.Root 'state/claude'
    $env:DISABLE_AUTOUPDATER = '1'
    $env:PYTHONNOUSERSITE = '1'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:PYTHONPATH = $null
    $env:PYTHONHOME = $null
}

function Get-NestraPortProcess {
    param([int]$Port)
    # Some Windows builds expose an active server as Bound/NONE rather than
    # Listen. Identify its peerless bound endpoint across all TCP states;
    # established clients and closing/TIME_WAIT connections have peers.
    $connections = @(Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue)
    $listeners = @($connections | Where-Object {
        [int]$_.RemotePort -eq 0 -and [string]$_.RemoteAddress -in @('0.0.0.0', '::', '')
    })
    if (-not $listeners.Count) { return $null }
    $owners = @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)
    if ($owners.Count -ne 1 -or [int]$owners[0] -le 0 -or @($listeners | Where-Object LocalAddress -NotIn @('127.0.0.1', '::1')).Count) {
        throw "Port $Port is occupied by a different or non-loopback service. Close or reconfigure that service manually; Nestra did not stop it."
    }
    $process = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + [int]$owners[0]) -ErrorAction Stop
    if ($null -eq $process -or [int]$process.ProcessId -ne [int]$owners[0] -or $null -eq $process.CreationDate) {
        throw "Cannot verify the live process owning port $Port."
    }
    return $process
}

function Assert-NestraExactProcess {
    param($Process, [string]$Executable, [string[]]$Arguments, [int]$Port, [string]$RedirectedExecutable = '')
    $argumentText = Join-NestraLaunchArguments $Arguments
    $commands = @(((ConvertTo-NestraLaunchArgument $Executable) + ' ' + $argumentText), ('"' + $Executable + '" ' + $argumentText))
    if ($RedirectedExecutable -and [string]$Process.ExecutablePath -ieq $RedirectedExecutable) {
        # Windows venv executables are redirectors. Accept the exact private
        # base runtime only through its still-live, verified venv parent.
        $parent = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + [int]$Process.ParentProcessId) -ErrorAction Stop
        if ($null -eq $parent -or [int]$parent.ProcessId -ne [int]$Process.ParentProcessId -or
            $null -eq $parent.CreationDate -or [DateTime]$parent.CreationDate -gt [DateTime]$Process.CreationDate) {
            throw 'The private Python redirector parent is absent or its PID was reused.'
        }
        Assert-NestraExactProcess $parent $Executable $Arguments $Port
        $commands += @(((ConvertTo-NestraLaunchArgument $RedirectedExecutable) + ' ' + $argumentText), ('"' + $RedirectedExecutable + '" ' + $argumentText))
    } elseif ([string]$Process.ExecutablePath -ine $Executable) {
        throw "Port $Port belongs to an unverified executable or command. Its process was preserved; review the private Nestra task and port owner."
    }
    if (-not $Process.ExecutablePath -or -not $Process.CommandLine -or
        -not (@($commands | Where-Object { [string]::Equals($_, [string]$Process.CommandLine, [StringComparison]::OrdinalIgnoreCase) }).Count)) {
        throw "Port $Port belongs to an unverified executable or command. Its process was preserved; review the private Nestra task and port owner."
    }
}

function Get-NestraBrokerArguments {
    param($Configuration)
    return @('-m', 'localpilot.broker', '--root', $Configuration.App, '--config', $Configuration.Config)
}

function Assert-NestraAuthenticatedBroker {
    param($Configuration, [bool]$RequireFullProcessIdentity = $false)
    $process = Get-NestraPortProcess ([int]$Configuration.Runtime.broker_port)
    if ($null -eq $process) { return $null }
    if ($RequireFullProcessIdentity) {
        Assert-NestraExactProcess $process $Configuration.Files.Python (Get-NestraBrokerArguments $Configuration) ([int]$Configuration.Runtime.broker_port) $Configuration.Files.BasePython
    }
    $tokenPath = Join-Path $Configuration.Data 'broker-token'
    if (-not (Test-Path -LiteralPath $tokenPath -PathType Leaf)) { return $null }
    Assert-NestraLaunchPath $tokenPath | Out-Null
    if ((Get-Item -LiteralPath $tokenPath).Length -gt 1024) { throw 'The private broker token is invalid.' }
    $token = (Get-Content -LiteralPath $tokenPath -Raw).Trim()
    if ($token -notmatch '^[A-Za-z0-9_-]{32,128}$') { throw 'The private broker token is invalid.' }
    try {
        $status = Invoke-RestMethod -Uri ($Configuration.Runtime.broker_host + '/v1/broker/status') -Headers @{ Authorization = 'Bearer ' + $token } -TimeoutSec 3
    } catch {
        # A just-bound owned server may not yet be accepting HTTP. An actual
        # HTTP rejection is different and must never trigger a fallback.
        $responseProperty = $_.Exception.PSObject.Properties['Response']
        if ($null -eq $responseProperty -or $null -eq $responseProperty.Value) { return $null }
        throw 'The service on the private broker port did not authenticate with this installation. Its process was preserved.'
    }
    $identity = $status.runtime_identity
    $expectedArguments = @(Get-NestraBrokerArguments $Configuration)
    $reported = @($identity.command_line)
    $created = ([DateTimeOffset]([DateTime]$process.CreationDate).ToUniversalTime()).ToUnixTimeMilliseconds() / 1000.0
    if ([int]$status.pid -ne [int]$process.ProcessId -or $status.elevated -ne $true -or
        [string]$identity.root -ine $Configuration.App -or [string]$identity.config_path -ine $Configuration.Config -or
        [string]$identity.data_dir -ine $Configuration.Data -or [string]$identity.executable -ine $Configuration.Files.Python -or
        [string]$identity.host_executable -notin @($Configuration.Files.Python, $Configuration.Files.BasePython) -or
        [string]$identity.desktop_host -cne '127.0.0.1' -or [int]$identity.desktop_port -ne 8766 -or
        [Math]::Abs([double]$identity.created_at_unix - $created) -gt 0.1 -or
        (Get-NestraAccountSid ([string]$identity.user)) -cne $Configuration.Tasks.owner_sid -or
        $reported.Count -ne ($expectedArguments.Count + 1) -or [string]$reported[0] -notin @($Configuration.Files.Python, $Configuration.Files.BasePython)) {
        throw 'The authenticated broker identity differs from the registered private Nestra installation. No process was stopped.'
    }
    for ($index = 0; $index -lt $expectedArguments.Count; $index++) {
        if ([string]$reported[$index + 1] -ine [string]$expectedArguments[$index]) { throw 'The authenticated broker has a different command line.' }
    }
    # Reobserve after HTTP authentication to reject an exited/reused PID.
    $again = Get-NestraPortProcess ([int]$Configuration.Runtime.broker_port)
    if ($null -eq $again -or $again.ProcessId -ne $process.ProcessId -or $again.CreationDate -ne $process.CreationDate) {
        throw 'The private broker changed during verification; retry its registered task.'
    }
    return $process
}

function Start-NestraPrivateProcess {
    param($Configuration, [string]$Executable, [string[]]$Arguments, [string]$Name)
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '-' + [Guid]::NewGuid().ToString('N')
    $stdout = Join-Path $Configuration.Logs ($Name + '-' + $stamp + '.stdout.log')
    $stderr = Join-Path $Configuration.Logs ($Name + '-' + $stamp + '.stderr.log')
    return Start-Process -FilePath $Executable -ArgumentList (Join-NestraLaunchArguments $Arguments) -WorkingDirectory $Configuration.App -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
}

function Wait-NestraPrivateBroker {
    param($Configuration, [bool]$RequireFullProcessIdentity = $false, $StartedProcess = $null)
    $deadline = [DateTime]::UtcNow.AddSeconds(90)
    do {
        if ($null -ne $StartedProcess -and $StartedProcess.HasExited) { throw 'The private Nestra broker exited during startup; inspect state/logs.' }
        $verified = Assert-NestraAuthenticatedBroker $Configuration $RequireFullProcessIdentity
        if ($null -ne $verified) { return $verified }
        Start-Sleep -Milliseconds 1000
    } while ([DateTime]::UtcNow -lt $deadline)
    throw 'The private Nestra broker did not become ready within 90 seconds. Inspect its registered Broker task and state/logs; no fallback broker was started.'
}

function Start-NestraBrokerRole {
    param($Configuration)
    $ollama = Get-NestraPortProcess ([int]$Configuration.Runtime.ollama_port)
    $startedOllama = $null
    if ($null -eq $ollama) {
        $startedOllama = Start-NestraPrivateProcess $Configuration $Configuration.Files.Ollama @('serve') 'ollama'
    } else { Assert-NestraExactProcess $ollama $Configuration.Files.Ollama @('serve') ([int]$Configuration.Runtime.ollama_port) }
    $deadline = [DateTime]::UtcNow.AddSeconds(60)
    $ready = $false
    do {
        if ($null -ne $startedOllama -and $startedOllama.HasExited) { throw 'Private Ollama exited during startup; inspect state/logs.' }
        $ollama = Get-NestraPortProcess ([int]$Configuration.Runtime.ollama_port)
        if ($null -ne $ollama) {
            Assert-NestraExactProcess $ollama $Configuration.Files.Ollama @('serve') ([int]$Configuration.Runtime.ollama_port)
            try { $inventory = Invoke-RestMethod -Uri ($Configuration.Runtime.ollama_host + '/api/tags') -TimeoutSec 3 } catch { $inventory = $null }
            if ($null -ne $inventory) {
                $accepted = @($inventory.models | Where-Object { $_.name -ceq 'nestra:p1.2' -and $_.digest -ceq 'c356171d9a54f773c0064a56b5af028b209f61355ba68f67119d857c83bf7f46' })
                if ($accepted.Count -ne 1) { throw 'Private Ollama does not report the exact accepted Nestra P1.2. Restore the bundled model; no download or substitution was attempted.' }
                $ready = $true; break
            }
        }
        Start-Sleep -Milliseconds 1000
    } while ([DateTime]::UtcNow -lt $deadline)
    if (-not $ready) { throw 'Private Ollama did not become ready within 60 seconds; inspect state/logs.' }
    $existing = Assert-NestraAuthenticatedBroker $Configuration $true
    if ($null -ne $existing) { (Get-Process -Id $existing.ProcessId -ErrorAction Stop).WaitForExit(); return }
    $broker = Start-NestraPrivateProcess $Configuration $Configuration.Files.Python (Get-NestraBrokerArguments $Configuration) 'broker'
    Wait-NestraPrivateBroker $Configuration $true $broker | Out-Null
    $broker.WaitForExit()
    if ($broker.ExitCode -ne 0) { throw "The private broker exited with code $($broker.ExitCode); inspect state/logs." }
}

$configuration = $null
try {
    $configuration = Get-NestraLaunchConfiguration $InstallDirectory
    if ($VerifyOnly) {
        [pscustomobject]@{ status = 'verified'; role = $Role; install_directory = $configuration.Root;
            model = 'nestra:p1.2'; guide_only = $true; broker_host = $configuration.Runtime.broker_host;
            ollama_host = $configuration.Runtime.ollama_host } | ConvertTo-Json -Compress
        exit 0
    }
    Assert-NestraLaunchTasks $configuration
    $principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    $elevated = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (($Role -eq 'Desktop' -and $elevated) -or ($Role -ne 'Desktop' -and -not $elevated)) {
        throw "Nestra $Role has the wrong Windows privilege. Use its registered task; Desktop must be Limited and Broker/Worker Highest."
    }
    New-Item -ItemType Directory -Path $configuration.Logs -Force | Out-Null
    Set-NestraPrivateEnvironment $configuration
    if ($Role -eq 'Broker') { Start-NestraBrokerRole $configuration; exit 0 }
    Wait-NestraPrivateBroker $configuration | Out-Null
    if ($Role -eq 'Worker') {
        $child = Start-NestraPrivateProcess $configuration $configuration.Files.Python @('-m', 'localpilot.background_worker', '--root', $configuration.App, '--config', $configuration.Config) 'worker'
    } else {
        $child = Start-NestraPrivateProcess $configuration $configuration.Files.Pythonw @('-m', 'localpilot.cli', '--config', $configuration.Config, 'desktop') 'desktop'
    }
    $child.WaitForExit()
    if ($child.ExitCode -ne 0) { throw "Nestra $Role exited with code $($child.ExitCode); inspect state/logs." }
} catch {
    if ($null -ne $configuration -and (Test-Path -LiteralPath $configuration.Logs -PathType Container)) {
        $failure = Join-Path $configuration.Logs ('launcher-' + $Role + '-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '-' + [Guid]::NewGuid().ToString('N') + '.error.txt')
        $_.Exception.Message | Set-Content -LiteralPath $failure -Encoding utf8
    }
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
    exit 1
}
