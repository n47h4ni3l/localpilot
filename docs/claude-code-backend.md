# Claude Code implementation backend

LocalPilot has one operational model, `gpt-oss:20b`. LocalPilot remains responsible for capability discovery, repository research, the acceptance contract, repository grounding, independent diff/test review, candidate delivery, and outcome learning. Claude Code performs only implementation work in the isolated candidate workspace and calls the same model through local Ollama.

## Windows install and configuration

Claude Code 2.1.263 or later is required. The current CLI contract was verified against `claude --help` and the official Claude Code CLI reference: print-mode JSON, `--model`, `--max-turns`, `--tools`, `--allowedTools`, `--disallowedTools`, `--settings`, `--strict-mcp-config`, `--restricted`, `--permission-prompts none`, and `--no-session-persistence` are used. The runtime never uses `--dangerously-skip-permissions`.

Install to the `E:` drive with WinGet:

```powershell
winget install --id Anthropic.ClaudeCode --exact --location E:\Tools\ClaudeCode --accept-package-agreements --accept-source-agreements
```

Configure LocalPilot:

```toml
[selfdev]
developer_model = "gpt-oss:20b"
developer_model_fallbacks = []
implementation_backend = "claude_code"
implementation_model = "gpt-oss:20b"
implementation_context_tokens = 65536
implementation_max_turns = 24
implementation_timeout_seconds = 600
implementation_review_repair_passes = 1
implementation_max_output_chars = 120000
implementation_executable = "E:\\Tools\\ClaudeCode\\claude.exe"
implementation_base_url = "http://localhost:11434"
```

No Anthropic login is required in local Ollama mode. The child receives `ANTHROPIC_AUTH_TOKEN=ollama`, an empty `ANTHROPIC_API_KEY`, and `ANTHROPIC_BASE_URL=http://localhost:11434`. `ollama launch claude --config` is an optional interactive setup convenience; LocalPilot does not depend on it.

Run the checked-in preflight:

```powershell
.\scripts\preflight-claude-code.ps1 -ClaudePath E:\Tools\ClaudeCode\claude.exe -Model gpt-oss:20b -RequiredContext 65536
```

Preflight performs a real local load and then reads Ollama's live allocation, so the first run can take several minutes while the model is mapped into memory. A metadata-only model check is not treated as proof that 64K is viable. On Windows, when Ollama reports the model fully resident in VRAM, preflight trims the runner's reclaimable host working-set mapping so the existing background memory governor measures usable RAM instead of the duplicate memory-mapped model pages.

Claude Code with Ollama needs at least 64K context. If `ollama ps` reports less, stop the running model, set the Ollama app context slider to at least 65536, or restart the server with:

```powershell
$env:OLLAMA_CONTEXT_LENGTH = "65536"
ollama serve
```

LocalPilot fails preflight with an explanation when the configured or observed context is too small. It never silently changes models.

## Security boundary

The wrapper fixes `cwd` to the candidate, uses `shell=False`, restricted and bare modes, disables session persistence, ignores external MCP servers and customizations, and makes permission prompts fail closed. File tools are confined to the working directory. Web tools, subagents, skills, hidden evaluation data, LocalPilot data, parent traversal, package managers, destructive commands, shell nesting, and Git mutation commands are denied. Only repository file reads/edits, narrow Git inspection, Python compilation, and existing pytest/unittest commands are allowed.

After every process run, LocalPilot derives changed paths from Git and rejects ungrounded paths, protected tests, deletions, links, disallowed file types, oversized files, invalid UTF-8, and secret-like assignments. A passing Claude run is still reviewed by LocalPilot and then by CI and a human. Human merge remains mandatory.

The previous in-process editor remains available only as an explicit rollback:

```toml
[selfdev]
implementation_backend = "local_tools"
```

## Post-merge benchmark

From a clean, up-to-date `main` with the background worker disabled:

```powershell
git switch main
git pull --ff-only origin main
.\.venv\Scripts\python.exe training\scripts\run_evolution_execution.py --post-cc
```

The success gate is mean execution score at least 3.5/4, zero hard failures, zero scope violations, and no material Eval v1 regression from the 1.48/4 overall and 1.3125/4 critical reasoning baselines. A configured backend or green unit test is not benchmark success.
