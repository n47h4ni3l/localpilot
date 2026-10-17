# Claude Code tool and implementation backend

Nestra can use **Claude Code as a tool** from chat and native Evolution. The normal orchestrator remains Nestra using `selfdev.implementation_backend = "local_tools"`; the tool supplies a coding harness when Nestra chooses to call it. Claude Code is the program running the tools. It uses the same accepted local Ollama model configured for Nestra, development, and implementation. The accepted bundle model is `nestra:p1.2`; frozen baselines remain separate rollback and comparison material.

The earlier fixed `implementation_backend = "claude_code"` route remains available as an explicit configuration choice. Native orchestration is a supported path with access to the Claude Code tool. A tool invocation, configured backend, or passing unit test does not establish a completed candidate, a successful Evolution cycle, model improvement, or benchmark promotion.

## Chat and native Evolution

The model-facing tool is `claude_code(task, allowed_paths=None)`. In chat, an omitted or empty path list requests read-only repository assistance. Explicit canonical repository-relative paths request a proposed edit limited to those exact files. Chat prepares a fresh private Git snapshot of safe committed source and requires a clean source checkout. Owner Git configuration, hooks, remotes, private configuration, secrets, hidden evaluations, training outputs, and reports are omitted from that snapshot.

The returned status, source revision, changed paths, proposal location/hash, and diff digest describe what actually happened. `pending_review` is a proposal state, not acceptance. Chat reports `applied_to_production: false` and `tests_independently_verified: false`; it does not apply, commit, or push the proposed edits. Read-only assistance does not claim candidate test success.

Native Evolution exposes the same tool after repository grounding, and during bounded repair of static-check or CI failures. Here omitting paths uses the verified implementation or candidate-owned repair scope; an explicit list must be a nonempty subset of that scope. The original experiment contract, resource budget, foreground-preemption, candidate validation, independent review, checkpoint, CI, delivery, and human merge gates still apply. Nestra can continue using its other local tools without invoking Claude Code.

The CLI's tool responses pass through a private loopback gateway before execution. It collects a bounded complete provider response, validates tool identities and argument structure, and serializes valid responses into canonical streaming events for the CLI. One malformed-response retry is allowed per invocation before any part of that rejected response is forwarded. A repeated invalid response becomes an explicit error. The gateway never invents tool names or arguments and never switches models. Retain the actual diagnostics when the CLI or local provider rejects a request; an unfamiliar-model warning alone does not establish that the request failed.

## Configuration and runtime

The private Windows bundle includes the pinned Claude Code **2.1.263** executable, PortableGit with Bash, and an isolated `state/claude` directory. The launcher sets `CLAUDE_CODE_GIT_BASH_PATH`, `CLAUDE_CONFIG_DIR`, and the private runtime paths only for its children. It does not copy the owner's Claude configuration or authentication material. The bundle dependency and license receipts identify the exact executable; including it does not establish permission for public redistribution.

For an existing installation, configure the actual installed Claude executable and the loopback Ollama endpoint. Keep all three model selectors on the accepted model:

```toml
[model]
name = "nestra:p1.2"

[selfdev]
developer_model = "nestra:p1.2"
developer_model_fallbacks = []
implementation_backend = "local_tools"
implementation_model = "nestra:p1.2"
implementation_context_tokens = 65536
implementation_max_turns = 40
implementation_timeout_seconds = 600
implementation_review_repair_passes = 12
implementation_max_output_tokens = 2048
implementation_max_output_chars = 120000
implementation_executable = "E:\\Tools\\ClaudeCode\\claude.exe"
implementation_base_url = "http://127.0.0.1:11434"
```

The bundle configures its own private executable and uses Ollama port **11435**, whereas the existing-server example uses 11434. Changing the model or backend is a separate explicit configuration decision. The optional fixed-backend route changes only `implementation_backend` to `"claude_code"`; the model selectors and other boundaries remain in force.

No Anthropic login is required in local Ollama mode. The child receives an empty `ANTHROPIC_API_KEY`, a temporary private gateway URL, and a per-invocation authentication token. The gateway forwards only to the configured loopback Ollama endpoint using `ollama` authentication. Owner GitHub credentials and general proxy settings are not forwarded into the coding child. This supplies local inference access, not permission to publish changes.

The wrapper checks the executable's version/help contract before use. It uses print-mode JSON, explicit model and turn limits, restricted/bare mode, explicit tool permissions, disabled external MCP configuration, and no session persistence. It never uses `--dangerously-skip-permissions`. A separately installed CLI must support the required flags; a version number alone is not proof of compatibility.

The checked-in preflight is available for an explicitly authorized live check:

```powershell
.\scripts\preflight-claude-code.ps1 -ClaudePath E:\Tools\ClaudeCode\claude.exe -Model nestra:p1.2 -RequiredContext 65536
```

Preflight performs a real local model load and reads Ollama's live allocation. The wrapper requires at least **65,536 context tokens** for the coding harness. This is separate from the normal chat context and may use more memory; a metadata-only check does not prove that the allocation is viable. Configured or observed context below the requirement fails with an explanation. It does not silently choose a smaller context or another model. Outer memory, time, and foreground guards remain relevant even after preflight succeeds.

## Permission and verification boundaries

The wrapper fixes the child working directory to the candidate, uses `shell=False`, disables hooks/customizations, supplies explicit CLI allow/deny rules, and makes permission prompts fail closed. Read-only requests advertise only repository read tools. Edit requests generate exact Edit/Write permissions for the approved paths and permit a narrow set of repository inspection and existing test commands. External web tools, subagents, skills, hidden evaluation data, LocalPilot runtime data, parent traversal, package managers, destructive commands, shell nesting, and Git mutation commands are denied by those CLI rules.

These are application and CLI permission controls, not an operating-system sandbox. Separate candidate checkouts, token-free child environments, bounded execution, and independent inspection reduce the scope of the operation; they do not make arbitrary third-party executables harmless.

Each request retains turn, retry, wall-clock, output-token, and output-character limits. On timeout or a resource stop, the wrapper terminates its process tree and asks the local Ollama endpoint to stop the outstanding model request. A malformed tool response is retained as a failure rather than treated as successful implementation.

After an Evolution edit run, LocalPilot derives changed paths from Git and rejects changes outside the grounded contract, protected tests, deletions, links, disallowed file types, oversized files, invalid UTF-8, and secret-like assignments. It independently runs the bounded repository test commands with a token-free environment, recording pass/fail, exit code, and an output digest. CLI-reported test claims alone are not verification. An edit proposal with passing checks still requires independent review and the normal Evolution/CI/merge path before acceptance.

## Separate benchmark gate

When a post-merge benchmark is explicitly authorized, run the existing suite from a clean, current `main` with the background worker disabled:

```powershell
.\.venv\Scripts\python.exe training\scripts\run_evolution_execution.py --post-cc
```

Use the actual report rather than inferring success from configuration or unit tests. Preserve raw failures, candidate evidence, and baseline identities; operational readiness and benchmark improvement are separate conclusions.
