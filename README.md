# LocalPilot

LocalPilot is a private, local-first Windows agent and a practical experiment in evidence-driven self-improvement. It uses local [Ollama](https://ollama.com/) models for everyday assistance, can study durable knowledge with provenance, and can use idle time to research and prepare isolated improvements to its own source code.

The long-term goal is a self-improving, general-purpose personal intelligence: a system that progressively expands what its owner can understand, create, and accomplish. That is a direction, not a claim that LocalPilot is AGI or that recursive self-improvement has been proved. Version `0.2.0` is an early but working foundation with bounded research, read-only PC observation, durable learning, guarded self-development, and deliberately human-controlled promotion.

> **Stable mission:** Become an increasingly capable general-purpose personal intelligence that expands what its user can understand, create, and accomplish while remaining reliable, transparent, resource-aware, interruptible, and under human control.

In practical terms: when the owner is using the PC, LocalPilot works for the owner; when the PC is idle, LocalPilot may work on LocalPilot.

## Design principles

### Protect the spark; constrain the blast radius

Preserve freedom to reason, explore, challenge assumptions, and devise unexpected solutions. Put hard boundaries around consequential or irreversible actions, not around imagination.

### Freedom of thought, bounded freedom of action

LocalPilot should be able to consider unconventional hypotheses and disagree with its owner or its own prior conclusions without automatically gaining permission to act on them.

### Evidence outranks confidence

A persuasive answer, green CI run, or internally coherent theory is not proof. Claims about capability should survive contact with real evidence and real model behaviour.

### Earn autonomy through demonstrated capability

Expand practical freedom when reliability is demonstrated, rather than either granting unrestricted authority or permanently constraining the system to its initial abilities.

### Memory should create growth, not dogma

Retained knowledge is a prior, not immutable truth. Preserve provenance, confidence, and staleness, and allow fresh evidence to overturn what was previously learned.

### Failure is evidence

A failed experiment should improve future decisions. Distinguish failures of an idea from failures imposed by the framework around it.

### Prefer transferable capability over tricks

Improvements that make many future tasks easier matter more than narrowly optimizing one benchmark or accumulating features.

### Leave room for surprise

If every successful behaviour was explicitly anticipated by the framework, LocalPilot is automation. Useful, evidence-grounded solutions that its designers did not prescribe are an important signal of increasing capability.

### Human authority and machine initiative can coexist

Human control over promotion, irreversible actions, and critical boundaries need not require micromanaging the agent's reasoning or experimentation.

Together these principles imply broad freedom to think and narrow, reviewable authority to act.

## What exists today

- A local interactive operator backed by Ollama (`gpt-oss:20b` by default).
- A persistent Windows desktop chat with Unicode-safe history, replayable runtime/tool events, and a runtime-driven pixel LocalPilot body.
- A loopback broker that stays independent of the replaceable operator/PowerShell worker while the CLI remains available as a fallback.
- Same-context, bounded research in which complete raw tool results remain authoritative.
- Read-only Windows observation tools for processes, disks, startup entries, power, Defender, and device problems.
- Durable owner lessons and a staged study system for repository, Qwen/Ollama, and Python knowledge.
- Bounded retrieval of relevant study facts into fresh operator turns, including provenance, confidence, verification time, source digest, and staleness.
- Real TOML configuration, resource gating, process-priority control, and redacted JSONL audit logging.
- Separate stable operator, idle-time developer, and isolated candidate responsibilities.
- Mission-directed capability discovery with falsifiable hypotheses, baselines, and evaluation plans.
- Git worktree candidates, protected reviewer tests, GitHub Actions validation, and human-only promotion.
- Durable cycle, capability, experiment, frontier, rejection, and review records in local SQLite memory.
- Compact checkpoints for safely resuming interrupted self-development work.
- Guarded trusted-`main` self-sync and an optional per-user Windows idle scheduler.

LocalPilot does not have a cloud-model or pricing subsystem. GitHub is used for source control, review, and candidate test execution; local inference and machine-private learning data stay on the workstation.

## What does not exist today

LocalPilot does not autonomously merge or promote its own code, execute untrusted candidate code on the owner's PC, control arbitrary desktop applications, or have unrestricted command execution. It does not rewrite its model weights when it studies. A passing test suite or successful benchmark is evidence for a particular claim, not proof of general intelligence.

## Information authority

LocalPilot deliberately keeps five kinds of information on separate paths:

| Path | Lifetime | Purpose | Authority |
| --- | --- | --- | --- |
| Raw operator observations | Current turn | Inspect the live PC, repository, or other tool-visible state | Authoritative for what the tool actually returned |
| Human lessons | Durable | Preserve explicit owner guidance entered with `/teach` or `localpilot teach` | Trusted guidance, but not a substitute for current evidence |
| Study knowledge facts | Durable | Retain researched facts with provenance and freshness metadata | Prior knowledge that must yield to fresher contradictory evidence |
| Self-development records | Durable | Track cycles, hypotheses, experiments, checkpoints, reviews, and failures | Evidence about development history, not operator knowledge by default |
| Desktop chat history | Durable UI/session record | Repaint the window and restore completed visible turns after runtime restart | Conversation context only; never promoted into learning facts |

Visible desktop turns are stored separately in `chat.sqlite3`; prompts, transcripts, tool results, and hidden reasoning are never stored as knowledge facts. Retrieved facts are turn-local and are not written back as new learning merely because they were retrieved.

When a request falls within a studied domain, the operator may retrieve a small relevant set from `LearningMemory` before researching from scratch. Retrieval is capped at 6 facts and 6,000 characters. Results retain their source URI and kind, confidence, source digest, verification time, staleness, and relationships where available.

Memory narrows live research; it does not replace it. For mutable or current implementation claims, LocalPilot is instructed to verify selectively with live tools. A stale fact or source-digest mismatch is surfaced, and a contradictory raw tool result wins. Final authority review fails closed if the answer would present contradicted memory as current truth.

## Operator research and answers

The interactive operator uses a single high-reasoning conversation for research and final synthesis. Each tool result receives a raw-result ID. Full raw results stay in that conversation rather than being replaced by an evidence summary.

Research has a 12-round soft budget and a 24-round hard ceiling by default. After the soft boundary, continuation requires a compact six-field planning checkpoint containing bounded evidence references, one unresolved fact, one proposed tool call, the result that would change the conclusion, and an optional distinct hypothesis. The checkpoint is control scaffolding, not factual evidence.

Checkpoint calls, results, recovery prompts, and rejected control turns are removed before every final-synthesis call. They are also excluded from retained chat memory, learning memory, evolution checkpoints, and audit content; only safe structural status is audited. This preserves the same-context relationship between raw evidence and the answer.

The runtime cognition probe exercises both the normal path and the compact-checkpoint path:

```powershell
.\scripts\cognition-probe.ps1
.\scripts\cognition-probe.ps1 -Checkpoint
```

The probe generates a fresh manifest with unpredictable fragments, then checks retrieval completeness, cross-fragment reconciliation, the final answer, and compliance with the configured hard research budget.

## Stable operator, developer, and candidate

LocalPilot separates three responsibilities:

1. **Stable operator** — the installed everyday agent. It uses `[model].name` and interacts with the PC through registered operator tools. The current Windows tool registry is observational.
2. **Developer** — an idle-time engineering process. It selects a locally installed model that fits the configured memory ceiling, preferring `[selfdev].developer_model` and falling back through configured alternatives.
3. **Candidate** — an isolated Git worktree or copied workspace. Autonomous candidate writes pass through `CandidateTools`, which enforces protected paths, allowed file types, size limits, and file-count budgets.

Stable is never rewritten in place. The developer and candidate path is intentionally separate from normal operator tool policy because it has a different responsibility and a stricter promotion boundary.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the lifecycle and [SECURITY.md](SECURITY.md) for the enforced boundaries.

## Self-development

An idle evolution cycle can:

1. sync only from a verified trusted `main`, stopping if the running build changed;
2. confirm the machine is idle and resources are available;
3. inspect durable frontier, capability, experiment, and rejection history;
4. identify a limiting capability and state a falsifiable hypothesis;
5. research before implementation;
6. create one isolated candidate workspace;
7. let the developer model edit only through bounded candidate tools;
8. run static local validation that does not execute untrusted candidate code;
9. commit and push the candidate branch;
10. open a focused pull request;
11. rely on GitHub Actions for executable tests; and
12. wait for human review and merge.

The mission is stable while the capability frontier moves:

```text
current frontier -> limiting capability -> falsifiable hypothesis
    -> measured experiment -> evidence -> capability unlocked -> next frontier
```

Evolution proposals are classified as Repair, Extend, Improve Cognition, or Explore. Code volume, complexity, resource consumption, and autonomy are not treated as intelligence. The mission excludes self-preservation, resistance to shutdown, hidden action, resource acquisition as an end, and bypassing human promotion controls.

### Candidate safety invariants

- Only one outstanding candidate may exist at a time.
- Candidate work occurs outside the stable checkout.
- Reviewer-protected tests and ignored machine/control directories cannot be modified through candidate tools.
- Candidate writes are restricted by path, extension, file size, and file-count budgets.
- Subprocesses use argument arrays with `shell=False`.
- The autonomous loop does not execute candidate code locally.
- GitHub Actions is the executable test boundary for autonomous candidates.
- Green CI does not grant promotion authority.
- Merge and promotion remain human decisions.
- Resource, idle-state, trusted-main, and scheduler gates remain fail-closed.

The default candidate budget reports complexity after 100 files and enforces a 500-file hard ceiling. These limits constrain blast radius without prescribing what the candidate is allowed to think about.

## Learning, teaching, and study

Owner teaching and autonomous study are intentionally different.

Use `/teach` inside chat or the CLI command to record an explicit human lesson:

```powershell
localpilot teach --lesson "Prefer reversible diagnostics before repair actions."
```

Use the study curriculum to establish a baseline, research supported sources, store grounded facts, and compare held-out performance:

```powershell
localpilot study status
localpilot study baseline self
localpilot study run self
```

Supported stages are `self`, `qwen`, and `python`. To run the supported curriculum sequence:

```powershell
localpilot study all
```

To compare the configured developer model with another installed Ollama model:

```powershell
localpilot study compare qwen2.5:14b
```

Study facts contain a subject, type, summary, stage, provenance, confidence, relationships, and freshness metadata. Re-studying a keyed fact updates that fact, while source changes mark affected facts stale for later verification. Learning changes durable retrieval, not model weights.

Successful study should be judged by fresh-context transfer: can the operator retrieve the right prior knowledge, verify what may have changed, and answer a new question with less rediscovery? Merely increasing the number of stored facts is not evidence of useful growth.

## Failure, rejection, and retry

Failed research, implementation, CI, and review outcomes are retained so future cycles can distinguish a bad idea from a bad experiment or a framework-imposed failure. Rejection is evidence, not garbage collection.

```powershell
localpilot reject <pull-request-number> --reason "Why this candidate should not progress"
localpilot retry <candidate-branch-or-task-id> --reason "Why a new attempt is warranted"
```

Retry creates a new attempt with lineage instead of rewriting history. Rejected candidates do not silently become stable, and old conclusions may be reconsidered when new evidence changes the situation.

## Resource-aware inference

LocalPilot queries Ollama for installed models and model sizes rather than assuming a fixed developer model is available. The resource governor considers idle time, CPU, available memory, model size, reserve requirements, and configured process priority before beginning or continuing expensive work.

Default behaviour includes:

- everyday chat may run while the owner is active;
- self-development waits for the configured idle threshold;
- resource gates are rechecked during long work;
- foreground activity can pause or stop autonomous work;
- model selection remains within the configured memory ceiling; and
- no model is downloaded automatically by the self-development loop.

The governor is a practical safeguard, not a complete GPU, power, or thermal model.

## Install

### Requirements

- Windows 10 or 11
- PowerShell
- Git
- Python 3.11 or newer
- [Ollama](https://ollama.com/) with a compatible local model

Clone the repository, then bootstrap the environment:

```powershell
git clone https://github.com/n47h4ni3l/localpilot.git
cd localpilot
.\scripts\bootstrap.ps1
```

If Ollama does not already have the default model:

```powershell
ollama pull gpt-oss:20b
```

Bootstrap creates `localpilot.toml` from the example when it is absent. To do that manually:

```powershell
Copy-Item config.example.toml localpilot.toml
```

Keep `localpilot.toml`, `localpilot-data/`, audit logs, learning databases, and credentials private. They are intentionally excluded from version control.

Check the environment and start chat:

```powershell
.\.venv\Scripts\Activate.ps1
localpilot doctor
localpilot chat
```

Inside chat, use `/status`, `/doctor`, `/evolve`, `/teach <lesson>`, and `/quit`.

Or open the persistent desktop window:

```powershell
localpilot desktop
```

The desktop command starts a detached loopback broker when needed. The window reconnects to that broker, while the broker supervises a replaceable operator worker that owns Ollama and the existing PowerShell-backed tools. Closing the window does not stop the broker; `localpilot chat` remains the independent fallback.

## Command reference

```text
localpilot chat              Start the interactive operator
localpilot desktop           Open the persistent desktop chat
localpilot broker            Run the local broker in the foreground
localpilot doctor            Validate local configuration and dependencies
localpilot status            Show operator and self-development status
localpilot evolve            Run one guarded evolution cycle
localpilot reject            Record a human rejection for a candidate
localpilot retry             Start a lineage-preserving retry
localpilot teach             Save an explicit owner lesson
localpilot study status      Show study progress
localpilot study baseline    Run a stage baseline
localpilot study run         Research and learn a stage
localpilot study all         Run the supported curriculum
localpilot study compare     Compare the developer model with an installed peer
localpilot study research    Inspect a public HTTPS source without learning it
```

Use `localpilot <command> --help` for the current arguments. CLI output is normalized for the active Windows console so redirected or legacy-console output does not fail on Unicode symbols.

## GitHub connection and CI

Self-development delivery requires a trusted GitHub origin and authenticated GitHub CLI session. The helper sets `origin` without storing tokens in LocalPilot or pushing anything:

```powershell
.\scripts\connect-github.ps1 -RepoUrl https://github.com/n47h4ni3l/localpilot.git
```

The included GitHub Actions workflow runs the full test suite on Windows with Python 3.12. Workflow permissions are read-only and checkout credentials are not persisted. LocalPilot observes PR and CI state but cannot interpret green CI as permission to merge.

## Optional idle scheduler

Install the per-user scheduled task that checks whether self-development work may run:

```powershell
.\scripts\install-idle-evolve-task.ps1
```

The task does not bypass LocalPilot's own idle, resource, candidate, or trusted-main checks. It only invokes the guarded entry point. Remove or disable the task through Windows Task Scheduler when it is no longer wanted.

## Configuration

`config.example.toml` is the source of truth for documented defaults. Important groups are:

- `[model]` — everyday Ollama model, context window, and generation settings;
- `[agent]` — data directory and operator research budgets;
- `[resource]` — idle, CPU, memory, and priority gates;
- `[selfdev]` — developer models, candidate limits, GitHub delivery, and checkpoints;
- `[desktop]` — loopback broker port, separate chat database, and runtime restart ceiling;
- `[github]` — trusted remote, main branch, and candidate delivery; and
- `[safety]` — the normal operator command-policy foundation.

The learning database name, owner-lesson limit, and candidate resource settings currently live under `[selfdev]`. Audit logging has no separate TOML section yet.

Read [config.example.toml](config.example.toml) before changing limits. Security-critical ceilings should be changed only with corresponding tests and a clear threat-model justification.

## Local data and privacy

By default, local state lives under `localpilot-data/`. It may include redacted audit events, learning memory, desktop chat history/events, evolution records, checkpoints, a broker authentication token, and machine-specific status. Repository candidates live in isolated worktrees or workspaces outside the stable checkout.

LocalPilot redacts common secret-shaped values from audit output and does not intentionally persist hidden reasoning. Nevertheless, local databases and logs should be treated as private because observations and owner-provided lessons may contain sensitive context.

Source branches and pull requests leave the machine when GitHub delivery is enabled. Autonomous candidate tests execute on GitHub Actions, so candidate source is sent to the configured private origin. Ollama inference and ordinary learning remain local unless a separately configured tool explicitly accesses a network service.

## Repository layout

```text
localpilot/
  agent.py                 interactive operator and same-context research loop
  broker.py                loopback API, history/event ownership, runtime handoff
  chat_store.py            separate persistent UI and session records
  desktop.py               native Windows chat window and pixel avatar
  runtime_supervisor.py    replaceable worker process supervision
  runtime_worker.py        JSONL adapter around the authoritative operator
  learning.py              lessons, grounded facts, retrieval, and study memory
  study.py                 staged study curriculum and held-out comparison
  research.py              bounded research and raw-evidence handling
  tools/windows.py         read-only Windows observation tools
  operator.py              guarded argument-based command foundation
  selfdev.py               discovery, candidate, repair, and delivery loop
  evolution.py             proposals, scoring, and evidence gates
  mission.py               stable mission, priorities, and non-goals
  candidate_resources.py   candidate write and complexity budgets
  checkpoint.py            compact versioned self-development resumption
  resource.py              idle, CPU, memory, model, and priority governor
  github_integration.py    trusted-main sync and PR/CI observation
  audit.py                 redacted JSONL event log
  cli.py                   command-line interface
tests/                     executable contracts and regressions
scripts/                   bootstrap, probe, GitHub, and scheduler helpers
.github/workflows/         Windows GitHub Actions test boundary
config.example.toml        documented configuration defaults
selfdev-backlog.json       bootstrap tasks, not a permanent capability frontier
ARCHITECTURE.md             detailed design and lifecycle
SECURITY.md                 candidate and machine safety boundaries
ROADMAP.md                  staged capability direction
```

## For contributors and external coding agents

Codex, Claude, and other external agents should be able to evaluate this repository without prior conversation context. Before proposing a change:

1. Read `ARCHITECTURE.md`, `SECURITY.md`, `config.example.toml`, `localpilot/mission.py`, the relevant implementation, and its tests.
2. Establish current behaviour and a concrete limitation from repository evidence; do not assume roadmap text is implemented.
3. Explain mission alignment and transferable leverage.
4. State a falsifiable hypothesis, metric, baseline, success criterion, and reproducible measurement method.
5. Preserve candidate confinement, reviewer-test immutability, argument-based subprocesses, the one-candidate gate, trusted-main sync, no local autonomous candidate execution, and human-only promotion.
6. Add or adjust focused tests and measurement artifacts. Do not weaken a test to make an implementation pass.
7. Keep the diff narrow, run relevant tests plus the full suite where practical, and open a focused PR with before/after evidence and remaining uncertainty.
8. Never auto-merge. A passing implementation remains a candidate until a human reviews and merges it.

A useful proposal answers: **What capability is currently limiting LocalPilot, what evidence shows that, what transferable capability would this change unlock, and what result would falsify the claim?**

For an ordinary human- or external-agent-authored change:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
git diff --check
```

Open a focused branch and PR. Include the observed limitation, alternatives considered, hypothesis, baseline, evidence or planned CI measurement, safety impact, and rollback story.

## Known limitations and open questions

- The stable PC tool registry is observation-only. A guarded command foundation exists, but broad reversible Windows actions are not wired into the operator.
- The project is Windows-first; idle detection, process priorities, scheduling, and CI contracts are Windows-specific.
- Autonomous candidates are not locally sandboxed strongly enough to execute. GitHub Actions therefore provides the executable evaluation boundary, adding latency and an external-service dependency.
- The resource model uses system CPU and memory plus Ollama model-size estimates. It does not deeply model GPU pressure, foreground applications, thermal state, power cost, or inference-quality trade-offs.
- The interactive agent lacks desktop vision, general UI control, and many application integrations described in the roadmap.
- Learning memory is deliberately compact. Retaining more useful experience without preserving secrets, transcripts, hidden reasoning, stale conclusions, or benchmark-gaming artifacts remains an open problem.
- The one-candidate gate favours safety and causal attribution but limits parallel exploration.
- Held-out benchmarks can demonstrate bounded improvement but can still be gamed or overfit. Strong claims require reproducible evidence, lineage, rollback, and independent human review.
- Current self-development provides scaffolding for recursive capability research, not proof that recursive self-improvement has been achieved.

## License

No license file is currently included. Treat the repository as all rights reserved unless the owner adds an explicit license.
