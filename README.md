# LocalPilot

LocalPilot is a private, local-first Windows agent and an experiment in evidence-driven, cumulative capability growth. It uses local [Ollama](https://ollama.com/) models for everyday assistance, persistent memory, bounded research, passive machine awareness, structured study, and isolated software-development candidates.

The current package version is `0.2.0`. LocalPilot is a working agent system, not a claim of AGI: it can operate persistently, learn through explicit memory and source-grounded study, and prepare proposed improvements to its own repository, but it does not train its model weights, execute arbitrary commands, or promote its own code.

> **Stable mission:** Become an increasingly capable general-purpose personal intelligence that expands what its user can understand, create, and accomplish while remaining reliable, transparent, resource-aware, interruptible, and under human control.

In practical terms, LocalPilot works for the owner while the PC is active and may work on LocalPilot when the PC is idle.

## Current status

| Capability | Current state |
| --- | --- |
| Local conversation through Ollama | Implemented |
| Persistent desktop chat and CLI fallback | Implemented |
| Repository, GitHub, PC, web, and optional library research | Implemented with bounded read-only tools |
| Passive Windows hardware and runtime awareness | Implemented through SystemSense |
| Explicit owner teaching and durable typed memory | Implemented |
| Benchmarked source-grounded study | Implemented for `self`, `qwen`, and `python` |
| Progressive background library reading | Implemented when the library is enabled |
| Idle capability discovery and candidate development | Implemented with resource and authority gates |
| Candidate branch, pull request, and CI lifecycle | Implemented when GitHub is configured |
| Autonomous merge or stable-code promotion | Not enabled |
| Arbitrary desktop or shell control | Not implemented |
| Local execution of autonomous candidate code | Disabled |
| Model-weight training or fine-tuning | Not implemented |
| Sustained recursive self-improvement | Not demonstrated |

That distinction matters. LocalPilot has substantial infrastructure for attempting measurable improvement, but infrastructure is not evidence that open-ended or recursive capability growth has occurred.

## Design principles

### Evidence outranks confidence

A persuasive answer, a coherent plan, or a green test run is evidence for a bounded claim, not proof of general intelligence or permission to act.

### Preserve initiative; constrain consequences

LocalPilot should be free to reason, explore, challenge assumptions, and propose unexpected solutions. Consequential actions remain typed, bounded, reviewable, and recoverable.

### Memory is revisable

Stored knowledge is prior context, not permanent truth. Provenance, confidence, source digests, verification time, and staleness allow current evidence to overturn earlier learning.

### Failure should teach

Failed research, rejected candidates, CI failures, and policy blocks remain development evidence. Later cycles should be able to distinguish a bad idea from a bad experiment or a framework-imposed failure.

### Prefer transferable capability

Improvements that help across many future tasks matter more than feature count, code volume, autonomy, or benchmark-specific tricks.

### Human authority and machine initiative can coexist

LocalPilot can choose what to investigate and propose while human review remains the boundary for merge, promotion, and other consequential authority.

## System architecture

LocalPilot separates three responsibilities.

### Operator

The Operator is the everyday agent. It owns conversation, local inference, evidence gathering, bounded tool use, and retrieval of relevant learning.

The Operator can:

- converse through the command line or persistent desktop interface;
- inspect the current repository, local Git state, authenticated GitHub metadata, and runtime lifecycle;
- observe Windows system, storage, process, startup, power, Defender, device, and SystemSense state;
- search the public web and read bounded public HTTPS text;
- search an enabled owner-managed local library;
- retrieve explicit owner lessons and relevant durable learning; and
- perform a small allow-list of reversible Windows actions.

The Operator does not receive unrestricted command execution, arbitrary filesystem access, generic process termination, or arbitrary desktop control.

### Developer

The Developer is the idle-time engineering process. It selects an installed Ollama model that fits configured resource limits, inspects current capability evidence, proposes a measurable improvement, researches it, and prepares a candidate.

Developer work is bounded by:

- trusted-`main` verification;
- foreground-use, CPU, memory, wall-clock, and tool-call gates;
- one outstanding candidate at a time;
- a persistent opportunity queue that rejects near-duplicate proposals;
- live repository grounding before write-capable tools are exposed; and
- isolated candidate workspaces.

### Candidate

A Candidate is a proposed version of LocalPilot, not the installed stable agent.

Candidate tools enforce path, symlink, file-type, file-size, archive, resource, and file-count restrictions. Reviewer-protected tests, Git metadata, CI definitions, virtual environments, caches, and private LocalPilot data are protected.

Local validation parses or compiles candidate files without importing or executing untrusted candidate code. Executable validation occurs in GitHub Actions. Passing CI still does not merge or promote the candidate.

For the complete lifecycle, read [ARCHITECTURE.md](ARCHITECTURE.md). For enforced authority boundaries, read [SECURITY.md](SECURITY.md).

## Operator research and information authority

LocalPilot keeps the raw evidence used during an interactive investigation in the same high-reasoning conversation as the final synthesis. Tool output is not replaced by a lossy evidence summary before the answer is written.

Operator research is bounded by default:

- 12 tool rounds form the soft boundary;
- 24 unique tool rounds form the hard ceiling;
- after the soft boundary, another observation requires a compact planning checkpoint; and
- exact duplicate read-only observations reuse the current-turn cache.

The checkpoint controls research; it is not factual evidence and is removed before synthesis.

Consequential repository and machine claims are checked after synthesis. Current paths, Python symbols, configuration fields, direct call relationships, selected lifecycle contracts, storage claims, and power-plan claims must be supported by the relevant live evidence. Unsupported claims are corrected or withheld rather than presented as established fact.

Public HTTPS reads validate the freshly resolved public address in the connection path while preserving hostname certificate verification. The same redirect and DNS-rebinding protection is used by Operator research, study-source inspection, and candidate-resource downloads.

Ordinary conversation is not forced through tool use. Recent operator behavior also distinguishes grounded claims from conversational judgment: LocalPilot can answer directly, offer an opinion, or make a choice without inventing personal experience, unseen activity, current external facts, or delivery deadlines it cannot verify.

## Information and memory paths

LocalPilot deliberately keeps different kinds of information separate.

| Path | Lifetime | Purpose | Authority |
| --- | --- | --- | --- |
| Raw tool observations | Current turn | Inspect current repository, PC, GitHub, library, or web state | Authoritative for what the tool returned |
| Human lessons | Durable | Preserve explicit owner guidance from `/teach` or `localpilot teach` | Trusted guidance, subject to current evidence |
| Study facts | Durable | Retain verified source-linked knowledge | Prior knowledge with provenance and freshness |
| Typed library learnings | Durable | Retain claims, concepts, heuristics, questions, hypotheses, and opinions without flattening them into facts | Type-specific prior context |
| Self-development records | Durable | Track opportunities, cycles, hypotheses, experiments, failures, reviews, and outcomes | Evidence about development history |
| Desktop chat history | Durable UI/session record | Restore visible conversations after restart | Conversation continuity only |

Visible desktop messages live in `chat.sqlite3`. Ordinary chat is not silently promoted into learning facts, and retrieved memory is not written back merely because it appeared in a prompt.

Retrieval is lexical by default. Optional semantic retrieval uses an explicitly installed local Ollama embedding model and retains the same provenance, stage, staleness, digest, evidence, and context-size controls. LocalPilot does not download an embedding model automatically. If embeddings fail, retrieval falls back to lexical matching.

## Teaching, study, and library learning

These are three separate learning mechanisms.

### Explicit owner teaching

Save a durable lesson from the CLI:

```powershell
localpilot teach --lesson "Prefer reversible diagnostics before repair actions."
localpilot teach --list
```

Inside chat, use:

```text
/teach Prefer reversible diagnostics before repair actions.
```

Owner teaching is concise guidance. It is not a transcript import and does not override stronger current evidence.

### Benchmarked study

The staged curriculum is `self -> qwen -> python`. Each stage records a held-out baseline, stores source-grounded facts, retests, and retains weak areas or failures for the next attempt.

```powershell
localpilot study status
localpilot study baseline self
localpilot study run self
localpilot study all
localpilot study compare qwen2.5:14b
```

Use `--allow-web` with `study run` or `study all` to permit authoritative public HTTPS research. A transient source can be inspected without promoting it to knowledge:

```powershell
localpilot study research https://docs.python.org/3/
```

Study changes durable knowledge and retrieval. It does not modify model weights.

### Owner-managed local library

The optional library indexes owner-provided PDF and UTF-8 text sources without changing them. The disposable full-text index lives under the private data directory.

```powershell
localpilot library status
localpilot library index
localpilot library search "query terms"
```

When the library is enabled and resources permit, the background runtime may read one bounded contiguous section, reflect on it, extract a small set of typed candidate learnings, verify each item against the exact source range and digest, and persist only verified results.

```text
read -> reflect -> extract -> verify passage and digest
     -> persist typed learning -> retrieve and use -> measure
```

A changed source digest makes earlier learning stale until it is verified again. Raw passages, full private notes, and hidden reasoning are not stored as authoritative knowledge.

See [docs/library-folder-readme.md](docs/library-folder-readme.md) for supported formats, indexing behavior, and privacy boundaries.

## SystemSense

SystemSense is a passive, read-only Windows observation layer. It samples bounded dynamic state, collects slower-changing inventory, stores local history, derives compact health signals and baselines, and can relate inference performance to observed workload conditions.

Depending on available Windows providers and sensors, it can expose:

- CPU, memory, storage, process, and contention state;
- hardware, firmware, network, storage, device, and driver inventory;
- sensor values from an available read-only hardware-monitor namespace;
- rolling anomalies and bounded history;
- model inference performance; and
- observational workload correlations.

SystemSense does not control devices, drivers, fans, clocks, voltages, or processes. Correlation does not establish causation, and inactive or older driver packages are review signals rather than deletion recommendations.

The desktop glance panel reads one authenticated loopback summary. It cannot trigger collection or mutate the machine.

See [docs/systemsense.md](docs/systemsense.md) for providers, retention, privacy, query surfaces, and coverage limits.

## Autonomous evolution

An evolution cycle can:

1. reconcile an existing candidate and its pull-request or CI state;
2. verify the stable checkout and guarded trusted-`main` state;
3. stop or defer when the owner is active or capacity is insufficient;
4. resume a valid checkpoint or select a novel queued opportunity;
5. identify a limiting capability and state a falsifiable hypothesis;
6. record a baseline, success criterion, and measurement method;
7. research with read-only repository and bounded web tools;
8. generate and validate a repository-claim manifest against the live candidate tree;
9. create or resume one isolated candidate workspace;
10. expose bounded candidate-writing tools;
11. perform non-executing local static validation and bounded repair;
12. commit and, when configured, push the candidate branch;
13. create or recover a focused GitHub pull request;
14. observe executable GitHub Actions results; and
15. wait for explicit human review and merge.

Proposals are classified as **Repair**, **Extend**, **Improve Cognition**, or **Explore**. The stable mission does not change, but the recorded capability frontier may move as experiments succeed or fail.

Default whole-cycle limits are 15 minutes, 32 total tool calls, and 8 public-web calls. Candidate complexity is reported after 100 files and blocked at the configured 500-file hard ceiling. These defaults are documented in [config.example.toml](config.example.toml).

Run one normal guarded cycle:

```powershell
localpilot evolve
```

A manual `--force` bypasses the idle-time requirement only. It does not bypass capacity, safety, trusted-repository, candidate, CI, or promotion controls.

### Failure, rejection, and retry

A pushed branch, passing static checks, and green CI are separate facts. None completes promotion.

Explicitly reject a managed candidate:

```powershell
localpilot reject <pull-request-number> --reason "Why this candidate should not progress"
```

Authorize a new lineage-preserving attempt only for a recorded framework-policy block:

```powershell
localpilot retry <candidate-branch-or-task-id> --reason "Why a new attempt is warranted"
```

Rejection and retry retain the earlier branch, pull request, outcome, and lesson. They do not rewrite history or grant merge authority.

## Desktop and runtime

`localpilot desktop` opens the WebView desktop companion. A native transparent avatar represents collapsed mode; expanded mode provides persistent chat, conversation selection, runtime events, and the read-only SystemSense glance panel. Use `localpilot desktop --tkinter` for the legacy Tkinter interface.

The desktop talks to a loopback-only broker authenticated with a per-install token. The broker owns visible chat persistence and supervises a replaceable runtime worker that owns Ollama and the registered operator tools.

A long request crossing the configured status threshold continues on the same worker and is reported as still running. Unexpected worker exit receives bounded recovery. Lifecycle events retain process identity, reason, affected request identifiers, and local Git state so later status answers can be based on evidence rather than guesswork.

Foreground conversation takes priority over background inference. Active foreground-turn state can defer autonomous work, and the everyday model may remain resident across normal conversation while background development models unload according to configuration.

The CLI remains an independent fallback even when the desktop or broker is not running.

## Install

### Requirements

- Windows 10 or Windows 11
- PowerShell
- Git
- Python 3.11 or newer
- [Ollama](https://ollama.com/)
- an installed Ollama chat model
- GitHub CLI only if candidate delivery to GitHub is required

Clone and bootstrap:

```powershell
git clone https://github.com/n47h4ni3l/localpilot.git
cd localpilot
.\scripts\bootstrap.ps1
```

Bootstrap creates `.venv`, installs LocalPilot with development dependencies, creates `localpilot.toml` from the example when needed, and offers to download the default `gpt-oss:20b` model if it is missing.

Activate the environment and verify the installation:

```powershell
.\.venv\Scripts\Activate.ps1
localpilot doctor
localpilot status
```

Start the CLI operator:

```powershell
localpilot
```

or:

```powershell
localpilot chat
```

Open the desktop companion:

```powershell
localpilot desktop
```

Inside CLI chat, the available slash commands are `/status`, `/doctor`, `/evolve`, `/teach <lesson>`, and `/quit`.

## Command reference

```text
localpilot                         Start the interactive Operator
localpilot chat                    Start the interactive Operator
localpilot desktop                 Open the WebView desktop companion
localpilot desktop --tkinter       Open the legacy Tkinter desktop
localpilot broker                  Run the loopback broker in the foreground
localpilot doctor                  Check configuration, models, Git, and GitHub readiness
localpilot status                  Show resource, repository, and evolution status
localpilot evolve [--force]        Run one guarded evolution cycle
localpilot reject <PR>             Record an explicit human rejection
localpilot retry <candidate>       Authorize a policy-blocked retry with lineage
localpilot teach                   Save or list explicit owner lessons
localpilot study                   Inspect or run the staged curriculum
localpilot library                 Inspect, index, or search the local library
```

Use `localpilot <command> --help` for exact arguments.

## Persistent background worker

The optional Windows scheduler installer requires a clean checkout on the configured trusted branch, an existing `.venv`, and `localpilot.toml`:

```powershell
.\scripts\install-idle-evolve-task.ps1
```

It registers `LocalPilot Background Worker`, starts one hidden `pythonw.exe` worker at user logon, and adds a one-minute watchdog trigger. The worker polls the guarded evolution entry point every 30 seconds by default and prevents overlapping cycles with an OS-backed lock.

The installer verifies that the replacement worker is running without a visible window before disabling the legacy `LocalPilot Idle Evolve` task. It refuses a dirty checkout or the wrong branch.

To request a clean stop, disable the scheduled task first:

```powershell
Disable-ScheduledTask -TaskName "LocalPilot Background Worker"
.\.venv\Scripts\python.exe -m localpilot.background_worker --root . --config .\localpilot.toml --stop
```

An active cycle stops at its established safe boundary. Trusted-`main` updates cause the persistent worker to exit so the watchdog can start a fresh interpreter from the new code.

## GitHub connection and CI

Self-development delivery requires a trusted `origin` and an authenticated GitHub CLI session. The helper configures the remote without storing a token in LocalPilot or pushing changes:

```powershell
.\scripts\connect-github.ps1 -RepoUrl https://github.com/n47h4ni3l/localpilot.git
```

The included workflow runs the full pytest suite on `windows-latest` with Python 3.12. Workflow permissions are read-only and checkout credentials are not persisted.

GitHub is an executable validation and review boundary, not a promotion authority. LocalPilot has no autonomous merge path, and `selfdev.auto_promote=true` is rejected by configuration validation.

## Configuration

`config.example.toml` documents the supported defaults:

- `[agent]`: private data directory and Operator research budgets;
- `[model]`: everyday Ollama model, context allocation, generation settings, keep-alive, and optional semantic retrieval;
- `[resource]`: active and idle priority plus CPU, memory, and idle gates;
- `[safety]`: read-only, reversible, and destructive-action policy flags;
- `[github]`: trusted remote, main branch, and candidate delivery;
- `[desktop]`: loopback broker, chat database, and restart limit;
- `[library]`: optional source root and bounded indexing limits;
- `[systemsense]`: collection cadence, retention, baselines, correlations, and compact context; and
- `[selfdev]`: developer models, cycle budgets, candidate limits, resources, repair, and learning storage.

The default everyday model is `gpt-oss:20b`. The preferred background developer model is `qwen2.5:32b`, with `qwen2.5:14b` configured as a fallback. LocalPilot uses installed model metadata and resource limits; it does not assume those models are available or download them automatically during evolution.

Read the example and corresponding tests before changing security-critical limits.

## Local data and privacy

Private state is stored under the configured `agent.data_dir`, `localpilot-data` by default. It can include:

- `audit.jsonl`;
- `chat.sqlite3`;
- `learning.sqlite3`;
- `systemsense.sqlite3`;
- library indexes and reading state;
- evolution opportunities, run state, and checkpoints;
- candidate workspaces and candidate resources;
- a broker authentication token; and
- machine-specific process and runtime state.

These files are excluded from version control and should not be published. LocalPilot does not intentionally persist hidden reasoning, but local observations, chat, lessons, and audit events may still contain sensitive context.

Ollama inference and machine-private learning remain local. Public-web tools contact selected HTTPS sources. Candidate branches and source leave the workstation when GitHub delivery is enabled, and executable candidate tests run on GitHub Actions.

## Testing

Run the repository suite:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

On Windows, a dedicated pytest temporary directory can avoid contention with a running LocalPilot process:

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp=".\.pytest-tmp"
```

Run the runtime cognition probe on both normal and post-soft-boundary paths:

```powershell
.\scripts\cognition-probe.ps1
.\scripts\cognition-probe.ps1 -Checkpoint
```

Before committing, also run:

```powershell
git diff --check
```

The current repository collects 512 pytest tests. Treat that as a point-in-time verification of this checkout, not a permanent project invariant.

## Repository layout

```text
localpilot/
  agent.py                     Operator orchestration and conversation loop
  agent_prompt.py              system prompt and behavioral contract
  agent_evidence.py            evidence routing and claim requirements
  agent_tools.py               tool-loop mechanics
  agent_runtime_support.py     runtime and postvalidation support
  authority.py                 deterministic information-authority checks
  broker.py                    authenticated loopback desktop API
  runtime_supervisor.py        runtime worker lifecycle and recovery
  runtime_worker.py            JSONL adapter around the Operator
  webview_app.py               WebView companion and native-avatar handoff
  chat_store.py                persistent visible chat and events
  foreground.py                active foreground-turn publication
  systemsense.py               telemetry storage, baselines, and queries
  systemsense_collectors.py    Windows, psutil, inventory, and sensor collectors
  learning.py                  lessons, facts, typed learning, and retrieval
  study.py                     staged curriculum and held-out evaluation
  background_reading.py        progressive idle library education
  research.py                  bounded research control
  selfdev.py                   candidate research, editing, repair, and delivery
  evolution.py                 proposals, experiments, and capability frontier
  evolution_orchestrator.py    opportunity queue and whole-cycle budgets
  candidate_resources.py       bounded inert resource storage
  checkpoint.py                resumable self-development state
  github_integration.py        trusted-main and candidate GitHub lifecycle
  resource.py                  idle, CPU, memory, model, and priority gates
  mission.py                   stable mission, priorities, and non-goals
  tools/                       bounded Operator observation and action surfaces
  cli.py                       command-line interface
tests/                         executable contracts and regressions
scripts/                       bootstrap, evaluation, GitHub, and worker helpers
docs/                          SystemSense, library, and behavior evidence
.github/workflows/tests.yml     Windows GitHub Actions test boundary
config.example.toml            documented configuration defaults
selfdev-backlog.json           bootstrap tasks, not the live capability frontier
ARCHITECTURE.md                detailed system design and lifecycle
SECURITY.md                    enforced authority and safety boundaries
ROADMAP.md                     staged direction; not a statement of current capability
```

## Contributing and external review

Before proposing a change:

1. Read `ARCHITECTURE.md`, `SECURITY.md`, `config.example.toml`, `localpilot/mission.py`, the relevant implementation, and its tests.
2. Establish current behavior and a concrete limitation from repository evidence. Do not treat roadmap text as implemented functionality.
3. Explain how the proposal supports the mission and what transferable capability it should unlock.
4. State a falsifiable hypothesis, metric, baseline, success criterion, and reproducible measurement method.
5. Preserve candidate confinement, reviewer-test protection, argument-based subprocesses, trusted-`main` synchronization, the one-candidate gate, the ban on local autonomous candidate execution, and human-only promotion.
6. Add or adjust focused tests without weakening existing contracts.
7. Run the focused tests, the full suite where practical, and `git diff --check`.
8. Open a focused branch and pull request containing the evidence, safety impact, remaining uncertainty, and rollback story.

A useful proposal answers: **What is limiting LocalPilot now, what evidence demonstrates that limitation, what reusable capability would the change create, and what result would falsify the claim?**

## Known limitations

- LocalPilot is Windows-first. Scheduling, foreground detection, process priority, several observation tools, and CI contracts are Windows-specific.
- Package metadata reports version `0.2.0`, while the bootstrap and CLI startup banners still identify the build as `0.1`.
- Stable PC mutation is intentionally narrow: four allow-listed app launches, five Settings destinations, and three installed built-in power-plan targets.
- The desktop has no screenshot vision, arbitrary pointer or keyboard control, or general application automation.
- Autonomous candidates are not locally sandboxed strongly enough to execute safely. GitHub Actions therefore adds latency and an external dependency.
- The resource governor models idle state, system CPU, memory, model size, and process priority, but not the full GPU, thermal, power, or foreground-application state.
- The owner-managed library is disabled by default and supports bounded PDF and UTF-8 text ingestion rather than arbitrary media.
- Durable learning is intentionally compact and typed. It is not a transcript store, unlimited long-term memory, or model training.
- One outstanding candidate at a time improves safety and causal attribution but limits parallel exploration.
- Held-out study and candidate benchmarks can still be gamed or overfit. Strong claims require reproducible evidence and human review.
- The project has built the machinery for autonomous capability experiments; it has not demonstrated sustained recursive self-improvement.

## Project status and scope

LocalPilot is experimental software under active development. The immediate engineering question is whether its existing persistence, evidence, learning, and candidate-development systems can repeatedly produce useful, measurable improvements that transfer to later work.

Claims should be evaluated against the current code, tests, recorded experiment evidence, and human-reviewed outcomes—not the mission statement or [ROADMAP.md](ROADMAP.md) alone.

## License

LocalPilot is distributed under the [LocalPilot Source-Visible License 1.0](LICENSE.md). The source is visible for transparency, study, discussion, and private non-commercial evaluation, but this is not an open-source license.

Redistribution, derivative works, commercial or hosted use, and use of LocalPilot source or project-specific artifacts to train another AI system require prior written permission from the copyright holder. Third-party components remain subject to their own licenses. Read [LICENSE.md](LICENSE.md) for the complete terms.
