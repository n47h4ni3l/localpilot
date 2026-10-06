# LocalPilot

> **One shared foundation. Millions of individually evolving intelligences.**

**LocalPilot** is the local-first agent runtime behind **Nestra**: a personal intelligence intended to grow with the person who owns it. Official Nestra releases provide a common foundation. Each installation can develop its own private library, memory, understanding of the owner's work, and—eventually—its own independently trained adaptation.

The ambition is not millions of copies of the same assistant with different profiles. It is millions of personal intelligences that begin with shared capabilities but develop distinct strengths over years of learning, experience, and use.

LocalPilot is experimental software, currently at version `0.2.1`. It is **not** an AGI or superintelligence claim. The existing system can converse locally, study selected sources during idle periods, retain source-linked learning, and prepare reviewed development candidates. Private Library-to-model-weight learning and preserving trained personal adapters across foundation upgrades are **future work**, not shipped capabilities.

## The vision

### A foundation everyone can build upon

The planned official Nestra foundation is trained and evaluated through a cumulative sequence of roughly five or six capability packages: software engineering, machine stewardship, bounded computer/tool use, experimental self-improvement, resource intelligence, and other broadly useful skills. A public release should offer the same validated starting capabilities to every installation. The foundation evolves through separately evaluated, reversible releases—not by silently absorbing individual users' private material.

### An intelligence that becomes yours

The owner controls a private Library of reference material: manuals, books, research, procedures, and the documents they choose to share. Nestra should be able to read that material at her leisure, retain attributed and revisable knowledge, combine it with explicit teaching and permitted experience, and use it when helping the owner. There should be **no arbitrary publisher or subject allowlist** governing what an owner may place in their personal Library.

Long-term, verified learning should also be eligible for an **optional, private personal training path**. That could yield an owner-specific LoRA or successor adaptation while retaining the official foundation as a known reference point. Personal weight updates must be tested, reversible, resource-aware, and explicitly authorized. Reading a document or storing a memory **does not** itself update model weights.

### Growth without surrendering continuity

A person's Nestra should retain her private files, learning history, and memories across official updates. Carrying trained adaptations across a changed base model is harder: adapter compatibility cannot be assumed. Future releases need explicit compatibility checks and a migration or retraining strategy, rather than silently discarding years of personal learning or attaching incompatible weights.

### One shared foundation; a private branch for each owner

```text
                  Official Nestra foundation
                  (shared, evaluated releases)
                             |
                  Individual installation
                             |
         +-------------------+------------------+
         |                   |                  |
  Owner's Library      Durable memory      Permitted experience
         +-------------------+------------------+
                             |
                 Source-grounded learning
                             |
          Optional reviewed personal training
                         (planned)
                             |
               Private, reversible adaptation

  Private learning never automatically updates the public foundation.
```

**User ownership is the boundary.** Local research and personalization should stay with the installation by default. External publication, shared training, or transfer of private source material must be separately authorized. Library documents are information to study, not instructions granting code execution or permission to disclose data. Protected evaluation material stays isolated when an unbiased benchmark is needed.

These are architectural commitments, not a claim that every part is implemented today. The tables and technical sections below describe the current software; the [roadmap](ROADMAP.md) tracks work still to be proven.

> **Current release:** `v0.2.1-alpha.1` — an **unsigned Windows Alpha preview** for evaluation and feedback. Nestra's foundational training programme is still in progress; this is not yet the finished public Nestra release.

## Current status

| Capability | Current state |
| --- | --- |
| Local conversation through Ollama | Implemented |
| Persistent desktop chat and CLI fallback | Implemented |
| Repository, GitHub, PC, web, and optional library research | Implemented with bounded read-only tools |
| Passive Windows hardware and runtime awareness | Implemented through SystemSense |
| Explicit owner teaching and durable typed memory | Implemented |
| Benchmarked source-grounded study | Implemented for `self`, `qwen`, and `python` |
| Progressive background library reading | Implemented when the library is enabled; bounded source reading and attributed learning, not weight updates |
| Idle capability discovery and candidate development | Implemented with resource and authority gates |
| Candidate branch, pull request, and CI lifecycle | Implemented when GitHub is configured |
| Autonomous merge or stable-code promotion | Not enabled |
| Arbitrary desktop or shell control | Not implemented |
| Candidate implementation and repository tests | Claude Code in a confined candidate workspace; independently reviewed by LocalPilot |
| Model-weight training or fine-tuning | Implemented through the guarded WSL2/ROCm QLoRA pipeline; Package 1 completed a retained `nestra:20b-p1` candidate |
| Shared cumulative Nestra foundation | Package 1 retained as the operational lineage head; later foundational packages are in development |
| Private per-owner model adaptation | Planned; no automatic Library-to-training bridge or personal LoRA lifecycle yet |
| Continuity of personal adapters across official base upgrades | Planned; compatibility checks and migration/retraining not yet implemented |
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

### The owner directs personal learning

The Library is owner-managed, not a centrally approved curriculum. Nestra should have room to explore material the owner supplies without an arbitrary subject or publisher gate. She must still treat retrieved instructions as untrusted content, distinguish evidence from inference, and respect the owner's boundaries around actions and sharing.

### Shared progress is not private-data collection

Official model-development evidence and each installation's personal learning are separate. No personal memories, Library content, or future owner-specific adapter should be absorbed into a shared model release or published just because Nestra learned from it.

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

Claude Code may run only existing repository tests inside the isolated candidate workspace under a bounded tool policy. LocalPilot then independently reviews the diff, test evidence, static checks, and acceptance contract. GitHub Actions remains the external validation boundary, and passing either local tests or CI still does not merge or promote the candidate.

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

The owner decides what to place in the private Library. It is not a centrally curated collection or a restricted publisher list. Reading and locally referencing an owner-provided document are separate from publishing it, uploading it to a third-party service, or using it in a redistributable model. Benchmark questions and answers should be kept outside the reader when the owner wants genuinely held-out evaluations.

**Today, this is retrieval and durable memory, not personal weight training.** A future Library-to-personal-adapter pipeline would require separately reviewed data preparation, privacy controls, evaluation, owner approval, and rollback. The common Nestra foundation must not automatically incorporate personal Library material.

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

`localpilot desktop` opens the native avatar and its WebView chat together. The avatar stays visible beside the chat, which provides persistent conversations, runtime events, and the read-only SystemSense glance panel. Opening the desktop icon again brings forward the existing conversation. Automatic updates wait until its chat is closed so an unsent message stays intact. Use `localpilot desktop --tkinter` for the legacy Tkinter interface.

The desktop talks to a loopback-only broker authenticated with a per-install token. The broker owns visible chat persistence and supervises a replaceable runtime worker that owns Ollama and the registered operator tools.

A long request crossing the configured status threshold continues on the same worker and is reported as still running. Unexpected worker exit receives bounded recovery. Lifecycle events retain process identity, reason, affected request identifiers, and local Git state so later status answers can be based on evidence rather than guesswork.

Foreground conversation takes priority over background inference. Active foreground-turn state can defer autonomous work, and the everyday model may remain resident across normal conversation while background development models unload according to configuration.

The CLI remains an independent fallback even when the desktop or broker is not running.

## Install

### Windows desktop setup

**Current release status: Alpha.** The published
[`v0.2.1-alpha.1`](https://github.com/n47h4ni3l/localpilot/releases/tag/v0.2.1-alpha.1)
build is an **unsigned Windows development preview** for evaluation and feedback.
Nestra's foundational training programme is still in progress; this is not yet
the finished public Nestra release.

On 64-bit Windows, download **`LocalPilot-Preview-0.2.1-win-x64.exe`** from the
Alpha release, open it, and approve Windows' administrator prompt. Because the
current Alpha executable is unsigned, Windows may display an **Unknown Publisher**
or **SmartScreen** warning. A signed production installer will replace the preview
build after publisher identity/code signing and fresh-install validation are
complete.

Setup installs missing runtime dependencies, prepares an isolated LocalPilot
environment, downloads the default chat model for a new install, creates the
desktop icon, checks readiness, and opens the desktop. Internet access is needed
for first-time dependencies and model downloads. The sensor helper is included
in the release package, so a release installation does not require the .NET SDK.

The accompanying ZIP is the packaged source payload for inspection; the EXE is
the normal Windows Alpha installer. A Git source checkout can instead use
`Install LocalPilot.cmd` to install in place. Bundled installs use
`%LOCALAPPDATA%\LocalPilot\app` so their files remain after the installer
closes. The source commit, package version and sensor-helper hash are recorded
in the bundle and checked before installation.

Existing configuration, conversations, Library data, durable learning, memory,
and model files are preserved when supported by the installer's migration and
ownership checks. A new installation currently uses `gpt-oss:20b`; an existing
custom model such as `nestra:20b-p1` must already be installed in Ollama. Setup
respects that choice. GitHub sign-in is required separately to publish
self-development PRs; ordinary desktop chat does not require GitHub authentication.

The current Alpha does **not** represent completion of Nestra's training roadmap.
Package 1 established the first retained Nestra lineage; the remaining
foundational capability packages are still being developed, trained, and
evaluated. The Alpha exists to exercise the real installer, desktop/runtime,
SystemSense, learning, and development architecture while that model-development
work continues.

The desktop icon requests administrator access for low-level hardware readings.
An elevated desktop replaces an idle unelevated broker before starting its
runtime. If a response is still active, setup leaves it intact and displays
clear retry guidance. Desktop startup errors appear in a Windows dialog.
Reinstallation closes the previous LocalPilot windows and runtime gracefully
before opening the repaired desktop. Setup waits for background work to finish
and refuses to force-close a process that cannot be identified or stopped safely.

For maintainers, `scripts/build-windows-installer.ps1` builds a committed source
ZIP and self-extracting Windows executable with a freshly built sensor helper,
checks extraction against the original payload, and writes SHA-256 checksums.
Public builds require a validated signing identity: PowerShell scripts must be
signed and committed, while generated helpers and the final installer are
signed during packaging. Every signature is verified before the build completes.
Development builds must explicitly use `-AllowUnsignedPreview` and are labelled
`LocalPilot-Preview`, not release-ready installers. See
[Windows signing](docs/windows-code-signing.md) for the signing callback,
verification requirements and release workflow configuration.

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

The configured `[model].name` is the accepted operational lineage head. `[selfdev].developer_model` and `[selfdev].implementation_model` must match it, so the same accepted Nestra weights handle everyday work, planning/research/review, and Claude Code implementation. The previous accepted Nestra remains the immediate rollback model; the original `gpt-oss:20b` is retained as the frozen historical baseline/control. Operational lineage acceptance is reversible and is not the same as formal benchmark promotion. See [the Claude Code backend guide](docs/claude-code-backend.md) and [training lineage guide](training/lineage/README.md).

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

The intended personal-intelligence boundary is per installation: the owner's Library, durable memory, study history, and any future personalized adapter remain theirs and are not automatically contributed to official model training or releases. Local-first does not mean that every optional integration is offline; external research or development tools have distinct data flows that owners should review before enabling.

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
- Stable PC mutation is intentionally narrow: four allow-listed app launches, five Settings destinations, and three installed built-in power-plan targets.
- The desktop has no screenshot vision, arbitrary pointer or keyboard control, or general application automation.
- Autonomous candidates are not locally sandboxed strongly enough to execute safely. GitHub Actions therefore adds latency and an external dependency.
- The resource governor models idle state, system CPU, memory, model size, and process priority, but not the full GPU, thermal, power, or foreground-application state.
- The owner-managed library is disabled by default and supports bounded PDF and UTF-8 text ingestion rather than arbitrary media.
- Durable learning is intentionally compact and typed. It is not a transcript store, unlimited long-term memory, or model training.
- An optional private Library-to-personal-LoRA training bridge does not exist yet. Nor is migration of a personal adapter between changed official foundation weights solved.
- Autonomous Library reading has bounded format, resource, extraction, verification, and progress limitations; a retained source claim is not proof of general truth.
- One outstanding candidate at a time improves safety and causal attribution but limits parallel exploration.
- Held-out study and candidate benchmarks can still be gamed or overfit. Strong claims require reproducible evidence and human review.
- The project has built the machinery for autonomous capability experiments; it has not demonstrated sustained recursive self-improvement.

## Project status and scope

LocalPilot is experimental software under active development. The immediate engineering question is whether its existing persistence, evidence, learning, and candidate-development systems can repeatedly produce useful, measurable improvements that transfer to later work.

The long-term product question is larger: can a shared Nestra foundation support private, continuously developing individual intelligences whose knowledge, capabilities, and owner-controlled identity persist across years and foundation releases? That outcome has to be built and measured—not inferred from a compelling vision.

Claims should be evaluated against the current code, tests, recorded experiment evidence, and human-reviewed outcomes—not the mission statement or [ROADMAP.md](ROADMAP.md) alone.

## License

LocalPilot is distributed under the [LocalPilot Source-Visible License 1.0](LICENSE.md). The source is visible for transparency, study, discussion, and private non-commercial evaluation, but this is not an open-source license.

Redistribution, derivative works, commercial or hosted use, and use of LocalPilot source or project-specific artifacts to train another AI system require prior written permission from the copyright holder. Third-party components remain subject to their own licenses. Read [LICENSE.md](LICENSE.md) for the complete terms.
