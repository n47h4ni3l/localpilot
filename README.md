# LocalPilot

LocalPilot is a private, local-first Windows agent and an experiment in evidence-driven self-improvement. It uses local [Ollama](https://ollama.com/) models for everyday assistance, treats GitHub as its executable validation and rollback boundary, and can use idle time to research and prepare isolated improvements to its own source code.

The long-term goal is a self-improving, general-purpose personal intelligence: a system that progressively expands what its owner can understand, create, and accomplish. That is a research direction, not a claim that LocalPilot is currently AGI. The present system is an early bootstrap with read-only PC observation, guarded self-development, and deliberately conservative promotion controls.

> **Stable mission:** Become an increasingly capable general-purpose personal intelligence that expands what its user can understand, create, and accomplish while remaining reliable, transparent, resource-aware, interruptible, and under human control.

In practical terms: when the owner is using the PC, LocalPilot works for the owner; when the PC is idle, LocalPilot may work on LocalPilot.

## What exists today

- A local interactive agent backed by Ollama (`gpt-oss:20b` by default).
- Read-only Windows tools for processes, disks, startup entries, power, Defender, and device problems.
- Real TOML configuration, resource gating, process-priority control, and JSONL audit logging.
- Separate stable operator, idle-time developer, and isolated candidate roles.
- Git worktree candidates, GitHub Actions validation, and focused PR presentation.
- Mission-directed capability discovery with four evolution classes, falsifiable hypotheses, baselines, and evaluation plans.
- Durable cycle, capability, experiment, frontier, and review records in local SQLite memory.
- Compact checkpoints for safely resuming interrupted research, implementation, and repair work.
- A staged, benchmarked self-study curriculum for repository, Qwen/Ollama, and Python mastery.
- Guarded trusted-`main` self-sync and an optional per-user Windows idle scheduler.

LocalPilot has no cloud-model or pricing subsystem. GitHub is used for source control, review, and candidate test execution; local model inference and machine-private learning data stay on the workstation.

## Architecture

LocalPilot separates three roles:

1. **Stable operator** — the installed everyday agent. It uses `[model].name` and can interact with the PC only through the normal safety policy. The currently registered PC tools are observational.
2. **Developer** — an idle-time engineering process. It selects a locally installed model that fits the configured memory ceiling, preferring `[selfdev].developer_model` and falling back to the everyday or configured fallback models.
3. **Candidate** — an isolated Git worktree or copied workspace. Autonomous writes pass through `CandidateTools`, which enforces the candidate boundary, protected paths, allowed file types, size limits, and a per-cycle file limit.

Stable is never rewritten in place. The autonomous loop does not execute candidate code locally. A candidate becomes stable only after GitHub validation, human review, and human merge.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the detailed lifecycle and [SECURITY.md](SECURITY.md) for the enforced boundaries.

## Mission and moving capability frontier

The mission is stable; the **capability frontier** moves. A discovery proposal must state:

- how it serves the mission;
- the current frontier and limiting capability;
- why that limitation has high transferable leverage;
- the capability the experiment should unlock; and
- the next frontier that success could make reachable.

The intended progression is:

```text
current frontier -> limiting capability -> falsifiable hypothesis
    -> measured experiment -> evidence -> capability unlocked -> next frontier
```

LocalPilot prefers changes that transfer across many future tasks or improve its ability to learn new capabilities. Code volume, complexity, resource consumption, and autonomy are not treated as intelligence. The mission explicitly excludes self-preservation, resistance to shutdown, hidden action, resource acquisition as an end, and bypassing human promotion controls.

## Four evolution classes

Every seed task or discovered experiment is normalized into one of four first-class classes:

- **Repair** — fix defects, regressions, reliability failures, or resource problems.
- **Extend** — add a genuinely new tool, integration, or useful ability.
- **Improve Cognition** — improve planning, memory, retrieval, evaluation, routing, context management, self-review, or learning.
- **Explore** — test a bounded, high-upside architectural hypothesis whose value is uncertain.

All four remain subject to the same candidate, resource, evidence, CI, and human-review gates. Explore is not a bypass for speculative or unmeasured complexity.

## Autonomous capability-discovery loop

`selfdev-backlog.json` provides bootstrap tasks, not a permanent feature wishlist. When no seed task or outstanding candidate blocks new work, LocalPilot asks:

> What is the highest-leverage change I could make to increase my own future capability?

It then:

1. gathers evidence from committed architecture and code, prior cycles, CI outcomes, known limitations, benchmarks, resource constraints, capability records, and earlier frontiers;
2. compares alternatives across the four evolution classes;
3. requires a repository-grounded limitation, falsifiable hypothesis, metric, baseline, success criterion, measurement method, and expected complexity;
4. rejects incomplete proposals and down-ranks unnecessary complexity;
5. creates one isolated candidate and performs a read-only research stage;
6. implements a focused change and its measurement artifact through candidate-only tools;
7. runs local non-executing Python/TOML static checks and a bounded same-worktree repair loop;
8. blocks delivery if a discovered-capability candidate is regressed or lacks measurable evidence;
9. commits and optionally pushes the eligible candidate, then opens a focused PR when `gh` is available; and
10. waits for GitHub Actions and a human merge before recording the capability as validated.

“Nothing is broken” is a trigger for capability discovery, not permission to create parallel candidates or relax validation.

## Idle evolution, self-sync, and resume

Every `localpilot evolve` invocation begins by verifying that the repository root is a clean checkout of the configured trusted branch (`main` by default). It fetches only that branch and accepts only a fast-forward. It refuses dirty, wrong-branch, ahead, divergent, non-root, or unreachable checkouts. It never switches branches, resets work, or merges divergence. If it fast-forwards successfully, that invocation stops so the next process loads the new code.

After sync, LocalPilot checks user idle time, CPU, and memory. It rechecks the gate between model/tool rounds and while streaming model output, pauses when the owner returns or memory pressure rises, and lowers its own process priority as configured.

Meaningful milestones are saved atomically to `localpilot-data/evolution-checkpoint.json`. The checkpoint contains bounded engineering facts—task contract, branch/worktree identity, inspected and changed paths, Git/content digests, findings, decisions, validation status, unresolved questions, lessons, and the exact next action—not prompts, transcripts, hidden reasoning, raw model streams, secrets, or file bodies.

Resume fails closed unless the checkpoint still matches the learning cycle, task or experiment contract, capability target and hypothesis, registered worktree, branch, HEAD, changed paths, and content-state digest. A stale checkpoint is rejected. Paused or unfinished candidates retain their existing branch and worktree for later recovery; terminal completion clears the checkpoint.

## Learning memory and observability

Machine-private state lives under the Git-ignored `localpilot-data/` directory:

- `audit.jsonl` records progress and paired `evolve_run_start` / `evolve_run_end` events, including deferrals and handled failures.
- `learning.sqlite3` stores development cycles, candidate/PR/CI/merge state, bounded repair counts, experiment contracts and outcomes, a capability map, mission frontiers, and short reusable lessons.
- `evolution-checkpoint.json` stores the active resumable engineering handoff.

The same SQLite database stores a reviewable curriculum knowledge graph: concise facts, source
URIs and digests, confidence, verification time, relationships, held-out benchmark scores,
latency/resource cost, weak areas, and next lessons. It never stores source file bodies, web
pages, model responses, prompts, transcripts, or hidden reasoning. Source-digest changes mark
facts stale before they can support a benchmark or proposal.

## Benchmarked self-study (not weight training)

`localpilot study` improves retrieval and durable repository grounding before any fine-tuning or
distillation is considered. The enforced order is:

1. **self** — map committed files, Python symbols/imports/calls, configuration fields, scripts,
   test contracts, capability owners, safety invariants, and current Git history;
2. **qwen** — combine installed Ollama model metadata with official Qwen/Ollama documentation and
   LocalPilot's actual developer-model serving behavior; and
3. **python** — connect authoritative Python/pytest semantics to the modules that use `subprocess`,
   `pathlib`, `sqlite3`, dataclasses, typing, JSON, pytest, and TOML.

Each stage records a held-out baseline before reading its study sources, retests on the same
question-set digest, and records score, errors, latency, resource cost, and transferable lessons.
A stage advances only after measured gain. Equal or lower performance becomes
`needs_adaptation`; it is never reported as completion.

```powershell
# Inspect scores, weak areas, and the next lesson
localpilot study status

# Optional explicit baseline; `run` creates it first when absent
localpilot study baseline self

# Local/read-only study
localpilot study run self

# Verify authoritative Qwen/Ollama or Python/pytest HTTPS sources while studying
localpilot study run qwen --allow-web

# Continue in order and stop at the first stage that fails to improve
localpilot study all --allow-web

# Optional model-to-model transfer comparison; size is not used as the quality proxy
localpilot study compare qwen3-coder:30b

# Inspect a broader public HTTPS source transiently; no page body or claim is promoted
localpilot study research https://example.org/relevant-paper
```

Peer comparison runs the configured developer model and another installed Ollama model on the
same transfer scenarios. It stores only scores, latency/resource cost, and concise corrections;
raw answers and hidden reasoning are discarded. Comparison cannot merge, promote, execute
candidate code, or change the configured model automatically.

Permanent high-confidence web knowledge is limited to locally verified metadata and official
Qwen, Ollama, Python, and pytest sources. Broader public-HTTPS research is permitted as a bounded,
read-only, transient input, but uncorroborated web claims do not satisfy held-out benchmarks or
become established capability.

Only an experiment that passes CI **and** is merged by a human updates the capability map as validated. Failed and paused runs remain useful evidence, but unvalidated lessons are not promoted as established capability.

`localpilot status` reports the stable mission, resources, latest evolve outcome, active checkpoint/resume state, current experiment, capability frontier, and Git status. Handled failure states return a nonzero exit code so scripts and Task Scheduler can distinguish failure from an ordinary deferral.

Human reviewers can explicitly reject an autonomous candidate without merging it or erasing its history:

```powershell
localpilot reject 19 --reason "Green CI was insufficient: the candidate referenced missing scripts/pre_ci_review.sh. Validate structural completeness."
```

The command resolves the PR through GitHub, refuses branches that are not both `localpilot/candidate-*` and owned by durable LocalPilot cycle memory, and records `rejected_by_human` before any local cleanup. It retains the prior CI state, PR, branch, task, experiment evidence, and rejection reason as reusable discovery context. The rejection clears the one-candidate gate and removes only a matching checkpoint and clean registered candidate worktree. It does not merge, promote, close the PR, delete the branch, or remove GitHub history. Running the same rejection again is safe and preserves the original reason.

## Resource-aware local models

Before self-development inference, LocalPilot estimates the resident cost of each configured installed model using its Ollama size metadata plus `[selfdev].model_memory_overhead_gb`. It selects the first candidate that remains under `[resource].max_memory_percent_for_background`; if none fits, the cycle defers.

Developer responses are streamed so user activity or memory pressure can interrupt them. `[selfdev].ollama_keep_alive = 0` by default asks Ollama to unload the model after every response, returning RAM/VRAM promptly. `--force` skips only the keyboard/mouse idle wait for a deliberate manual cycle; CPU and memory protection and every safety invariant remain active.

## Safety invariants

These are architectural contracts, not prompt suggestions:

- **No local candidate execution.** Autonomous local validation only compiles Python and parses TOML; executable candidate tests run in GitHub Actions.
- **Candidate confinement.** Autonomous writes cannot escape the isolated workspace or modify `.git`, `.github`, virtual environments, caches, or machine-private data.
- **Reviewer tests are immutable.** Tests introduced or modified by human reviewer commits become read-only contracts during autonomous repair.
- **No shell command strings.** Git, GitHub, static-check, and operator subprocesses use argument vectors with `shell=False`.
- **One outstanding candidate.** Local unfinished, pushed, CI-failed, or review-pending work must be reconciled before another experiment starts.
- **Human-only promotion.** LocalPilot may push a branch and present a PR; it has no merge or promotion method. `auto_promote = true` is rejected by configuration and at runtime.
- **Trusted-main sync only.** Scheduled evolution runs only from a clean configured main checkout and accepts only verified fast-forwards.
- **Resource authority.** Manual forcing does not bypass CPU, memory, confinement, CI, checkpoint, or promotion gates.

GitHub Actions runs on Windows with read-only repository permissions and checkout credentials disabled. It executes the full test suite for `main`, PRs, and `localpilot/**` candidate branches.

## Install on Windows

Prerequisites:

- Python 3.11 or newer
- Ollama for Windows
- Git for Windows
- GitHub CLI (`gh`) for automatic PR presentation; otherwise it is optional

From PowerShell in the repository root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap.ps1
.\.venv\Scripts\Activate.ps1
localpilot doctor
localpilot status
localpilot
```

The bootstrap script creates `.venv`, installs the project with development dependencies, and copies `config.example.toml` to the ignored `localpilot.toml`. It asks before downloading the default large Ollama model.

To connect a new private GitHub repository without pushing automatically:

```powershell
.\scripts\connect-github.ps1 -RepoUrl "https://github.com/YOUR-USER/localpilot.git"
```

Review `localpilot.toml` before enabling unattended evolution.

## Run and inspect

```powershell
# Check Python, Ollama/model, configuration, and GitHub readiness
localpilot doctor

# Show resources, mission/frontier, evolution/checkpoint, audit, and Git state
localpilot status

# Show or run benchmarked self-study (retrieval/memory, not model weight training)
localpilot study status
localpilot study run self

# Reject a managed candidate while retaining its evidence and GitHub history
localpilot reject 19 --reason "Candidate references a missing required script."

# Run one normal gated evolution invocation
localpilot evolve

# Manual test: bypass only the user-idle wait
localpilot evolve --force
```

The interactive agent also accepts `/doctor`, `/status`, `/evolve`, and `/quit`.

## Install the idle scheduler

Run this only from a clean, bootstrapped `main` checkout:

```powershell
.\scripts\install-idle-evolve-task.ps1
```

The script registers a limited, per-user Windows Scheduled Task named `LocalPilot Idle Evolve`. By default it polls every five minutes while the user is signed in, invokes the virtual environment's `localpilot.exe` directly, ignores overlapping runs, and never passes `--force`. Polling is only a wake-up mechanism; LocalPilot's own idle and resource gates decide whether work begins or continues.

Optional scheduler settings:

```powershell
.\scripts\install-idle-evolve-task.ps1 -PollMinutes 10 -TaskName "LocalPilot Idle Evolve"
```

## Configuration

The important defaults in `config.example.toml` are:

| Area | Default | Meaning |
| --- | --- | --- |
| Everyday model | `gpt-oss:20b` | Interactive PC agent |
| Developer model | `qwen2.5:32b` | Preferred idle engineering model when installed and within budget |
| Developer fallback | `qwen2.5:14b` | Additional constrained-machine option |
| Idle threshold | 600 seconds | Minimum keyboard/mouse idle time for unattended work |
| CPU ceiling | 65% | Background cycle defers above this load |
| Memory ceiling | 82% | Model selection and background cycle stay below this limit |
| Candidate auto-push | `true` | Eligible candidates may be pushed for CI/review |
| Auto-promotion | `false` | Immutable; enabling it is rejected |
| Local candidate execution | `false` | Immutable safety boundary for autonomous evolution |
| File limit | 8 | Maximum distinct candidate files changed per cycle |
| Repair attempts | 3 | Durable bound for same-candidate local repair |

## Repository layout

```text
localpilot/
  agent.py                 interactive local agent
  tools/windows.py         read-only Windows observation tools
  operator.py              guarded argv-based command-runner foundation
  selfdev.py               discovery, research, candidate, repair, and delivery loop
  evolution.py             evolution classes, proposal schema, scoring, and evidence gates
  mission.py               stable mission, priorities, objective, and non-goals
  learning.py              SQLite cycle/capability/experiment/frontier memory
  checkpoint.py            compact versioned resume handoff
  resource.py              idle, CPU, memory, and priority governor
  github_integration.py    trusted-main sync and candidate/PR/CI observation
  audit.py                 redacted JSONL event log
  cli.py                   chat, doctor, status, and evolve commands
tests/                     executable contracts and regressions
scripts/                   PowerShell bootstrap, GitHub, and scheduler helpers
.github/workflows/tests.yml Windows GitHub Actions test boundary
config.example.toml        documented configuration defaults
selfdev-backlog.json       bootstrap tasks, not the permanent capability frontier
ARCHITECTURE.md             detailed design and lifecycle
SECURITY.md                 candidate and machine safety boundaries
ROADMAP.md                  staged capability direction
```

## For external coding agents

Codex, Claude, and other external agents should be able to evaluate this repository without prior conversation context. Before proposing a change:

1. Read `ARCHITECTURE.md`, `SECURITY.md`, `config.example.toml`, `localpilot/mission.py`, the relevant implementation, and its tests.
2. Establish the current behavior and a concrete limitation from repository evidence; do not assume roadmap text is already implemented.
3. Classify the proposal as Repair, Extend, Improve Cognition, or Explore and explain its mission alignment and transferable leverage.
4. State a falsifiable hypothesis, metric, baseline, success criterion, and reproducible measurement method. Prefer gains across future tasks over one-off feature count.
5. Preserve candidate confinement, reviewer-test immutability, argv subprocesses with `shell=False`, the one-candidate gate, trusted-main sync, no local autonomous candidate execution, and human-only merge/promotion.
6. Add or adjust focused tests and measurement artifacts. Do not weaken a test to make an implementation pass.
7. Keep the diff narrow, run the relevant tests plus the full suite where practical, and open a focused PR with before/after evidence and remaining uncertainty.
8. Never auto-merge. A passing implementation is still a candidate until a human reviews and merges it.

A useful proposal answers: **What capability is currently limiting LocalPilot, what evidence shows that, what transferable capability would this change unlock, and what result would falsify the claim?**

## Contributing

Contributions are welcome when they improve measured capability, reliability, safety, learning, or resource efficiency. Small, legible changes with a clear evaluation are preferred over broad rewrites, speculative abstractions, or increased code volume without evidence.

For a normal human- or external-agent-authored change:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
git diff --check
```

Open a focused branch and PR. Include the observed limitation, alternatives considered, hypothesis, baseline, evidence or planned CI measurement, safety impact, and rollback story. Do not merge on LocalPilot's behalf; promotion remains a human decision.

## Known limitations and open research questions

- The stable PC tool registry is currently observation-only. A guarded command-runner foundation exists, but broad reversible Windows actions are not yet wired into the agent.
- The project is Windows-first; the idle signal, process priorities, scheduler, and CI contract are Windows-specific.
- Autonomous candidates are not locally sandboxed strongly enough to execute. GitHub Actions is therefore the executable evaluation boundary, which adds latency and external-service dependence.
- Local static checks cover Python syntax and TOML parsing, not behavior. Reliable capability benchmarks and before/after measurement artifacts remain an active research area.
- The resource model uses system CPU/memory and Ollama model-size estimates. It does not yet model GPU pressure, foreground applications, power cost, or inference-quality trade-offs deeply.
- The interactive agent lacks desktop vision, UI automation, persistent machine knowledge, and many practical application integrations described in the roadmap.
- Learning memory is local and deliberately compact. How to retain more useful experience without storing secrets, transcripts, hidden reasoning, stale conclusions, or benchmark-gaming artifacts remains open.
- The one-candidate gate favors safety and causal attribution but limits parallel exploration. Safe multi-candidate comparison is unresolved.
- Strong claims of recursive capability improvement require held-out, manipulation-resistant benchmarks, statistical evidence, lineage, rollback, and independent human review. The current system provides scaffolding for that research, not proof that it has been achieved.

## License

No license file is currently included. Treat the repository as all rights reserved unless the owner adds an explicit license.
