# LocalPilot v0.2

LocalPilot is a private, local-first Windows agent intended to grow into a capable computer operator while using GitHub as its engineering and rollback layer.

The end goal is simple:

> When the owner is using the PC, LocalPilot works for the owner. When the PC is idle, LocalPilot can work on LocalPilot.

v0.1 is the bootstrap stage. It already contains the architecture needed to begin that process, but it deliberately does **not** give the first untested build unrestricted system-write access.

## What v0.1 does

- Runs a local Ollama model (`gpt-oss:20b` by default).
- Lets the agent inspect Windows, processes, disks, startup entries, power state, Defender and device problems.
- Loads real settings from `localpilot.toml`.
- Writes an audit log to `localpilot-data/audit.jsonl`.
- Detects whether the PC is active or idle and keeps background development gated by CPU/memory/idle thresholds.
- Integrates with local Git and a GitHub remote.
- Maintains **stable / developer / candidate** separation.
- Treats Repair, Extend, Improve Cognition, and Explore as first-class evolution classes.
- Can select seed work from `selfdev-backlog.json` or discover its own measured capability-growth question when no candidate blocks it.
- Confines autonomous source edits to an isolated candidate workspace.
- Runs non-executing local static checks, then can commit/push a candidate branch for full testing in GitHub Actions.
- Does not automatically promote an experimental candidate over stable.

There is no pricing or credit subsystem and no cloud-model dependency in this starter.

## 1. Extract it

Extract the ZIP somewhere you intend to keep it, for example:

```text
C:\LocalPilot
```

Do not run it directly from inside the ZIP.

## 2. Prerequisites

Install:

- Python 3.11 or newer
- Ollama for Windows
- Git for Windows
- GitHub CLI (`gh`) is optional initially

You do **not** need administrator PowerShell for the v0.1 bootstrap.

## 3. Bootstrap

Open PowerShell in the LocalPilot directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap.ps1
```

The script creates `.venv`, installs LocalPilot and copies `config.example.toml` to `localpilot.toml`.

If the configured Ollama model is missing, the script asks before starting the large model download. It does not silently pull it.

Then:

```powershell
.\.venv\Scripts\Activate.ps1
localpilot doctor
localpilot
```

Useful interactive commands:

```text
/status
/doctor
/evolve
/quit
```

`/evolve` waits for the configured idle/resource conditions. For a deliberate manual test you can run:

```powershell
localpilot evolve --force
```

That bypasses only the keyboard/mouse idle wait for that cycle. CPU and memory ceilings remain authoritative, as do candidate path confinement, stable/candidate separation, and the guarded main-branch sync. LocalPilot does not automatically install its scheduler. An external scheduler may invoke `localpilot evolve`, but the checkout must be the clean configured `main` branch. Each invocation fetches only that branch from the configured remote and permits only a fast-forward. Dirty, wrong-branch, ahead, divergent, or unreachable checkouts stop before candidate work. If a fast-forward occurs, that invocation exits so the next scheduled run starts with the newly loaded code.

To let LocalPilot poll dynamically and begin work only when its own idle/resource gate permits it, register the included per-user Windows task from a clean `main` checkout after bootstrap:

```powershell
.\scripts\install-idle-evolve-task.ps1
```

The task invokes `localpilot evolve` every five minutes while you are signed in, ignores overlapping invocations, and never passes `--force`. Polling is only a wake-up mechanism: returning to the PC or exceeding a resource limit still defers or pauses model/tool work.

Every invocation now writes paired `evolve_run_start` / `evolve_run_end` rows to `localpilot-data/audit.jsonl`, including early deferrals, capability discovery, handled failures, and candidate delivery. `localpilot status` shows the newest terminal outcome plus the current evolution class, capability target, hypothesis, evaluation plan, and latest experiment outcome. Handled failure states return a nonzero process exit so Task Scheduler's `LastTaskResult` no longer reports success for an internally failed cycle.

Meaningful evolution milestones also update `localpilot-data/evolution-checkpoint.json`. This is a compact, versioned engineering handoff containing the task contract, candidate branch/worktree, inspected and changed paths, concise findings and decisions, Git/diff/static-check/test status and failure markers, unresolved questions, lessons and the exact next action. It intentionally contains no prompts, hidden reasoning, raw model output streams, chat transcripts, secrets or copied file contents. `localpilot status` shows whether a checkpoint exists and the latest resume outcome.

Before resume, LocalPilot requires the checkpoint to match the learning cycle, current seed-or-experiment contract, capability target/hypothesis, registered Git worktree, branch, HEAD, changed paths and a content-state digest. Missing or inconsistent checkpoints are rejected and the candidate is reconstructed only from current safety-validated Git/task state. Checkpoints are retained across resource pauses and Ollama unloads, then removed after terminal candidate completion.

## GitHub connection

Create a **private, empty** GitHub repository named `localpilot`, then from the extracted project:

```powershell
.\scripts\connect-github.ps1 -RepoUrl "https://github.com/YOUR-USER/localpilot.git"
```

The script configures the remote but intentionally does not push without you reviewing the initial project. Then:

```powershell
git add .
git commit -m "Initial LocalPilot v0.1"
git push -u origin main
```

After that, candidate self-development cycles can use Git worktrees/branches instead of plain copied workspaces. By default, statically valid candidates are pushed automatically so GitHub Actions can execute the test suite away from your workstation. GitHub Actions runs on `main`, PRs and `localpilot/**` branches.

## Resource behaviour

Defaults in `localpilot.toml`:

- LocalPilot uses below-normal process priority while the PC is active.
- Background self-development requires 10 minutes of keyboard/mouse idle time.
- It defers background work when CPU is above 65% or memory usage is above 82%.
- Before loading Ollama, it estimates each configured developer model's resident footprint and selects the first model that stays under the same memory ceiling. On a constrained machine this can choose the everyday or configured fallback model instead of starting an oversized preferred model.
- Developer responses are streamed so user activity or new hardware pressure can cancel a long inference promptly. Ollama receives `keep_alive = 0` by default for self-development, returning model RAM/VRAM after each response.

These values are deliberately editable. The next development phase should add GPU and foreground-application awareness so gaming, CAD and other demanding work cause an even faster yield.

## Self-development model

`selfdev-backlog.json` contains seed tasks, not a permanent capability wishlist. One `localpilot evolve` cycle:

1. verifies the process is running from the clean configured `main` checkout and fast-forwards it from the configured remote when safe;
2. checks idle/load conditions;
3. resumes or repairs the one outstanding candidate before considering new work;
4. selects an eligible seed task or, when none remains, asks **“What is the highest-leverage change I could make to increase my own future capability?”**;
5. derives its own capability target from committed architecture, prior cycle/CI evidence, known limitations, benchmarks and resource constraints;
6. classifies the work as Repair, Extend, Improve Cognition, or Explore, compares alternatives, states a falsifiable hypothesis, and defines a metric, baseline, success criterion and measurement method;
7. creates an isolated candidate workspace and gives the local model candidate-only file tools;
8. performs read-only research, implements a limited candidate, and adds the measurement artifact needed for comparison;
9. runs non-executing syntax/config checks locally and feeds failures back through the bounded same-worktree repair loop;
10. rejects regressed or unmeasured capability candidates, then commits/pushes an eligible candidate and opens a focused human-review PR when GitHub CLI is available;
11. lets GitHub Actions execute candidate tests and benchmarks away from the workstation; and
12. retains validated capability evidence and lessons only through the existing CI-plus-human-merge lifecycle while leaving stable untouched.

“Nothing is broken” is not a terminal condition. It is the trigger for capability discovery. This does not authorize parallel candidates, local candidate execution, automatic merge, or automatic promotion; the existing resource, checkpoint, confinement, reviewer-test, and one-outstanding-candidate gates still apply.

Automatic candidate promotion is disabled. Candidate code is also not executed locally by the autonomous self-development loop in v0.1; GitHub Actions is the executable test sandbox. That is intentional until a stronger local sandbox/rollback layer exists.

## Why PC control starts read-only

The intended LocalPilot is not a permanently restricted support bot. It is meant to gain broad operating freedom. The first build itself, however, has not yet earned permission to change Windows indiscriminately.

The first backlog item is the guarded Windows operator foundation. Once that layer has tests, rollback hooks and command confinement, we can start adding real reversible PC actions without turning every harmless task into a confirmation-dialog exercise.

See `ARCHITECTURE.md`, `ROADMAP.md` and `SECURITY.md` for the design.


## Learning and evolution architecture

Everyday operation and self-development now use separate model roles:

- `[model].name` remains the everyday PC agent (`gpt-oss:20b` by default).
- `[selfdev].developer_model` prefers the existing local `qwen2.5:32b` for engineering cycles and falls back to the everyday model when it is unavailable.

Evolution is staged as a capability discovery loop: observe current limitations → identify a limiting capability → research alternatives → state a falsifiable hypothesis → define a measurable evaluation and baseline → implement an isolated candidate → compare against the baseline → retain validated lessons → present the result for human review. A read-only research pass produces an evidence brief, then an implementation pass edits only the isolated candidate. If direct tool editing stalls, the developer must return a strict structured change plan; every proposed file is still validated and applied through `CandidateTools.write_project_file`, so it cannot bypass candidate path or file limits.

Research and implementation can span scheduler invocations. The resume input is the validated structured checkpoint, not the previous model conversation. A resumed model receives only the concise engineering facts needed to continue, while current path confinement, reviewer-test immutability, non-executing local checks and resource gates are re-established first.

Cycle outcomes, a lightweight capability map, experiment hypotheses, evidence, baselines, evaluation plans/outcomes and short reusable lessons are stored locally in `localpilot-data/learning.sqlite3`. The schema intentionally excludes prompts, transcripts, model thinking, hidden reasoning, raw token streams, secrets and duplicated file contents. Failed and paused cycles are useful evidence too; only CI-and-human-merge validated experiment outcomes update the capability map. Unpushed Git candidates also retain their branch, worktree and durable repair-attempt count, so a later `evolve` invocation resumes the existing candidate instead of silently starting a replacement. The attempt limit prevents an endless repair loop; an exhausted candidate remains held for human review or a manual correction.

A pushed branch does not complete a backlog task. LocalPilot holds the current task while its candidate awaits a PR, records GitHub Actions status, and advances only after checks pass **and** the PR is merged. There is no automatic promotion path, even if a configuration file attempts to enable one.
