# LocalPilot v0.1

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
- Can run one autonomous self-development cycle against `selfdev-backlog.json`.
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

That bypasses only the idle gate for that cycle; candidate path confinement and stable/candidate separation still apply.

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

These values are deliberately editable. The next development phase should add GPU and foreground-application awareness so gaming, CAD and other demanding work cause an even faster yield.

## Self-development model

`selfdev-backlog.json` contains seed tasks for LocalPilot to work toward. One `localpilot evolve` cycle:

1. checks idle/load conditions;
2. selects the next todo task;
3. creates an isolated candidate workspace;
4. gives the local model candidate-only file tools;
5. lets it inspect and modify a limited number of source files;
6. runs non-executing syntax/config checks locally;
7. if Git is connected, commits and pushes a passing candidate branch;
8. GitHub Actions runs the executable test suite away from the workstation;
9. leaves stable untouched.

Automatic candidate promotion is disabled. Candidate code is also not executed locally by the autonomous self-development loop in v0.1; GitHub Actions is the executable test sandbox. That is intentional until a stronger local sandbox/rollback layer exists.

## Why PC control starts read-only

The intended LocalPilot is not a permanently restricted support bot. It is meant to gain broad operating freedom. The first build itself, however, has not yet earned permission to change Windows indiscriminately.

The first backlog item is the guarded Windows operator foundation. Once that layer has tests, rollback hooks and command confinement, we can start adding real reversible PC actions without turning every harmless task into a confirmation-dialog exercise.

See `ARCHITECTURE.md`, `ROADMAP.md` and `SECURITY.md` for the design.
