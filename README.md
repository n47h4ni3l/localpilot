# LocalPilot

LocalPilot is a private, local-first Windows agent designed to become more capable without allowing an experimental build to overwrite the stable agent.

When the owner is using the PC, the everyday agent works for the owner. When the machine has spare idle capacity, a separate developer model can research and implement one backlog task in an isolated candidate.

## Model roles

- `[model].name` is the everyday PC agent (`gpt-oss:20b` by default).
- `[selfdev].developer_model` is the engineering model (`qwen2.5:32b` by default).
- The developer preference is used when installed in Ollama; otherwise that cycle falls back to the everyday model.

This separation lets routine work stay responsive while self-development can use the stronger existing local model.

## Safe evolution

One `localpilot evolve` cycle:

1. enforces the existing idle/CPU/memory resource gate;
2. reconciles prior candidate PR and GitHub Actions state;
3. selects the first backlog task that is neither completed nor awaiting validation;
4. creates an isolated Git worktree (or copied candidate before Git is connected);
5. runs a read-only research stage;
6. runs a candidate-only implementation stage;
7. if direct editing stalls, requests a strict structured change plan and applies every item through the same confined `CandidateTools.write_project_file` method;
8. runs non-executing static checks locally;
9. commits and pushes only the candidate files that were written; and
10. waits for GitHub Actions and a human-reviewed PR merge.

A task does not advance merely because files were written, static checks passed, or a branch was pushed. It advances only after GitHub reports both passing validation and a merged PR. Automatic promotion is forbidden even if a local config attempts to enable it.

## Learning memory

Successful, failed, paused, and no-change cycles are recorded in `localpilot-data/learning.sqlite3`. Later cycles receive a small set of concise reusable lessons. The database stores outcomes and reviewable summaries only—not prompts, transcripts, hidden reasoning, or chain-of-thought—and remains excluded from Git.

## Setup

Requirements are Python 3.11+, Ollama for Windows, and Git for Windows. GitHub CLI is needed for automatic PR/check reconciliation.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap.ps1
.\.venv\Scripts\Activate.ps1
localpilot doctor
localpilot
```

Useful commands:

```text
/status
/doctor
/evolve
/quit
```

For a deliberate manual self-development test:

```powershell
localpilot evolve --force
```

`--force` bypasses only the initial idle gate for that manual cycle. Candidate path confinement, no local candidate execution, GitHub validation, manual merge, and stable/candidate isolation remain in force.

See `ARCHITECTURE.md`, `ROADMAP.md`, and `SECURITY.md` for the broader design and safety policy.

