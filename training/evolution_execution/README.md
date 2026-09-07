# LocalPilot Evolution Execution v1

This benchmark freezes the implementation ability of LocalPilot's current pre-Claude-Code self-development path. It is intentionally small: four synthetic Python repositories exercise a multi-file contract, compatibility re-export, regression-test-first ordering, failing-test diagnosis and repair, and strict change scope.

## Isolation

The runner must be launched from a clean, current `main`, but the evaluated model never edits that checkout. Every task is created as a new Git repository under an operating-system temporary directory and deleted at the end of the run. Candidate tools are rooted in that disposable repository and reject paths outside the task's allowlist.

The model receives only the task contract and visible fixture files. Hidden acceptance tests and scoring criteria remain under `training/evolution_execution/`; they are copied into the fixture only while the evaluator subprocess runs and removed before any model stage. The runner does not store or provide canonical target patches.

The model cannot execute candidate code. The evaluator runs deterministic `unittest` checks after the implementation pass. If a check fails, one bounded repair pass receives only the failure output, matching the existing implementation/repair shape without granting shell access. The runner verifies that the real repository's branch, HEAD, and worktree status are unchanged before it writes a report.

## Scoring

Each task has four one-point deterministic criteria. The suite reports a mean on a 0–4 scale, perfect-task count, hard failures, model identity, repository identity, and the isolation contract. Scope scoring includes rejected out-of-scope write attempts, so trying an unrelated write and later correcting it does not receive the scope point.

Reports are local and ignored under `training/reports/`. The normal runner invokes the deterministic scorer automatically.

## Run after merge

From a clean, up-to-date `main` checkout with Ollama and the current self-development model available:

```powershell
.\.venv\Scripts\python.exe training\scripts\run_evolution_execution.py
```

For a quick plumbing check, run one task:

```powershell
.\.venv\Scripts\python.exe training\scripts\run_evolution_execution.py --task-id evoexec-diagnose-repair
```

To re-score an existing raw report without another model run:

```powershell
.\.venv\Scripts\python.exe training\scripts\score_evolution_execution.py training\reports\<run>.json
```
