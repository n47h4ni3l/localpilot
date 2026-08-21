# LocalPilot architecture

LocalPilot has three isolated roles and two model responsibilities:

1. **Stable operator** — the installed everyday agent. It uses `[model].name` and may interact with the PC only through the normal safety policy.
2. **Developer** — an idle-time engineering process. It prefers `[selfdev].developer_model` (`qwen2.5:32b` by default) when that model is installed, otherwise it explicitly falls back to the everyday model.
3. **Candidate** — an isolated Git worktree or copied workspace. Autonomous source writes are possible only through `CandidateTools.write_project_file` inside this boundary.

Stable is never rewritten in place, candidate code is not executed locally by the autonomous loop, and there is no automatic promotion path.

## Self-development cycle

```text
resource gate
    -> reconcile earlier candidate PR/check state
    -> select first eligible backlog task
    -> create isolated candidate
    -> research stage (list/read tools only)
    -> implementation stage (candidate tools)
    -> structured JSON change-plan fallback, if direct editing stalls
    -> non-executing local static checks
    -> candidate commit/push
    -> GitHub Actions
    -> human-reviewed PR merge
    -> task becomes complete in local learning state
```

The fallback is data, not a privileged editing channel. Its paths and complete file contents are parsed into a `ChangePlan`, and every item is applied by calling `CandidateTools.write_project_file`. It therefore keeps the same path confinement, protected-directory rules, file limits, type allow-list, and size limit as ordinary candidate tool calls.

GitHub validation and merge are separate facts. A passing local static check or a pushed branch does not complete a task. The next backlog item is eligible only after the current candidate's GitHub checks pass and its PR is merged. LocalPilot can observe that state; it has no method that merges or promotes.

## Local learning memory

`localpilot-data/learning.sqlite3` stores durable, machine-local development facts:

- task, candidate branch, everyday/developer model names;
- cycle outcome, static-check result, push/PR/merge state;
- concise implementation summary; and
- a short reusable lesson for later cycles.

The schema deliberately has no prompts, transcripts, messages, thinking, or chain-of-thought fields. Research and model scratch work are transient. Machine-private learning data remains under the already ignored `localpilot-data/` directory and is never committed to GitHub.

## Resource and process safety

The existing `ResourceGovernor` remains in charge of self-development eligibility. LocalPilot checks user idle time, CPU, and memory before the cycle and between every model/tool round, lowering its own Windows priority and pausing promptly when the owner returns.

All Git/GitHub/static-check process calls use argument arrays with `shell=False`. Full executable tests remain in GitHub Actions. Stable, developer, and candidate boundaries do not depend on model cooperation: the tool surface enforces them.

