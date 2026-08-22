# LocalPilot architecture

LocalPilot has three isolated roles and two model responsibilities:

1. **Stable operator** — the installed everyday agent. It uses `[model].name` and may interact with the PC only through the normal safety policy.
2. **Developer** — an idle-time engineering process. It prefers `[selfdev].developer_model` (`qwen2.5:32b` by default) when that model is installed, otherwise it explicitly falls back to the everyday model.
3. **Candidate** — an isolated Git worktree or copied workspace. Autonomous source writes are possible only through `CandidateTools.write_project_file` inside this boundary.

Stable is never rewritten in place, candidate code is not executed locally by the autonomous loop, and there is no automatic promotion path.

## Self-development cycle

```text
guarded fetch/fast-forward of the clean trusted main checkout
    -> stop and reload on update, or fail closed when sync is unsafe
    -> resource gate
    -> reconcile earlier candidate PR/check state
    -> resume an unfinished local candidate, when present
    -> enforce the one-outstanding-candidate gate
    -> select an eligible seed task, or discover a capability-growth question
    -> classify Repair / Extend / Improve Cognition / Explore
    -> record limitation, alternatives, falsifiable hypothesis and measured baseline
    -> create isolated candidate
    -> save versioned candidate/task/capability checkpoint
    -> research stage (list/read tools only)
    -> checkpoint concise findings and exact next action
    -> implementation stage (candidate tools)
    -> structured JSON change-plan fallback, if direct editing stalls
    -> non-executing local static checks
    -> bounded same-worktree static-repair/recheck loop, if needed
    -> reject regressed or unmeasured complexity
    -> candidate commit/push
    -> focused human-review PR presentation
    -> GitHub Actions
    -> human-reviewed PR merge
    -> validated experiment outcome updates local capability memory
```

On a later invocation, an active checkpoint is considered only after guarded main sync, the resource gate and candidate reconciliation. Resume fails closed unless its version, cycle/task/branch/worktree identity, seed-or-experiment contract fingerprint, capability target/hypothesis, Git HEAD, changed-path set and candidate content digest all match current state. A rejected checkpoint is deleted; LocalPilot may then rebuild a handoff from the independently validated learning and Git state, but never trusts stale findings or decisions.

The self-sync gate acts only on the repository root while it is checked out on the configured main branch. It never switches branches, resets changes, merges divergent history, or enters a candidate worktree. Git operations use argument vectors with `shell=False`. A successful fast-forward ends the current invocation so no evolution work continues in a process that loaded the previous build.

The fallback is data, not a privileged editing channel. Its paths and complete file contents are parsed into a `ChangePlan`, and every item is applied by calling `CandidateTools.write_project_file`. It therefore keeps the same path confinement, protected-directory rules, file limits, type allow-list, and size limit as ordinary candidate tool calls.

GitHub validation and merge are separate facts. A passing local static check or a pushed branch does not complete a task. The next backlog item is eligible only after the current candidate's GitHub checks pass and its PR is merged. An unfinished unpushed Git candidate is pending too: its exact branch and worktree are reused on the next invocation, its existing changes are revalidated against path, type, size, reviewer-test and file-count protections, and it is never committed or pushed until static checks pass. LocalPilot can observe promotion state; it has no method that merges or promotes.

Explicit human rejection is a separate terminal outcome, never inferred from an arbitrary closed PR. `localpilot reject <PR> --reason <text>` resolves the PR to a `localpilot/candidate-*` branch and requires that exact branch to be owned by durable cycle memory. The database transaction records `rejected_by_human`, the prior validation state, PR identity, reason, and experiment lesson before the outstanding-candidate gate can clear. Later lifecycle reconciliation cannot overwrite that state. Only afterward may LocalPilot clear the matching checkpoint and remove a clean, registered matching worktree; it never deletes the branch or GitHub history.

## Capability-growth mandate

The scheduler does not treat “nothing is broken” as a terminal state. When no seed task or outstanding candidate blocks new work, the discovery planner asks what highest-leverage change could increase LocalPilot's future capability. It chooses from evidence in committed architecture, prior runs, failures, CI outcomes, resource constraints, benchmarks and recorded capability gaps—not from a hard-coded feature wishlist.

Every proposal belongs to one of four first-class classes: **Repair** for defects/reliability/resources, **Extend** for new tools or abilities, **Improve Cognition** for planning/memory/retrieval/evaluation/routing/context/learning, and **Explore** for bounded high-upside architectural hypotheses. A proposal is ineligible without a falsifiable hypothesis, alternatives, metric, baseline, success criterion and measurement method. Complexity is penalized during selection, and a capability candidate cannot be delivered when its evaluation is regressed or unmeasured.

Exploration remains candidate-only, resource bounded and human controlled. GitHub CI is the executable evaluation boundary; LocalPilot may open a PR for review but has no merge or promotion method.

## Local learning memory

`localpilot-data/learning.sqlite3` stores durable, machine-local development facts:

- task, candidate branch, everyday/developer model names;
- cycle outcome, static-check result, push/PR/merge/rejection state and prior CI evidence;
- candidate worktree and durable bounded local-repair attempt count;
- concise implementation summary;
- a short reusable lesson for later cycles;
- capability names and known limitations; and
- experiment class/target/question, alternatives, hypothesis, baseline, evaluation evidence and outcome.

The schema deliberately has no prompts, transcripts, messages, thinking, or chain-of-thought fields. Research and model scratch work are transient. Machine-private learning data remains under the already ignored `localpilot-data/` directory and is never committed to GitHub.

`localpilot-data/evolution-checkpoint.json` complements the cycle database while work is active. It stores bounded, scrubbed engineering facts: checkpoint version/time, task objective and acceptance criteria, evolution class, capability target, hypothesis, evaluation plan, cycle/branch/worktree, inspected and changed path names, concise research findings and decisions, Git/diff/static-check/test status and failure markers, unresolved questions, exact next action and reusable lessons. It stores no file bodies, prompts, messages, transcripts, raw token streams, hidden reasoning or known secret-bearing text. Saves use atomic replacement after meaningful tool/milestone progress and immediately before a resource pause or unhandled exit. Version-1 checkpoints migrate to the current reviewable schema, and terminal completion removes the checkpoint.

## Resource and process safety

The existing `ResourceGovernor` remains in charge of self-development eligibility. LocalPilot checks user idle time, CPU, and memory before the cycle and between every model/tool round, lowering its own Windows priority and pausing promptly when the owner returns.

All Git/GitHub/static-check process calls use argument arrays with `shell=False`. Full executable tests remain in GitHub Actions. Stable, developer, and candidate boundaries do not depend on model cooperation: the tool surface enforces them.



## Mission-directed capability frontier

LocalPilot's self-development is directed by a stable mission rather than by defect repair alone.
The mission is to become an increasingly capable general-purpose personal intelligence that expands
what its user can understand, create, and accomplish while remaining reliable, transparent,
resource-aware, interruptible, and under human control.

The mission is stable; the capability frontier is not. During idle capability discovery LocalPilot
must identify its current frontier, the limiting capability, why overcoming it has high transferable
leverage, the capability expected to be unlocked, and the next frontier that could become reachable.

Capability growth is therefore evaluated as:

current frontier -> limiting capability -> hypothesis -> measurable experiment -> evidence ->
capability unlocked -> next frontier

Code volume, complexity, autonomy, and resource consumption are not treated as intelligence.
Human review and merge remain the promotion boundary.
