# LocalPilot architecture

LocalPilot has three isolated roles and two model responsibilities:

1. **Stable operator** — the installed everyday agent. It uses `[model].name` and may interact with the PC only through the normal safety policy.
2. **Developer** — an idle-time engineering process. It prefers `[selfdev].developer_model` (`qwen2.5:32b` by default) when that model is installed, otherwise it explicitly falls back to the everyday model.
3. **Candidate** — an isolated Git worktree or copied workspace. Autonomous construction is possible only through candidate tools inside this boundary: file writes, free inert directory scaffolding, bounded ZIP creation, and provenance-tracked HTTPS resources stored outside the repository.

Stable is never rewritten in place, candidate code is not executed locally by the autonomous loop, and there is no automatic promotion path.

## Stable-operator research context

Operator research keeps the owner's request, streamed same-model reasoning, assistant tool calls, and complete raw tool results in one live context through final synthesis. Raw results receive current-turn `obs-NNN` and `result-NNN` identifiers. They are never replaced by a bounded ledger or fresh summary context.

After the advisory 12-round soft budget, each further unique observation requires a model-authored compact information-gain delta. It contains only a bounded list of bare current-turn `evidence_refs`, one `unresolved_fact`, one `proposed_tool`, `proposed_arguments` as a real object, the result that would change the conclusion, and an optional distinct hypothesis. It has no verified-fact prose, unresolved-question history, or inspected-observation history. Prior inspection state and semantic redundancy are derived internally from `ObservationRecord`.

Semantically similar searches and overlapping reads prompt for the distinct new hypothesis being tested. A justified follow-up remains allowed. Exact duplicate read-only calls retain the existing result cache and consume no unique research round. The hard ceiling remains 24 rounds.

Checkpoint tool calls/results, checkpoint-required messages, malformed-call recovery prompts/partial reasoning, and rejected control turns are removed before final synthesis, not merely after the owner turn. The final high-reasoning model context therefore contains the original request, legitimate assistant/tool trajectory, and every complete raw tool result—but no checkpoint prose or planning scaffold and no fresh summary context. Only safe structural counts/status may enter audit output; checkpoint content is excluded from retained chat memory, learning SQLite, evolution checkpoints, and audit content. A checkpoint probe mode uses a fresh manifest and three unpredictable fragments to cross the soft boundary and verify the real compact protocol, retrieval, reconciliation, and clean raw-evidence synthesis within the hard budget.

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

Candidate file volume uses a configurable 100-file soft budget and 500-file hard ceiling. Crossing the soft budget adds explicit complexity evidence but does not block work; directories never count. ZIP members must originate in the candidate or its resource store and are normalized, member-count bounded, byte bounded, and never executed. Candidate resources use an 8 GiB default quota and 512 MiB per-file bound under `localpilot-data/candidate-resources`, with source/final URL, time, hash, MIME/extension, size, task, branch, cycle, and stale-source provenance.

A human-authorized policy retry is represented by a new development cycle linked to the preserved prior cycle. The prior record becomes `policy_blocked` with `framework_policy` attribution. An unpushed candidate may retain its branch; a pushed-but-unmerged candidate must have its remote branch and PR state verified, then receives a deterministic new candidate branch from clean current trusted `main`. The old pushed branch, commit, PR, workspace and cycle evidence remain untouched, while only the linked retry is nonterminal. The retry retains the same task, resets only bounded local repair/write-integrity counters, and re-enters research/implementation. Repeated authorization returns the existing linkage and never promotes the result.

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

## Staged repository-grounded study

The curriculum is a separate, read-only learning path, not an autonomous candidate and not model
weight training:

```text
held-out baseline -> read-only study -> concise sourced fact graph -> held-out retest
    -> measured gain: unlock next stage
    -> no gain: retain errors and adapt current stage
```

Stages are locked in the order `self -> qwen -> python`. Facts record only bounded summaries,
source/provenance, confidence, last verification time, source digest, and relationships. Repository
source changes invalidate prior facts. Benchmark question identifiers and evaluators are held
separately from study facts so indexing cannot simply persist the answer set.

The self map is extracted from committed structure and Python ASTs. It supports explicit checks for
invented APIs/config fields, duplicate subsystem proposals, disconnected code, missing files or
commands, wrong integration points, call-graph mismatches, and missing test contracts. Capability
discovery receives only bounded curriculum status and verified fact counts; its prompt requires
current repository evidence for these claims.

Qwen study prefers the local Ollama `show` metadata for the artifact actually installed. Optional
web verification is HTTPS/read-only and restricted to official Qwen/Ollama/Python/pytest sources
for high-confidence durable facts. Python study ties authoritative library semantics back to the
project paths importing those APIs.

Optional peer-model comparison runs identical transfer scenarios through two installed Ollama
models and records scores, latency/resource cost, and concise lessons. Model size is not a scoring
input, raw responses are not retained, and no comparison changes configuration or promotion state.
