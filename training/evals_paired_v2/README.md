# Nestra paired evaluation v2

This is a **new comparison**, not a rewrite of Eval v1 or of its historical
baseline scores. The goal is to distinguish a change in **model weights** from
the gains provided by LocalPilot's tooling. All three conditions use the same
tracked source snapshot, prompts, task IDs, 0–4 rubrics, context settings and
model-requested reasoning settings. The two scaffold arms use the **same**
LocalPilot code and identical authorized read-only repository tools.

| Arm | Weight identity | Tooling |
| --- | --- | --- |
| `base_direct` | `gpt-oss:20b` | Direct Ollama call, no LocalPilot tools |
| `base_localpilot` | `gpt-oss:20b` | Identical current LocalPilot scaffold |
| `nestra_localpilot` | `nestra:20b-p1` (replaceable future candidate) | Identical current LocalPilot scaffold |

**These are three configurations but only two distinct weight sets.** The
weight-change claim is determined **only** by `nestra_localpilot` compared
against `base_localpilot`. The direct arm separately describes the benefits
of assistance from tools and orchestration; repository evidence tasks are at a
structural disadvantage without tools, so that contrast is not a pure model
intelligence comparison. Once P2 is trained, the candidate tag can be changed
in a new report, not inside a running one. To compare **three weight sets**
directly, perform separate matched runs and compare their scaffold arms.

## Coverage and workload

The new suite **adds 51** project-curated held-out scenarios under
`training/evals_paired_v2/` to the **25 existing Eval v1 tasks**, for **76
distinct tasks** over seven areas: repository reasoning, debugging, tool use,
research, evolution, epistemics and generalization. Scenarios vary their
constraints and failure mechanisms; they are not simply copies with different
wording. The questions/rubrics are separate from model target answers, and
are excluded from training and from the isolated repository snapshot.

Each task runs **three times in each of the three arms**, for **684 total
inference cells**. A new agent/data directory is used per scaffold cell. Arm
order rotates each repetition to reduce systematic order effects. Each
response is stored immediately so testing can be resumed without repeating
completed cells. The run never auto-promotes a model. The suite is expensive
on a 16 GB GPU: plan for long idle periods and small sessions.

Before consequential conclusions, an independent reviewer should inspect the
curated rubrics, especially any case that appears to decide a model regression.
This suite is a stronger comparative experiment, not a mathematical guarantee
that a small measured difference must be real.

## Run on the PC (after PR merge)

Use a clean, current `main`, with Ollama available and both model tags
installed. Do not switch the current accepted Operational model or alter any
training artifact. The runner checks exact model digests before evaluation.

First preview the plan without running inference:

```powershell
.\.venv\Scripts\python.exe training\scripts\run_eval_matrix.py --plan-only
```

Pilot a small complete subset to test the workflow:

```powershell
.\.venv\Scripts\python.exe training\scripts\run_eval_matrix.py --limit 6 --output training\reports\paired_pilot.json
```

That pilot covers **54 cells** (6 tasks × 3 runs × 3 conditions). For the
full suite, use a new output path:

```powershell
.\.venv\Scripts\python.exe training\scripts\run_eval_matrix.py --output training\reports\paired_full.json --max-new-cells 27
```

Resume repeatedly (same `--output`, same repository commit, identical model
digests and settings) during later idle sessions:

```powershell
.\.venv\Scripts\python.exe training\scripts\run_eval_matrix.py --output training\reports\paired_full.json --resume --max-new-cells 27
```

Omit `--max-new-cells` when the system can finish the remaining cells.
The runner stops early after three runtime exceptions to avoid grinding through
a broken environment; the result is preserved for investigation. When paused,
the report is intentionally incomplete and cannot be scored. It can only be
resumed with the same selected tasks and source state. The suite is
**manually triggered**, not an automatic background GPU workload.

## Blind review and score

Once the report contains all expected cells, prepare a blinded scorecard:

```powershell
.\.venv\Scripts\python.exe training\scripts\score_eval_matrix.py training\reports\paired_full.json --prepare training\reports\paired_full_review.jsonl
```

The review file shuffles cases and hides arm names and repetition numbers.
Each case contains the prompt, expected behaviour rubric, actual response,
and any runtime error. Assign a 0–4 `score`, boolean `hard_failure`,
non-empty `rationale`, `reviewer` identifier, and `failure_origin` from:

- `none`: no hard failure or attributable system failure;
- `model`: attributable model output/behaviour failure;
- `tool`: observed malfunction in an authorized tool;
- `harness`: runner, parser or orchestration failure;
- `environment`: system capacity/service/platform failure; or
- `uncertain`: insufficient evidence to attribute cause.

**Do not infer model regression from a runtime exception or a zero score.**
A correct tool refusal is not a software fault. A hallucinated result is not
excused merely because a tool existed. All origins require independent review.
A reviewer should not use the evaluated model to grade itself.

After scoring, calculate results without further model calls:

```powershell
.\.venv\Scripts\python.exe training\scripts\score_eval_matrix.py training\reports\paired_full.json --scorecard training\reports\paired_full_review.jsonl --output training\reports\paired_full_scored.json
```

Scores are paired **by task**, averaging the three runs for each arm. The
primary comparison bootstraps *whole tasks*, 5,000 repetitions, for a
95% percentile confidence interval (not 684 independent observations). The
report includes per-category differences, model-attributed hard failures,
infrastructure fault counts, task wins/losses, and an excluded-case ledger.
A task with unresolved/tool/harness/environment errors is omitted from the
model comparison **and reported explicitly**, so it must be rerun or
investigated before trusting any conclusion. A confidence interval excluding
zero is evidence of a stable difference on this task population—not proof of
out-of-sample AGI or real-world superiority.

## Investigate the three P1 hard failures first

The committed P1 lineage manifest contains the aggregate hard-failure count,
not the affected task responses or full review evidence. No diagnosis can be
made from that aggregate. Use the original **local** P1 Eval v1 response
report and scored JSON (plus the original base run and scores, if available):

```powershell
.\.venv\Scripts\python.exe training\scripts\triage_eval_v1_failures.py --candidate-run training\reports\p1_run.json --candidate-scores training\reports\p1_scored.json --output training\reports\p1_failure_triage.json
```

Replace illustrative paths with the actual report filenames. If available,
append `--baseline-run ... --baseline-scores ...` for task-level side-by-side
evidence. Read each failed task's response, exception, reviewer rationale and
tool/broker trace. Attribute failure to **model**, **tool**, **harness**,
**environment**, **review**, or **undetermined** only after inspecting the
original observations and, where useful, repeating under matched conditions.
Do **not** modify historic scorecards to make regression disappear.

## Isolation and maintenance

- Evaluation source tree is physically removed from each tracked Git snapshot.
- Library, semantic embeddings, existing memories, SystemSense, external
  research/GitHub, self-development and reversible actions are disabled.
- Temporary per-cell agent state and the complete snapshot are discarded.
- Raw answers, local rubrics-with-answers, human scores and diagnostics remain
  in ignored `training/reports/`; **never commit or upload them by default**.
- Running on a different Git commit or swapping model tags/digests mid-report
  is rejected. Scorecards protect prompt/rubric/response evidence from edits.
- Do not tune training examples or change this suite to chase its scores.
  Freeze the evaluator for before/after work; preserve genuinely novel,
  untouched future tests for confirmation.
- This tool does **not** train, modify LoRA tensors, register a new Ollama
  model, change LocalPilot's accepted P1 lineage or promote any candidate.
