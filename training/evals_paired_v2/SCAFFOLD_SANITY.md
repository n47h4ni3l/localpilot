# Nestra P1: direct versus LocalPilot sanity check

This is a **small exploratory test**, separate from the formal P1-vs-original
GPT-OSS evaluation. It compares **the same exact Nestra P1 weights** without
LocalPilot against Nestra P1 run through the repaired LocalPilot scaffold. It
asks how the software around the model affects delivered answer quality.

Eight deliberately varied **existing held-out** tasks cover repository reasoning,
diagnostics, lineage validation, current-vs-historical evidence, tool authority,
CI/promotion, epistemic failure analysis and cross-device generalization.
There are three repetitions per configuration: **8 × 3 × 2 = 48 answers**.

The P1 weight tag and immutable digest are identical in both arms. Inference
temperature, context length, reasoning settings, prompt and expected rubric
stay matched. LocalPilot may provide its **read-only isolated repository tools**;
the direct arm has no tool privileges and is told not to claim live observation.
That is the difference under test, not a controlled difference in model training.

The selected tasks are already part of the 76-task suite. **Do not reuse these
48 results as a new independent model-weight benchmark or P2 promotion gate.**
The exploratory sample is too small for a reliable population confidence
interval, and the direct model intentionally has fewer tools.

## Running the sanity check on the Windows PC

Use a clean, updated `main` containing the approved scaffold repair.
Ollama must have `nestra:20b-p1` installed and runnable. Nothing here starts
training or changes model weights, accepted lineage or production settings.

Preview the planned cases at zero inference cost:

```powershell
.\.venv\Scripts\python.exe training\scripts\run_scaffold_sanity.py --plan-only
```

Run a short first session if the PC is busy:

```powershell
.\.venv\Scripts\python.exe training\scripts\run_scaffold_sanity.py --max-new-cells 12
```

Resume from the **same repository revision/model digest**:

```powershell
.\.venv\Scripts\python.exe training\scripts\run_scaffold_sanity.py --resume
```

Or omit `--max-new-cells` on a fresh run to run all 48 cases.
Default output is `training/reports/nestra_scaffold_sanity.json`, excluded
from source commits. Every response includes the raw Ollama or LocalPilot
intermediate turn evidence, terminal answer, audit log, timing and any exception.
A stop after three execution faults prevents repeatedly wasting GPU time on
a broken environment. **Do not overwrite/resume across scaffold changes.**

The immediate console summary only includes error/withholding counts and
latency. **Those numbers alone cannot establish whether the answer is good.**
For proper assessment, prepare anonymous arm-blind rubric review cards:

```powershell
.\.venv\Scripts\python.exe training\scripts\score_scaffold_sanity.py training\reports\nestra_scaffold_sanity.json --prepare training\reports\nestra_scaffold_sanity_review.jsonl
```

Provide the review cases to an independent evaluator who must not see the
direct/scaffold arm labels or repetition indices. Fill in numeric `score`
from 0–4, boolean `hard_failure`, `failure_origin`
(`none`, `model`, `tool`, `harness`, `environment` or `uncertain`),
nonempty `rationale` and `reviewer`. Retain other fields unchanged.

```powershell
.\.venv\Scripts\python.exe training\scripts\score_scaffold_sanity.py training\reports\nestra_scaffold_sanity.json --scorecard training\reports\nestra_scaffold_sanity_review.jsonl --output training\reports\nestra_scaffold_sanity_scored.json
```

The score report gives paired **task-level** wins, losses, ties, score
difference, failures by origin, category differences and automated diagnostic
signals. It never silently excludes a scaffold fault: a fault affecting the
user-visible answer remains part of delivered-system quality and is separately
flagged for review. Check draft/model-turn versus delivered-answer evidence in
scaffold regressions, particularly false-source requirements, mistaken
refusals, unhelpful recovery and incorrect synthesis. A good-looking result
does not itself prove the scaffold is harmless on unseen tasks.

## Relationship to the 684-cell evaluation rerun

The old 684-cell report is **frozen historical evidence**. PR #183 changed
instrumentation, PR #184 changed answer routing. Both alter the evaluated
environment, so **do not resume the old report** on the repaired code.

After approving the repair, start a **new full evaluator run**, new run ID,
new output path, both original GPT-OSS and P1 models, with the repaired
scaffold pinned to its exact repository revision. Compare:
1. Original GPT-OSS + LocalPilot versus P1 + LocalPilot (primary weight effect).
2. Original GPT-OSS direct as optional descriptive context.
3. P1 direct versus P1 + LocalPilot (this separate sanity check).

No automatic model promotion, adapter rewrite or P2 training is permitted
by these evaluations.
