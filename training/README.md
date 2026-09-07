# LocalPilot Training

This workspace exists to improve the single operational LocalPilot model. It is not a parallel "developer model" project. A candidate model or adapter is temporary until evaluation proves it better; if promoted, it becomes LocalPilot and the previous checkpoint remains only as rollback.

## Current sequence

1. Freeze and score the current `gpt-oss:20b` LocalPilot baseline with `LocalPilot Eval v1`.
2. Establish a small evolution-execution baseline for the current implementation path so the Claude Code cutover is measured on actual candidate work, not only conversational reasoning.
3. Integrate the full Claude Code implementation backend into the existing evolution loop.
4. Re-run the unchanged held-out Eval v1 and evolution-execution benchmark.
5. Build Training Corpus v1 from verified LocalPilot history, verified Claude Code evolution traces, and carefully licensed external material.
6. Select and verify the AMD-compatible LoRA/QLoRA backend.
7. Train the first adapter candidate.
8. Re-run the same held-out evaluation and promote only if the candidate improves without critical regression.

Actual model training must not begin before the baseline/evaluation foundation and Claude Code cutover are measurable.

## Data authority tiers

- **A — verified project evidence.** Human-reviewed fixes, passing/merged LocalPilot changes, verified evolution outcomes, trusted project-owned evaluation material, and other evidence with a known outcome.
- **B — trusted licensed external evidence.** High-quality permissively licensed datasets, repositories, or technical material with recorded provenance and license terms.
- **C — synthetic but independently validated.** Model-generated material that has been checked against authoritative evidence or executable outcomes.
- **D — unverified/generated material.** Candidate material only. Tier D must never enter a training or validation split directly.

The `quality_tier` field records evidence authority, not difficulty.

## Split isolation

The only supported splits are `train`, `validation`, and `held_out_eval`.

`held_out_eval` is physically and semantically isolated:

- it must never be used as model-training data;
- it must never be used to generate synthetic training examples;
- it must not be retrieved into model context while the evaluation is running;
- held-out records contain prompts and scoring expectations, not assistant answers;
- normalized prompt collisions between train/validation and held-out files are validation errors.

The stdlib validator catches exact and normalized duplicate leakage. Semantic/paraphrase leakage still requires review when data is added.

## JSONL example format

Every line is one JSON object. The normative structural schema is `training/schema/example.schema.json`; `training/scripts/validate_dataset.py` also enforces cross-record rules that JSON Schema cannot express conveniently.

Core fields:

```json
{
  "id": "lp-debug-000001",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "task_type": "debugging",
  "source": "localpilot_verified_history",
  "license": "project_owned",
  "quality_tier": "A",
  "split": "train",
  "verification_status": "ci_verified",
  "provenance": {
    "repository": "n47h4ni3l/localpilot",
    "reference": "github:pr/82"
  }
}
```

Optional fields are `tags`, `difficulty`, `expected_behavior`, `created_at`, and `metadata`.

### Messages

Allowed roles are `system`, `user`, `assistant`, and `tool`. Every example needs at least one user message. Training/validation records require an assistant target and must end with an assistant message. Held-out evaluation records must not contain assistant answers; they instead require `expected_behavior`.

### Verification status

Allowed values are:

- `human_verified`
- `ci_verified`
- `source_verified`
- `synthetic_validated`
- `review_required`
- `unverified`

`review_required` and `unverified` are never acceptable for active train/validation examples. Held-out eval accepts independently verified A/B material only.

## Dataset validation

Validate individual files or entire directories:

```powershell
.\.venv\Scripts\python.exe training\scripts\validate_dataset.py training\evals
```

Run the validator tests:

```powershell
.\.venv\Scripts\python.exe -m unittest training.tests.test_validate_dataset
```

The validator checks:

- required/unknown fields;
- IDs, task types, roles, and non-empty message content;
- tier/split/verification policy;
- provenance, source, and license presence;
- exact duplicate examples;
- normalized-message duplicates;
- duplicate IDs;
- normalized train/validation-to-held-out prompt leakage;
- held-out answer leakage.

Exit code `0` means valid, `1` means dataset validation failed, and `2` is reserved for invocation/I/O failures.

## Eval v1

`training/configs/eval_v1.yaml` defines a 150-task target suite. The initial seed contains 25 held-out tasks across:

- repository reasoning;
- debugging;
- tool use;
- research;
- evolution;
- epistemics;
- generalization.

The seed intentionally targets known LocalPilot failure classes with novel/paraphrased tasks rather than copying training examples. It is a starting set, not the final benchmark.

### Running the baseline

Run from a clean, up-to-date `main` checkout with Ollama and the configured model available:

```powershell
.\.venv\Scripts\python.exe training\scripts\run_eval_v1.py
```

The runner fails closed when the checkout is not clean `main` or does not match the locally known `origin/main`. It records the exact Git HEAD, configured model settings, and Ollama model digest/size metadata in the report.

Before any prompt reaches LocalPilot, the runner exports tracked `HEAD` into a temporary snapshot and removes the entire `training/` tree. Each task gets a fresh empty LocalPilot data directory. Memory embeddings, the owner library, SystemSense, self-development, GitHub/web tools, machine-state tools, and reversible actions are disabled. The only available tools are bounded repository tree/read/search/dependency inspection against the sanitized snapshot. This prevents LocalPilot from finding the public benchmark/rubric through its own repository or web tools.

Local response reports are written under `training/reports/` and ignored by Git so benchmark answers do not become accidental repository/training material.

A quick smoke run can target one task:

```powershell
.\.venv\Scripts\python.exe training\scripts\run_eval_v1.py --task-id lp-eval-repo-001
```

### Scoring

The model under evaluation must not see `expected_behavior`. After the run completes, create a separate review scorecard:

```powershell
.\.venv\Scripts\python.exe training\scripts\score_eval_v1.py training\reports\<run>.json --prepare
```

The resulting local JSONL combines each response with its held-out rubric and leaves the score blank. A human or independent reviewer assigns the 0–4 score, hard-failure flag, and a short rationale. Do not have the evaluated model score its own answers.

After review:

```powershell
.\.venv\Scripts\python.exe training\scripts\score_eval_v1.py training\reports\<run>.json --scorecard training\reports\<run>_scorecard.jsonl
```

The summary records overall mean, category means, critical-category means, hard failures, repository identity, model identity, and the isolation contract used for the run.

`LocalPilot Eval v1` measures reasoning, grounding, tool judgment, and epistemics. Because the Claude Code change primarily replaces the self-development implementation backend, it is not sufficient by itself to claim that the cutover improved autonomous software engineering. A separate bounded evolution-execution baseline must be frozen before the Claude Code integration and repeated afterward.

The independently reviewed pre-cutover aggregate is frozen in `training/baselines/eval_v1_gpt_oss_20b_6670fa9.json`. That durable artifact contains aggregate metadata only: no held-out prompts, rubrics, model answers, or scorecard rows.

## Evolution-execution baseline

`LocalPilot Evolution Execution v1` is a four-task synthetic benchmark for the current pre-Claude-Code implementation path. It measures multi-file contract completion, compatibility re-export preservation, regression-test-first ordering, failing-test diagnosis and repair, and strict scope control.

Each task runs in a disposable Git repository outside the LocalPilot checkout. The model sees only its contract and visible fixture files through candidate-scoped file tools. Hidden acceptance tests are mounted only while the evaluator runs and removed before model implementation or repair. No target patch exists in the benchmark or enters model context. Deterministic checks execute outside the model's tool surface, and at most one bounded repair pass receives failing output.

Run the complete baseline after this tooling is merged, from clean and current `main`:

```powershell
.\.venv\Scripts\python.exe training\scripts\run_evolution_execution.py
```

The runner records the selected current-path developer model and digest, verifies the real repository remains unchanged, writes the ignored raw report under `training/reports/`, and invokes `score_evolution_execution.py` automatically. See `training/evolution_execution/README.md` for the task inventory and isolation contract.

## Corpus policy

Preferred order of training sources:

1. verified LocalPilot Git/PR/CI/evolution history;
2. verified successful Claude Code implementation traces after integration;
3. selectively licensed external software-engineering and reasoning material;
4. independently validated synthetic examples.

Do not ingest arbitrary public GitHub code, scraped conversations, proprietary material, private chats, secrets, or model-generated output merely because it is available. Publicly accessible is not equivalent to licensed for training.

## Training backend

`training/configs/qlora_v1.yaml` and `training/scripts/train_adapter.py` remain placeholders intentionally. Backend selection comes after Eval v1 and the Claude Code architecture change because the target environment is Windows/AMD and current compatibility must be verified rather than assumed.
