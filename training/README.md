# LocalPilot Training

This workspace exists to improve the single operational LocalPilot model. It is not a parallel "developer model" project. A candidate model or adapter is temporary until evaluation proves it better; if promoted, it becomes LocalPilot and the previous checkpoint remains only as rollback.

## Current sequence

1. Freeze and score the current `gpt-oss:20b` LocalPilot baseline with `LocalPilot Eval v1`.
2. Integrate the full Claude Code implementation backend into the existing evolution loop.
3. Re-run the same held-out Eval v1 to measure the architectural change.
4. Build Training Corpus v1 from verified LocalPilot history, verified Claude Code evolution traces, and carefully licensed external material.
5. Select and verify the AMD-compatible LoRA/QLoRA backend.
6. Train the first adapter candidate.
7. Re-run the same held-out evaluation and promote only if the candidate improves without critical regression.

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

## Validation

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

## Corpus policy

Preferred order of training sources:

1. verified LocalPilot Git/PR/CI/evolution history;
2. verified successful Claude Code implementation traces after integration;
3. selectively licensed external software-engineering and reasoning material;
4. independently validated synthetic examples.

Do not ingest arbitrary public GitHub code, scraped conversations, proprietary material, private chats, secrets, or model-generated output merely because it is available. Publicly accessible is not equivalent to licensed for training.

## Training backend

`training/configs/qlora_v1.yaml` and `training/scripts/train_adapter.py` remain placeholders intentionally. Backend selection comes after Eval v1 and the Claude Code architecture change because the target environment is Windows/AMD and current compatibility must be verified rather than assumed.
