# LocalPilot Training

This workspace exists to improve the single operational LocalPilot model. It is not a parallel "developer model" project. A candidate model or adapter is temporary until evaluation proves it better; if promoted, it becomes LocalPilot and the previous checkpoint remains only as rollback.

## Current sequence

1. **Complete:** freeze the pre- and post-Claude-Code aggregate benchmarks.
2. **Complete:** build the first verified, project-owned Corpus v1 seed and held-out hash manifest.
3. **Proposed:** validate the pinned WSL2/ROCm/Unsloth adapter environment on the target machine.
4. Change `qlora_v1.yaml` from `proposed` only after an exact dry-run passes locally and its report matches the config and corpus digests.
5. Train the first adapter in a separate, explicitly authorized run.
6. Re-run the unchanged Eval v1 and Evolution Execution v1 benchmarks.
7. Consider promotion only from held-out and execution evidence, with human review and merge.

This change prepares training but does not install PyTorch, download model weights, or launch training.

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

The corresponding post-Claude-Code aggregate is frozen in `training/baselines/eval_v1_post_cc_gpt_oss_20b_f619abb.json`: overall 1.88/4, critical 1.6875/4, and one hard failure. It points back to the existing pre-cutover aggregate for comparison and likewise contains no held-out content.

## Evolution-execution baseline

`LocalPilot Evolution Execution v1` is a four-task synthetic benchmark for the current pre-Claude-Code implementation path. It measures multi-file contract completion, compatibility re-export preservation, regression-test-first ordering, failing-test diagnosis and repair, and strict scope control.

Each task runs in a disposable Git repository outside the LocalPilot checkout. The model sees only its contract and visible fixture files through candidate-scoped file tools. Hidden acceptance tests are mounted only while the evaluator runs and removed before model implementation or repair. No target patch exists in the benchmark or enters model context. Deterministic checks execute outside the model's tool surface, and at most one bounded repair pass receives failing output.

Run the complete baseline after this tooling is merged, from clean and current `main`:

```powershell
.\.venv\Scripts\python.exe training\scripts\run_evolution_execution.py
```

The runner records the selected current-path developer model and digest, verifies the real repository remains unchanged, writes the ignored raw report under `training/reports/`, and invokes `score_evolution_execution.py` automatically. See `training/evolution_execution/README.md` for the task inventory and isolation contract.

Aggregate-only execution records preserve the corrected pre-cutover result (3.0/4, two perfect tasks, two hard failures) and the post-Claude-Code result (3.75/4, three perfect tasks, zero hard failures, zero scope violations). They contain no contracts, hidden acceptance fixtures, model output, or target patch.

## Corpus v1 seed

`training/sources/corpus_v1_seed_sources.jsonl` is the reviewable, hand-curated source recipe. `build_localpilot_history_dataset.py` validates every record against its cited Git commit, merged PR number, changed files, and tests before emitting `training/datasets/corpus_v1_seed.jsonl`.

The initial seed has 24 Tier A, project-owned, source-verified examples: three examples in each of software engineering, debugging, agent planning/tool use, epistemics/self-correction, AI/LLM/agent systems, architecture/code review, LocalPilot architecture, and evolution/experimental reasoning. The split is 20 train and 4 validation. These are curated derivations of verified project changes, not raw conversations or claims that CI verified the wording. Immutable Git blob IDs identify the files and tests supporting each answer. This small seed validates the pipeline; its size alone is not evidence that a 20B adapter will improve.

The builder uses an explicit project-owned path allowlist and rejects `training/evals/**`, scorecards, reports, hidden acceptance fixtures, rubrics, target patches, and benchmark answers before reading them. Tier D and non-training splits are fatal. This history builder also refuses Tier C because it does not implement an independent synthetic-validation executor. Exact/normalized duplicates and repeated prompts across splits are rejected. The generated corpus is checked against a schema-validated held-out manifest for IDs, normalized prompts/content, role-independent prompt/rubric fragments, and shared 16-token shingles, including text nested in metadata. Symlinks, junctions, conflicting IDs, malformed or empty manifests, and output paths outside the designated folders fail closed.

The builder requires complete Git history and a LocalPilot origin. It verifies each commit is on known `origin/main` (or local `main`), its subject matches the declared merged PR, and cited source paths are regular Git files. New source recipes require source review; substring/hash checks cannot detect every paraphrase or prove the semantic truth of an answer. The held-out files must never be opened to generate or repair corpus examples.

Generate and verify the artifacts from the repository root:

```powershell
.\.venv\Scripts\python.exe training\scripts\build_eval_manifest.py --check
.\.venv\Scripts\python.exe training\scripts\build_localpilot_history_dataset.py --check
.\.venv\Scripts\python.exe training\scripts\validate_dataset.py training\datasets\corpus_v1_seed.jsonl
```

Corpus statistics are written to the ignored `training/reports/corpus_v1_stats.json`, including category/tier/split/source/license counts, provenance completeness, duplicate and leakage rejections, size estimates, and split ratio.

## Promotion comparison

`compare_models.py` compares a baseline and candidate across both Eval v1 and Evolution Execution v1. It reports category deltas and requires no overall Eval regression, no material regression in the critical aggregate or any critical category, no increase in Eval or execution hard failures, execution performance at least at baseline, and zero scope violations. Complete matching task/category coverage, frozen suite hashes, and matching model identity across each model's two benchmark runs are required. Missing evidence produces an incomplete comparison, never a recommendation. Historical aggregates remain useful for deltas, but lack enough recorded metadata to authorize a model promotion. The scoring tools now emit the coverage, scope, and suite metadata needed for future comparisons. Training loss is deliberately not a promotion signal; final promotion requires human review.

## Corpus policy

Preferred order of training sources:

1. verified LocalPilot Git/PR/CI/evolution history;
2. verified successful Claude Code implementation traces after integration;
3. selectively licensed external software-engineering and reasoning material;
4. independently validated synthetic examples.

Do not ingest arbitrary public GitHub code, scraped conversations, proprietary material, private chats, secrets, or model-generated output merely because it is available. Publicly accessible is not equivalent to licensed for training.

## Training backend and dry-run

See `training/BACKEND_DECISION.md` for the current primary-source compatibility research and exact E:-backed Windows/WSL2 setup commands. The proposed backend is Ubuntu 24.04 under WSL2, AMD ROCm 7.14.1/PyTorch 2.12, and the current Unsloth AMD gpt-oss path. Native Windows is retained as an experimental fallback, not the primary path.

The proposed `qlora_v1.yaml` pins model revisions and all adapter, data, optimization, validation, checkpoint, seed, and resource settings. It remains `status: proposed` until the actual target environment passes:

```powershell
wsl -d LocalPilot-Training --cd <repo> -- bash -lc '~/.venvs/localpilot-training/bin/python training/scripts/train_adapter.py --dry-run --allow-downloads'
```

The dry-run performs no training. It verifies the WSL/Ubuntu/Python environment, pinned backend versions, a small BF16 GPU operation, the HIP Triton target, free VRAM/RAM/storage, local dataset schema and splits, the frozen manifest digest and leakage, all cached model shards, tokenizer/chat-template lengths, PEFT configuration, output safety, and the exact resolved training command. `--allow-downloads` explicitly permits downloading the pinned model weights and tokenizer into the configured cache in the E:-stored WSL distro. Omit it for an offline cache check. The dry-run does not allocate the 20B model or measure its training peak; the memory estimate remains unmeasured.

Real training additionally requires an approved config, a matching passing local dry-run report, `--train`, and an explicit confirmation string. Changing only the approval status preserves the tested training-settings digest; changing any model/data/resource/training setting invalidates it. The entry point rechecks the environment, corpus, manifest, cached model, and output immediately before loading. Do not approve or execute training in this phase.
