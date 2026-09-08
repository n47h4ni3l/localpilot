# Training Manifest

Status: **Training Phase 1 prepared; adapter configuration proposed; no model training authorized yet**.

## Active artifacts

- `schema/example.schema.json` — normative per-record JSON Schema.
- `scripts/validate_dataset.py` — stdlib dataset/schema/duplicate/leakage validator.
- `tests/test_validate_dataset.py` — focused validator regression tests.
- `configs/eval_v1.yaml` — Eval v1 category targets, metrics, isolation, and promotion criteria.
- `evals/*/eval_v1_seed.jsonl` — 25 held-out seed tasks with no assistant answers.
- `scripts/run_eval_v1.py` — isolated LocalPilot response runner that freezes repository/model identity and removes the training tree before model access.
- `scripts/score_eval_v1.py` — post-run scorecard preparation and aggregate scoring; the evaluated model never receives its rubric.
- `tests/test_run_eval_v1.py` — checkout, snapshot-isolation, held-out loading, and tool-surface tests.
- `tests/test_score_eval_v1.py` — scorecard and aggregate-scoring tests.
- `baselines/eval_v1_gpt_oss_20b_6670fa9.json` — durable aggregate-only current-model baseline; contains no held-out task content or scorecard rows.
- `baselines/eval_v1_post_cc_gpt_oss_20b_f619abb.json` — aggregate-only post-Claude-Code Eval v1 result.
- `baselines/evolution_execution_pre_cc_gpt_oss_20b_c213be3.json` — corrected aggregate-only pre-Claude-Code execution baseline.
- `baselines/evolution_execution_post_cc_gpt_oss_20b_f619abb.json` — aggregate-only post-Claude-Code execution result.
- `evolution_execution/cases.json` — four synthetic implementation contracts plus declarative deterministic criteria; no target patches.
- `evolution_execution/acceptance/*` — evaluator-only acceptance tests, mounted temporarily and never exposed through model tools.
- `scripts/run_evolution_execution.py` — disposable-fixture runner for the current pre-Claude-Code candidate implementation and bounded repair path.
- `scripts/score_evolution_execution.py` — deterministic 0–4 aggregate scorer for execution reports.
- `tests/test_evolution_execution.py` — isolation, fixture, scope, ordering, and scoring regression tests.
- `manifests/eval_v1_manifest.json` — frozen held-out IDs plus one-way normalized/raw/shingle hashes; no plaintext held-out content.
- `sources/corpus_v1_seed_sources.jsonl` — reviewed source recipe with commit/PR/file/test provenance.
- `datasets/corpus_v1_seed.jsonl` — 24-example Tier A seed (20 train, 4 validation).
- `scripts/build_eval_manifest.py` — the sole training-preparation utility allowed to hash `training/evals/**`.
- `scripts/build_localpilot_history_dataset.py` — allowlisted, provenance-verified, duplicate/leakage-checked corpus builder.
- `scripts/compare_models.py` — Eval v1 plus Evolution Execution v1 promotion-gate comparison.
- `configs/qlora_v1.yaml` — proposed, revision-pinned AMD/Unsloth QLoRA configuration.
- `scripts/train_adapter.py` — environment/data/model/resource dry-run and separately gated real-training entry point.
- `BACKEND_DECISION.md` — dated primary-source backend research and exact E:-backed setup commands.
- `outputs/.gitignore` — keeps adapters and checkpoints out of Git.
- `tests/test_eval_manifest.py`, `tests/test_history_dataset.py`, `tests/test_compare_models.py`, `tests/test_train_adapter.py`, and `tests/test_phase1_artifacts.py` — focused safety and behavior coverage.
- `reports/.gitignore` — keeps local responses/scorecards/summaries out of Git and future training data by default.
- `README.md` — data authority, split, provenance, licensing, isolation, scoring, and rollout policy.

## Split policy

- `train`: verified A/B/C examples only; assistant target required.
- `validation`: verified A/B/C examples only; assistant target required.
- `held_out_eval`: independently verified A/B tasks only; assistant answers forbidden.

Held-out examples must never enter training, synthetic generation, or retrieval context during evaluation.

## Data authority

- Tier A: verified LocalPilot/project evidence with known outcomes.
- Tier B: trusted, appropriately licensed external material.
- Tier C: synthetic material independently validated against evidence/outcomes.
- Tier D: unverified/generated candidate material; never direct training input.

Every record must carry `source`, `license`, `verification_status`, and `provenance.reference`.

## Eval v1 seed inventory

| Category | Seed | Target |
| --- | ---: | ---: |
| repository_reasoning | 4 | 30 |
| debugging | 4 | 30 |
| tool_use | 4 | 20 |
| research | 3 | 20 |
| evolution | 4 | 20 |
| epistemics | 4 | 20 |
| generalization | 2 | 10 |
| **Total** | **25** | **150** |

## Baseline isolation contract

`run_eval_v1.py` requires clean, up-to-date `main` by default and records the exact Git HEAD plus Ollama model identity. It evaluates a `git archive HEAD` snapshot with the entire `training/` tree removed. Each task receives a fresh empty data directory. Memory embeddings, owner library, SystemSense, self-development, GitHub/web access, machine-state tools, and reversible actions are disabled. Only bounded repository tree/read/search/dependency tools remain.

This prevents the public held-out benchmark or its scoring rubric from being retrieved by LocalPilot during the run.

## Next gates

1. Review and merge Training Phase 1 without running the adapter trainer.
2. Create the documented Ubuntu 24.04 WSL2 distribution and Python environment on E:.
3. Run the exact local backend preflight and dry-run; retain its ignored report.
4. Review dry-run evidence before changing the config from `proposed`.
5. Run training only as a separate, explicitly authorized activity.
6. Evaluate the candidate on unchanged Eval v1 and Evolution Execution v1, then apply the comparison gates and human review.
