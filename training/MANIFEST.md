# Training Manifest

Status: **baseline evaluation tooling active; no model training authorized yet**.

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
- `reports/.gitignore` — keeps local responses/scorecards/summaries out of Git and future training data by default.
- `README.md` — data authority, split, provenance, licensing, isolation, scoring, and rollout policy.

## Intentionally still placeholders

- `configs/qlora_v1.yaml`
- `scripts/train_adapter.py`
- `scripts/build_localpilot_history_dataset.py`
- `scripts/build_eval_manifest.py`
- `scripts/compare_models.py`

These remain inactive until the current baseline is captured and the full Claude Code implementation backend is integrated and evaluated.

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

1. CI-validate the baseline runner/scorer and existing dataset validator.
2. Run the 25-task current `gpt-oss:20b` baseline locally and save the ignored response report.
3. Prepare the scorecard and obtain an independent/human 0–4 review for every response.
4. Freeze a small **evolution-execution baseline** for the current Qwen/self-development implementation path; Eval v1 alone does not directly measure candidate-building performance.
5. Implement the full Claude Code evolution backend.
6. Re-run both the unchanged Eval v1 and the execution benchmark.
7. Only then begin Corpus v1 construction and backend-specific adapter training work.
