# Training Manifest

Status: **evaluation foundation in progress; no model training authorized yet**.

## Active artifacts

- `schema/example.schema.json` — normative per-record JSON Schema.
- `scripts/validate_dataset.py` — stdlib dataset/schema/duplicate/leakage validator.
- `tests/test_validate_dataset.py` — focused validator regression tests.
- `configs/eval_v1.yaml` — Eval v1 category targets, metrics, isolation, and promotion criteria.
- `evals/*/eval_v1_seed.jsonl` — 25 held-out seed tasks with no assistant answers.
- `README.md` — data authority, split, provenance, licensing, and rollout policy.

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

## Next gates

1. Run validator tests and validate all held-out JSONL files.
2. Freeze the exact current LocalPilot model/config/repository baseline.
3. Implement a baseline evaluation runner/scorer without exposing held-out expectations to the model.
4. Score current LocalPilot.
5. Implement the full Claude Code evolution backend.
6. Re-score the unchanged held-out suite.
7. Only then begin Corpus v1 construction and backend-specific adapter training work.
