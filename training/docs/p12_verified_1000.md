# Nestra P1.2 — Verified 1,000-target curriculum

**State:** source-selected, source/hash manifest frozen; PC tokenizer, semantic leakage and actual training have not run. This is a candidate training package, not a promoted model.

## Composition

The 591 original JSONL records yield **1,000 assistant completion targets** under `training/scripts/train_adapter.py`: 900 training, 100 validation.

| Source family | Training targets | Validation targets | Total |
|---|---:|---:|---:|
| Project-owned, verified LocalPilot seed (24 records) | 20 | 4 | 24 |
| Code generation | 228 | 24 | 252 |
| Code review | 140 | 14 | 154 |
| Software-engineering repair | 14 | 2 | 16 |
| Execute / observe / revise traces | 180 | 20 | 200 |
| Structured tool calling | 158 | 17 | 175 |
| Multi-tool execution | 160 | 19 | 179 |
| **Total** | **900** | **100** | **1,000** |

The 24 project-owned seed records are Tier A. The other 567 source records are licensed Tier B and retain their source repository, license, upstream revision and provenance. The 1,000 targets are **not** 1,000 independent human-reviewed corrections. Source verification establishes traceable provenance and format, not that every target is individually correct.

Selection: stable FNV-1a ordering of `p12-v1:` + record ID, then a bounded subset-sum over the true assistant completion counts for each task type and original source split. Original records and messages are copied without rewriting or paraphrasing. Original family splits from external corpus v1 are retained. Existing Eval-v1 and private 48-case evaluation material is not used as training source.

**Limitations:** This source-derived curriculum strengthens coverage across software engineering and tool use but only contains 24 project-specific examples. P1 has previously seen much of the source corpus, so this is not proof that retraining will improve autonomous self-development. In particular, do not overclaim 250 new self-development examples; those have not been authored or reviewed. Consider further targeted project-grounded source work as a separate future package if P1.2 fails to improve.

## Training specification

- `training/configs/qlora_p12_verified_1000.yaml` starts from the immutable accepted P1 adapter (`training/lineage/package-1.json`), **not** P1.1's candidate and not a fresh LoRA from the GPT-OSS base.
- Same validated native Unsloth/ROCm recipe as the completed 11,112-step P1: rank 8, alpha 16, NF4 double quantization/BF16, AdamW 8-bit, learning rate 0.0001, microbatch 1, gradient accumulation 4, seed 3407, native single-worker compilation, 1,040-token sequence limit and three epochs.
- This smaller dataset requires 675 expected optimizer steps at batch 4 for three epochs. Independent checkpointing is every 225 steps; in-Trainer evaluation is off, as in the proven P1 recovery configuration. No extra pilot or 51-step micro-training.
- Training data and reference snapshots are not mutated. Original P1 remains the production model and direct rollback reference.

## Required sequence (no pilot)

1. Checkout this merged change in the existing Windows/WSL training environment. Run CI, exact tokenizer length checks, schema/source verification and full held-out leakage screening before training. This is the original recipe's preflight, **not a separate training pilot**.
2. Obtain a matching successful guarded dry-run report using `training/scripts/train_adapter.py --dry-run` with the P1.2 config and a report under `training/reports/`. Review report identity and accepted parent adapter hash. Approve the existing config status under the established training gate only after preflight.
3. Run the entire planned 675-step training using the established `training/scripts/start_guarded_adapter.ps1` launcher and `train_adapter.py --train` gated by the matching dry-run report and explicit confirmation. Preserve all checkpoints and completion evidence.
4. Merge/export to a **new** distinct Ollama candidate tag with the existing P1 exporter. Verify token names/layout, tensor readback, exact template/parameters and the candidate digest. Keep production `nestra:20b-p1` untouched.
5. Run fresh identical-seed paired P1 vs P1.2 candidate tests, including the exact previously failing P1.1 reasoning-loop seeds, final-answer completion, research/tool use, coding, autonomy, `/evolve` capability discussion, plus untouched holdout cases with independent review. Preserve failed outputs. Do not promote automatically.

## Guardrails

No `/evolve` or workbench code is to train autonomously without explicit instruction, and no extra rule-based answer filters are permitted. Leave the strict historical scaffold unchanged. Do not copy P1.1 artifacts into P1.2 parent lineage. No held-out prompts, private scorecards or blind-review answers should enter training records.
