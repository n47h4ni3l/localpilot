# Nestra P1.2 — locked 1,000-record capability curriculum

Status: **corpus prepared for verification; no training or export started**.

## Intent

Continue the **accepted original P1 adapter**, not the experimental P1.1
candidate. The requested goal is preserved ordinary conversation, correct
code, useful evidence/tool handling, capable self-development and reliably
delivered final answers. The 51-step P1.1 candidate is retained separately
for failure analysis, not promoted or used as the training parent.

This curriculum emphasizes **completed outcomes** over repeated internal
reasoning, confidence slogans or permission mantras. It does not include
generated chain-of-thought supervision. It is deliberately not a duplicate of
the failed P1.1 micro-training corpus.

## Contents

- `training/manifests/p12_selection.json`: locked source-ID selection,
  exact Git blob identities, data composition and expected expansion counts.
- `training/datasets/p12_curated_cases.jsonl`: 50 distinct source-informed,
  author-written examples about completing answers, evidence use,
  self-development, engineering and trainable lineage. These have **not**
  received independent expert grading, and are explicitly marked
  `synthetic_validated`, not `human_verified`.
- `training/scripts/build_p12_corpus.py`: deterministic materializer. It
  rejects changed source blobs, missing/duplicated IDs, duplicate prompts,
  cross-split source families, empty terminal assistant answers, mismatched
  source/split/expansion counts and any collision with the **frozen Eval v1**
  hashed manifest. The builder never reads private P1.1 holdout answers.
- `training/configs/qlora_p12_curated_1000.yaml`: proposed, P1-parented
  QLoRA training configuration. Same pinned base/backend, native compile,
  LoRA, optimizer, LR, batch, BF16, epochs and max sequence length as the
  accepted P1 recovery recipe. The new corpus/output identity and
  final-only checkpoint cadence are the intentional differences.

| Source | Train records | Validation records | Total |
|---|---:|---:|---:|
| Verified LocalPilot project-history seed | 20 | 4 | 24 |
| New source-informed completion/self-development cases | 45 | 5 | 50 |
| OpenCodeInstruct Python tasks with 10/10 recorded verifier score | 683 | 77 | 760 |
| Nemotron agentic tool-use, error-free and completed traces | 136 | 14 | 150 |
| Nemotron SWE issue/repair examples | 16 | 0 | 16 |
| **Total** | **900** | **100** | **1,000** |

Some source entries are externally licensed; their per-record licence and
source provenance are preserved. Automatic filtering and a prior source
verifier score do not constitute a fresh independent semantic audit of every
answer, and cannot establish that the model will not regress.

The 1,000 records expand to **1,604 train and 165 validation assistant
targets** under LocalPilot's native-conversation expander. Three epochs at
effective batch size four yield **1,203 optimizer steps**. One checkpoint
is scheduled at step 1,203. There are **no early or periodic checkpoints**.

## Build and check locally

From `E:\LLM_HOME\src\localpilot` after this branch is available:

```powershell
python training/scripts/build_p12_corpus.py
python training/scripts/build_p12_corpus.py --check
python training/scripts/validate_dataset.py training/datasets/p12_corpus_1000.jsonl
```

This creates the locked materialized dataset
`training/datasets/p12_corpus_1000.jsonl` and a SHA-256-bound trainer
manifest `training/manifests/p12_corpus_manifest.json`. The builder refuses
to overwrite an existing different artifact. CI performs the same checks
on a fresh checkout.

The training specification remains `proposed` until a **full, real target-PC
dry-run** validates GPU/environment, model and parent adapter hashes,
tokenization and held-out leakage using the established P1 runner. No
preliminary training pilot is requested.

## Training and evaluation boundaries

1. Preserve `nestra:20b-p1`, its accepted Package-1 lineage/checkpoints,
   the original GPT-OSS base, and the P1.1 failed candidate unchanged.
2. Dry-run the new configuration using the **existing** P1 training
   orchestration. Do not alter optimizer/LoRA settings or restart from P1.1.
3. Train **once**, with only the natural final checkpoint. If interrupted,
   restart that experiment from the beginning rather than treating an
   incomplete adapter as finished.
4. Use the proven adapter merge, GGUF conversion/quantization and **new
   candidate Ollama tag**. Never overwrite P1 as an export shortcut.
5. Compare frozen P1 against the new candidate with identical prompt
   framing, decoding, seeds, context and final-turn budgets, including
   previously observed non-final reasoning loops plus **fresh non-training**
   cases. Separate model weaknesses from harness/template/runtime effects.
   Retain empty/timeout/length outcomes. Promotion is a separate owner
   decision after evidence review.

No absolute non-regression claim is possible from corpus composition alone.
In particular, training on the original verified external families does
not prove freedom from overlap with any **private** holdout whose hashes
are unavailable to this builder.
