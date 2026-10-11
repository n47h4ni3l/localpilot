# Nestra P1.2 — 1,000-record source-verified corpus

Candidate corpus, not a claim that 1,000 answers were newly written or individually reviewed. Reuses the previously accepted provenance-checked sources: 24 project-owned Tier A and 976 external Tier B. The purpose is to continue accepted ORIGINAL P1, not failed P1.1.

## Frozen specification

- Exactly 1,000 records: 898 train and 102 validation.
- 3,003 expanded assistant targets: 2,696 train and 307 validation.
- Three epochs / effective batch 4 / 2,022 optimizer steps.
- Parent: accepted P1 adapter from training/lineage/package-1.json, loaded with PEFT trainable adapter by the existing trainer.
- Corpus SHA-256: 8eb2f9300eda78137fc6f1204d34beb0ec2d65059e4e374ec5d42ffa29655801
- Source mix: 24 LocalPilot Tier A; 400 OpenCodeInstruct; 180 xLAM; 190 Nemotron Agentic; 78 CodeAct; 110 SWE-CARE; 18 Nemotron SWE.
- Source material was first verified for licensing, provenance, structural integrity and selection rules upstream. Not all responses are independently graded for correctness.

## Assembly on the user's PC (no training)

In the checkout containing these artifacts, with standard Python:

    python training/scripts/assemble_p12_corpus.py

This concatenates 12 pinned source shards to training/datasets/nestra_p12_curated_1000.jsonl, verifying SHA-256, schema, unique family/prompt, split counts, expanded assistant targets and Eval v1 held-out leakage. It refuses to overwrite an existing file with different bytes.

## Full training: original recipe, no separate pilot

From the same pinned Windows/WSL environment as original P1, perform the mandatory *nontraining dry-run*. This is a data/model/VRAM/lineage check, not a test training batch.

    python training/scripts/train_adapter.py --config training/configs/nestra_p12_curated_1000.yaml --dry-run --report training/reports/nestra_p12_curated_1000_dry_run.json

Only after passing preflight, change config.status from proposed to approved_after_dry_run, with no other spec modification; user already authorized full training. Use the proven resource guard and pinned native-compile environment. Then:

    python training/scripts/train_adapter.py --config training/configs/nestra_p12_curated_1000.yaml --train --dry-run-report training/reports/nestra_p12_curated_1000_dry_run.json --confirm TRAIN_LOCALPILOT_ADAPTER_V1

The original training recipe retains rank 8, LoRA alpha 16, BF16, adamw_8bit, native compile, batch=1, accumulation=4, LR 1e-4, 3 epochs, accepted base identity and same trainable P1 parent adapter. Preflight requires exact baseline source and accepted parent hashes. Stop on any failed preflight.

## Checkpoint policy

**No intermediate checkpoints.** The existing natural-completion contract requires ONE final checkpoint plus adapter and completion marker at optimizer step 2,022. Training that crashes earlier must restart from the beginning; this is intentional. No separate pilot or mid-run saves.

## Candidate evaluation and promotion

Preserve live nestra:20b-p1 and the failed P1.1 for comparison. After exact trained-adapter and export verification, import P1.2 as a separately named Ollama candidate. Compare original P1 vs candidate with the same prompts, seeds, context, generation settings and fresh conversations, including the exact P1.1 blank-answer loops, coding, ordinary dialogue, tool use, and autonomous self-development awareness. Preserve every failure. Check both delivered answer quality and reasoning-loop rate; include a private unseen holdout. Do not promote until serious regressions are excluded and owner approves.

## Limitations

This is a carefully filtered subset of historical source-verified records, not a newly human-graded 1,000-example self-development curriculum. Its bulk supports code and tool capabilities rather than specifically self-development; only the project-owned subset directly teaches LocalPilot architecture. Old Eval v1 hashes are screened at assembly, but private holdout and semantic overlap need independent checking. P1.1's root-cause investigation remains relevant.
