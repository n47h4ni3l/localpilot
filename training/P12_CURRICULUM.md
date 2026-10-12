# Nestra P1.2: fresh online 1,000-record curriculum

Status: **owner-requested candidate rebuild; independent semantic review
pending; no training, export or promotion performed**.

The accepted original P1 adapter remains the proposed continuation parent.
The owner's 12 October request replaces the earlier mostly repeated P1 corpus
with fresh online examples, including a small concrete self-improvement mix.

## Curriculum

| Capability | Records | Train | Validation |
|---|---:|---:|---:|
| Code with executed source assertions | 200 | 180 | 20 |
| Concrete self-improvement: failure, scoped repair, checks | 50 | 45 | 5 |
| Completed native tool trajectories | 150 | 135 | 15 |
| Everyday substantive conversation | 100 | 90 | 10 |
| Instruction-following chat, reasoning off | 100 | 90 | 10 |
| General conversation and practical tasks | 100 | 90 | 10 |
| Completed text rewriting | 125 | 113 | 12 |
| Source-grounded summaries | 125 | 112 | 13 |
| Constrained completed responses | 50 | 45 | 5 |
| **Total** | **1,000** | **900** | **100** |

Actual native expansion: **1577 train / 172
validation assistant targets**, with **1185
optimizer steps** at the original three epochs and effective batch four.
The counts-derived recipe schedules one final checkpoint and no pilot.

Sources are BigCode's execution-filtered StarCoder2 self-alignment data;
the newly generated Apache-licensed SmolTalk subsets; Hugging Face everyday
conversations; NVIDIA instruction-following chat's reasoning-off release;
and ToolACE. Each is downloaded from a pinned upstream training release.
Only the stated SmolTalk components are imported; the heterogeneous full
mixture is not assigned a blanket license.

## Concrete self-improvement

Fifty source-derived cases show actual confined CPU test evidence. One
operator expression inside an implementation is changed; every test assertion
is preserved. The mutant must fail and the restored implementation must pass,
with all distinct assertions actually visited during execution. The prompt
contains the objective, candidate, checks and observed failure. The answer
delivers a corrected implementation and a concise measured outcome.
These adaptations are Tier C, `synthetic_validated`, with semantic review
pending. They are not verbatim upstream conversations or human-verified traces.
They train concrete progress without adding hidden chain-of-thought targets.

## Source and selection evidence

The original selected rows are retained in
`sources/p12_fresh_source_records.jsonl`. Dataset cards, full licenses,
attribution and transformation notices are bundled under `licenses/p12_fresh/`.
The acquisition lock identifies upstream revisions, immutable URLs, complete
file hashes and the bounded NVIDIA prefix hash. Per-record row identities,
source hashes and transformations are checked again during materialization.

`scripts/curate_p12_fresh.py` selects deterministically from verified snapshots.
It screens blank/incomplete trajectories, unsupported native tool syntax,
hidden-reasoning markers, placeholders, unsupported action claims, numeric
additions in text transformations, source-test failures, excessive context
and close lexical/code duplicates against P1 and other selected examples.
Known content defects discovered during inspection are rejected explicitly
in `manifests/p12_fresh_content_rejections.json`.

Every expanded example is rendered through the pinned cached Unsloth
tokenizer without model weights. Over-1,040-token examples are rejected,
with no truncation. The frozen Eval v1 check uses its one-way manifest;
no evaluation answers or private holdout are read.

The local acquisition workspace preserves downloaded sources, rejected IDs
and reasons, and content-addressed intermediate phases. The lock and curation
report bind those assets by hash. Intermediate attempts are not approved
datasets. The initial PR audit is retained in
`reviews/p12_initial_corpus_audit_20261012.md`; prior sources and Git history
remain unchanged.

## Reproduce materialization

```powershell
python training/scripts/build_p12_corpus.py
python training/scripts/build_p12_corpus.py --check
python training/scripts/validate_dataset.py training/datasets/p12_corpus_1000.jsonl
python -m unittest discover -s training/tests -p 'test_p12*.py' -v
```

The materializer refuses to overwrite an existing different corpus or trainer
manifest. The selection binds the new candidate bytes and source bundle.
Curation needs PyArrow and the tokenizer; normal materialization and CI need
neither downloads nor model weights.

## Remaining review and training boundaries

The original P1 base, adapter parent, LoRA, optimizer, learning rate, BF16,
micro-batch, accumulation, three epochs and native compile settings are
preserved. Only corpus identity and counts-derived checkpoint cadence change.
P1, P1.1, their evidence and the production alias remain unchanged.

Source, schema, assertion and tokenizer checks are integrity evidence. They
are not independent semantic approval, complete decontamination or proof of
improved model behavior. Remaining gates are an independent content review
bound to the final corpus digest, separate private-holdout/similarity work,
and an exact target-PC preflight. Any subsequently authorized training run
needs paired P1/candidate conversation, final-answer, coding, tool and
self-development evaluation with failures retained. Export and promotion
remain separate owner decisions.
