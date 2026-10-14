# Nestra P1.2: fresh online 1,000-record curriculum

Status: **historical P1.2 corpus preparation record**. The owner reported
on 14 October 2026 that P1.2 was successfully trained **from P1.1** and
promoted on the evening of **13 October 2026**. This operational report has
now been reconciled with [verified immutable lineage receipts](lineage/README.md)
from the training machine. The frozen preparation config still names the earlier P1
adapter as its proposed parent; **do not infer actual run lineage from it**.

The owner's 12 October request replaced an earlier mostly repeated P1 corpus
with fresh online examples, including a small concrete self-improvement mix.
The corpus selection, source receipts and tokenizer figures below describe
that preparation; they do not prove the parent adapter used by the eventual
P1.2 run or the quality of the promoted model.

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
delivers a corrected implementation and a concise explanation; measured
execution receipts remain in metadata.
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

## Preparation boundaries and later promotion report

This checked-in preparation recipe preserved the **original P1** base/adapter
parent, LoRA, optimizer, learning rate, BF16, micro-batch, accumulation,
three epochs and native compile settings; only the corpus identity and
counts-derived checkpoint cadence changed at preparation time. These are
historical configuration facts, **not proof of the actual P1.2 run parent**.
The owner has since specified **P1.1** as the actual parent.

The training configuration records that the exact target-PC dry-run passed
30 checks on 12 October 2026 and that the owner authorized training.
The owner separately reports that P1.2 training and promotion completed
on 13 October. Neither the original preparation file nor the CI corpus
checks authenticate an output adapter, Ollama digest or production alias.

Source, schema, assertion and tokenizer checks are integrity evidence; they
do not provide independent semantic approval, complete decontamination or
proof of improved model behavior. **Do not retroactively edit the frozen
training inputs or fabricate hashes to reconcile this mismatch.** Capture
the actual P1.1 parent manifest/adapter identity, P1.2 run configuration,
checkpoint/completion receipts, output SHA-256 hashes, Ollama model digest,
promotion decision and rollback target as a new immutable lineage record.
Compare promoted P1.2 against P1.1 with failures retained. Historical P1
and P1.1 model evidence must remain available for audit/rollback.
