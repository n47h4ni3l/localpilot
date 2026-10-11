# Nestra P1.2 corpus — preparation in progress

**Not training-ready.** This branch currently contains 500 source-preserving examples in five 100-record shards. The intended target is 1,000 unique records, split 900 training and 100 validation. Do not begin training from this incomplete branch.

Source: the existing repository-controlled `training/datasets/external_corpus_v1.jsonl` (source-verified corpus, tracked blob SHA `23fe83a13a130335ab869577e544a581837087bd`), and the 20 training-split project-owned examples from `training/datasets/corpus_v1_seed.jsonl`. Examples retain their original messages, source, license, metadata, split and provenance. No synthetic or rewritten model answers were introduced.

The initial selection policy favored existing provenance, completed assistant answers, verified coding tests, successful bounded tool execution, distinct problem/family identifiers, and higher-ranked relevance. Candidate evaluation failures and private holdout answers were not used as training sources. The initial distribution was *planned*, not fully validated end-to-end.

## Status and requirements

- Five shards committed, each with 100 JSONL records.
- Remaining shards are **not present**. An upload of the next shard was rejected; do not represent the corpus as complete.
- This is a preliminary machine-filtered selection, **not** independent human review or verified absence of semantic leakage.
- Independently inspect accepted examples for correctness, harmful/misleading patterns, incomplete or tool-only answers, final-answer training balance, source licensing and family overlap before approving a complete corpus.
- Generate a full corpus and a hash-bound manifest with 900 train / 100 validation examples, verify with existing project corpus validators and tokenizer-native sequence checks, and run leakage checks against the frozen eval manifests and private holdouts held separately.
- Avoid repeating P1.1's 80-sample distribution, which did not establish reliable generalization. Do not train on private P1.1 failed evaluation prompts.

## Training notes

Use original accepted P1 package-1 adapter as parent, not the unpromoted P1.1 candidate. Keep the historically successful P1 recipe except for approved dataset/schedule changes. No intermediate saves; retain the verified final checkpoint and final adapter. User requested no separate pilot and candidate evaluation after full training.

**No training, candidate export, model promotion, or changes to production have been executed in this branch.**
