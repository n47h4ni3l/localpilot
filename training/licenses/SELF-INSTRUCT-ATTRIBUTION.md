# Third-party attribution — Self-Instruct human seed examples

This directory includes four source examples extracted from the human-authored
seed task collection published in:

**Self-Instruct: Aligning Language Models with Self-Generated Instructions**
Yizhong Wang et al., 2022. Original code and seed tasks:
https://github.com/yizhongw/self-instruct

Exact upstream source revision:
`0b26ccaa415992100fa32df62d41b994cf928e23`
Source path: `data/seed_tasks.jsonl`
Source IDs: `seed_task_92`, `seed_task_70`, `seed_task_37`,
`seed_task_19`.

These examples are reproduced in
`training/sources/nestra_p1_selfdev_external_candidates.jsonl`.

**Changes from the original:** Four records were extracted from the original
human-written seed pool; each instruction plus its non-empty input was
converted into a user message separated by two newlines, and each original
output was converted into an assistant message. Local record IDs, split
labels, dataset metadata, quality/review flags and explicit source identifiers
were added. The upstream text was not independently authored by LocalPilot.
The exact original prompt and output are retained for review.

**License:** Apache License, Version 2.0. A verbatim copy of the upstream
project's `LICENSE` at the revision above is included here as
`self-instruct-Apache-2.0-LICENSE.txt` and governs these imported examples.
The rest of LocalPilot retains its own licence; see root `LICENSE.md`
third-party-material exceptions. The upstream revision does not include a
separate NOTICE file. This attribution file is LocalPilot's own added
documentation and is not represented as an upstream NOTICE.

**Review status:** The four examples were accepted at the previous
independent-review revision, but the revised two-file corpus is pending
fresh digest-bound confirmation. Inclusion in the proposed corpus does not
authorize running training or replacing the deployed P1 adapter.
