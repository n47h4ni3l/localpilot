# PR 205 initial corpus audit, 12 October 2026

This report concerns the frozen PR head `d27ddc57f9405a9b76095ed411a2929ca274c896`, before the owner requested a fresh-source rebuild. Its corpus SHA-256 is `1bdcf065bf0c04d48378449302fb21c4b0b6a8eae4ee2719f8bd0ba14f9bd406`. Verdict: **request changes to the content; preserve the preparation infrastructure**.

## Confirmed content defects

1. Selected record `ext-v1-nemotron-swe-v2-dd4e27d355b6521b` says Flask's debugger reloader is unsupported on Windows and removes `debug=True` to address “Run tests on Windows in CI.” This is an unsupported diagnosis and does not demonstrate a CI repair. Werkzeug 3.1.3's [reloader implementation](https://github.com/pallets/werkzeug/blob/3.1.3/src/werkzeug/_reloader.py) explicitly handles Windows entry points. The record is at the frozen `training/datasets/external_corpus_v1.jsonl:1208` and is selected at `training/manifests/p12_selection.json:78`.
2. `ext-v1-open-code-instruct-cbcda74732249419` has an incorrect minimum-stack sample. Executing its inspected target on the sample operations returns `[3,1,2,2]`; the prompt claims `[3,1,2,5]`. Its metadata still says `average_test_score=1.0`. The frozen source line is 5818, and the selection line is 391. A recorded perfect upstream score does not establish consistency between the prompt, target and sample.
3. Minimum-stack variants `cbcda74732249419` (train) and `9f7d34d4ec87533b` (validation) teach the same operation sequence and auxiliary-stack algorithm. The `262ed0eff42a4f9a` / `683c31a8b17621e4` pair repeats this problem across splits too. Their different prompt-derived family hashes evade the builder's family check. A lexical screen found 71 candidate pairs, including 23 crossing splits; these are candidate counts, not 71 independently confirmed duplicates.
4. All 16 selected agentless SWE prompts ask for `<think>` and `<solution>` blocks, while none of their targets includes `<solution>`. Removing hidden reasoning from a target without reconciling the visible formatting request leaves contradictory instruction supervision.

## Curriculum weaknesses

The 50 new cases are advisory questions about behavior rather than executed demonstrations of the intended skills. For example, `lp-p12-curated-003` asks the assistant to answer a completed calculation without giving the calculation, and its target tells it to state the result instead of supplying a result. The general principle is reasonable; the record is weak supervision for actually completing a task.

Exactly 835 of the 900 training records are unchanged rows from P1's external training corpus, with the same training split. They expand to 1,539 of 1,604 targets (95.95%). The 45 new training cases are 2.81% of targets and 1,539 of 214,155 labeled assistant tokens (0.72%). These are corpus measurements, not a causal prediction of training influence. P1's pinned recovery configuration uses that external corpus, and its recorded 11,112 optimizer steps agree with the original expansion and recipe.

The original mix contains 760 code records, 150 agentic records, 16 agentless SWE records, 24 local history records and 50 newly authored cases. All 150 agentic records have `recovered_error_count=0`. Ordinary conversation and useful recovery after a failed tool operation are not adequately represented as explicit curriculum components.

## Checks and scope

- Exact source blobs and LF-byte materialization verified; 1,000 records / 900 train / 100 validation.
- Unchanged P1.2 unit tests: 3 passed. Dataset validator: zero errors and warnings. Builder's frozen Eval v1 hash check passed without reading evaluation answers.
- Actual native expansion: 1,604 train / 165 validation; 1,203 optimizer steps under the original three-epoch, effective-batch-four recipe.
- CPU-only tokenizer check used the locally cached pinned Unsloth snapshot, revision `093fba6992ef5a7152481afec0bdfca1ac486998`, and Transformers 4.56.2. All 1,769 expanded examples were prefix-aligned, nonempty and within 1,040 tokens; maximum 1,029. No model weights were loaded.
- All 760 code targets parsed as Python; no exact normalized answer or unmodified AST duplicates were found. These checks do not establish semantic correctness.
- 130 local provenance anchors existed or matched their recorded Git blobs. Generic source references do not substantiate every target sentence.
- Read all 50 new cases, 24 local seeds and 16 selected SWE targets; performed systematic checks across all records and targeted external spot checks. This is not an independent semantic approval of all 1,000 records.

The PR was open, draft, unchanged at the audited head, with green CI and no submitted GitHub reviews. No private holdout was inspected, and no training, export, promotion, model change or production checkout edit was performed during the audit.

The subsequent owner request authorizes replacing the candidate corpus with fresh online examples, including a small concrete self-improvement component. It does not authorize training or promotion. This report remains evidence about the prior candidate.
