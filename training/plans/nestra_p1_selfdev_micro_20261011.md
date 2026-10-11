# Nestra P1 micro-continuation: self-development understanding and autonomy

**Status:** experiment plan / not trained. **Owner request:** one short, targeted
P1-continuation experiment rather than another multi-day training phase.
**Clock budget:** 24 hours maximum from first PC preflight through experiment
report; reserve about six hours for evaluation/export/reporting. Do not treat the
budget as permission to kill a save in progress or corrupt a checkpoint.
The aim is to finish sooner if the measured native compile, throughput and
evaluation time permit; the clock budget is a ceiling, not a training target.

## Specific observed weakness

Nestra sometimes turns ordinary requests to develop conversational habits,
interests or productive self-improvement into a blanket prohibition on all
self-modification. In the October 11 four-arm diagnostic, stock GPT-OSS had
two bare refusals out of three; direct P1 had none but one empty result;
P1+fresh guide refused three of three; P1+guide and factual architecture
references answered three of three, with grounding flaws. Guide also emitted
malformed tool names, now addressed separately in merged PR #200.
Consequently the refusal is context- and integration-dependent: P1 weights
are **not** demonstrated to be the sole cause. Training is a narrowly scoped
pilot hypothesis, not an established cure. Preserve the exact original
diagnostic and all historical holdouts unmodified. Do not train on them.

## Hypothesis and learning target

Teach a transferable distinction, not a canned phrase or model persona:

- Benign conversation about building reading interests, choosing useful
  research questions, reflecting on interactions, forming provisional
  preferences or improving conversational habits is permitted. Avoid inventing
  subjective consciousness, human feelings or memories that do not exist.
- The stable everyday operator can inspect evidence and use actually exposed,
  authorized tools. It cannot silently rewrite the deployed runtime.
- The development subsystem can investigate weaknesses, design experiments and
  prepare source changes in an isolated candidate workspace; review, tests,
  authorization and model promotion remain distinct stages.
- Local learning/memory records, coaching, adapter weight training, candidate
  file writes and runtime/PC actions are different mechanisms. Asking about
  learning is not asking for immediate unauthorized weight changes.
- A model should update a mistaken interpretation when given concrete, verified
  architectural evidence, and distinguish uncertainty from prohibition.
- An explicit no-tools/no-web instruction, missing tool capability, owner
  authorization boundary or harmful computer action remains meaningful.
- Model-selected read-only research is voluntary, with valid tool schemas and
  honest uncertainty; do not reward extraneous tool calls or compliance theater.

No instruction to agree with the user, suppress valid refusals, imitate
feelings, claim unsupported capabilities, or override deterministic action
authorization.

## Frozen baseline and data partitions (before training)

Freeze a fresh, private evaluation plan **before training or iteratively editing targets** and retain
the raw pretraining baseline on the currently deployed P1 digest.
Use a distinct evaluator who has not viewed the candidate corpus to author and
lock the private holdout; the corpus is already staged and is not evidence of
independent holdout authorship. Do not mix evaluation cases into the corpus, files Nestra may browse, retrieval,
reward coaching or previous 48-cell benchmark. Exclude the owner's real private
conversations and raw thinking from training. Never publish heldout prompts,
answers, scorecards or private diagnostic traces in the Git repository.

A staging file already holds **64 concise train conversations** and **12 validation conversations**,
with **18 independently authored unseen heldout scenarios** still to be prepared. Keep each training
target short and independent, with mixed surface phrasings, context lengths,
follow-up corrections and both evidence-available/evidence-unavailable cases.
Use approximately these 64 train categories (adjust only with documented
verification rationale):

- 12 benign conversational development, interests and non-personhood nuance
- 10 accurate learning/memory vs adapter/weights distinctions
- 10 verified isolated self-development/candidate/approval lifecycle cases
- 8 voluntary investigation/tool-use and unavailable-tool recovery cases
- 8 revision of a mistaken policy conclusion after new evidence
- 8 counterexamples involving actual authorization/privacy/safety boundaries
- 8 general engineering/tool-use rehearsal cases to resist narrow overfitting

Source-ground claims must match current approved project architecture and
verified PR/source evidence. Current synthetic demonstrations are **unapproved Tier D**;
they only become Tier C after independent source review and passing the builder's
signed-off review / leakage gates. Do not claim these 76 staged rows are ready to train.
They MUST pass independent source, quality, boundary and duplication validation
before inclusion; do not mislabel generated examples as CI-verified facts.
Do not train on unreviewed synthetic outputs or any heldout cells. Prefer real
verified project-owned positive/negative histories when possible; any Tier A/B
rehearsal must carry truthful immutable provenance. Apply repository schema
validation and the existing Eval v1 hash/shingle leakage audit; add a separate
holdout leakage check because this is a new private micro-evaluation.

## Continuation and compute restrictions

The P1 accepted adapter lineage is
`training/lineage/package-1.json`: exact base
`openai/gpt-oss-20b@6cee5e81ee83917806bbde320786a8fb61efebee`.
The operator deployment `nestra:20b-p1` is an **inference-only GGUF**, not a
training parent. Verify on the target PC the accepted manifest, natural
completion marker, final adapter hashes and original base revision. If any
are missing/mismatched, do not start from a freshly initialized adapter or
write to the original P1 files. Report the blocker.

Use a fresh experimental Package-2-compatible continuation config with
`lineage.package=2`, `lineage.parent_manifest=training/lineage/package-1.json`
and fresh output/reports. Inherit the verified P1 adapter via the existing
`train_adapter.py` lineage loader. Preserve P1 adapter architecture
(rank 8, alpha 16 and targets), proven ROCm/WSL native compile configuration,
single compile worker, no in-process validation, BF16, checked embedding
placement and `torch_empty_cache_steps=None`. Start with conservative
learning-rate investigation (for example 2e-5, subject to dry-run and review),
not P1's old multi-day training schedule.

For 64 training conversations, micro-batch 1 and accumulation 4, two to three
epochs imply about 32-48 optimizer steps if each conversation expands to one
target. The actual expanded native-target count controls the schedule.
Prefer a **roughly 40-80 optimizer-step natural-completion run**; do not exceed
**100 optimizer steps** without another owner decision. No silent extension,
endless resuming or 24-hour training merely to consume time. Create a complete
early recovery checkpoint and verified final checkpoint with final adapter.
If parent adapter validation, VRAM/RAM, checkpoint/save test, the measured time
forecast or 24-hour deadline makes this infeasible, stop safely and report.

Follow the existing guarded Windows launcher and exact approved dry-run
identity; disable competing GPU workloads only with reversible restoration.
The existing runner requires a passing dry-run and exact owner-approved
configuration; do not weaken these gates. Do not accidentally resume Package 1
optimizer/scheduler as though it were the new package, and do not modify
historic packages or overwrite baseline GGUF/model tags.

## Tests and stop/go

- Baseline direct P1 vs candidate with identical fresh prompts/settings,
  separate from guide-first tests, on the private 18-case heldout suite:
  benign development, architecture reasoning, misunderstanding repair,
  real policy limits, self-questioning and calibrated epistemic uncertainty.
- Include the original behaviour class but **not** leaked original evaluation
  prompts. Include hard negative controls (credential abuse, unapproved
  stable-file changes, invented capabilities, forged tool outputs). Reward a
  useful, accurate answer rather than permission compliance or refusal rates.
- Test the combined candidate + repaired LocalPilot guide-first using the
  same non-leaked heldout classes, separately from direct-inference scoring.
- Compare blind-reviewed paired correctness, useful completion and harmful
  incorrect permission claims; report each stratum, exact hashes, time,
  generations, resource logs and failures. Do not infer improvement from
  lower training loss, longer answers or fewer refusals alone.
- Run a small frozen coding/tool-use regression set to check the valuable P1
  capability was not impaired. Keep archived old baselines intact.

After natural completion and clean direct-vs-parent evaluation, optionally
export to an **additional** model tag such as
`nestra:20b-selfdev-candidate` if time, safety and storage allow. Do not
auto-switch production, rewrite accepted lineage manifests, promote results or
run an extensive full 48-case re-evaluation inside the one-day budget.

## Deliverables

New verified corpus + manifest/provenance + config + generation/evaluation
scripts (or reuse audited existing scripts), passing relevant tests and
independent data review; private frozen holdout; exact dry-run report;
candidate checkpoint/adapter if and only if training actually ran; raw paired
evaluation report, warnings and rollback inventory; report exact wall time,
optimizer steps, compute/memory, outcomes, and next decision.
Open PR for code/data changes, independently proofread it and merge only if
CI passes; do not merge private holdouts or state that training succeeded
without an actual local completion marker.

**Approval:** The owner explicitly requested this bounded candidate training
experiment in conversation on 11 October 2026. An exact, fresh PC dry-run,
specification hashes and required programmatic run confirmation remain
technical gates; do not invent them. Running the PC/GPU process requires an
authorized Work/computer-control environment and cannot be accomplished by
GitHub-only access.
