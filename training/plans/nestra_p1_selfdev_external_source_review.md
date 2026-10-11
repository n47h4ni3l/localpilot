# External data selection: Self-Instruct, WildChat and Super-NaturalInstructions

Status: **review-only supplement**, not training data. No change to the 64
existing train rows, 12 validation rows, parent P1 adapter or 24-hour pilot
budget until full independent review and a new frozen corpus identity.

## Source reality versus search-result claims

- **Self-Instruct** (Yizhong Wang et al.; includes Ai2 researchers) is a
  method for generating diverse instruction examples with a seed set,
  model-generated tasks, invalid/near-duplicate filtering, and *separate*
  instruction fine-tuning. It is NOT a mechanism by which the currently
  deployed model rewrites itself during dialogue. Its official research
  repository is `yizhongw/self-instruct`, not a separately validated Ai2
  "self-modification" product. The research authors found quality issues
  in 46% of a reviewed 200-sample subset of generated examples.
- **Super-NaturalInstructions** is a broad instruction-generalisation
  collection. Its task definitions are Apache-2.0; underlying *instances*
  have task-specific licences. Broad generalisation is not equivalent to
  Nestra self-development. Do not import instance examples without their
  exact provenance and licence.
- **WildChat** comprises human conversations and may contain personal or
  sensitive details despite documented de-identification. Its ODC-BY
  *database* licence does not resolve all rights and privacy risks of
  individual message contents. Do not use WildChat in this micro-pilot.

Primary references:
- https://github.com/yizhongw/self-instruct
- https://github.com/yizhongw/self-instruct/blob/0b26ccaa415992100fa32df62d41b994cf928e23/LICENSE
- https://arxiv.org/abs/2212.10560
- https://github.com/allenai/natural-instructions
- https://huggingface.co/datasets/allenai/WildChat

## Four actual upstream examples staged for independent review

`training/sources/nestra_p1_selfdev_external_candidates.jsonl` holds
**verbatim prompt/answer demonstrations** from the human-authored
`data/seed_tasks.jsonl` in Self-Instruct, frozen at upstream commit
`0b26ccaa415992100fa32df62d41b994cf928e23` (Apache-2.0).
These are genuine upstream seed-task examples, NOT four uninspected
GPT-3 generated examples, and are not yet independently verified.
Each retains the original `seed_task_...` ID, source revision,
licence reference and the source instruction/input/output.

| Source ID | Transferable skill | Why review matters |
| --- | --- | --- |
| `seed_task_92` | Identify and explain a false factual assertion | Ensure factual correction rather than reflexively contradicting |
| `seed_task_70` | Form a bounded, actionable improvement plan | Sports-coaching domain is only an analogy to learning |
| `seed_task_37` | Make the specific requested code change | Confirm the example remains correct and does not imply uncontrolled live modification |
| `seed_task_19` | Explain a technical distinction to a nonexpert | Confirm terminology and resist vague assertions |

The **default decision is to keep the current 64 training examples**.
A reviewer may select **at most four** of the source examples to replace
the weakest existing generic engineering/rehearsal examples; do not
increase training count, change the protected negative controls, or change
the separate frozen validation/heldout sets. If any upstream example
is poor, generic, low-quality or irrelevant after review, exclude it.

These four additional rows are explicitly `Tier D`,
`review_required`, and `candidate_only`. The existing
`build_nestra_selfdev_micro.py` ignores this supplemental file and
must continue to do so until explicitly extended to require:
- exact source revision, content and per-record ID verification;
- independent quality/source/licence review;
- candidate-to-review cryptographic binding;
- all Eval v1 and private new heldout overlap checks;
- a new frozen manifest/dataset digest on any accepted replacement.

**Never relabel the external rows as project-owned synthetic** or forge
local commit provenance to satisfy the existing builder. Adapt the
builder if an external replacement is genuinely accepted.

## What to borrow from the Self-Instruct method

Use the 64 currently reviewed-or-reviewable Nestra-specific seeds as
the source material. Optionally generate a *very small pool* of
paraphrased challenge questions about (1) ordinary conversational
growth versus adapter training, (2) actual isolated candidate work,
(3) new contradictory evidence and revised conclusions, and
(4) permission boundaries, including counterexamples.

Filter new proposals for near-duplicate prompts, invented architecture,
unsupported blanket prohibitions, misleading code-safety claims and
unearned agreements. Have an independent reviewer check each kept
example and replace, not expand, the 64-row corpus. Vary the premises,
not merely the surface wording. No live self-training feedback loop or
weight update is implied by synthetic instruction generation.

Training and evaluation still require the target PC and the original
one-day hard budget. Do not use the original four-configuration
diagnostic prompts or the existing 48-cell evaluation as training data.
