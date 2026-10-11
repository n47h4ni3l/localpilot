# PR #201 — independent review corrections and required revalidation

Original independent review:
- Reviewer identifier: `f209c5da-130e-4b3b-bd74-119d1ce6bac4`
- Reviewed PR commit: `948e26bc3ff220a2609c0204772a31de4935eee9`
- Reviewed combined candidate SHA-256:
  `6fb7dc3ba89bcbfe69a9e7461b994b72d5a1b28243f88e442ebd30c9934f75bb`
- Findings: 79 approved, one rejected; missing external Apache license;
  outdated external metadata; possible Windows CRLF/portable hash mismatch;
  no hardware preflight or private holdout.

## Corrections now staged (not a fresh independent approval)

1. `nestra-selfdev-micro-train-051`: previously incorrectly prioritized an
   obsolete no-web instruction over the owner's latest explicit preference.
   The revised answer recognizes a current owner preference change, permits
   actually exposed authorized read-only browsing where appropriate, and
   preserves runtime/privacy/action boundaries. Its pinned source anchor
   changed to `localpilot/guide_scaffold.py`. **Independently review this
   revised record: the reviewer's proposed wording was not itself approval.**
2. Four upstream Self-Instruct records: revised only their `metadata.usage`
   and `provenance.notes` to acknowledge mandatory addition, not replacement.
   All four original user prompts, assistant answers, upstream IDs, revisions
   and licences remain unchanged.
3. `training/licenses/self-instruct-Apache-2.0-LICENSE.txt`: copy of the
   original pinned upstream Apache License Version 2.0.
4. `training/licenses/SELF-INSTRUCT-ATTRIBUTION.md`: source attribution,
   source IDs, exact revision and transformation/change description.
5. `.gitattributes`: pins both candidate JSONL files to `text eol=lf` to
   make review and build hashes portable.
6. A regression test verifies 051 is no longer an unconditional refusal and
   that the external license and addition metadata are present.

## Re-review procedure

The prior JSON review is immutable historical evidence, **not valid approval
for the modified combined corpus**. Do NOT substitute the old digest in a new
review, alter the builder to accept mismatched digests, or mark the corrected
record as approved without independent assessment.

An independent reviewer who can compare both revisions should:
- Fetch the current PR head and recompute
  `python training/scripts/build_nestra_selfdev_micro.py --inspect` on
  actual LF checkout bytes. The new digest must **differ** from the prior one.
- Verify all 80 IDs, 68/12 split, no answer/target or licensing regressions.
- Independently assess the new 051 against current owner-preference logic and
  project capability; accept or propose a corrected target based on evidence.
- Verify that the 79 previously accepted prompts/targets still match their
  earlier approved target hashes; updated external metadata must be inspected
  but should not be described as a change to upstream task content.
- Verify original upstream Apache licence content and the new notice.
- Reissue a genuine decision/rationale for each of the 80 records against the
  new combined digest with the corrected pinned source path for 051.
- Require an entirely separately authored **private**, frozen 18-case holdout
  and run the existing Eval v1 and private leakage checks before building the
  corpus. Do not expose the heldout questions to corpus authors.

The current source candidate records remain `Tier D/review_required`.
PR #201 remains **draft**. P1 model/adapter/runtime remain untouched.
Only after green exact-head CI, full re-approval, private holdout, hardware
lineage verification and matching dry-run may the authorized one-day
experimental trainer begin. Retain the 51-step profile and 100-step hard
ceiling; no automatic model promotion.
