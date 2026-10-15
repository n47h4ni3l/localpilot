# Incremental continuation of an Evolution PR

An open managed candidate PR is **not finished merely because CI is green**.
After checkpoint recovery, local candidate repair, CI repair, and any pending
owner-authenticated `NESTRA-REVISE:` request, an idle Evolution invocation may
continue the same open PR with one bounded implementation step.

## Behaviour

- Verify that the candidate was previously pushed and tracked in LocalPilot's
  durable development history, its GitHub PR is open and green, its head is in
  this repository against trusted main, and the local worktree is clean and at
  precisely the same SHA as the remote PR head.
- Recover the existing branch/worktree. Do not create a replacement PR or
  merge the existing one.
- Give the developer a **small step**, not the entire long-running acceptance
  contract. Preserve the original research question, measurement contract,
  previous result, and outstanding larger goal. Focus on functional code,
  reproducible fixtures, or targeted tests, not cosmetic changes.
- Preserve all normal source-grounding, path confinement, independent review,
  resource limits, reviewer-controlled tests, and human-only merge rules.
- Record a private receipt in `localpilot-data/autonomous-pr-iterations.json`
  **before** inference. An unchanged PR SHA can be attempted only once.
  Even with changed commits, stop after six autonomous iterations per branch;
  the owner can still issue a new `NESTRA-REVISE:` instruction through the
  separate existing review workflow.
- A failed or paused local candidate is handled by the existing recovery
  pipeline *before* selecting another automatic step; a failing CI run goes
  through the existing CI repair path.

This is a bounded opportunity to learn by doing, **not an automatic acceptance
criterion** for a complete capability. Neither an intermediate commit nor
passing CI proves improved factuality, and the model is never auto-promoted.
The original objective and human merge decision remain authoritative.

## Scope and limitations

The six-step bound controls wasted inference but can stop a productive project
before it is finished. The owner can extend it using an authenticated PR
review instruction; there is intentionally no silent reset of the receipt.
This patch does not create an independent semantic evaluator or promise that
a small implementation step will be useful. Live local-model behaviour must
still be observed on the target PC. It also does not rewrite or repair an
existing PR as part of installation.
