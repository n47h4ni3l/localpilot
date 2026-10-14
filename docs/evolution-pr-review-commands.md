# Owner-directed Evolution PR revision

An ordinary GitHub PR Conversation comment can request a revision of an existing
LocalPilot-managed, unmerged candidate **without making CI fail**.

The repository owner writes a **new top-level PR comment** beginning at the
first character with the exact marker:

```text
NESTRA-REVISE:
The proposed baseline is not yet implemented. Add a reproducible factuality
benchmark with labelled reference answers, score the actual responses, and
report sample size and uncertainty. Preserve safety and existing tests.
```

Everything after `NESTRA-REVISE:` in that *same comment* becomes bounded
owner feedback (maximum 4,000 characters). The codeword is a command prefix,
**not a secret/password**. Author identity is verified against the repository
owner returned by GitHub; arbitrary commenters, quoted markers, empty bodies,
edited PR descriptions and unrelated issue threads are not triggers.

## What the next Evolution invocation does

After guarded main sync, candidate/CI reconciliation, checkpoint recovery and
failed-CI repair, Evolution checks its existing pushed, green, open, managed
candidate PRs. A new owner revision request can start a **new durable
cycle on the existing PR branch**. The original objective remains in place;
the comment augments acceptance criteria and guides scoped implementation.
The current source-grounding gate, candidate file limitations, reviewer test
protections, resource limits and independent local acceptance review still
apply. LocalPilot pushes new commits **to the same branch** and does not
create another PR.

The private `localpilot-data/owner-pr-revisions.json` receipt ledger stores
the PR number, comment ID, branch, task, feedback and cycle state. A comment
ID is consumed only once. A further owner-authored comment with the marker
is needed for another requested revision after the previous one completes.
If Evolution pauses, the existing compact checkpoint and worktree recovery
take priority over accepting a second revision command.

Before editing, LocalPilot requires the existing PR to be open, same-repository,
targeting trusted main, on a locally tracked candidate branch, with passing
GitHub checks. It refuses dirty candidate worktrees or any mismatch between
local candidate HEAD and the remote PR branch. A comment alone never
authorizes merging, promotion, modifying stable main, disabling policies,
or reading secrets. No new GUI controls are introduced.

This feature does not mean CI passing proves correctness. The owner must
still review the revised PR, and new changes must pass CI. The next Evolution
invocation must run on the owner's PC; adding a GitHub comment does not cause
GitHub Actions to perform the local model-inference cycle.
