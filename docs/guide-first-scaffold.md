# Guide-first scaffold (experimental)

This branch introduces a **reversible alternative** to the old rigid LocalPilot
postvalidation loop. Nestra P1 remains the reasoning agent. LocalPilot supplies
research tools, private on-device context, useful memory and **pre-answer
method guidance** only. It does not inspect, correct, append to or withhold
her final answer based on unverifiable heuristics.

## Activation

In the owner's existing `localpilot.toml` file:

```toml
[agent]
scaffold_mode = "guide_first"
```

The default is still `"strict"` until a new blind A/B trial demonstrates
reliable improvement. To roll back, set `"strict"` or remove the option.
The historical evaluator forcibly pins strict mode, avoiding silent changes to
its primary model-versus-model comparison. Never resume a frozen run under a
different mode or runtime revision.

## Behavior differences

| Concern | Strict comparison mode | Guide-first experiment |
| --- | --- | --- |
| Answer quality checks | Multiple corrective inference calls and possible complete withholding | Nestra's answer is delivered byte-for-byte; no grading, warnings, rewrite or veto |
| Missing source | Mandatory acquisition/recovery, may replace answer with generic fallback | Model decides which permitted research to attempt and communicates its own evidence limits |
| Synthesis | Long final source-authority postcondition prompt | Short, supportive synthesis prompt |
| Numeric/citation uncertainty | Reject if unsatisfied | Teach source verification *before* answering; do not inspect or annotate the output |
| Research access | Controlled by the existing tool registry and research budget | Same permission registry and hard budget; soft budget advises, without a mandatory notebook checkpoint |
| Security/actions | Risk-class policy at execution | **Identical** hard execution policy, destructive confirmation, protected data/owner no-web limits |

Guide-first **never examines, rewrites, scores, appends warnings to, or withholds** a nonempty
model-authored final response. Quality guidance goes into Nestra's context
*before* she works, not into the answer afterward. It uses a standalone
research-mentor system prompt **instead of** the strict examiner's prompt:
merely stacking a friendly instruction on top of mandatory old rules would
leave the interference in place. As a result, unsupported
numeric constants and high-confidence mistakes can appear in answers: this is
an intentional tradeoff to measure fairly, not evidence they are correct.
For consequential matters use primary-source verification and keep actual
tool/action permissions at the execution boundary.

Guide-first does not force source acquisition through the strict evaluator's
`evidence_requirements` list and bypasses the deterministic fast-answer route
for free-form conversation. Nestra can choose whether a read-only source
would materially improve the answer, bounded by unchanged tool authority
and hard budgets.

The tool-result success classifier now treats quoted phrases such as
`Tool error:` and `No matches found.` in a legitimate repository file as
data, not as a failed tool call. A genuine empty repository search remains
classified as not having acquired useful source evidence.

## Evaluation gate

Before switching the default, compare the *same frozen P1 digest* in:

1. P1 direct, no LocalPilot
2. P1 + LocalPilot, strict scaffold
3. P1 + LocalPilot, guide-first scaffold

Use 8-12 independently authored prompts (including new repository, numeric,
historic/live evidence, unfamiliar scripts, web/no-web and prompt-injection
situations), two or three repeats per arm and matched effective generation
settings. Grade delivered answers independently and retain the raw first
draft, tool evidence, warning notes and final answer. Reuse the original ten
withheld test cases only as **diagnostic regressions**, never as independent
new benchmark samples. Investigate unsupported high-confidence claims as
carefully as unnecessary withholding. Actual inference must run locally
against the installed P1 model and cannot be inferred from GitHub CI.

Keep P1 weights, original 48- and 684-cell reports, action permissions,
self-development lineage and quality-reward isolation unchanged.
