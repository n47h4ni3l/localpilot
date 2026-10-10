# Guide-first scaffold (experimental)

This branch introduces a **reversible alternative** to the old rigid LocalPilot
postvalidation loop. Nestra P1 remains the reasoning agent. LocalPilot should
supply research tools, private on-device context, useful memory and occasional
verification notes rather than repeatedly attempting to regrade/rewrite answers
or withholding a substantive explanation for an unverified line citation.

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
| Answer quality checks | Multiple corrective inference calls and possible complete withholding | A single model-authored draft is preserved; heuristic issues are advisory |
| Missing source | Mandatory acquisition/recovery, may replace answer with generic fallback | Still attempts explicitly required live reads; if unavailable, retains useful draft and scopes live claims |
| Synthesis | Long final source-authority postcondition prompt | Short, supportive synthesis prompt |
| Numeric/citation uncertainty | Reject if unsatisfied | Keep answer, append concise verification note for flagged specifics |
| Research access | Controlled by the existing tool registry and research budget | Same tool registry and execution budget, more discretionary answer completion |
| Security/actions | Risk-class policy at execution | **Identical** hard execution policy, destructive confirmation, protected data/owner no-web limits |

The guide-first code does **not** automatically correct a wrong numerical
constant. It also cannot guarantee that a draft with a verification note is
accurate. These known limitations are exactly why the mode is experimental.
The verification note is not a citation and cannot substitute for an actual
source. For high-stakes claims the owner should rely on checked evidence.

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
