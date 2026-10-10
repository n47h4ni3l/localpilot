# Tool-name transport recovery (guide-first only)

LocalPilot's normal operator inference may receive a correctly structured
Ollama tool call whose `function.name` is malformed. In the 11 October
conversation investigation, nine names ended in
`<|channel|>commentary` and none of the intended tool calls executed.
This is a **transport/interface problem**, not grounds to rewrite Nestra's
thinking or final answer.

Guide-first now performs a narrow pre-dispatch resolution:

1. First retain any already-correct registered name unchanged.
2. If the original name ends in exactly one recognized GPT-OSS channel
   suffix, remove only that suffix. Case-only errors and surrounding ASCII
   whitespace are also tolerated when they uniquely identify a tool.
3. The observed legacy `repo_browser.search` spelling maps to
   `search_repository` only with the already-supported repository search
   arguments (required nonempty `query`, optional `path` and
   `max_results`). No input argument names or values are modified.
4. Only a unique registered **read-only** tool that was *actually exposed to
   the model on the current inference* may be recovered. The optional remote
   mentor is excluded. Reversible actions, non-exposed tools, unregistered
   names, fuzzy matches, multiple suffixes and unfamiliar namespaces remain
   unrecovered. A malformed request does not acquire a permission.
5. The recovered call travels through the existing normal registry lookup,
   permission check, limits, result handling and audit. The original input
   object is not modified. A bounded audit event records only original tool
   name, resolved name and recovery mechanism, not tool arguments or replies.

The original strict comparison control and final-answer logic are unchanged.
Recovery does not change model weights, learning records, memory, tools,
tool schemas, safety policy, available research budgets or final text.

Regression tests use mocked tool streams, test exact arguments, execute an
actual read-only registered stub through `LocalPilotAgent.ask()`, and prove
that forbidden/unadvertised tools and the external mentor are not inferred
from an altered name.

**Limitations:** This is deliberately *not* general fuzzy inference of tool
intent, nor a repair for raw parser exceptions that prevent a structured
call from reaching LocalPilot. It does not solve the original P1 conversational
refusal; that remains a separately diagnosed behaviour.

PR #200 additionally repairs bounded recent-reading progress metadata
available to guide-first responses. Those records are supplied only when
relevant and removed after the turn. This reading correction and tool
transport recovery should be evaluated on their own acceptance criteria,
not represented as a fix for unrelated conversational refusals.
