# Nestra's hosted AI mentor — optional, remote, and guidance-only

Nestra remains the only local reasoning agent. A remote model runs on a
provider's machines, not on the owner's GPU. Nestra normally solves problems
herself, including unrestricted public web research when useful. If she reaches
a specific impasse despite her own investigation, she may choose to ask the
hosted model for a second opinion. It is never an automatically required step.
She then evaluates its non-authoritative advice and decides what to do.
There is no second local model, local weight download, browser scripting,
public message-board post, model fine-tuning, answer rewriting or remote
tool authorization.

## Default: free-tier GroqCloud

Sign up at https://console.groq.com/ and create an API key. Free-plan
quotas and hosted model availability can change. The currently documented
free limits for GPT-OSS 120B include 1,000 requests/day and 200,000 tokens/day,
with separate rate limits. This is a hosted 120B model, not ChatGPT/GPT-6.

Store the key as the "GROQ_API_KEY" environment variable for the LocalPilot
runtime. Do **not** put credentials in localpilot.toml or GitHub.

Edit the owner's existing localpilot.toml:

    [agent]
    scaffold_mode = "guide_first"

    [mentor]
    enabled = true
    provider = "groq"
    model = "openai/gpt-oss-120b"
    max_requests_per_session = 3

Default config has mentor.enabled = false; guide-first is opt-in and strict
remains the benchmark control. Only enable mentor after the guide-only test
is recorded so extra inference does not contaminate that comparison.

## Alternative: free OpenRouter models

OpenRouter offers a free hosted model variant. Create an API key at
https://openrouter.ai/ and store it only in the process environment as
"OPENROUTER_API_KEY". Set:

    [mentor]
    enabled = true
    provider = "openrouter"
    model = "openai/gpt-oss-120b:free"
    max_requests_per_session = 3

The integration **rejects** non-":free" OpenRouter models so switching the
provider cannot accidentally select a paid one. Free accounts are usually
limited to 50 requests/day, and provider capacity may be unavailable.

The previous OpenAI Responses API option remains supported as a separately
billed provider, but is not necessary for hosted advice.

    [mentor]
    enabled = true
    provider = "openai"
    model = "gpt-5.5"

This optional path requires an "OPENAI_API_KEY" with its own billing.
A ChatGPT subscription does not provide API credits.

## Privacy and control

- The consultation accepts an abstract question (12-2,000 characters) and
  a self-assessed impasse (24-600 characters) describing approaches she
  considered and the unresolved issue. Only the abstract question is sent
  remotely; the impasse stays local, is not logged verbatim and does not
  certify that no other local solution exists. She can simply continue
  working or answer without consulting a mentor.
- There is no blanket internet prohibition on Nestra. Normal public research
  tools remain available in guide-first conversations, including hypotheticals.
  A **specific owner instruction** not to use the web is still respected.
- Nestra does not attach her chat history, memory, private source files,
  SystemSense results, tool result transcripts or access to local tools.
- Obvious secrets, e-mail addresses, raw code blocks and local Windows user
  paths are blocked, but this **is not complete private-data detection**.
  Keep every outgoing question abstract and safe to disclose.
- A bounded number of requests (1-10; default 3) is allowed per running
  LocalPilot process; provider-side free quotas also apply. This is *not*
  an exact financial budget on optional paid providers.
- No provider receives credentials other than the relevant API key via
  the corresponding HTTPS Authorization header; keys are never placed in
  prompts, repository contents or durable audit events.
- The known endpoints are fixed, network requests are time bounded, remote
  HTTP redirects are rejected, responses are size-limited and non-authoritative.
  No remote model receives function tools.
- External advice is voluntary, never mandatory, and Nestra retains authorship
  of her answer. Owner instructions and protected system boundaries still apply.
- Remote advice isn't automatically stored in long-term learning, promoted
  as verified evidence or used as a grading oracle.

## Evaluation / rollout

First merge and locally test guide-first PR #196 independently, while keeping
its original strict control. Use identical P1 checkpoint, prompts and budgets
in direct / strict / guide-first arms. Only after recording that comparison
should external mentorship be evaluated as an **additional** fourth arm,
measuring benefit, latency, privacy and quota usage. Unit tests only use
mocked network responses; they cannot guarantee provider account eligibility
or prove answer-quality improvement.

The free service is a remote programmatic chat interface, not a bypass of a
website's interactive UI or an attempt to automate ChatGPT.com sessions.
No local inference installation or additional model weights are involved.
