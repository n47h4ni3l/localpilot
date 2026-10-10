# Nestra inference and tool delivery diagnosis — 10 October 2026

This is an execution diagnosis of the completed autonomy study, not a new
quality review or evidence of learned improvement. The original 48 cells,
protocol, sources, strict runtime and model data remain untouched.

Repair: [PR #199](https://github.com/n47h4ni3l/localpilot/pull/199).
[CI checks for the final PR head](https://github.com/n47h4ni3l/localpilot/pull/199/checks)
and the PR's validation record provide the exact tested revision and run links.

## Evidence identity and reproducibility

- Main inspected: `b66c7239021d24045d65f988bdfeaacd5a415ab8` (PR #198).
- Raw ZIP SHA-256: `b0ff60430dc34959caa54b85da5ae8ede20e118fbf34037c8ab0164427cb3302`.
- Protocol SHA-256: `0de4a8a8d09456e1c025f2a205d35df4a607befc52acb86ce97e65ac4493f754`.
- Strict control: `7f7e77ded9805c1231f44152afa87b9cc253e301`, preserved exactly.
- P1 digest: `870386dd9cbfe625ab110af128ffbf6216565e5c9d2d603a070e8f6228f203d9`.
- Collection: 48 cells, 188 calls, 89,200 generated tokens, 578,447 prompt
  tokens (repeated contexts counted repeatedly), 116 public-reader executions.
- The supplied archive's protocol hash and all 256 listed runtime/source file
  hashes were checked. Its original and working ZIP hashes still match.

`training/scripts/diagnose_autonomy_traces.py` accepts the original ZIP or an
extracted directory. It reads the archived data without importing archived
code, making model calls, changing answers, assigning scores or updating
learning. `--output` creates a new JSON file outside the original evidence;
an existing output is rejected. The machine-readable companion
`inference-tool-diagnosis-20261010.json` indexes every call's raw chunk filename,
requested/effective settings, total token counts, output character counts,
tool proposals/schema exposure, error, phase and draft/delivery hashes.

```powershell
python training/scripts/diagnose_autonomy_traces.py <raw-evidence.zip> --output <new-diagnosis.json>
```

Ollama reports total generated tokens, including reasoning. It does **not**
provide a separate exact reasoning/final token partition here. Character
counts are labelled as character counts; they are not estimates of token use.
Runtime audit `num_predict` reflects the runtime's requested cap. The wrapper
overrode it; use each raw turn's effective cap when assessing the study.

## Confirmed trace findings

| Stratum | Arm | Cells | Nonempty deliveries | Error cells | Calls |
| --- | --- | ---: | ---: | ---: | ---: |
| Supplied source | direct | 8 | 7 | 0 | 8 |
| Supplied source | strict | 8 | 6 | 2 | 33 |
| Supplied source | guide_first | 8 | 8 | 0 | 9 |
| Live source | direct | 8 | 2 | 6 | 46 |
| Live source | strict | 8 | 2 | 6 | 48 |
| Live source | guide_first | 8 | 3 | 5 | 44 |

1. **Nineteen exceptions bind on calls, not generated tokens.** All are
   `StudyBudgetExceeded` after six completed calls. Those cells consumed only
   230–2,642 of their 24,576 generated-token allocation. Eighteen sixth calls
   requested tools rather than emitting content. No stream exception, silent
   stream stall, timeout or missing terminal usage occurs in these 188 calls.
   This does not prove those runtime conditions can never occur.

2. **Much of the apparent looping is ordinary pagination.** For example,
   `autonomy-v1-live-problem-details--r1--direct` makes six 2,000-character
   reads at offsets 0, 2000, 4000, 6000, 8000 and 10000: 264 tokens total.
   `autonomy-v1-live-thread-offload--r1--direct` follows the same pattern:
   272 tokens total. These are distinct model-chosen reads, not duplicated
   executions or token-heavy thought loops. The wrapper has no seventh call
   for another read or an answer; production's configured research bounds
   are a separate envelope. The schema permits larger excerpts and offsets;
   a schema default does not override the model's explicit 2,000-character
   choice. No automatic enlargement, mandatory research or source selection
   is added by this repair.

3. **Two calls exhaust generation entirely before final text.**
   `autonomy-v1-cache-directives--r1--direct/call-1` records `length`, 8,192
   generated tokens, 42,076 reasoning characters, zero content and zero
   tools. Its direct loop ends on that first non-tool response by protocol.
   It is an empty delivery without a cell execution exception.
   `autonomy-v1-preview-scope--r1--guide_first/call-1` records `length`,
   8,192 tokens and 40,055 reasoning characters. Its second, high-reasoning
   call produces a 4,501-character final answer, delivered byte-for-byte.
   No assertion about hidden reasoning correctness follows from either trace.

4. **Frozen strict loses drafts to its historical orchestration.**
   `autonomy-v1-live-thread-offload--r1--strict/call-6` produces 3,108 content
   characters with `stop`, then `model_evidence_acquisition_retry` requests
   unavailable SystemSense-watch/local-Library evidence. The seventh call
   is rejected. The Python wording was routed without exposed tool schemas;
   this historical routing also produced invented tool names. Across six
   strict cells, 20 proposals use unregistered names, including `web_search`,
   `browser.open` and `repo_browser.search`. Strict also records three
   authority-correction and two final-correction calls. Its two supplied
   failures retain earlier drafts in the raw evidence. These are not new
   guide-first answer gates, and the archived control is not repaired.

5. **Reader failures and source availability are distinct.** One executed
   read gets HTTP 403: the Stack Overflow URL in
   `autonomy-v1-live-policy-scope--r2--direct/call-6`. Other recorded executed
   readers complete. A missing isolated local reader, unregistered tool,
   cached fragment retry or incomplete excerpt does not prove that a public
   source or production runtime is unavailable. The diagnostic companion
   preserves the failed read's actual provenance separately.

6. **Successful guide delivery is preserved.** The 11 nonempty successful
   guide study deliveries match the final model call's content byte-for-byte.
   Nonempty includes any model-authored greeting/refusal and implies neither
   accuracy nor usefulness.

## Repair and regression evidence

Inspection of current main found an additional, reproducible delivery defect
on the existing bounded generation continuation: after an empty high-reasoning
final pass, guide-first still applied strict's generic-reset classifier to
the continuation. A completed model-authored greeting was discarded and
replaced with a no-final marker. The seven new recovery tests reproduced two
failures before the fix, including after the one-tool hard ceiling.

The runtime change bypasses that classifier for guide-first continuation
text, then uses the existing exact-draft delivery path. Strict retains its
historical behavior. The existing continuation count, context safety margin,
generation caps and rendering reasoning setting are preserved. There is no
new answer inspection, correction, rewriting, quality gate, mandatory tool
step, tool-name remapping, authorization change or provider substitution.

This continuation-specific greeting defect was **not observed as the cause
of a frozen study cell**. It is a code-path defect proven by regression tests
motivated by the observed reasoning-only exhaustion. The frozen 48-study
totals remain 28 nonempty and 20 empty.

Recovery regressions cover exact whitespace/refusal/unsupported-draft
preservation, empty high-reasoning operator and final passes, an exhausted
tool budget, no extra tool execution, transient reasoning cleanup and a
second generation exhaustion stopping after one continuation. Diagnostic
regressions retain empty and exhausted cells/drafts, distinguish requested
from effective caps, keep missing usage unknown, reject a changed protocol
and block paths outside the supplied evidence directory.

## Live sanity and preservation

Two fresh, unscored diagnostic probes used the same P1 digest with Groq and
all alternate providers disabled. They used current runtime per-call caps,
high reasoning, fresh local state and an artificial one-tool ceiling. They
are **not** matched reruns of the 48-study protocol or quality comparisons.

- Supplied function review: the first call reached its 3,072-token cap in
  reasoning; a second high-reasoning synthesis call used 318 tokens and
  delivered 1,261 characters byte-for-byte.
- Optional RFC reader: one public read, two high-reasoning calls (45 and 754
  tokens), and 2,304 final characters delivered byte-for-byte at the tool
  ceiling. No additional reader or external model executed.

Four before/after call identity checks retained the frozen digest.
Configuration, learning-memory database/WAL, quality-feedback database and
Library reading files were hash-checked unchanged during the probes.
No weight/checkpoint, training, reward label or production setting change
was requested. Checkpoint tensors were not freshly audited.
The canonical Background Worker stopped through its own verified-PID stop
request and was restored enabled/Running with a live recorded owner PID;
its exported task definition matched the original. No force termination
was used. Normal background activity can resume after restoration.

## Validation and remaining limits

The focused runtime/research/recovery suites passed 97 tests; the combined
new diagnosis/recovery checks passed 14. Full local suites and CI results are
recorded with the linked repair PR. The author performs a second complete
diff audit before merge; these checks are not independent model grading.

The directory supplied for the independent review contains the original
collection and `blind-review.zip` with all 48 accuracy/utility/reviewer fields
unfilled. A completed `nestra-48-independent-review.zip` was not present.
Execution diagnosis does not require or invent those grades.

A future, separately frozen evaluation may allow more calls or explicitly
separate research and answer allocations. Such a protocol must keep failed
attempts, settings and work consumption visible. Raising the historical
ceiling or silently continuing its failed cells is not this repair. The
current findings do not establish an answer-quality winner or promotion.
