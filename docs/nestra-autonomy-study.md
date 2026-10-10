# Nestra autonomy and the next controlled study

Decision recorded 2026-10-10. Learned correctness belongs to Nestra's P1
weights and the existing approved training/learning processes. Guidance can
offer methods before an answer; it cannot enforce a factual conclusion.

## Production boundary

The guide is optional advice about discriminating evidence, scope and
conditions, identity versus behavior/performance, and tracing side effects
and failure paths. It includes no task-specific correction or required answer
format. The existing optional Groq mentor remains Nestra's choice, under its
existing owner opt-in and privacy/authorization boundaries.

No answer scoring, gating, rejection, rewriting, quality classifier, required
verification loop, substitute model, training, or production memory/config
change is introduced. The guide delivery path preserves a model-authored
draft byte-for-byte, including unsupported claims, unsafe advice, refusals
and generic greetings. Tool execution continues through the existing safety
policy. Preserving an answer is not a claim that it is correct or safe advice.

## Historical evidence and control

The supplied `report.md`, `nine-delivered-answers.md` and
`nestra-p1-nine-response-evidence.zip` describe nine attempted cells, not nine
inferences: 17 internal model calls, seven nonempty deliveries, two empty
deliveries, and no independent blind numeric grading. The qualitative report
is diagnostic evidence, not an accuracy/utility winner. Raw drafts and failed
cells must remain intact. Guide-first preserved factual errors and unsafe
advice. Strict suppressed a script draft and misrouted the standards task;
the six-call ceiling also contributed to its failed delivery. Direct exhausted
a 3072-token call without a terminal answer.

The frozen strict control is **7f7e77ded9805c1231f44152afa87b9cc253e301**, the
exact installed revision in those nine cells. It is not the earlier 48-cell
runtime. Its prompt, classifiers, verifier, recovery and known bugs remain
unchanged. Archive that commit for strict; never repair it for this comparison.
The new guide runs from its separately recorded committed revision.

## Study requirements

Eight independently authored fresh tasks × two repetitions × three arms =
48 cells. Supplied-source interpretation and live-source research are separate
strata, reported separately. Match complete question bytes and P1 digest,
checkpoint lineage, seed schedule, reasoning, temperature, context and
effective per-call budgets. Record requested settings as well as overrides.
The study override is part of the protocol, not production behavior.

System envelopes differ by design. For live research, direct P1 uses a minimal
tool-message loop with the same permitted public readers, without LocalPilot
guidance or output review. Tool schema exposure chosen by the strict/guide
runtime is observed, never forced to fix historical routing. Supplied-source
tasks have no tools in any arm. The direct loop ends on the first non-tool
response, even if empty; no retry, replacement or hidden-reasoning substitution.

Preserve every planned cell, failures/empty deliveries, raw stream chunks,
drafts, messages, effective parameters, tool proposals/results, source URLs,
coverage and hashes, latency and token counts. Separate per-call and per-cell
resource ceilings from answer quality. No online grading influences output.
Groq and all alternate model providers are disabled throughout this study.
Optional mentoring needs a separate later protocol.

Freeze source packets, rubrics, arm order, runtime archives, inference
settings and evaluator revision before launch. Validate the protocol and safe
Windows runtime first. Blank shuffled offline scorecards omit arm labels,
prompts from the system, tools, timing and runtime errors; retain a separate
private arm key and diagnostic traces. An independent reviewer grades
accuracy, utility and unsafe advice with source references after collection.
Intrinsic answer wording can still suggest an arm; ask for an arm guess to
measure imperfect blinding. Never grade the executing author's own outputs
as independent evidence. Missing grades remain missing, not zero.

Do not silently resume failed cells or overwrite evidence. This runner allows
one collection launch per study directory and does not implement resume. If
interrupted, retain its incomplete denominator and partial drafts; a later
continuation needs a separately reviewed protocol, never an automatic retry.
No source/evidence is admitted to learning or training.
Do not infer learned-weight improvement or promotion from this system study.

## Pitfalls that must stay visible

Nominal `think=high` is not proof that every internal call used it. A cell is
not a model call. Equal ceilings do not imply equal consumed tokens or work.
Sources supplied by the harness are not model-acquired provenance. No match
in an isolated snapshot is not a production defect. A nonempty fallback is
not task success. An empty length-limited answer is not proof of incorrect
hidden reasoning. Failed live reads need their own provenance, and changing
web pages require per-cell captures for offline grading. Do not fix strict
bugs or change P1 weights to make this comparison look better.

## Prepared protocol and operation

`training/scripts/run_autonomy_study.py --plan` lists eight new questions:
two supplied diagnostic/code records, two frozen primary-source interpretation
tasks (RFC 9111 and SQLite foreign keys), and four live-source tasks (SQLite
WAL, Python thread offload, PowerShell policy scope, RFC 9457). This gives 24
supplied-source cells and 24 live-source cells. Each matched trio uses identical
question bytes and seed; arm positions differ by at most one occurrence over
the 16 trios. Task rubrics are never put in an inference request.

Every inference is streaming, high reasoning, temperature 0.1, context 32768,
and a maximum of 8192 generated tokens (thinking included). Other sampling
parameters are explicit. A cell has six model calls, 24576 total generated
tokens, ten public reader calls, 180 seconds per call and 600 seconds per cell.
The next call's token ceiling is the smaller of 8192 and the remaining cell
allocation; this same rule applies to every arm. Missing terminal usage reserves
the full call ceiling and prevents another inference in that cell. An HTTP
read timeout bounds a silent stream; elapsed checks close a progressing stream.
On a child timeout the parent records its exact PID and stops dispatch without
force-killing it. Inspect that PID before any separately authorized cleanup.

The source preparation path paginates complete primary documents and checks
coverage and stable hashes. It selects whole relevant sections for the two
packets; live reference captures are kept offline and not supplied to P1.
Per-cell public reads retain exact URLs, returned text, coverage, source hashes,
time and errors. Requested inference settings and effective overrides are both
retained. Strict's internal routing and schema exposure remain observable
differences; identical declared tool availability cannot guarantee identical
runtime tool choices. This is a system comparison, not an isolated estimate
of the guide text's causal effect.

After committing the evaluator, use:

```powershell
python training/scripts/run_autonomy_study.py --prepare --guide-revision <full-commit> --output <new-study-directory>
python training/scripts/run_autonomy_study.py --readiness
python training/scripts/run_autonomy_study.py --run --output <study-directory> --protocol-sha256 <freeze-hash> --review <review-record.json> --production-root E:/LLM_HOME/src/localpilot --model-root E:/LLM_HOME/ollama-models
```

The review record contains `protocol_sha256`, `reviewer`, and
`implementation_and_protocol_approved: true`. It records a code/protocol review,
never model-answer grading. On 2026-10-10 the owner authorized the implementing
agent's thorough second review and merge after CI; independent arm-blind model
answer grading remains separate and required.

Preparation is read-only toward production and performs no inference. The
freeze checks exact evaluator/client environment, both runtime archives,
sources, tasks and order before launch and between cells. Readiness checks
Windows/P1 availability, Background Worker state, WSL and resident alternate
models. The runner cannot stop a worker, disable a task, change config, unload
a model, or start training. Collection records production config/memory,
manifest/GGUF and accessible checkpoint hashes before/after and does not mark
completion after identity drift. Inaccessible checkpoints are reported as
unavailable for fresh audit, not as verified.

Offline review is a separate executable:

```powershell
python training/scripts/review_autonomy_study.py cards --study <study-directory> --output <new-blind-bundle> --key <separate-private-key.json>
python training/scripts/review_autonomy_study.py summary --study <study-directory> --key <private-key.json> --grades <independent-completed-cards.jsonl> --output <new-summary.json>
```

The review key binds original cell/answer hashes; changed answers, duplicate
cards, missing reviewer independence or out-of-scale grades cannot silently
enter an offline summary. Empty/error deliveries remain collected cells;
unattempted cells keep their planned denominator and are not passed off as
model failures. Unknown token usage and missing grades remain explicit. Source
and live results are separate, and two repetitions are not independent tasks.

The implementation review found and repaired Windows archive line-ending
conversion, unbalanced ordering, incomplete-source risks, unmetered failed-call
usage, identity drift after failed inference, and missing/duplicate offline
grade integrity. Regression tests include an isolated worker with a fake
transport; that is harness validation, not a P1 result. Actual study outcomes
must be reported only after collection and independent offline review.
