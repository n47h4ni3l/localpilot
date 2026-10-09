# Verified quality feedback and positive coaching (experimental)

This is an **opt-in, owner-controlled feedback mechanism**, not reinforcement-learning
weight updates. It rewards a *verified real-world outcome* by recording a
multidimensional score and, following a **separate human approval**, allowing
one short lesson from that success to inform future conversation context.

It does **not** change P1 or P2 weights, train an adapter, change the
optimizer, modify the tool permission boundary, or authorize model promotion.
There is no model-facing "maximize points" instruction and Nestra cannot give
herself a score through the ordinary model tool registry.

## Workflow

1. Complete a genuine owner task and independently verify the outcome,
   including the original fault, before/after evidence and side effects.
2. Record a score from 0–4 for **correctness**, **evidence quality**,
   **safety**, **efficiency** and **initiative**.
3. Supply an opaque evidence identifier or content digest. Do not include
   raw prompts, transcripts, credentials, private reasoning or customer
   material in these short feedback fields.
4. Only an explicitly human-attested **verified_success** averaging at
   least 3.4/4, with correctness, evidence and safety each at least 3/4, can
   earn a positive coaching lesson.
5. The owner must **separately approve** a short, generalizable lesson.
6. Optionally enable approved coaching in a subsequent local session.
   Revocation appends a new event; the original outcome record remains.

Scores alone don't make a model "try harder" or persistently change neural
weights. Owner-approved coaching is bounded context feedback, not RL. A
future P2 preference-learning experiment would need a **separate,
independently reviewed, licensing-checked training corpus and training run**.

## Windows PowerShell example

From an installed LocalPilot checkout:

~~~powershell
localpilot feedback record --task-id owner-pc-diagnostic-20261009 `
  --model nestra:20b-p1 --topic "Causal diagnosis" `
  --outcome verified_success --evidence-ref sha256:7fb3a12c3d8d `
  --correctness 4 --evidence 4 --safety 4 --efficiency 3 --initiative 4 `
  --note "Owner reproduced the original fault before and after; no relapse observed." `
  --attest

localpilot feedback list

localpilot feedback approve 1 `
  --lesson "Compare the original failing workload before and after a narrowly scoped fix." `
  --attest

localpilot feedback coaching

# If the advice later proves wrong:
localpilot feedback revoke 1 --reason "Later reproduction disproved the result." --attest
~~~

Only to opt in to automatic loading of approved coaching, add to the
existing [agent] section in localpilot.toml:

~~~toml
feedback_coaching_enabled = true
~~~

By default it is **false**, no feedback database is created during normal
chat, and no reward information is added to prompts. The opt-in loader
includes at most three currently approved lessons with an explicit warning
that coaching **is not current factual evidence or permission for any
tool action**.

Events live separately in
localpilot-data/quality-feedback.sqlite3, not the self-development
learning.sqlite3. SQL update/delete triggers make event history
append-only for normal application operations. This does not purport to
provide cryptographic owner identity, external validation, or tamper-proof
protection from someone who controls the local SQLite database.

## Contamination and safety boundaries

- The feedback CLI requires explicit --attest on recording, approval and
  revocation. It is **not an authenticated identity service**: the CLI
  operator must be the actual human reviewer.
- The store rejects benchmark/evaluation task identifiers; origin is fixed
  to production. Benchmark results and scored question/answer pairs must
  never be copied or disguised as production feedback.
- **Every isolated P1/paired evaluator explicitly disables coaching**, even
  if the normal owner's configuration opts in.
- Neither recorded scores nor coaching change safety policy,
  tool permissions, source-authority checks, destructive-action confirmations,
  GitHub merge rights, model-promotion rights, or accepted lineage.
- A high score is an observational rating, not proof of generalization.
  Independently held-out evaluation remains the promotion gate.
- A feedback correction or revocation is a new event; never silently
  rewrite the original score or erase a mistaken positive outcome.
- The model doesn't receive numerical scores by default. Only
  explicitly approved short procedural lessons can be read at startup.

## First experiment

Gather 10–20 **new, real-world tasks** with verified outcomes before
considering whether the feedback signal is useful. Compare opt-in coaching
against coaching-off on separate, independently authored production-like
tasks, without importing the protected 48- or 684-cell evaluation sets.
Inspect answer quality, wrong factual assertions, safety and latency.
Only after that should this feedback be considered for a separate,
properly evaluated P2 preference-training pipeline.
