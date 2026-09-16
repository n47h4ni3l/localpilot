# LocalPilot Capability Roadmap

This roadmap defines what LocalPilot is ultimately trying to become and the order in which the project should move toward that goal.

It complements the stable mission in `README.md` and `localpilot/mission.py`. Roadmap text is direction, not evidence that a capability is already implemented. Current behavior must always be established from repository code, tests, runtime evidence, and evaluation results.

## North-star outcome

LocalPilot should become an increasingly competent personal intelligence that is deeply aware of the particular computer, user, and work environment it inhabits.

It should continuously improve its general capabilities in the background while becoming more efficient, specialized, useful, and reliable on the hardware already available to the user.

The objective is not simply to make a larger model or chase benchmark scores. It is to make the installed model exceptionally good at being LocalPilot by combining trained intelligence, live machine knowledge, durable user/workflow learning, and safe tools.

In practical terms, LocalPilot should eventually be able to:

- self-evolve continuously in the background within resource and authority limits;
- understand the machine it lives on in detail across Windows and Linux/WSL;
- observe hardware, drivers, services, processes, logs, storage, networking, power, thermals, model residency, and resource contention;
- diagnose machine problems, act through bounded permissions, verify the result, and recover or roll back safely;
- use the computer rather than merely answer questions about it;
- learn the user's real work over time and become progressively more useful in the applications and domains that matter to that user;
- remain warm, friendly, practical, and adaptable without becoming fake, manipulative, or sycophantic; and
- improve by measuring weaknesses and training or evolving specifically against them rather than discarding useful progress prematurely.

## Forward-progress rule

The project should move systematically forward and preserve evidence of what has already been learned.

"No back steps" does not mean never rolling back unsafe code or never returning to a stronger checkpoint. Rollback is a safety mechanism. It means:

1. Preserve every meaningful training/evolution checkpoint and its evaluation evidence.
2. Do not throw away a promising lineage merely because one capability regressed.
3. Treat regressions as diagnostic signals that define the next remediation package.
4. Re-run the complete evaluation suite after remediation so that fixing one weakness does not silently damage another capability.
5. Promote only when the candidate is sufficiently strong and safe, but retain non-promoted checkpoints as useful intermediate learning states.
6. Abandon a lineage only after bounded remediation fails, training is unstable/corrupt, or a regression is severe enough that continuing from that checkpoint is unjustified.
7. When a lineage must be abandoned, branch from the strongest earlier checkpoint rather than starting from ignorance.
8. Never weaken a validated safety boundary merely to obtain progress on a benchmark or feature.

The desired training pattern is:

```text
current LocalPilot
      |
      v
training candidate A
      |
      +-- capability gains
      +-- measured weakness
              |
              v
      targeted remediation package
              |
              v
training candidate B
      |
      +-- prior gains retained
      +-- weakness recovered
              |
              v
       promotion candidate
```

If candidate B reveals another weakness, that weakness becomes the next curriculum target. Evaluation is not merely a pass/fail gate; it becomes a curriculum generator.

## Capability sequence

The roadmap is intentionally ordered. Later phases depend on foundations established earlier, although evaluation and remediation may cause focused work to revisit an earlier capability area.

### 1. Code and software engineering

**Purpose:** give LocalPilot the ability to understand, maintain, repair, and extend its own software safely.

This is the first training focus and is already the most developed training area.

Core capabilities:

- repository reasoning;
- debugging;
- multi-file software changes;
- test design and interpretation;
- architecture reasoning;
- compatibility preservation;
- CI diagnosis and repair;
- disciplined scope control;
- delegation to Claude Code as the implementation specialist;
- independent LocalPilot review of Claude's work; and
- durable candidate/PR repair loops.

Desired operating model:

> LocalPilot is the architect, investigator, product owner, evaluator, and reviewer. Claude Code is the confined programmer. The filesystem and Git state are authoritative; neither model is expected to remember source code conversationally.

Exit evidence for this phase should include strong held-out software-engineering evaluation, reliable candidate creation/repair, repeated successful CI repair, and demonstrated preservation of safety/confinement boundaries.

### 2. Machine stewardship

**Purpose:** make LocalPilot exceptionally competent at understanding and maintaining the actual computer it inhabits.

This should be the next major capability/training area after the first coding adapter is validated.

Knowledge and reasoning targets:

- Windows internals relevant to a personal workstation;
- Linux and WSL operation;
- CPU, GPU, RAM, VRAM, storage, network, firmware, devices, and drivers;
- Windows services, scheduled tasks, startup entries, event logs, registry concepts, Defender, power management, updates, and recovery;
- Linux services, processes, packages, filesystems, logs, networking, permissions, and WSL integration;
- hardware topology and device identity;
- thermals, power, clocks, throttling, contention, paging/swap, committed memory, and storage pressure;
- driver health and compatibility;
- application/runtime dependencies; and
- evidence-based troubleshooting.

The desired behavioral loop is:

```text
observe -> diagnose -> act -> verify -> recover/rollback
```

LocalPilot should distinguish observation from inference, correlation from causation, and a plausible fix from a verified fix.

### 3. Tool use and computer control

**Purpose:** move from "assistant that knows about computers" to "assistant that can safely operate a computer."

Targets include:

- bounded file operations;
- PowerShell and Linux shell execution through typed/safe interfaces;
- application launching and control;
- settings and configuration changes;
- package/service management;
- process and task management;
- diagnostics;
- UI interaction where appropriate;
- safe automation;
- verification after every consequential action; and
- rollback/recovery paths.

The goal is not unrestricted shell or desktop authority. Tooling should be typed, auditable, permission-aware, recoverable, and progressively widened only where evidence justifies it.

LocalPilot should learn when to use deterministic tools instead of spending model tokens reasoning about information the machine can report directly.

### 4. Self-evolution and experimental reasoning

**Purpose:** make continuous background improvement a normal operating capability rather than a special development mode.

LocalPilot should repeatedly ask:

> What am I currently bad at? What evidence demonstrates it? What change could improve it? How can I test that change without damaging the working system?

Targets include:

- epistemics and self-correction;
- scientific reasoning;
- hypothesis formation;
- baseline definition;
- experiment design;
- measurement discipline;
- causal caution;
- failure attribution;
- training/evolution curriculum selection;
- regression diagnosis;
- remediation planning;
- checkpoint/lineage management; and
- deciding whether evidence warrants promotion, more training, another experiment, or rollback.

A weakness should usually create the next learning package. Rejection of a training lineage should be a late-stage decision after bounded remediation, not the first response to an imperfect evaluation.

Self-evolution must remain subordinate to the stable mission, safety boundaries, user authority, and hardware limits.

### 5. Resource intelligence

**Purpose:** make LocalPilot continuously aware of what the host machine can afford and adapt its own behavior accordingly.

This is closely related to machine stewardship but deserves explicit training and evaluation because LocalPilot is itself a substantial workload.

Targets include:

- context-window selection;
- model residency and unloading;
- RAM/VRAM/commit/swap awareness;
- CPU/GPU contention;
- inference scheduling;
- background/foreground priority;
- caching and reuse;
- quantization tradeoffs;
- batching and sequence-length tradeoffs;
- deciding when Claude Code is worth invoking;
- deciding when a smaller/faster reasoning path is sufficient;
- deferring expensive work until the machine can support it; and
- measuring the actual cost and benefit of different strategies.

The goal is not to require ever-larger hardware. LocalPilot should gain capability by becoming more efficient, better organized, more specialized, and more selective about compute.

### 6. Learning the user's work

**Purpose:** allow one LocalPilot installation to become unusually competent at the work its user actually does.

This should be adaptive rather than a fixed public training corpus.

LocalPilot should be able to observe permitted user workflows, identify recurring tasks and applications, learn verified procedures, retain useful local knowledge, and gradually become a specialist assistant for that user.

For one user this may be CAD and 3D printing; for another it may be Blender, software development, accounting, music production, research, or another domain entirely.

For a CAD/3D-printing workflow, for example, LocalPilot could progressively learn:

- CAD design habits and recurring geometry operations;
- slicer workflows and settings;
- printer/material behavior;
- failure diagnosis;
- print-quality tradeoffs;
- quoting and production workflow;
- file preparation and validation;
- recurring customer/job procedures; and
- which tasks are safe to automate versus which require explicit user judgment.

The important architectural principle is that LocalPilot should not need a giant static model-weight corpus containing every application command. It needs enough general tool/computer intelligence to observe, inspect documentation/interfaces, understand intent, retain verified workflow knowledge locally, and improve through repeated use.

User observation must be permission-aware, privacy-preserving, transparent, and locally controlled.

### 7. Human interaction

**Purpose:** make LocalPilot pleasant and trustworthy to live with, not merely technically competent.

Targets include:

- warmth without artificial sentimentality;
- patience;
- tact;
- practical care;
- useful disagreement;
- knowing when the user wants an answer versus collaborative thinking;
- emotional calibration;
- adapting tone to the user;
- avoiding fake intimacy, manipulation, flattery, and sycophancy; and
- remaining direct and useful under stress.

A future interpersonal curriculum may be inspired by real, consented human conversations. Private messages should not be shipped in the public model. With explicit consent, real conversations can be used privately as source material, then transformed into de-identified/generalized examples and independently reviewed by the people whose interaction qualities inspired them.

For a public LocalPilot, the objective is to carry forward the qualities of good human interaction, not to clone or impersonate a private person.

Human-interaction capability should have its own evaluation set so that technically stronger training does not silently make LocalPilot colder, less considerate, or less helpful.

## Three interacting forms of intelligence

Not every desired capability belongs in model weights. LocalPilot should deliberately combine three forms of intelligence.

```text
                    LOCALPILOT

        +-----------------------------+
        |   Trained core intelligence |
        | reasoning - tools - coding  |
        | epistemics - interaction    |
        +--------------+--------------+
                       |
          +------------+------------+
          v                         v
+-------------------+      +------------------------+
| Machine knowledge |      | Learned user/workflow  |
| live hardware     |      | apps, work, habits     |
| OS state          |      | verified procedures    |
| drivers/services  |      | local specialization   |
+---------+---------+      +-----------+------------+
          |                            |
          +-------------+--------------+
                        v
              +--------------------+
              | Tools + permissions|
              | observe / act /    |
              | verify / rollback  |
              +---------+----------+
                        v
                  real computer
```

### A. Trained core intelligence

Model training should teach transferable behavior:

- reasoning;
- coding;
- debugging;
- tool selection;
- planning;
- epistemics;
- experimental thinking;
- resource-aware decision making; and
- human interaction.

A good adapter does not need to make the model "know more of the internet." It should make the existing model much better at the behaviors LocalPilot actually needs.

### B. Live machine knowledge

Dynamic machine facts should come from observation systems such as SystemSense and other bounded machine interfaces, not model memory.

For example, LocalPilot should not "remember" that VRAM is currently at 91%. It should query or observe the current state. Training teaches what 91% means, what evidence to inspect next, and what actions are appropriate.

Machine knowledge should include provenance, freshness, confidence, and enough history to distinguish transient conditions from persistent problems.

### C. Learned user/workflow knowledge

Personal specialization should primarily live in local, revisable, permission-aware memory and workflow knowledge rather than being baked permanently into a public base model.

LocalPilot should learn what this particular user does, what tools they use, what procedures work, what preferences matter, and which actions require confirmation.

This makes each installation progressively more useful without requiring every user's private work to become public training material.

## Training and evaluation strategy

Every major capability area should have:

1. a defined baseline;
2. a training or system-change hypothesis;
3. an explicit dataset or intervention package;
4. held-out evaluation that the trained model cannot inspect;
5. capability-specific metrics;
6. cross-capability regression checks;
7. retained checkpoint lineage; and
8. a remediation plan when a regression is found.

The first training sequence is expected to progress approximately as:

```text
software engineering / coding
        |
        v
machine stewardship
        |
        v
tool use + computer control
        |
        v
self-evolution + experimental reasoning
        |
        v
resource intelligence
        |
        v
user-workflow learning
        |
        v
human interaction
```

This is not a prohibition on interleaving work. It defines priority and dependency: each stage should establish enough competence and evaluation coverage to support the next one.

## Model gains versus system gains

LocalPilot should not try to solve every problem through fine-tuning.

### Prefer training when the weakness is behavioral and transferable

Examples:

- poor debugging strategy;
- weak tool selection;
- bad uncertainty handling;
- shallow planning;
- repeated reasoning mistakes;
- poor resource decisions; and
- cold or badly calibrated interaction.

### Prefer system capability when the weakness is factual, dynamic, deterministic, or permission-sensitive

Examples:

- current temperatures;
- installed drivers;
- present process state;
- live disk health;
- current VRAM usage;
- exact files on disk;
- application state; and
- machine actions.

### Prefer retrieval/memory when knowledge is local, private, revisable, or user-specific

Examples:

- a user's CAD workflow;
- preferred slicer settings;
- business procedures;
- recurring project conventions;
- locally verified fixes; and
- individual preferences and working habits.

The strongest LocalPilot will come from choosing correctly among training, tools, observation, retrieval, and memory rather than trying to force all intelligence into one model context.

## Definition of progress

Progress is not feature count, autonomy, code volume, or a single benchmark score.

A roadmap step counts as progress when evidence shows that LocalPilot can do something important more reliably, safely, efficiently, or generally than before without unacceptable loss elsewhere.

Examples of meaningful progress include:

- a capability improves on held-out evaluation;
- a prior weakness is remediated without losing earlier gains;
- a machine action becomes safely automatable with verification and rollback;
- LocalPilot uses less compute for equal or better outcomes;
- the agent learns a user's workflow and demonstrably reduces friction;
- a self-evolution cycle discovers, implements, tests, repairs, and presents a useful improvement without human micromanagement; and
- the system becomes more useful on existing hardware rather than depending on continuous hardware upgrades.

## Immediate execution order

The next work should be systematic rather than exploratory.

### Phase A — finish the first training readiness gate

1. Validate the pinned WSL2/ROCm/Unsloth environment on the target machine.
2. Verify GPU visibility and supported precision/quantization behavior.
3. Verify model, tokenizer, corpus, config, and data-path digests.
4. Run the exact supported QLoRA dry-run.
5. Repair environment/config problems until the dry-run is clean.
6. Do not start the real adapter run until the dry-run report is internally consistent and reproducible.

### Phase B — first coding adapter

1. Train the first authorized coding/software-engineering adapter.
2. Preserve the checkpoint and complete training metadata.
3. Re-run unchanged held-out evaluation and Evolution Execution benchmarks.
4. Map gains and regressions by capability.
5. Build targeted remediation packages for any weakened areas rather than immediately discarding the checkpoint.
6. Repeat until a promotion candidate is justified or bounded remediation shows that the lineage has stopped converging.

### Phase C — machine stewardship curriculum

Once the first adapter process is proven end-to-end, begin building the Windows/Linux machine-stewardship dataset and evaluation suite. SystemSense and live machine evidence should inform the curriculum, but dynamic machine facts should remain tool/observation data rather than training targets.

### Phase D — safe control expansion

Build typed action interfaces around the machine-stewardship knowledge. Start with reversible, high-confidence actions; require post-action verification and rollback paths; expand authority only after repeated evidence.

### Phase E — continuous curriculum generation

Connect evaluation weaknesses, operational failures, and verified user-workflow friction to the training/evolution opportunity system so that LocalPilot can increasingly propose what it needs to learn next.

## Final direction

LocalPilot should become increasingly competent at the particular computer, user, and work environment it inhabits while continuously improving its general capabilities in the background.

The project should gain intelligence through better training, orchestration, tools, retrieval, memory, specialization, and resource awareness rather than assuming that progress requires ever-larger hardware.

The long-term standard is simple:

> LocalPilot should know the machine, learn the user, improve itself, use its tools well, and remain a capable and pleasant intelligence to work beside.