# LocalPilot

LocalPilot is a persistent, local-first Windows agent designed to learn, reason, observe its environment, and improve its own software capabilities over time.

It is built around local inference through Ollama, durable memory, evidence-grounded research, passive system awareness, progressive study, and an autonomous software-development loop that prepares isolated candidate improvements for review.

LocalPilot is not a wrapper around a chat model. The model is one component inside a larger persistent agent architecture.

## What LocalPilot is

LocalPilot currently combines:

- a persistent desktop assistant;
- local high-reasoning inference through Ollama;
- evidence-grounded repository, PC, library, and web research;
- durable typed learning with provenance and freshness;
- passive Windows hardware and software awareness through SystemSense;
- progressive background reading and study;
- a persistent autonomous background worker;
- runtime lifecycle supervision and recovery;
- an autonomous evolution orchestrator;
- isolated self-development candidates;
- GitHub-backed testing and review;
- human-controlled promotion of stable code.

Its purpose is cumulative capability growth.

The central design question is not whether LocalPilot can produce convincing answers, but whether it can repeatedly observe a limitation, form a useful hypothesis, test an improvement, retain what it learns, and use that improvement to perform better later.

## Architecture

LocalPilot separates three major roles.

### Operator

The Operator is the everyday LocalPilot that interacts with the owner.

It can:

- maintain persistent desktop conversations;
- reason with local models;
- use read-only and explicitly allowed Windows tools;
- inspect the repository and runtime;
- retrieve relevant durable learning;
- research using local sources and bounded public HTTPS;
- observe SystemSense environmental state;
- ground consequential claims against current evidence.

The Operator is not given unrestricted command execution.

### Developer

The Developer performs bounded autonomous engineering work when the machine is available.

It can:

- inspect LocalPilot's current capability frontier;
- identify candidate limitations;
- form falsifiable improvement hypotheses;
- research possible solutions;
- prepare structured implementation plans;
- create isolated candidate workspaces;
- modify candidate code;
- run non-executing local validation;
- commit and push candidate branches;
- create pull requests.

The Developer does not rewrite the stable runtime in place.

### Candidate

A Candidate is an isolated proposed version of LocalPilot.

Candidate writes are bounded by:

- repository path restrictions;
- protected files and reviewer tests;
- allowed file extensions;
- file-count and file-size limits;
- frontend security rules;
- resource limits;
- candidate workspace confinement.

Executable candidate validation occurs through GitHub Actions rather than by casually running untrusted candidate code on the owner's workstation.

## Evolution

LocalPilot's autonomous evolution system is persistent rather than a one-shot coding prompt.

A typical evolution cycle can:

1. verify the trusted repository state;
2. confirm resource and foreground-use conditions;
3. inspect previous opportunities, experiments, failures, rejections, and capability history;
4. select a queued opportunity or identify a new limiting capability;
5. state a measurable hypothesis;
6. research before implementation;
7. create an isolated candidate;
8. verify the proposed change against the real repository;
9. expose bounded candidate-writing tools;
10. prepare the implementation;
11. validate the candidate locally without executing untrusted candidate code;
12. push the candidate;
13. create a GitHub pull request;
14. rely on CI for executable validation;
15. wait for human promotion.

The Evolution Orchestrator applies whole-cycle resource and tool budgets and maintains an opportunity ledger to reduce repetitive or near-duplicate self-improvement work.

Failures and rejected candidates remain useful development evidence rather than simply disappearing.

## Learning and memory

LocalPilot uses several intentionally distinct information paths.

### Human teaching

Explicit owner guidance can be stored as durable human lessons.

### Study

Structured study can create verified source-linked knowledge about areas such as LocalPilot itself, Python, and relevant models or tooling.

### Library learning

LocalPilot can progressively read indexed local material, extract candidate learnings, verify them against exact source passages, and retain supported information with provenance.

### Self-development memory

Evolution cycles retain information about:

- hypotheses;
- experiments;
- candidate outcomes;
- failures;
- capability frontiers;
- rejected attempts;
- review results;
- opportunity history.

### Conversation history

Desktop chat history is persistent for continuity, but ordinary conversation is not silently converted into factual learning.

## Evidence and epistemic control

LocalPilot distinguishes memory from current evidence.

Durable learning can help narrow research, but it does not automatically override live observations.

Stored knowledge may carry:

- source identity;
- epistemic type;
- confidence;
- verification time;
- source digest;
- staleness state;
- relationships to other learning.

For mutable claims, LocalPilot can selectively verify the current repository, runtime, PC state, GitHub state, or other live sources.

An empty search does not prove that a broader capability is absent.

A remembered fact contradicted by stronger current evidence should yield to the current evidence.

## SystemSense

SystemSense provides low-overhead passive awareness of the Windows environment.

Depending on available sensors and platform support, LocalPilot can reason from information including:

- CPU state;
- GPU state;
- memory pressure;
- storage pressure;
- hardware inventory;
- device identifiers;
- firmware information;
- driver inventory;
- process state;
- historical observations;
- inference-performance trends;
- workload correlations;
- runtime and repository state.

SystemSense is observational. Correlation is not automatically treated as causation, and inactive hardware or drivers are not automatically classified as unnecessary.

## Runtime

LocalPilot's desktop interface, broker, runtime worker, and autonomous worker are separated so that individual components can recover without unnecessarily destroying the entire application.

Long reasoning is not supposed to restart the runtime merely because a normal request exceeds a soft response threshold.

Runtime lifecycle events can be recorded and exposed as evidence so LocalPilot can inspect its own process history rather than guessing whether it restarted.

The Windows autonomous worker is persistent and windowless. It runs periodically inside one long-lived process rather than repeatedly opening PowerShell windows.

## Research

Interactive research keeps complete raw tool results available to the reasoning context.

Research is bounded by soft and hard budgets.

Beyond the soft research boundary, LocalPilot must become increasingly selective about additional observations rather than simply issuing unlimited tool calls.

Repository and system claims may pass through deterministic evidence checks before being presented as established facts.

## Security model

LocalPilot is designed around broad freedom to reason and narrower authority to perform consequential actions.

Current boundaries include:

- no autonomous merge or promotion of stable code;
- no unrestricted shell execution by the normal operator;
- no arbitrary process termination;
- no automatic execution of untrusted candidate code on the workstation;
- isolated candidate workspaces;
- path and file restrictions for candidate writes;
- resource-aware autonomous work;
- trusted-main verification;
- explicit handling of public HTTPS safety, including DNS-rebinding resistance;
- GitHub Actions as an external executable validation boundary.

These boundaries are intended to constrain blast radius rather than prescribe what LocalPilot is allowed to think about.

## Current capabilities

LocalPilot can currently demonstrate meaningful capability in:

| Area | Status |
|---|---|
| Persistent local conversation | Working |
| High-reasoning local inference | Working |
| Repository inspection | Working |
| Evidence-grounded research | Working |
| Persistent typed learning | Working |
| Progressive library study | Working |
| Passive Windows awareness | Working |
| Runtime self-observation | Working |
| Autonomous evolution scheduling | Working |
| Opportunity tracking | Working |
| Isolated self-modification | Working |
| Candidate PR creation | Working |
| CI-backed candidate testing | Working |
| Human-controlled promotion | Working |
| Arbitrary desktop operation | Limited |
| Unrestricted autonomous execution | Not enabled |
| Model-weight self-training | Not implemented |
| Sustained recursive self-improvement | Not yet demonstrated |

## What remains unproven

LocalPilot has the machinery required for meaningful autonomous improvement, but the most important question remains empirical.

A complete long-term success would look like:

observe limitation
    ↓
form hypothesis
    ↓
design experiment
    ↓
implement candidate
    ↓
measure result
    ↓
retain transferable learning
    ↓
use improved capability to solve the next problem better
    ↓
repeat

The architecture exists to attempt this.
Sustained recursive capability growth has not yet been demonstrated.
Development philosophy
LocalPilot follows several principles:
Evidence over confidence.
A persuasive model response is not proof.
Capability over feature count.
A useful general improvement matters more than another isolated function.
Failure should teach.
Failed experiments should improve later decisions.
Memory should remain revisable.
Learning is prior knowledge, not permanent truth.
Preserve initiative.
LocalPilot should have room to find useful solutions its designers did not explicitly prescribe.
Keep consequential action observable and recoverable.
Increasing intelligence should not require making failures impossible to inspect or undo.
Installation
Requirements:
- Windows 10 or Windows 11
- Python 3.11+
- Git
- PowerShell
- Ollama
- a compatible installed local model
Clone and bootstrap:
git clone https://github.com/n47h4ni3l/localpilot.git
cd localpilot
.\scripts\bootstrap.ps1
Local configuration and learned state are intentionally kept outside Git.
Do not publish:
- localpilot.toml
- localpilot-data
- learning databases
- audit data
- local credentials
Testing
Run the Windows test suite with:
.\.venv\Scripts\python.exe -m pytest
When concurrent LocalPilot background processes interfere with pytest's temporary directory on Windows, a dedicated pytest base directory can be used:
.\.venv\Scripts\python.exe -m pytest --basetemp=".\.pytest-tmp"
GitHub Actions provides an additional independent validation layer for repository changes.
Project status
LocalPilot is an experimental system under active development.
It should be evaluated by what it can repeatedly demonstrate, not by ambitious terminology.
The immediate milestone is no longer simply adding more subsystems.
It is proving that LocalPilot can use the systems that now exist to generate useful, measurable, increasingly independent improvements over time.
