# Security

LocalPilot is a private, local-first agent with broad freedom to reason and deliberately narrow authority to perform consequential actions on its owner's Windows workstation.

The security model is designed to keep changes observable, bounded, and recoverable. It does not treat model confidence as authorization.

## Core trust boundary

LocalPilot separates three roles:

- the stable Operator, which assists the owner through a small typed tool surface;
- the Developer, which researches and prepares bounded improvements;
- isolated Candidates, which contain proposed source changes until review and promotion.

The stable runtime is never overwritten by the autonomous candidate loop. Promotion remains a human decision.

## Stable Operator actions

PC observation is read-only. Stable Operator mutation is limited to a small explicit allow-list of reversible actions.

The Operator does not accept arbitrary executables, arguments, filesystem paths, process IDs, URIs, power-plan identifiers, or shell commands.

Allowed actions follow these rules:

- application launches and Windows Settings destinations come from fixed typed allow-lists;
- Settings tools may open a page but do not change a setting;
- app-launch actions are reversed by closing the visible window;
- a power-plan target must be one of three Microsoft built-in GUIDs and must already appear in `powercfg /LIST`;
- a power-plan change records the prior active GUID, verifies the new plan, and returns a random one-use rollback token;
- rollback rejects stale tokens, verifies that the prior plan is still installed, and verifies restoration;
- failed post-change verification triggers a bounded automatic restoration attempt;
- every process launch uses an argument vector, `shell=False`, and a bounded timeout;
- audit output is presentation-safe, and rollback tokens are redacted from durable audit previews;
- `SafetyPolicy.auto_allow_reversible` controls whether reversible tools are visible to the model; and
- no destructive Operator action is registered.

## Candidate development boundary

Autonomous source changes are confined to isolated candidate workspaces.

Candidate writes are subject to these controls:

- absolute paths, `..` traversal, and symlink-backed paths are rejected;
- `.git`, `.github`, `.venv`, caches, `localpilot-data`, and other protected paths cannot be modified;
- writes are limited to approved source, data, configuration, and archive file types;
- file-count, file-size, resource, and workspace limits are enforced;
- candidate directories are unrestricted inside the isolated workspace, subject to the configured 500-file hard ceiling;
- complexity is reported when a candidate exceeds 100 files;
- ZIP creation accepts only bounded candidate or resource-store inputs and emits normalized, non-absolute members without `..`;
- archives are not extracted or executed;
- a generated claim manifest must resolve against the live candidate tree before write-capable tools are exposed;
- manifest checks cover paths, AST symbols, configuration fields, existing tests, integration points, and direct call relationships;
- malformed or contradicted plans are audited and rejected;
- local autonomous validation compiles or parses candidate source without importing or executing it;
- full executable validation runs in GitHub Actions;
- CI checkout credentials are not persisted and repository permissions are limited to read; and
- a Candidate cannot merge, promote, or overwrite stable code.

## Network and resource boundary

Research and candidate resources are limited to bounded public HTTPS access.

Third-party candidate resources:

- resolve and connect through the same public-address validation path, preventing a hostname from passing an earlier check and rebinding to a private address at connection time;
- stream through the resource governor into quota-bound storage outside the repository;
- retain source, task, and content-hash provenance;
- are subject to type and size restrictions; and
- reject obvious executables and installers.

Network access does not grant shell, package-installation, archive-extraction, or local-execution authority.

## Desktop and runtime boundary

The desktop, broker, runtime worker, and autonomous worker are separated so that one component can recover without unnecessarily replacing the whole application.

Desktop controls include:

- the broker accepts loopback configuration only;
- broker API requests require a per-install token stored under ignored local data;
- visible user and assistant turns, plus presentation-safe events, are stored in `chat.sqlite3`;
- chat history is not silently converted into durable factual learning;
- hidden reasoning and raw tool results do not cross the worker protocol;
- runtime and PowerShell workers start with argument vectors, UTF-8 pipes, and `shell=False`;
- worker recovery cannot merge, promote, or weaken candidate confinement; and
- the CLI uses the same `LocalPilotAgent`, tool registry, and safety policy independently of the GUI.

## SystemSense boundary

SystemSense is observational.

Collectors use psutil, native COM/WMI/CIM, an optional read-only hardware-monitor WMI namespace, and a fixed PnPUtil enumeration argument vector.

SystemSense does not expose raw SQL, device control, fan, clock or voltage control, driver installation or removal, or process termination.

Additional controls include:

- the desktop glance panel uses one bearer-authenticated loopback `GET` summary route;
- the glance route cannot trigger collection and has no corresponding mutation route;
- inactive, orphaned, or older driver-package classifications are review signals only;
- driver observations always report `safe_to_delete=false`;
- correlations are treated as observational and do not establish causation or authorization;
- compact context is transient;
- raw serials, process rows, sensor inventories, and device topology stay outside normal prompts;
- `systemsense.sqlite3` is local private data with bounded retention;
- SystemSense data remains separate from chat, learning, and library databases; and
- collector or provider failure is isolated from Operator startup and does not place exception text or command output in model context.

## Learning and retrieval boundary

Human lessons, structured study, library learning, self-development evidence, and conversation history remain distinct information paths.

Learning retrieval is lexical by default. Optional semantic retrieval sends only bounded fact documents and the current query to the owner's local Ollama service. It does not download models or call a remote embedding service.

Cached fact vectors remain in the ignored local learning database. Query text and query vectors are not persisted.

Semantic ranking cannot bypass stage, provenance, staleness, source-digest, test-evidence, or context-size controls. If embeddings fail, retrieval falls back to lexical matching.

## Evidence grounding

Memory is prior knowledge, not current authority.

For consequential repository answers, LocalPilot applies an in-process information-authority postcondition after synthesis. Repository literals and direct-call claims are checked against a content-digest-invalidated live index of paths, syntax, and configuration.

Structured high-impact flow and lifecycle contracts are enabled only when their repository anchors pass `RepositoryGroundingValidator`.

These checks add no tool authority and no research rounds. When live ground truth is incomplete, a current repository claim fails closed. Audit output retains issue codes, claim classes, and bounded evidence identifiers rather than rejected draft sentences.

## What these boundaries do not authorize

These controls are foundations for bounded autonomy. They do not authorize:

- unrestricted shell execution;
- arbitrary desktop control;
- arbitrary process termination;
- destructive filesystem actions;
- execution of untrusted Candidate code on the workstation;
- autonomous merge or promotion;
- silent weakening of security controls; or
- treating correlation, memory, a model response, or a passing test as proof.

LocalPilot may reason broadly. Consequential action remains typed, bounded, reviewable, and under human control.
