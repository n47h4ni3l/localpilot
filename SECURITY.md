# Security boundaries

LocalPilot is a private personal agent, but broad PC access still needs engineering boundaries so a model mistake does not damage the workstation.

PC observation remains read-only. Stable operator mutation is limited to a small explicit
**reversible action allow-list**, while the developer instance may still modify **candidate source code
only**.

Stable operator action rules:

- no arbitrary executable, argument, filesystem path, PID, URI, power GUID, or shell command is accepted;
- app launches and Settings destinations come from fixed typed allow-lists and are reversed by closing the visible window;
- Settings tools open a page but never change a setting;
- a power-plan target must be one of three Microsoft built-in GUIDs and must already appear in `powercfg /LIST`;
- a power change records the exact prior active GUID, verifies the new active GUID, and returns a random one-use in-session rollback token that is redacted from durable audit previews;
- rollback refuses a stale token when the active plan changed independently, verifies that the prior plan remains installed, and verifies restoration;
- failed post-change verification triggers a bounded automatic restoration attempt and reports separately if restoration cannot be verified;
- every process uses argv with `shell=False`, a bounded timeout, and a presentation-safe audit event; and
- `SafetyPolicy.auto_allow_reversible` gates model visibility. No destructive operator action is registered.

Candidate path rules:

- absolute paths are rejected;
- all `..` traversal and symlink-backed paths are rejected before writing;
- `.git`, `.venv`, caches and `localpilot-data` are protected;
- autonomous file writes are restricted to approved source/data/config/archive file types;
- directories are free inside the isolated candidate; file complexity is reported above 100 files and the configurable 500-file hard ceiling remains enforced;
- ZIP creation accepts only bounded in-candidate/resource-store inputs and produces normalized non-absolute members without `..`; archives are never extracted or executed;
- public-HTTPS resources stream through the resource governor into quota-bound storage outside the repository, retain hash/source/task provenance, and block obvious executables/installers;
- before implementation, a read-only generated claim manifest must resolve against the live candidate tree's paths, AST symbols, configuration fields, existing tests, integration points, and direct call relationships; a malformed or contradicted plan is audited and rejected before write-capable tools are exposed;
- autonomous local validation compiles/parses candidate source without importing or executing it;
- `.github/` is protected from autonomous editing so the candidate cannot rewrite its own CI sandbox;
- full executable tests run in GitHub Actions, with checkout credentials not persisted and repository permissions limited to read;
- stable is never overwritten by the candidate loop.

These are foundations for bounded PC autonomy, not authority for arbitrary or destructive control.

Desktop boundary rules:

- the broker accepts only loopback configuration and authenticates API requests with a per-install token under ignored local data;
- the desktop persists visible user/assistant turns and presentation-safe events in `chat.sqlite3`, never in learning facts;
- hidden reasoning and raw tool results do not cross the worker protocol;
- the runtime/PowerShell worker is started with an argument vector, UTF-8 pipes, and `shell=False`;
- a worker restart cannot merge, promote, or weaken candidate confinement; and
- the CLI continues to use the same `LocalPilotAgent`, tool registry, and safety policy independently of the GUI.

SystemSense boundary rules:

- collectors perform observation only through psutil, native COM/WMI/CIM, an optional read-only hardware-monitor WMI namespace, and a fixed PnPUtil enumeration argv;
- SystemSense exposes six bounded `READ_ONLY` tools and no raw SQL, device control, fan/clock/voltage control, driver installation/removal, or process termination;
- inactive, orphan and older driver-package classifications are review signals, always report `safe_to_delete=false`, and cannot authorize removal;
- correlations are explicitly observational and cannot establish causal authority for a setting or hardware change;
- compact context is transient and raw serials, process rows, sensor inventories and device topology remain outside normal prompts;
- `systemsense.sqlite3` is local private data with bounded retention and is distinct from chat, learning and library databases; and
- collector/provider failure is isolated from operator startup and records no exception text or command output in model context.

Learning retrieval is lexical by default. Optional semantic retrieval sends only bounded fact documents
and the current retrieval query to the owner's local Ollama service; it does not download models or call a
remote embedding service. Cached fact vectors stay in the ignored local learning database, while query
text and query vectors are not persisted. Semantic ranking cannot bypass stage, provenance, staleness,
digest, test-evidence, or context-size controls, and any embedding failure falls back to lexical retrieval.

Current repository answers use an in-process information-authority postcondition after synthesis. It
checks repository literals and direct call claims against a content-digest-invalidated live AST/path/config
index, while structured high-impact flow/lifecycle contracts are enabled only after their own repository
anchors pass `RepositoryGroundingValidator`. This adds no tool capability and no research rounds. If live
ground truth is incomplete, a current repository claim fails closed. Audit output retains issue codes,
claim classes and bounded repository evidence identifiers, never the rejected draft sentences.
