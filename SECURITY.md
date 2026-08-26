# Security boundaries

LocalPilot is a private personal agent, but broad PC access still needs engineering boundaries so a model mistake does not damage the workstation.

v0.1 deliberately keeps **PC system tools read-only** while allowing the developer instance to modify **candidate source code only**.

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

These are foundations for future full PC autonomy, not a statement that LocalPilot should remain read-only.

Desktop boundary rules:

- the broker accepts only loopback configuration and authenticates API requests with a per-install token under ignored local data;
- the desktop persists visible user/assistant turns and presentation-safe events in `chat.sqlite3`, never in learning facts;
- hidden reasoning and raw tool results do not cross the worker protocol;
- the runtime/PowerShell worker is started with an argument vector, UTF-8 pipes, and `shell=False`;
- a worker restart cannot merge, promote, or weaken candidate confinement; and
- the CLI continues to use the same `LocalPilotAgent`, tool registry, and safety policy independently of the GUI.
