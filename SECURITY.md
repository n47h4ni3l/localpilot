# Security boundaries

LocalPilot is a private personal agent, but broad PC access still needs engineering boundaries so a model mistake does not damage the workstation.

v0.1 deliberately keeps **PC system tools read-only** while allowing the developer instance to modify **candidate source code only**.

Candidate path rules:

- absolute paths are rejected;
- `..` escapes are rejected after path resolution;
- `.git`, `.venv`, caches and `localpilot-data` are protected;
- autonomous file writes are restricted to text/source/config file types;
- each cycle has a file-write count limit;
- autonomous local validation compiles/parses candidate source without importing or executing it;
- `.github/` is protected from autonomous editing so the candidate cannot rewrite its own CI sandbox;
- full executable tests run in GitHub Actions, with checkout credentials not persisted and repository permissions limited to read;
- stable is never overwritten by the candidate loop.

These are foundations for future full PC autonomy, not a statement that LocalPilot should remain read-only.
