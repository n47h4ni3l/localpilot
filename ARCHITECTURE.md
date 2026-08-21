# LocalPilot architecture

LocalPilot is intentionally split into three roles:

1. **Stable** — the installed agent that interacts with the PC.
2. **Developer** — an idle-time agent allowed to modify an isolated candidate workspace.
3. **Candidate** — an experimental build that must pass tests before it can ever be considered for promotion.

The stable runtime must never rewrite itself in place.

## Runtime layers

- **Agent**: local Ollama model and tool loop.
- **PC tools**: Windows-native observations first; operator tools are added behind risk policy later.
- **Resource governor**: keeps LocalPilot low priority while the owner is active and permits heavier work only after an idle threshold and load checks.
- **Local data**: audit logs, candidate workspaces and future machine memory live under `localpilot-data/` and are ignored by Git.
- **GitHub**: source, issues, branches, CI, review and rollback history. Machine-private state does not belong in GitHub.
- **Self-development**: creates a Git worktree when possible, or a copied candidate workspace before Git is connected. Candidate tools are path-confined. Local validation is non-executing; full candidate tests run in GitHub Actions.

## Autonomy philosophy

The target is **autonomy + observability + rollback**, not a confirmation dialog for every harmless action.

- Read-only observations: automatic.
- Reversible operations: intended to be automatic once implemented and tested.
- Destructive/irreversible operations: explicit confirmation.
- Self-development: automatic only inside a candidate workspace and only while resources permit.
- Promotion to stable: manual in v0.1 and remains a separate gate until the candidate system is mature.

## PC responsiveness

A background evolution cycle is eligible only when:

- the keyboard/mouse idle timer exceeds the configured threshold;
- CPU utilisation is below the configured ceiling; and
- memory pressure is below the configured ceiling.

LocalPilot also lowers its own Windows process priority while the owner is active. Future versions should add GPU/foreground-app awareness and resumable inference.
