# Roadmap

## v0.1 — bootstrap

- Local Ollama agent with real TOML config loading.
- Read-only Windows observation tools.
- Audit logging with basic secret redaction.
- Resource governor for active-vs-idle behaviour.
- Git/GitHub integration primitives.
- Stable/developer/candidate separation.
- Autonomous candidate editing confined to a candidate workspace.
- Local static validation + GitHub Actions candidate tests; no automatic promotion.

## v0.2 — guarded operator

- Structured Windows command runner.
- Reversible system actions and automatic pre-change state capture.
- Explicit destructive-operation gate.
- Better process/foreground-app awareness.

## v0.3 — persistent machine knowledge

- SQLite machine inventory and change history.
- Known-good configuration snapshots.
- Problem/solution memory with evidence and timestamps.

## v0.4 — desktop agent

- Screenshot/vision observation.
- UI automation only when native tools/APIs are insufficient.
- Application-specific adapters.

## v0.5 — continuous self-development

- Evidence-driven autonomous capability discovery beyond seed tasks.
- First-class Repair, Extend, Improve Cognition and bounded Explore experiments.
- Durable capability map, falsifiable hypotheses, baselines and reusable validated lessons.
- Resumable idle development cycles.
- Candidate benchmarking and regression checks.
- Automatic candidate push/PR presentation with human-only merge and promotion.
- Carefully designed promotion and rollback path.
