from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from localpilot.agent import LocalPilotAgent
from localpilot.audit import AuditLog
from localpilot.checkpoint import CheckpointStore
from localpilot.config import load_config
from localpilot.doctor import doctor
from localpilot.github_integration import GitHubIntegration
from localpilot.learning import LearningMemory
from localpilot.mission import mission_context
from localpilot.resource import ResourceGovernor
from localpilot.selfdev import SelfDeveloper


_FAILED_EVOLVE_STATUSES = {"failed", "sync_blocked", "candidate_needs_work"}


def evolve_exit_code(status: str) -> int:
    """Expose internally handled failures to Task Scheduler and scripts."""
    return 1 if status in _FAILED_EVOLVE_STATUSES else 0


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _show_status(console: Console, config, root: Path) -> None:
    governor = ResourceGovernor(config.resource)
    state = governor.sample()
    gh = GitHubIntegration(root, config.github)
    table = Table(title="LocalPilot status")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("Model", f"{config.model.provider}:{config.model.name}")
    mission = mission_context()
    table.add_row(
        "Mission",
        (
            f"{mission['mission']}\n"
            f"Evolution objective: {mission['evolution_objective']}"
        )[:1400],
    )
    table.add_row("User idle", f"{state.idle_seconds:.0f}s")
    table.add_row("CPU", f"{state.cpu_percent:.1f}%")
    table.add_row("Memory", f"{state.memory_percent:.1f}%")
    table.add_row("Background self-dev", "available" if state.background_allowed else f"deferred — {state.reason}")
    audit = AuditLog(root / config.agent.data_dir / "audit.jsonl")
    last_evolve = audit.latest("evolve_run_end")
    if last_evolve:
        timestamp = str(last_evolve.get("timestamp") or "")
        try:
            timestamp = datetime.fromisoformat(timestamp).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
        except ValueError:
            pass
        status = str(last_evolve.get("status") or "unknown")
        branch = str(last_evolve.get("branch") or "")
        summary = str(last_evolve.get("summary") or "").splitlines()[0]
        detail = f"{status} at {timestamp}"
        if branch:
            detail += f"\nbranch: {branch}"
        if summary:
            detail += f"\n{summary[:240]}"
        table.add_row("Last evolve", detail)
    else:
        table.add_row("Last evolve", "No completed invocation has been recorded yet.")
    checkpoint_store = CheckpointStore(root / config.agent.data_dir / "evolution-checkpoint.json")
    memory = LearningMemory(root / config.agent.data_dir / config.selfdev.learning_database)
    checkpoint = None
    try:
        checkpoint = checkpoint_store.load()
    except Exception as exc:
        table.add_row("Checkpoint", f"Invalid — {type(exc).__name__}: {exc}"[:300])
    else:
        latest_resume = audit.latest("selfdev_checkpoint_resume")
        if checkpoint:
            detail = (
                f"v{checkpoint.version} {checkpoint.milestone} at {checkpoint.updated_at}\n"
                f"branch: {checkpoint.branch}\n"
                f"task: {checkpoint.task_id} — {checkpoint.objective}\n"
                f"next: {checkpoint.next_action}"
            )
            if latest_resume and latest_resume.get("branch") == checkpoint.branch:
                detail += f"\nlast resume: {latest_resume.get('status', 'unknown')}"
            table.add_row("Checkpoint", detail[:1200])
        elif latest_resume:
            table.add_row(
                "Checkpoint",
                f"None active; last resume {latest_resume.get('status', 'unknown')}: "
                f"{str(latest_resume.get('reason') or '')[:240]}",
            )
        else:
            table.add_row("Checkpoint", "No active evolution checkpoint.")
    experiment = memory.latest_experiment()
    if checkpoint:
        evolution_detail = (
            f"class: {checkpoint.evolution_class}\n"
            f"target: {checkpoint.capability_target}\n"
            f"hypothesis: {checkpoint.hypothesis or '(legacy checkpoint; not recorded)'}\n"
            f"evaluation: {checkpoint.evaluation_plan or '{}'}"
        )
        if experiment and experiment.task_id == checkpoint.task_id:
            evolution_detail += f"\nlatest outcome: {experiment.status} — {experiment.outcome or 'pending'}"
        table.add_row("Evolution", evolution_detail[:1800])
    elif experiment:
        evaluation = (
            f"{experiment.metric}; baseline: {experiment.baseline}; "
            f"success: {experiment.success_criterion}; method: {experiment.measurement_method}"
        )
        table.add_row(
            "Evolution",
            (
                f"class: {experiment.evolution_class}\n"
                f"target: {experiment.capability_target}\n"
                f"hypothesis: {experiment.hypothesis}\n"
                f"evaluation: {evaluation}\n"
                f"latest outcome: {experiment.status} — {experiment.outcome or 'pending'}"
            )[:1800],
        )
    else:
        table.add_row(
            "Evolution",
            "No experiment recorded yet; the next ungated idle cycle will discover a capability-growth question.",
        )
    frontier = memory.latest_frontier()
    if frontier:
        table.add_row(
            "Capability frontier",
            (
                f"current: {frontier.current_frontier}\n"
                f"mission alignment: {frontier.mission_alignment}\n"
                f"why high leverage: {frontier.why_high_leverage}\n"
                f"unlocks: {frontier.capability_unlocked}\n"
                f"next: {frontier.next_frontier}"
            )[:2200],
        )
    else:
        table.add_row(
            "Capability frontier",
            "No frontier recorded yet; the next ungated capability-discovery cycle will establish one.",
        )
    table.add_row("Git", gh.status())
    console.print(table)


def _show_doctor(console: Console, config, root: Path) -> int:
    table = Table(title="LocalPilot doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    failures = 0
    for name, ok, detail in doctor(config, root):
        if not ok and name not in {"GitHub CLI (optional)", "Local Git repo"}:
            failures += 1
        table.add_row(name, "OK" if ok else "WARN", detail)
    console.print(table)
    return 1 if failures else 0


def _progress(console: Console):
    return lambda message: console.print(f"[dim]{message}[/dim]")


def _chat(console: Console, config, root: Path) -> None:
    console.print(f"[bold]{config.agent.name} 0.1[/bold] — local-first Windows agent")
    console.print(f"Model: {config.model.name} via Ollama. Commands: /status /doctor /evolve /quit\n")
    agent = LocalPilotAgent(config, root)
    while True:
        try:
            prompt = console.input("[bold cyan]you>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not prompt:
            continue
        command = prompt.lower()
        if command in {"/quit", "/exit"}:
            return
        if command == "/status":
            _show_status(console, config, root)
            continue
        if command == "/doctor":
            _show_doctor(console, config, root)
            continue
        if command == "/evolve":
            result = SelfDeveloper(config, root, progress=_progress(console)).run_once(force=False)
            console.print(f"[bold]{result.status}[/bold]\n{result.summary}")
            if result.workspace:
                console.print(f"Candidate: {result.workspace}")
            continue
        try:
            answer = agent.ask(prompt)
            console.print(Markdown(answer))
        except Exception as exc:
            console.print(f"[red]Agent error:[/red] {type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="localpilot")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("chat", help="Start the interactive agent")
    sub.add_parser("doctor", help="Check prerequisites and GitHub/model readiness")
    sub.add_parser("status", help="Show resource and Git status")
    evolve = sub.add_parser("evolve", help="Run one isolated self-development cycle")
    evolve.add_argument("--force", action="store_true", help="Ignore idle-resource gate for this manual cycle")
    parser.add_argument("--config", default=None, help="Path to localpilot.toml")
    args = parser.parse_args()

    root = _root()
    config = load_config(args.config)
    console = Console()
    if args.command in {None, "chat"}:
        _chat(console, config, root)
    elif args.command == "doctor":
        raise SystemExit(_show_doctor(console, config, root))
    elif args.command == "status":
        _show_status(console, config, root)
    elif args.command == "evolve":
        result = SelfDeveloper(config, root, progress=_progress(console)).run_once(force=args.force)
        console.print(f"[bold]{result.status}[/bold]\n{result.summary}")
        if result.workspace:
            console.print(f"Candidate workspace: {result.workspace}")
        raise SystemExit(evolve_exit_code(result.status))
