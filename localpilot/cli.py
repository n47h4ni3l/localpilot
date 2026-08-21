from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from localpilot.agent import LocalPilotAgent
from localpilot.config import load_config
from localpilot.doctor import doctor
from localpilot.github_integration import GitHubIntegration
from localpilot.resource import ResourceGovernor
from localpilot.selfdev import SelfDeveloper


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
    table.add_row("User idle", f"{state.idle_seconds:.0f}s")
    table.add_row("CPU", f"{state.cpu_percent:.1f}%")
    table.add_row("Memory", f"{state.memory_percent:.1f}%")
    table.add_row("Background self-dev", "available" if state.background_allowed else f"deferred — {state.reason}")
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
