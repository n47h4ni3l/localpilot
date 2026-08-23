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
from localpilot.selfdev import CandidateRejectionError, CandidateRetryError, SelfDeveloper
from localpilot.study import STAGES, StudyEngine


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
    resource_files = root / config.agent.data_dir / "candidate-resources" / "files"
    resource_usage = sum(
        path.stat().st_size
        for path in resource_files.iterdir()
        if resource_files.is_dir() and path.is_file() and not path.is_symlink()
    ) if resource_files.is_dir() else 0
    resource_quota = int(config.selfdev.candidate_resource_quota_gb * 1024**3)
    table.add_row(
        "Candidate resources",
        f"{resource_usage / 1024**3:.2f} GiB / {resource_quota / 1024**3:.2f} GiB",
    )
    table.add_row(
        "Candidate file budget",
        f"soft {config.selfdev.candidate_file_soft_budget}; hard {config.selfdev.candidate_file_hard_ceiling}; directories free",
    )
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
    rejected = memory.latest_rejected_candidate()
    if rejected:
        table.add_row(
            "Last rejected candidate",
            (
                f"PR #{rejected.rejection_pull_request_number or '?'} — {rejected.branch}\n"
                f"task: {rejected.task_id}\n"
                f"prior CI: {rejected.rejection_prior_validation_state or 'unknown'}\n"
                f"reason: {rejected.rejection_reason}"
            )[:1800],
        )
    else:
        table.add_row("Last rejected candidate", "No explicit human rejection recorded.")
    retry = memory.latest_policy_retry()
    if retry:
        table.add_row(
            "Human-authorized policy retry",
            (
                f"{retry.branch}\n"
                f"task: {retry.task_id}\n"
                f"retry cycle: {retry.cycle_id}; prior cycle: {retry.retry_of_cycle_id}\n"
                f"reason: {retry.retry_reason}"
            )[:1800],
        )
    else:
        table.add_row("Human-authorized policy retry", "No policy-blocked candidate retry recorded.")
    curriculum = [memory.curriculum_state(stage) for stage in STAGES]
    active = next((item for item in curriculum if item.status != "improved"), curriculum[-1])
    curriculum_lines = []
    for item in curriculum:
        baseline = "—" if item.baseline_score is None else f"{item.baseline_score:.1f}"
        latest = "—" if item.latest_score is None else f"{item.latest_score:.1f}"
        curriculum_lines.append(
            f"{item.stage}: {item.status} (baseline {baseline}, latest {latest})"
        )
    if active.known_weak_areas:
        curriculum_lines.append(f"weak: {active.known_weak_areas[0][:220]}")
    if active.next_lesson:
        curriculum_lines.append(f"next: {active.next_lesson[:300]}")
    table.add_row("Study curriculum", "\n".join(curriculum_lines))
    comparison = memory.latest_peer_model_comparison()
    if comparison:
        table.add_row(
            "Latest peer comparison",
            (
                f"{comparison.subject_model}: {comparison.subject_score:.1f} vs "
                f"{comparison.peer_model}: {comparison.peer_score:.1f}\n"
                f"latency: {comparison.subject_latency_ms}ms vs {comparison.peer_latency_ms}ms"
            ),
        )
    table.add_row("Git", gh.status())
    console.print(table)


def _show_study_status(console: Console, memory: LearningMemory) -> None:
    table = Table(title="LocalPilot study curriculum")
    for name in ("Stage", "Status", "Baseline", "Latest", "Weak areas", "Next"):
        table.add_column(name)
    for stage in STAGES:
        state = memory.curriculum_state(stage)
        table.add_row(
            stage,
            state.status,
            "—" if state.baseline_score is None else f"{state.baseline_score:.1f}",
            "—" if state.latest_score is None else f"{state.latest_score:.1f}",
            "; ".join(state.known_weak_areas[:3]) or "—",
            state.next_lesson or "Establish the held-out baseline.",
        )
    console.print(table)


def _show_study_outcome(console: Console, outcome) -> None:
    gain = outcome.latest.score - outcome.baseline.score
    console.print(
        f"[bold]{outcome.stage}: {outcome.state.status}[/bold]\n"
        f"Baseline: {outcome.baseline.score:.1f} ({outcome.baseline.correct}/{outcome.baseline.total})\n"
        f"Latest: {outcome.latest.score:.1f} ({outcome.latest.correct}/{outcome.latest.total})\n"
        f"Measured gain: {gain:+.1f}\n"
        f"Facts added: {outcome.facts_written}; stale facts invalidated: {outcome.stale_facts}\n"
        f"Next: {outcome.state.next_lesson}"
    )


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localpilot")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("chat", help="Start the interactive agent")
    sub.add_parser("doctor", help="Check prerequisites and GitHub/model readiness")
    sub.add_parser("status", help="Show resource and Git status")
    evolve = sub.add_parser("evolve", help="Run one isolated self-development cycle")
    evolve.add_argument("--force", action="store_true", help="Ignore idle-resource gate for this manual cycle")
    reject = sub.add_parser(
        "reject",
        help="Explicitly reject a LocalPilot-managed candidate PR",
    )
    reject.add_argument("pull_request", type=int, help="GitHub pull request number")
    reject.add_argument(
        "--reason",
        default=None,
        help="Durable, non-interactive reason retained as learning evidence",
    )
    retry = sub.add_parser(
        "retry",
        help="Human-authorize retry of a framework-policy-blocked local candidate",
    )
    retry.add_argument("candidate", help="Managed local candidate branch or task id")
    retry.add_argument(
        "--reason",
        required=True,
        help="Durable attribution and authorization reason",
    )
    study = sub.add_parser(
        "study",
        help="Run benchmarked self-study; this does not train model weights",
    )
    study_sub = study.add_subparsers(dest="study_action")
    study_sub.add_parser("status", help="Show curriculum progress and weak areas")
    baseline = study_sub.add_parser(
        "baseline", help="Record the held-out baseline before studying a stage"
    )
    baseline.add_argument("stage", choices=STAGES)
    run_study = study_sub.add_parser(
        "run", help="Study one stage and retest against its held-out benchmark"
    )
    run_study.add_argument("stage", choices=STAGES)
    run_study.add_argument(
        "--allow-web",
        action="store_true",
        help="Read authoritative HTTPS documentation and verify it before persistence",
    )
    run_all = study_sub.add_parser("all", help="Run stages in enforced curriculum order")
    run_all.add_argument(
        "--allow-web",
        action="store_true",
        help="Read authoritative HTTPS documentation and verify it before persistence",
    )
    compare = study_sub.add_parser(
        "compare",
        help="Compare the developer model with another installed model on transfer tasks",
    )
    compare.add_argument("peer_model", help="Installed Ollama model used as the peer")
    compare.add_argument(
        "--subject-model",
        default=None,
        help="Model being evaluated; defaults to selfdev.developer_model",
    )
    research = study_sub.add_parser(
        "research",
        help="Inspect any public HTTPS source transiently without promoting it to knowledge",
    )
    research.add_argument("url", help="Public HTTPS source to inspect read-only")
    parser.add_argument("--config", default=None, help="Path to localpilot.toml")
    return parser


def main() -> None:
    args = build_parser().parse_args()

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
    elif args.command == "reject":
        developer = SelfDeveloper(config, root, progress=_progress(console))
        try:
            result = developer.reject_candidate(
                args.pull_request,
                reason=args.reason,
            )
        except CandidateRejectionError as exc:
            console.print(f"[red]Rejection refused:[/red] {exc}")
            raise SystemExit(1) from exc
        state = "already rejected" if result.already_rejected else "rejected"
        console.print(
            f"[bold]PR #{result.pull_request_number} {state}[/bold]\n"
            f"Branch: {result.branch}\n"
            f"Task: {result.task_id}\n"
            f"Reason: {result.reason}\n"
            f"Local cleanup: {result.worktree_cleanup}\n"
            "GitHub branch/history retained; no merge or promotion was performed."
        )
    elif args.command == "retry":
        developer = SelfDeveloper(config, root, progress=_progress(console))
        try:
            result = developer.retry_candidate(args.candidate, reason=args.reason)
        except CandidateRetryError as exc:
            console.print(f"[red]Retry refused:[/red] {exc}")
            raise SystemExit(1) from exc
        state = "already authorized" if result.already_authorized else "authorized"
        console.print(
            f"[bold]Candidate retry {state}[/bold]\n"
            f"Branch: {result.branch}\n"
            f"Task: {result.task_id}\n"
            f"Prior cycle: {result.prior_cycle_id}; retry cycle: {result.retry_cycle_id}\n"
            f"Mode: {result.resume_mode}\n"
            f"Reason: {result.reason}\n"
            "Prior failure evidence was retained and attributed to framework policy. "
            "No merge, promotion, or candidate execution was performed."
        )
    elif args.command == "study":
        memory = LearningMemory(root / config.agent.data_dir / config.selfdev.learning_database)
        action = args.study_action or "status"
        if action == "status":
            _show_study_status(console, memory)
            return
        engine = StudyEngine(
            root,
            memory,
            config,
            allow_web=(
                bool(getattr(args, "allow_web", False)) or action == "research"
            ),
        )
        try:
            if action == "baseline":
                run = engine.baseline(args.stage)
                console.print(
                    f"[bold]{args.stage} baseline recorded[/bold]\n"
                    f"Score: {run.score:.1f} ({run.correct}/{run.total})\n"
                    f"Held-out set: {run.question_set_digest[:12]}"
                )
            elif action == "run":
                _show_study_outcome(console, engine.run_stage(args.stage))
            elif action == "all":
                outcomes = engine.run_all()
                for outcome in outcomes:
                    _show_study_outcome(console, outcome)
                if not outcomes:
                    console.print("All curriculum stages already record measured improvement.")
            elif action == "compare":
                result = engine.compare_models(
                    args.peer_model,
                    subject_model=args.subject_model,
                )
                console.print(
                    f"[bold]Peer comparison recorded[/bold]\n"
                    f"{result.subject_model}: {result.subject_score:.1f} in {result.subject_latency_ms}ms\n"
                    f"{result.peer_model}: {result.peer_score:.1f} in {result.peer_latency_ms}ms\n"
                    f"Lesson: {result.transferable_lessons[0]}"
                )
            elif action == "research":
                source = engine.inspect_web_source(args.url)
                authority = "authoritative" if source.authoritative else "unverified"
                console.print(
                    f"[bold]Transient web source inspected[/bold]\n"
                    f"Final URL: {source.final_url}\n"
                    f"Bytes: {source.bytes_read}; digest: {source.source_digest[:12]}\n"
                    f"Tier: {authority}; confidence ceiling: {source.confidence_ceiling:.2f}\n"
                    "No page body or claim was persisted. Unverified sources require corroboration."
                )
        except (RuntimeError, ValueError) as exc:
            console.print(f"[red]Study refused:[/red] {exc}")
            raise SystemExit(1) from exc
