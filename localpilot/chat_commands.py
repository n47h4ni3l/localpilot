from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from rich.console import Console

if TYPE_CHECKING:
    from localpilot.agent import LocalPilotAgent
    from localpilot.config import Config


@dataclass(frozen=True, slots=True)
class ParsedChatCommand:
    name: str
    argument: str
    source: str


@dataclass(frozen=True, slots=True)
class ChatCommandResult:
    text: str


COMMANDS: tuple[tuple[str, str, str], ...] = (
    ("/help", "", "Show the commands available in desktop chat."),
    ("/status", "", "Show LocalPilot, resource, evolution, and repository status."),
    ("/doctor", "", "Run LocalPilot's prerequisite and runtime health checks."),
    ("/teach", "<lesson>", "Save an explicit durable owner teaching and load it now."),
    ("/evolve", "", "Request one normal gated self-development cycle."),
    ("/clear", "", "Start a fresh conversation without changing durable learning."),
)


def parse_chat_command(value: str) -> ParsedChatCommand | None:
    """Parse slash-prefixed desktop input without ever treating it as model text."""
    source = str(value).strip()
    if not source.startswith("/"):
        return None
    parts = source.split(maxsplit=1)
    name = parts[0].casefold()
    if name == "/new":
        name = "/clear"
    return ParsedChatCommand(
        name=name,
        argument=parts[1].strip() if len(parts) == 2 else "",
        source=source,
    )


def render_help() -> str:
    lines = [
        "### LocalPilot commands",
        "",
        "Slash commands are executed directly by LocalPilot; they are not sent to the language model as ordinary prompts.",
        "",
    ]
    for name, usage, description in COMMANDS:
        suffix = f" {usage}" if usage else ""
        lines.append(f"- `{name}{suffix}` — {description}")
    lines.extend(
        [
            "",
            "`/clear` is handled by the desktop interface itself. Conversation history and durable LearningMemory remain separate.",
        ]
    )
    return "\n".join(lines)


def _render_rich(call: Callable[[Console], object]) -> str:
    """Render existing deterministic CLI diagnostics as plain desktop text."""
    stream = io.StringIO()
    console = Console(
        file=stream,
        width=112,
        color_system=None,
        force_terminal=False,
        highlight=False,
        soft_wrap=False,
    )
    call(console)
    return stream.getvalue().strip()


def _diagnostic_block(title: str, rendered: str) -> str:
    return f"### {title}\n\n```text\n{rendered}\n```"


def execute_chat_command(
    command: ParsedChatCommand,
    *,
    agent: LocalPilotAgent | None,
    config: Config,
    root: str | Path,
    progress: Callable[[str], None] | None = None,
) -> ChatCommandResult:
    """Execute one trusted slash command outside model inference."""
    root = Path(root).resolve()
    progress = progress or (lambda _message: None)

    if command.name == "/help":
        return ChatCommandResult(render_help())

    if command.name == "/status":
        # Imported lazily to avoid making the runtime worker depend on CLI setup
        # during normal model turns while still reusing the authoritative status
        # implementation rather than maintaining a second status definition.
        from localpilot.cli import _show_status

        rendered = _render_rich(lambda console: _show_status(console, config, root))
        return ChatCommandResult(_diagnostic_block("Current status", rendered))

    if command.name == "/doctor":
        from localpilot.cli import _show_doctor

        rendered = _render_rich(lambda console: _show_doctor(console, config, root))
        return ChatCommandResult(_diagnostic_block("Doctor", rendered))

    if command.name == "/teach":
        if not command.argument:
            return ChatCommandResult("Usage: `/teach <durable lesson>`")
        if agent is None:
            raise RuntimeError("The active LocalPilot agent is unavailable for /teach")
        try:
            record = agent.teach(command.argument, topic="chat")
        except ValueError as exc:
            return ChatCommandResult(f"Teaching refused: {exc}")
        return ChatCommandResult(
            f"Teaching #{record.id} saved and loaded into this conversation.\n\n"
            f"**{record.lesson}**"
        )

    if command.name == "/evolve":
        from localpilot.evolution_reliability import SelfDeveloper

        progress("Starting a gated self-development cycle")
        result = SelfDeveloper(config, root, progress=progress).run_once(force=False)
        lines = [f"### Evolution: {result.status}", "", result.summary]
        if result.workspace:
            lines.extend(["", f"Candidate workspace: `{result.workspace}`"])
        return ChatCommandResult("\n".join(lines))

    if command.name == "/clear":
        # The webview intercepts this before submission so the current command
        # normally never reaches the runtime. Keeping the deterministic result
        # here makes typed/runtime tests and non-webview desktop adapters safe.
        return ChatCommandResult(
            "Starting a fresh conversation. Durable teachings and LearningMemory are unchanged."
        )

    if command.name in {"/quit", "/exit"}:
        return ChatCommandResult(
            "`/quit` is a CLI command. In the desktop companion, use the close control instead."
        )

    return ChatCommandResult(
        f"Unknown command `{command.name}`. Type `/help` to see available commands."
    )
