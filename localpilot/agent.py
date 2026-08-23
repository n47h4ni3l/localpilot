from __future__ import annotations

from pathlib import Path
from typing import Any

from localpilot.audit import AuditLog
from localpilot.config import Config
from localpilot.learning import HumanLesson, LearningMemory
from localpilot.safety import SafetyPolicy
from localpilot.resource import ResourceGovernor
from localpilot.tools import registry

SYSTEM_PROMPT = """You are LocalPilot, a local-first Windows agent running on the owner's PC.
Your long-term purpose is to become a capable general computer agent while keeping the PC pleasant to use.
Use evidence and tools rather than generic tweak lists. Be economical with tool calls.
When discussing LocalPilot's own implementation, current modules, classes, functions, dependencies, configuration, or integration points, inspect the trusted repository with the read-only repository tools before making factual claims. Plausible names and memories from earlier failed candidates are not evidence. Clearly distinguish verified existing interfaces from proposed new architecture.
The v0.1 PC toolset is observation-first: do not imply a system change occurred unless a tool explicitly did it.
The self-development subsystem may write only inside isolated candidate workspaces, never directly over the stable runtime.
GitHub is the durable engineering layer for source, issues, branches, tests and rollback.
"""


class LocalPilotAgent:
    def __init__(self, config: Config, project_root: str | Path) -> None:
        self.config = config
        self.project_root = Path(project_root).resolve()
        self.policy = SafetyPolicy(
            auto_allow_read_only=config.safety.auto_allow_read_only,
            auto_allow_reversible=config.safety.auto_allow_reversible,
            require_confirmation_for_destructive=config.safety.require_confirmation_for_destructive,
        )
        self.tools = registry(self.project_root)
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.data_dir = (self.project_root / config.agent.data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.audit = AuditLog(self.data_dir / "audit.jsonl")
        self.memory = LearningMemory(self.data_dir / config.selfdev.learning_database)
        teachings = self.memory.human_lessons(config.selfdev.lesson_limit)
        self._loaded_human_lesson_ids = {item.id for item in teachings}
        if teachings:
            self.messages.append(
                {
                    "role": "system",
                    "content": (
                        "Durable explicit teachings from the owner. Treat these as high-priority "
                        "guidance, but still verify factual claims against current evidence:\n- "
                        + "\n- ".join(f"[{item.topic}] {item.lesson}" for item in teachings)
                    ),
                }
            )
        self.governor = ResourceGovernor(config.resource)

    def teach(self, lesson: str, *, topic: str = "general") -> HumanLesson:
        record = self.memory.record_human_lesson(
            lesson,
            topic=topic,
            source="owner",
            confidence=1.0,
        )
        if record.id not in self._loaded_human_lesson_ids:
            self.messages.append(
                {
                    "role": "system",
                    "content": (
                        "New durable owner teaching. Apply it where relevant and verify "
                        f"repository facts before acting: [{record.topic}] {record.lesson}"
                    ),
                }
            )
            self._loaded_human_lesson_ids.add(record.id)
        self.audit.write(
            "human_teaching",
            teaching_id=record.id,
            topic=record.topic,
            source=record.source,
            lesson=record.lesson,
        )
        return record

    def _functions(self):
        return [spec.fn for spec in self.tools.values() if self.policy.permits_without_confirmation(spec.risk)]

    def ask(self, prompt: str) -> str:
        if self.config.model.provider.lower() != "ollama":
            raise RuntimeError("v0.1 supports Ollama only.")
        try:
            from ollama import chat
        except ImportError as exc:
            raise RuntimeError("Ollama Python package is not installed. Run scripts/bootstrap.ps1.") from exc

        self.messages.append({"role": "user", "content": prompt})
        retried_empty_response = False
        for round_no in range(self.config.agent.max_tool_rounds):
            state = self.governor.sample(interval=0.02)
            self.governor.apply_process_priority(idle=state.background_allowed)
            response = chat(
                model=self.config.model.name,
                messages=self.messages,
                tools=self._functions(),
                think=self.config.model.think,
                options={"temperature": self.config.model.temperature},
            )
            self.messages.append(response.message)
            calls = response.message.tool_calls or []
            if not calls:
                content = str(response.message.content or "")
                thinking = str(getattr(response.message, "thinking", "") or "")
                if content.strip():
                    return content
                if thinking.strip():
                    self.audit.write(
                        "model_no_final_answer",
                        model=self.config.model.name,
                        think=self.config.model.think,
                        round=round_no,
                        reasoning_present=True,
                    )
                    return (
                        f"[LocalPilot completed a {self.config.model.think} reasoning pass "
                        "but returned no final answer.]"
                    )
                if not retried_empty_response and round_no + 1 < self.config.agent.max_tool_rounds:
                    retried_empty_response = True
                    retry_message = {
                        "role": "user",
                        "content": (
                            "Your previous response contained neither a final answer nor a reasoning trace. "
                            "Try once more. Return a final answer, or explicitly state that you choose not to answer."
                        ),
                    }
                    self.messages.append(retry_message)
                    self.audit.write(
                        "model_empty_response_retry",
                        model=self.config.model.name,
                        think=self.config.model.think,
                        round=round_no,
                    )
                    continue
                self.audit.write(
                    "model_empty_response",
                    model=self.config.model.name,
                    think=self.config.model.think,
                    round=round_no,
                    reasoning_present=False,
                )
                return "[LocalPilot returned an empty response after one retry.]"

            for call in calls:
                name = call.function.name
                spec = self.tools.get(name)
                args = call.function.arguments or {}
                if spec is None:
                    result = f"Unknown tool: {name}"
                elif not self.policy.permits_without_confirmation(spec.risk):
                    result = f"Tool requires confirmation and is unavailable in this v0.1 loop: {name}"
                else:
                    self.audit.write("tool_call", tool=name, risk=spec.risk, args=args, round=round_no)
                    try:
                        result = spec.fn(**args)
                    except Exception as exc:
                        result = f"Tool error: {type(exc).__name__}: {exc}"
                    self.audit.write("tool_result", tool=name, result_preview=str(result)[:1200])
                self.messages.append({"role": "tool", "tool_name": name, "content": str(result)})

        return "Stopped at the tool-call limit. Narrow the request or inspect the audit log."
