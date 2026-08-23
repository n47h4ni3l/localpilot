from pathlib import Path

from localpilot.agent import LocalPilotAgent
from localpilot.cli import build_parser
from localpilot.config import Config
from localpilot.learning import LearningMemory


def test_human_teaching_is_durable_deduplicated_and_reviewable(tmp_path: Path):
    memory = LearningMemory(tmp_path / "learning.sqlite3")
    first = memory.record_human_lesson(
        "Before proposing an API, verify that it exists in the repository.",
        topic="repository-grounding",
    )
    duplicate = memory.record_human_lesson(
        "Before proposing an API, verify that it exists in the repository.",
        topic="repository-grounding",
    )

    assert duplicate.id == first.id
    reopened = LearningMemory(memory.path)
    lessons = reopened.human_lessons(query="repository API", limit=5)
    assert len(lessons) == 1
    assert lessons[0].topic == "repository-grounding"
    assert lessons[0].source == "owner"
    assert lessons[0].confidence == 1.0
    assert reopened.human_lesson_count() == 1

    context = reopened.discovery_context()
    assert context["human_teachings"][0]["lesson"].startswith("Before proposing")
    reusable = reopened.reusable_lessons(limit=4, query="API integration")
    assert reusable[0].startswith("Human teaching [repository-grounding]:")

    with reopened._connect() as connection:
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(human_lessons)").fetchall()
        }
    assert not columns & {
        "reasoning", "chain_of_thought", "thinking", "prompt",
        "transcript", "messages", "raw_tokens",
    }


def test_agent_loads_and_can_add_durable_owner_teaching(tmp_path: Path):
    config = Config()
    config.agent.data_dir = "data"
    memory = LearningMemory(tmp_path / config.agent.data_dir / config.selfdev.learning_database)
    stored = memory.record_human_lesson(
        "Green CI is insufficient when new code is never exercised.",
        topic="evaluation",
    )

    agent = LocalPilotAgent(config, tmp_path)
    system_text = "\n".join(
        str(message.get("content") or "")
        for message in agent.messages
        if isinstance(message, dict)
    )
    assert stored.lesson in system_text

    new = agent.teach(
        "Prefer actual repository interfaces over invented model and dataset APIs.",
        topic="repository-grounding",
    )
    assert new.id != stored.id
    reopened = LearningMemory(memory.path)
    assert any(item.id == new.id for item in reopened.human_lessons(limit=10))


def test_teach_cli_contract():
    args = build_parser().parse_args([
        "teach", "--topic", "repository-grounding", "--lesson",
        "Verify every proposed symbol before implementation.",
    ])
    assert args.command == "teach"
    assert args.topic == "repository-grounding"
    assert args.lesson.startswith("Verify every")

    listing = build_parser().parse_args(["teach", "--list"])
    assert listing.command == "teach"
    assert listing.list is True
