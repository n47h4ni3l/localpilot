from pathlib import Path

from localpilot.selfdev import (
    CandidateTools,
    build_read_context,
    parse_change_plan,
    apply_change_plan,
)


def test_read_context_contains_only_files_model_inspected(tmp_path: Path):
    (tmp_path / "read.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "unread.py").write_text("secret = 2\n", encoding="utf-8")

    tools = CandidateTools(tmp_path)
    tools.read_project_file("read.py")

    context = build_read_context(tools)

    assert "--- read.py ---" in context
    assert "value = 1" in context
    assert "unread.py" not in context
    assert "secret = 2" not in context


def test_structured_repair_plan_still_uses_candidate_write_tool(tmp_path: Path):
    (tmp_path / "operator.py").write_text("broken = True\n", encoding="utf-8")
    tools = CandidateTools(tmp_path)

    plan = parse_change_plan(
        """{
          "summary": "Repair timeout handling",
          "reusable_lesson": "Use structured fallback when direct tool editing stalls.",
          "changes": [
            {
              "path": "operator.py",
              "content": "broken = False\\n",
              "reason": "Repair the candidate."
            }
          ]
        }"""
    )

    apply_change_plan(plan, tools)

    assert (tmp_path / "operator.py").read_text(encoding="utf-8") == "broken = False\n"
    assert len(tools.files_written) == 1
