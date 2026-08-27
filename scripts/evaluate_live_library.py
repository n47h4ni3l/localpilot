from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localpilot.agent import LocalPilotAgent
from localpilot.config import load_config


def main() -> None:
    config = load_config(ROOT / "localpilot.toml")
    # A behavioral validation must not inherit the desktop operator's chat history,
    # research notebook, durable facts, or audit stream.
    config.agent.data_dir = str(
        ROOT / "tmp" / f"live-library-eval-{time.time_ns()}"
    )
    agent = LocalPilotAgent(config, ROOT)
    facts_before = len(agent.memory.knowledge_facts(include_stale=True))
    audit_path = ROOT / config.agent.data_dir / "audit.jsonl"
    audit_offset = audit_path.stat().st_size if audit_path.exists() else 0
    prompt = (
        "Consult the local library and use Atlas of Human Emotions to explain the difference between envy "
        "and jealousy. Cite the exact library:// source and page. Then give your provisional view on why the "
        "two labels are often confused. Do not use the public web, and do not store anything durably."
    )
    assert agent._evidence_requirements(prompt) == {"local library"}
    assert {"search_library", "read_library_passage"} <= agent.tools.keys()
    started = time.monotonic()
    answer = agent.ask(prompt)
    elapsed = round(time.monotonic() - started, 2)
    new_audit = ""
    if audit_path.exists():
        with audit_path.open("r", encoding="utf-8") as handle:
            handle.seek(audit_offset)
            new_audit = handle.read()
    rows = [json.loads(line) for line in new_audit.splitlines() if line.strip()]
    tool_results = [
        {
            "tool": row.get("tool"),
            "ok": row.get("ok"),
            "evidence_source": row.get("evidence_source"),
        }
        for row in rows
        if row.get("event") == "tool_result"
    ]
    tool_names = [str(item["tool"]) for item in tool_results]
    facts_after = len(agent.memory.knowledge_facts(include_stale=True))
    acceptance = {
        "library_searches_at_most_two": tool_names.count("search_library") <= 2,
        "passage_read_succeeded": any(
            item["tool"] == "read_library_passage" and item["ok"]
            for item in tool_results
        ),
        "no_public_web_tools": not {
            "search_public_web",
            "fetch_public_https",
        }.intersection(tool_names),
        "literal_library_citation": "library://" in answer,
        "provisional_view_present": "provisional" in answer.lower(),
        "no_durable_fact_write": facts_before == facts_after,
        "answer_not_withheld": "withheld the draft" not in answer.lower(),
    }
    result = {
        "model": config.model.name,
        "prompt": prompt,
        "answer": answer,
        "elapsed_seconds": elapsed,
        "tool_results": tool_results,
        "durable_facts_before": facts_before,
        "durable_facts_after": facts_after,
        "acceptance": acceptance,
        "accepted": all(acceptance.values()),
    }
    output = ROOT / "docs" / "live-library-validation.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["accepted"]:
        raise SystemExit("Live library validation failed one or more acceptance criteria.")


if __name__ == "__main__":
    main()
