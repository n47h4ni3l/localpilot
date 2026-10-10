"""Read-only execution diagnosis; never rerun, score, or repair a frozen study."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PureWindowsPath
import zipfile


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Evidence:
    def __init__(self, source: Path):
        self.source = source.resolve()
        self.archive = zipfile.ZipFile(source) if source.is_file() else None

    def bytes(self, name: str) -> bytes:
        if Path(name).anchor or PureWindowsPath(name).anchor or ".." in Path(name).parts or "\\" in name:
            raise ValueError("Invalid evidence path")
        if self.archive:
            return self.archive.read(name)
        target = (self.source / name).resolve()
        if not target.is_relative_to(self.source):
            raise ValueError("Evidence path leaves its source directory")
        return target.read_bytes()

    def json(self, name: str):
        return json.loads(self.bytes(name).decode("utf-8"))

    def close(self):
        if self.archive:
            self.archive.close()


def diagnose(evidence: Evidence) -> dict:
    protocol = evidence.json("protocol.json")
    frozen_hash = evidence.json("freeze.json")["protocol_sha256"]
    if sha(evidence.bytes("protocol.json")) != frozen_hash:
        raise ValueError("Frozen protocol hash mismatch")
    checked = 0
    for arm, runtime in protocol["runtimes"].items():
        for name, expected in runtime["files"].items():
            if sha(evidence.bytes(f"runtime/{arm}/{name}")) != expected:
                raise ValueError(f"Frozen runtime hash mismatch: {arm}/{name}")
            checked += 1
    for name, expected in protocol["source_files"].items():
        if sha(evidence.bytes(f"sources/{name}.json")) != expected:
            raise ValueError(f"Frozen source hash mismatch: {name}")
        checked += 1

    cells = []
    for entry in protocol["plan"]:
        row = evidence.json(f"cells/{entry['id']}.json")
        if row["id"] != entry["id"] or row["arm"] != entry["arm"]:
            raise ValueError("Cell identity mismatch")
        audits = [json.loads(line) for line in row.get("audit_jsonl", "").splitlines() if line]
        streams = [a for a in audits if a.get("event") == "model_stream_complete"]
        # Failed streams can add turns without a completion audit. Do not shift
        # subsequent phases to the wrong call when that happens.
        aligned = len(streams) == len(row["turns"])
        calls = []
        for index, turn in enumerate(row["turns"]):
            chunks = [json.loads(line)["chunk"] for line in evidence.bytes(
                "cells/" + turn["chunks_file"]).decode("utf-8").splitlines() if line]
            messages = [c.get("message") or {} for c in chunks]
            content = "".join(m.get("content") or "" for m in messages)
            reasoning_chars = sum(len(m.get("thinking") or "") for m in messages)
            proposals = [call for m in messages for call in m.get("tool_calls") or []]
            requested, effective = turn["requested"], turn["effective"]
            terminal = turn.get("terminal") or {}
            exposed = [s["function"]["name"] for s in effective.get("tools") or []]
            calls.append({
                "call": index + 1, "chunks_file": turn["chunks_file"],
                "phase": streams[index].get("phase") if aligned else None,
                "requested_think": requested.get("think"), "effective_think": effective.get("think"),
                "requested_num_predict": requested.get("options", {}).get("num_predict"),
                "effective_num_predict": effective.get("options", {}).get("num_predict"),
                "generated_tokens": terminal.get("eval_count"),
                "prompt_tokens": terminal.get("prompt_eval_count"),
                "done_reason": terminal.get("done_reason"), "seconds": turn.get("seconds"),
                "reasoning_chars": reasoning_chars, "content_chars": len(content),
                "content_sha256": sha(content.encode("utf-8")),
                "output_kind": ("tool_request" if proposals else "content" if content
                                else "reasoning_only" if reasoning_chars else "empty"),
                "exposed_tools": exposed, "tool_proposals": proposals,
                "proposals_without_schema": [p["function"]["name"] for p in proposals
                                             if p["function"]["name"] not in exposed],
                "error": turn.get("error"),
            })
        metering, error = row.get("metering", {}), row.get("error")
        limits = protocol["budgets"]
        ceiling = None
        if error and error["type"] == "StudyBudgetExceeded":
            if metering.get("calls", 0) >= limits["model_calls"]:
                ceiling = "model_calls"
            elif metering.get("generated_tokens_reserved_or_observed", 0) >= limits["generated_tokens"]:
                ceiling = "generated_tokens"
            elif metering.get("tools", 0) >= limits["tool_calls"]:
                ceiling = "tool_calls"
            else:
                ceiling = "elapsed_or_unknown_usage"
        delivered = row.get("response") or ""
        cells.append({
            "id": row["id"], "arm": row["arm"], "stratum": row["stratum"],
            "error": error, "ceiling": ceiling, "metering": metering, "calls": calls,
            "delivered_chars": len(delivered), "delivered_sha256": sha(delivered.encode("utf-8")),
            "nonempty": bool(delivered.strip()),
            "nonempty_draft_call_numbers": [c["call"] for c in calls if c["content_chars"]],
            "tool_executions": len(row["tool_results"]),
            "tool_errors": [t for t in row["tool_results"] if t.get("error")],
            "audit_events": dict(Counter(a["event"] for a in audits)),
        })
    calls = [c for row in cells for c in row["calls"]]
    groups = []
    for stratum, arm in sorted({(r["stratum"], r["arm"]) for r in cells}):
        rows = [r for r in cells if (r["stratum"], r["arm"]) == (stratum, arm)]
        groups.append({"stratum": stratum, "arm": arm, "cells": len(rows),
                       "nonempty": sum(r["nonempty"] for r in rows),
                       "error_cells": sum(bool(r["error"]) for r in rows),
                       "calls": sum(len(r["calls"]) for r in rows)})
    return {"schema": 1, "purpose": "execution_diagnosis_only_no_quality_grades",
            "protocol_sha256": frozen_hash, "frozen_files_verified": checked,
            "strict_revision": protocol["runtimes"]["strict"]["revision"],
            "token_partition": "Ollama reports total generated tokens; reasoning/content are character counts, not token counts.",
            "cells": cells, "groups": groups,
            "totals": {"cells": len(cells), "calls": len(calls),
                       "nonempty": sum(r["nonempty"] for r in cells),
                       "errors": dict(Counter(r["error"]["type"] for r in cells if r["error"])),
                       "ceilings": dict(Counter(r["ceiling"] for r in cells if r["ceiling"])),
                       "generated_tokens": sum(c["generated_tokens"] or 0 for c in calls),
                       "unknown_token_calls": sum(c["generated_tokens"] is None for c in calls),
                       "reasoning_only_length_calls": sum(c["done_reason"] == "length" and
                                                          c["output_kind"] == "reasoning_only" for c in calls)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, help="Original ZIP or extracted study directory")
    parser.add_argument("--output", type=Path, help="New JSON file outside the original evidence")
    args = parser.parse_args()
    source = args.evidence.resolve()
    if args.output:
        output = args.output.resolve()
        if output == source or (source.is_dir() and output.is_relative_to(source)):
            parser.error("Output must be outside the frozen evidence")
    evidence = Evidence(source)
    try:
        result = diagnose(evidence)
    finally:
        evidence.close()
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
        print(json.dumps(result["totals"]))
    else:
        print(payload)


if __name__ == "__main__":
    main()
