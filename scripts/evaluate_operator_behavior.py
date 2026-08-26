from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from localpilot.agent import SYSTEM_PROMPT
from localpilot.behavior_eval import SCENARIOS, score_response, summarize_scores


def _system_prompt(root: Path, revision: str) -> str:
    if revision == "working-tree":
        return SYSTEM_PROMPT
    result = subprocess.run(
        ["git", "show", f"{revision}:localpilot/agent.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    module = ast.parse(result.stdout)
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SYSTEM_PROMPT"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise RuntimeError(f"SYSTEM_PROMPT was not found at {revision}")


def evaluate(root: Path, revision: str, model: str) -> dict[str, object]:
    from ollama import chat

    prompt = _system_prompt(root, revision)
    scores = []
    for scenario in SCENARIOS:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": scenario.prompt},
        ]
        response = chat(
            model=model,
            messages=messages,
            think="high",
            options={"temperature": 0.1, "seed": 42, "num_ctx": 65536, "num_predict": 4096},
        )
        message = response.get("message", response)
        content = str(
            message.get("content", "")
            if isinstance(message, dict)
            else getattr(message, "content", "")
        )
        if not content.strip():
            thinking = str(
                message.get("thinking", "")
                if isinstance(message, dict)
                else getattr(message, "thinking", "")
            )
            continuation = chat(
                model=model,
                messages=[
                    *messages,
                    {"role": "assistant", "content": "", "thinking": thinking},
                    {
                        "role": "user",
                        "content": "Continue from that reasoning and give only the concise final answer.",
                    },
                ],
                think="high",
                options={"temperature": 0.1, "seed": 42, "num_ctx": 65536, "num_predict": 4096},
            )
            message = continuation.get("message", continuation)
            content = str(
                message.get("content", "")
                if isinstance(message, dict)
                else getattr(message, "content", "")
            )
            if not content.strip():
                final_thinking = str(
                    message.get("thinking", "")
                    if isinstance(message, dict)
                    else getattr(message, "thinking", "")
                )
                rendered = chat(
                    model=model,
                    messages=[
                        *messages,
                        {"role": "assistant", "content": "", "thinking": thinking},
                        {"role": "user", "content": "Continue and give the final answer."},
                        {"role": "assistant", "content": "", "thinking": final_thinking},
                        {
                            "role": "user",
                            "content": "Render only the concise user-visible answer now, with no hidden reasoning.",
                        },
                    ],
                    think=False,
                    options={"temperature": 0.1, "seed": 42, "num_ctx": 65536, "num_predict": 1200},
                )
                message = rendered.get("message", rendered)
                content = str(
                    message.get("content", "")
                    if isinstance(message, dict)
                    else getattr(message, "content", "")
                )
        scores.append(score_response(scenario, content))
    return {
        "revision": revision,
        "model": model,
        "summary": summarize_scores(scores),
        "scenarios": [score.as_dict() for score in scores],
    }


def rescore(results: list[dict[str, object]]) -> list[dict[str, object]]:
    scenarios_by_name = {scenario.name: scenario for scenario in SCENARIOS}
    rescored = []
    for result in results:
        scores = [
            score_response(
                scenarios_by_name[str(item["name"])],
                str(item.get("response", "")),
            )
            for item in result.get("scenarios", [])
            if str(item.get("name", "")) in scenarios_by_name
        ]
        rescored.append({
            "revision": result.get("revision"),
            "model": result.get("model"),
            "summary": summarize_scores(scores),
            "scenarios": [score.as_dict() for score in scores],
        })
    return rescored


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the bounded LocalPilot initiative/evidence behavioral evaluation."
    )
    parser.add_argument("--revision", action="append")
    parser.add_argument("--rescore", type=Path)
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rescore:
        results = rescore(json.loads(args.rescore.read_text(encoding="utf-8")))
    elif args.revision:
        results = [evaluate(ROOT, revision, args.model) for revision in args.revision]
    else:
        parser.error("provide --revision at least once or use --rescore")
    rendered = json.dumps(results, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(rendered)


if __name__ == "__main__":
    main()
