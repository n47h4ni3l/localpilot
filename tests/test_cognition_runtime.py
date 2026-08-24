from types import SimpleNamespace

from localpilot.agent import LocalPilotAgent
from localpilot.cognition_probe import run_cognition_probe
from localpilot.config import Config


def _chunk(
    *,
    content: str = "",
    thinking: str = "",
    tool_calls=None,
    done=None,
    done_reason=None,
    prompt_eval_count=None,
    eval_count=None,
    total_duration=None,
    eval_duration=None,
):
    return SimpleNamespace(
        message=SimpleNamespace(
            content=content,
            thinking=thinking,
            tool_calls=list(tool_calls or []),
        ),
        done=done,
        done_reason=done_reason,
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
        total_duration=total_duration,
        eval_duration=eval_duration,
    )


def _call(name: str, arguments: dict | None = None):
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=dict(arguments or {}))
    )


def test_stream_runtime_preserves_termination_and_token_metadata(tmp_path):
    config = Config()
    agent = LocalPilotAgent(config, tmp_path)

    def fake_chat(**kwargs):
        return iter(
            [
                _chunk(thinking="private reasoning"),
                _chunk(
                    done=True,
                    done_reason="length",
                    prompt_eval_count=31000,
                    eval_count=4096,
                    total_duration=123456,
                    eval_duration=100000,
                ),
            ]
        )

    message = agent._stream_chat_message(
        fake_chat,
        think="high",
        options={"num_predict": 4096},
        phase="test_synthesis",
        turn_no=7,
    )

    assert message["content"] == ""
    runtime = agent._last_stream_runtime
    assert runtime["phase"] == "test_synthesis"
    assert runtime["turn"] == 7
    assert runtime["done"] is True
    assert runtime["done_reason"] == "length"
    assert runtime["prompt_eval_count"] == 31000
    assert runtime["eval_count"] == 4096
    assert runtime["num_predict"] == 4096
    assert runtime["runtime_classification"] == "generation_limit"
    assert runtime["context_used_percent"] == round(100.0 * 31000 / 32768, 2)
    assert runtime["reasoning_chars"] == len("private reasoning")

    audit = agent.audit.latest("model_stream_complete")
    assert audit is not None
    assert audit["runtime_classification"] == "generation_limit"
    assert audit["done_reason"] == "length"
    assert audit["eval_count"] == 4096


def test_tool_result_success_distinguishes_real_read_failures():
    assert LocalPilotAgent._tool_result_success("Repository tree: .\nlocalpilot/") is True
    assert LocalPilotAgent._tool_result_success(
        "Private GitHub pull request (read-only).\nGitHub read failed: authentication required"
    ) is False
    assert LocalPilotAgent._tool_result_success("Tool error: FileNotFoundError: missing") is False


def test_cognition_probe_requires_unpredictable_tool_then_validates_reasoned_answer(tmp_path):
    config = Config()
    calls = []
    streams = iter(
        [
            [
                _chunk(
                    thinking="I should discover the manifest first.",
                    tool_calls=[_call("get_probe_manifest")],
                    done=True,
                    done_reason="stop",
                    prompt_eval_count=200,
                    eval_count=50,
                )
            ],
            [
                _chunk(
                    thinking="I need every distinct fragment before reconciling.",
                    tool_calls=[
                        _call("get_probe_fragment", {"fragment_id": "frag-a"}),
                        _call("get_probe_fragment", {"fragment_id": "frag-b"}),
                        _call("get_probe_fragment", {"fragment_id": "frag-c"}),
                    ],
                    done=True,
                    done_reason="stop",
                    prompt_eval_count=320,
                    eval_count=80,
                ),
            ],
            [
                _chunk(thinking="The XOR check reconciles, so I can add and reconstruct the nonce."),
                _chunk(
                    content='{"sum": 42, "nonce": "fresh-run-value"}',
                    done=True,
                    done_reason="stop",
                    prompt_eval_count=320,
                    eval_count=80,
                ),
            ],
        ]
    )

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return iter(next(streams))

    result = run_cognition_probe(
        config,
        tmp_path,
        chat=fake_chat,
        facts={
            "left": 19,
            "right": 23,
            "nonce": "fresh-run-value",
            "fragment_ids": ["frag-a", "frag-b", "frag-c"],
        },
    )

    assert result.ok is True
    assert result.stage == "passed"
    assert result.tool_called is True
    assert len(calls) == 3
    assert calls[0]["think"] == "high"
    assert calls[1]["think"] == "high"
    assert calls[2]["think"] == "high"
    assert "tools" in calls[0]
    assert "tools" in calls[1]
    assert "tools" in calls[2]
    assert result.observation_count == 4
    assert result.tool_rounds == 2
    assert result.tool_rounds <= result.hard_budget
    assert result.runtime["done_reason"] == "stop"


def test_cognition_probe_reports_generation_limit_instead_of_guessing(tmp_path):
    config = Config()
    streams = iter(
        [
            [
                _chunk(
                    tool_calls=[_call("get_probe_manifest")],
                    done=True,
                    done_reason="stop",
                    prompt_eval_count=200,
                    eval_count=40,
                )
            ],
            [
                _chunk(
                    tool_calls=[
                        _call("get_probe_fragment", {"fragment_id": "one"}),
                        _call("get_probe_fragment", {"fragment_id": "two"}),
                        _call("get_probe_fragment", {"fragment_id": "three"}),
                    ],
                    done=True,
                    done_reason="stop",
                    prompt_eval_count=350,
                    eval_count=60,
                )
            ],
            [
                _chunk(thinking="reasoning that consumes the budget"),
                _chunk(
                    done=True,
                    done_reason="length",
                    prompt_eval_count=500,
                    eval_count=4096,
                ),
            ],
        ]
    )

    result = run_cognition_probe(
        config,
        tmp_path,
        chat=lambda **kwargs: iter(next(streams)),
        facts={
            "left": 7,
            "right": 11,
            "nonce": "another-fresh-value",
            "fragment_ids": ["one", "two", "three"],
        },
    )

    assert result.ok is False
    assert result.stage == "generation_limit"
    assert result.runtime["runtime_classification"] == "generation_limit"
    assert result.runtime["eval_count"] == 4096
    assert result.observation_count == 4


def test_cognition_probe_fails_closed_when_multistep_research_exceeds_hard_budget(tmp_path):
    config = Config()
    config.agent.research_hard_tool_rounds = 1
    streams = iter(
        [
            [_chunk(tool_calls=[_call("get_probe_manifest")], done=True, done_reason="stop")],
            [
                _chunk(
                    tool_calls=[_call("get_probe_fragment", {"fragment_id": "x"})],
                    done=True,
                    done_reason="stop",
                )
            ],
        ]
    )

    result = run_cognition_probe(
        config,
        tmp_path,
        chat=lambda **kwargs: iter(next(streams)),
        facts={
            "left": 2,
            "right": 3,
            "nonce": "unpredictable",
            "fragment_ids": ["x", "y", "z"],
        },
    )

    assert result.ok is False
    assert result.stage == "hard_budget"
    assert result.observation_count == 1
    assert result.tool_rounds == result.hard_budget == 1
