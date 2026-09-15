from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import external_corpus_v1 as core

BUILD_SPEC = importlib.util.spec_from_file_location("external_builder", SCRIPTS / "build_external_corpus_v1.py")
assert BUILD_SPEC is not None and BUILD_SPEC.loader is not None
builder = importlib.util.module_from_spec(BUILD_SPEC)
sys.modules[BUILD_SPEC.name] = builder
BUILD_SPEC.loader.exec_module(builder)

ACQUIRE_SPEC = importlib.util.spec_from_file_location("external_acquirer", SCRIPTS / "acquire_external_corpus_v1.py")
assert ACQUIRE_SPEC is not None and ACQUIRE_SPEC.loader is not None
acquirer = importlib.util.module_from_spec(ACQUIRE_SPEC)
sys.modules[ACQUIRE_SPEC.name] = acquirer
ACQUIRE_SPEC.loader.exec_module(acquirer)


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, truncation=False, tools=None):
        assert tokenize and not truncation
        size = sum(len((item.get("content") or "").split()) + 5 * len(item.get("tool_calls", [])) for item in messages)
        return list(range(size + 4))


def candidate(identifier: str, *, prompt: str = "Inspect the failure before editing.", target: str = "Run focused tests and report the result.", family: str = "family") -> core.Candidate:
    return core.Candidate(
        dataset="codeact_instruct",
        original_id=identifier,
        source_path="codeactinstruct/full_std.jsonl",
        messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": target}],
        task_type="agentic_code_execution",
        difficulty="hard",
        family=family,
        category="code_generation_apps",
        subskill="execute_observe_revise",
        scores=core.score_components(verification=1, difficulty=0.8, relevance=0.9, diversity=0.7),
        metadata={"action_observation_cycles": 2},
    )


class GlobalFilterTests(unittest.TestCase):
    def test_rejects_malformed_roles_empty_outputs_and_invalid_unicode(self) -> None:
        self.assertEqual(core.global_content_rejection([{"role": "alien", "content": "x"}]), "malformed_messages")
        self.assertEqual(core.global_content_rejection([{"role": "user", "content": "x"}]), "malformed_roles")
        self.assertEqual(
            core.global_content_rejection([{"role": "user", "content": "x"}, {"role": "assistant", "content": "\ud800"}]),
            "invalid_unicode",
        )

    def test_rejects_placeholders_truncation_boilerplate_and_prompt_copying(self) -> None:
        base = [{"role": "user", "content": "Please solve this carefully."}]
        self.assertEqual(core.global_content_rejection(base + [{"role": "assistant", "content": "<TODO>"}]), "unresolved_placeholder")
        self.assertEqual(core.global_content_rejection(base + [{"role": "assistant", "content": "answer [truncated]"}]), "truncated_output")
        repeated = "\n".join(["same useful-looking line"] * 12)
        self.assertEqual(core.global_content_rejection(base + [{"role": "assistant", "content": repeated}]), "boilerplate_heavy")
        prompt = " ".join(f"word{index}" for index in range(30))
        self.assertEqual(core.global_content_rejection([{"role": "user", "content": prompt}, {"role": "assistant", "content": prompt}]), "prompt_copying")

    def test_rejects_provider_shaped_credentials_without_recording_the_value(self) -> None:
        credentials = (
            "ghp_" + "A" * 36,
            "hf_" + "B" * 30,
            "AKIA" + "C" * 16,
            "-----BEGIN " + "PRIVATE KEY-----",
        )
        for credential in credentials:
            with self.subTest(prefix=credential[:4]):
                messages = [
                    {"role": "user", "content": "Inspect this configuration."},
                    {"role": "assistant", "content": "The embedded value is " + credential},
                ]
                self.assertEqual(core.global_content_rejection(messages), "sensitive_credential_pattern")

    def test_training_messages_must_end_with_an_assistant_target(self) -> None:
        messages = [
            {"role": "user", "content": "Inspect this."},
            {"role": "assistant", "content": "I will inspect it."},
            {"role": "user", "content": "Tool result: finished"},
        ]
        self.assertEqual(core.global_content_rejection(messages), "malformed_final_role")


class DuplicateAndSplitTests(unittest.TestCase):
    def test_exact_near_and_code_duplicates_are_rejected(self) -> None:
        exact = core.Deduplicator()
        first = candidate("one")
        exact.add(first)
        self.assertEqual(exact.reason(candidate("two")), "exact_duplicate")

        near = core.Deduplicator()
        near.add(candidate("three", target="Use evidence from the repository and run the focused tests before finalizing the repair."))
        self.assertEqual(
            near.reason(candidate("four", target="Use evidence from the repository and run the focused tests before finalizing the fix.")),
            "near_duplicate",
        )

        code = core.Deduplicator()
        code.add(candidate("five", target="```python\ndef add(a, b):\n    return a + b\n```"))
        self.assertEqual(code.reason(candidate("six", target="```python\ndef add(x, y):\n    return x + y\n```")), "code_duplicate")

    def test_near_duplicates_are_found_across_different_family_keys(self) -> None:
        dedup = core.Deduplicator()
        first = candidate(
            "one", family="family-one",
            prompt="Investigate why the parser rejects empty arrays before changing the implementation.",
            target="Inspect the parser, correct the empty-array branch, and run all focused parser tests.",
        )
        duplicate = candidate(
            "two", family="family-two",
            prompt="Investigate why the parser rejects empty arrays before changing the implementation.",
            target="Inspect the parser, correct the empty-array branch, and run all focused parser tests successfully.",
        )
        dedup.add(first)
        self.assertEqual(dedup.reason(duplicate), "near_duplicate")

    def test_near_duplicate_search_ignores_injected_system_tool_boilerplate(self) -> None:
        shared = "Available tools:\n" + " ".join(f"schema_token_{index}" for index in range(200))
        first = candidate(
            "one", prompt="Find weather for Adelaide", target="Call weather for Adelaide", family="one",
        )
        second = candidate(
            "two", prompt="Create a calendar event in Tokyo", target="Call calendar for Tokyo", family="two",
        )
        first.messages.insert(0, {"role": "system", "content": shared})
        second.messages.insert(0, {"role": "system", "content": shared})
        dedup = core.Deduplicator()
        dedup.add(first)
        self.assertIsNone(dedup.reason(second))

    def test_family_grouped_split_is_deterministic_and_never_splits_a_family(self) -> None:
        values = [candidate(str(index), family=f"family-{index // 2}") for index in range(20)]
        first = core.grouped_split(values)
        second = core.grouped_split(reversed(values))
        self.assertEqual(first, second)
        self.assertEqual(set(first), {value.family for value in values})
        self.assertTrue(1 <= list(first.values()).count("validation") <= 3)

    def test_family_grouped_split_avoids_giant_validation_overshoot(self) -> None:
        values = [candidate(f"giant-{index}", family="giant") for index in range(90)]
        values.extend(candidate(f"small-{index}", family=f"small-{index}") for index in range(10))
        split = core.grouped_split(values)
        validation_rows = sum(1 for value in values if split[value.family] == "validation")
        self.assertEqual(validation_rows, 10)

    def test_cross_source_problem_variants_stay_in_one_split(self) -> None:
        values = [candidate(f"other-{index}", family=f"other-{index}") for index in range(10)]
        repair = candidate("repair", family="nemotron-family")
        tests = candidate("tests", family="different-source-family")
        repair.metadata["problem_identity_sha256"] = "shared-problem"
        tests.metadata["problem_identity_sha256"] = "shared-problem"
        split = core.grouped_split(values + [repair, tests])
        self.assertEqual(split[repair.family], split[tests.family])

    def test_candidate_pool_is_insertion_order_independent(self) -> None:
        values = [candidate(str(index)) for index in range(20)]
        for index, value in enumerate(values):
            value.scores["weighted_total"] = float(index)
        forward, reverse = core.CandidatePool({"x": 5}), core.CandidatePool({"x": 5})
        for value in values:
            forward.add("x", value)
        for value in reversed(values):
            reverse.add("x", value)
        self.assertEqual([item.original_id for item in forward.finish()["x"]], [item.original_id for item in reverse.finish()["x"]])

    def test_dataset_keys_and_issue_caps_are_enforced_quality_first(self) -> None:
        dedup = core.Deduplicator(include_dataset_keys=True)
        first = candidate("first", prompt="First distinct request", target="First distinct response", family="issue")
        first.metadata["dataset_duplicate_key_sha256"] = "shared"
        dedup.add(first)
        duplicate = candidate("duplicate", prompt="Another request", target="Another response", family="other")
        duplicate.metadata["dataset_duplicate_key_sha256"] = "shared"
        self.assertEqual(dedup.reason(duplicate), "dataset_key_duplicate")

        issues = core.Deduplicator(include_dataset_keys=True)
        repair = candidate("repair", prompt="Repair request alpha", target="Repair result alpha", family="same-issue")
        repair.dataset, repair.subskill = "nemotron_swe_v2", "repair"
        issues.add(repair)
        same = candidate("same", prompt="Different wording beta", target="Different result beta", family="same-issue")
        same.dataset, same.subskill = "nemotron_swe_v2", "repair"
        self.assertEqual(issues.reason(same), "same_family_subskill_duplicate")
        tests = candidate("tests", prompt="Test request gamma", target="Test result gamma", family="same-issue")
        tests.dataset, tests.subskill = "nemotron_swe_v2", "test_generation"
        self.assertIsNone(issues.reason(tests))
        issues.add(tests)
        localization = candidate("localize", prompt="Locate request delta", target="Locate result delta", family="same-issue")
        localization.dataset, localization.subskill = "nemotron_swe_v2", "localization"
        self.assertEqual(issues.reason(localization), "same_family_example_cap")

    def test_same_problem_cap_applies_to_every_dataset(self) -> None:
        dedup = core.Deduplicator(include_dataset_keys=True)
        first = candidate("first", prompt="Request alpha", target="Response alpha", family="source-family-one")
        second = candidate("second", prompt="Request beta", target="Response beta", family="source-family-two")
        first.task_type = second.task_type = "agentic_code_execution"
        first.metadata["problem_identity_sha256"] = "same-problem"
        second.metadata["problem_identity_sha256"] = "same-problem"
        dedup.add(first)
        self.assertEqual(dedup.reason(second), "same_family_subskill_duplicate")

    def test_duplicate_match_reports_retained_representative_cluster(self) -> None:
        dedup = core.Deduplicator()
        retained = candidate("retained", target="```python\ndef add(a, b):\n    return a + b\n```")
        dedup.add(retained)
        left = candidate("left", family="left", target="```python\ndef add(x, y):\n    return x + y\n```")
        right = candidate("right", family="right", target="```python\ndef add(first, second):\n    return first + second\n```")
        left_match = dedup.match(left)
        right_match = dedup.match(right)
        self.assertEqual(left_match[0], "code_duplicate")
        self.assertEqual(right_match[0], "code_duplicate")
        self.assertEqual(left_match[1], right_match[1])
        self.assertIn(retained.exact_key, str(left_match[1]))


class DatasetSpecificFilterTests(unittest.TestCase):
    def test_tool_calls_require_known_tools_required_args_and_plausible_types(self) -> None:
        tools = [{"name": "fetch", "parameters": {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}}]
        self.assertIsNone(core.validate_calls(tools, [{"name": "fetch", "arguments": {"count": 2}}]))
        self.assertEqual(core.validate_calls(tools, [{"name": "missing", "arguments": {"count": 2}}]), "unknown_tool")
        self.assertEqual(core.validate_calls(tools, [{"name": "fetch", "arguments": {}}]), "missing_required_tool_argument")
        self.assertEqual(core.validate_calls(tools, [{"name": "fetch", "arguments": {"count": "two"}}]), "implausible_tool_argument_type")

    def test_tool_contracts_preserve_semantics_in_prompt_and_signature(self) -> None:
        weather = [{"name": "lookup", "description": "Fetch weather", "parameters": {"type": "object", "properties": {"unit": {"type": "string", "enum": ["c", "f"], "description": "Temperature unit"}}, "required": ["unit"]}}]
        stocks = [{"name": "lookup", "description": "Fetch stock price", "parameters": {"type": "object", "properties": {"unit": {"type": "string", "enum": ["usd"], "description": "Currency"}}, "required": ["unit"]}}]
        compact = core.compact_tools(weather)[0]
        self.assertEqual(compact["description"], "Fetch weather")
        self.assertEqual(compact["parameters"]["properties"]["unit"]["enum"], ["c", "f"])
        self.assertNotEqual(core.canonical_tool_signature(weather), core.canonical_tool_signature(stocks))

    def test_xlam_one_call_requires_distractor_tools(self) -> None:
        self.assertTrue(core.xlam_trivial_one_call(1, 1))
        self.assertFalse(core.xlam_trivial_one_call(1, 4))
        self.assertFalse(core.xlam_trivial_one_call(2, 1))

    def test_xlam_caps_each_called_contract_across_distractor_sets(self) -> None:
        weather = {"name": "weather_lookup", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
        first_set = [weather, {"name": "news_lookup", "parameters": {"type": "object", "properties": {}}}]
        second_set = [weather, {"name": "stock_lookup", "parameters": {"type": "object", "properties": {}}}]
        signature = core.canonical_tool_contracts(first_set)["weather_lookup"]
        self.assertEqual(signature, core.canonical_tool_contracts(second_set)["weather_lookup"])
        counters = defaultdict(Counter)
        counters["called_tool_signature"][signature] = 10
        item = candidate("weather")
        item.dataset = "xlam_60k"
        item.tool_signature = core.canonical_tool_signature(second_set)
        item.metadata = {"called_tool_signatures": [signature], "api_categories": ["weather_lookup"]}
        self.assertEqual(builder._cap_reason(item, [], counters), "api_signature_cap")

    def test_xlam_final_category_cap_uses_full_api_name_and_final_tranche(self) -> None:
        values = []
        for index in range(100):
            item = candidate(f"xlam-{index}", prompt=f"Prompt {index}", target=f"Answer {index}")
            item.dataset = "xlam_60k"
            category = "weather_lookup" if index < 10 else f"unique_api_{index}"
            item.metadata = {"api_categories": [category]}
            values.append((item, 50))
        kept = builder._prune_xlam_api_categories(values, builder.Stats())
        counts = Counter(category for item, _ in kept for category in item.metadata["api_categories"])
        cap = max(1, int(sum(item.dataset == "xlam_60k" for item, _ in kept) * 0.02))
        self.assertLessEqual(max(counts.values()), cap)
        self.assertIn("weather_lookup", counts)

    def test_xlam_native_prompt_does_not_duplicate_textual_tool_schema(self) -> None:
        row = {
            "id": "xlam-1",
            "query": "Look up Adelaide weather and return the temperature.",
            "tools": json.dumps([
                {"name": "weather_lookup", "description": "Fetch weather", "parameters": {
                    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"],
                }},
                {"name": "news_lookup", "parameters": {"type": "object", "properties": {}}},
                {"name": "stock_lookup", "parameters": {"type": "object", "properties": {}}},
                {"name": "time_lookup", "parameters": {"type": "object", "properties": {}}},
            ]),
            "answers": json.dumps([{"name": "weather_lookup", "arguments": {"city": "Adelaide"}}]),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "xlam-function-calling-60k"
            source.mkdir()
            (source / "xlam_function_calling_60k.json").write_text(json.dumps([row]), encoding="utf-8")
            with mock.patch.object(builder, "sha256_file", return_value=core.DATASETS["xlam_60k"]["expected_sha256"]):
                pools, _ = builder.load_xlam(root, builder.Stats())
        item = next(iter(pools.values()))[0]
        self.assertEqual(item.metadata["native_messages"], [{"role": "user", "content": row["query"]}])
        record = core.candidate_record(item, "train", 100)
        examples = builder.expand_training_examples([record])
        self.assertEqual(examples[0]["prompt"], [{"role": "user", "content": row["query"]}])
        self.assertNotIn("Available tools", json.dumps(examples[0]["prompt"]))

    def test_agentless_subskills_need_mechanical_evidence(self) -> None:
        repair = [
            {"role": "user", "content": "Fix the bug in src/a.py where the current line is old."},
            {"role": "assistant", "content": "diff --git a/src/a.py b/src/a.py\n@@ -1 +1 @@\n-old\n+new"},
        ]
        self.assertEqual(core.classify_agentless(repair), "repair")
        self.assertTrue(core.verify_agentless("repair", repair))
        superficial = [
            {"role": "user", "content": "Fix src/a.py."},
            {"role": "assistant", "content": "diff --git a/src/a.py b/src/a.py"},
        ]
        self.assertFalse(core.verify_agentless("repair", superficial))
        search_replace = [
            {"role": "user", "content": "Fix src/a.py containing old_value = 1."},
            {"role": "assistant", "content": "```diff\n### src/a.py\n<<<<<<< SEARCH\nold_value = 1\n=======\nnew_value = 2\n>>>>>>> REPLACE\n```"},
        ]
        self.assertEqual(core.classify_agentless(search_replace), "repair")
        self.assertTrue(core.verify_agentless("repair", search_replace))
        localization = [
            {"role": "user", "content": "Provide a list of files. Repository: src/a.py and tests/test_a.py"},
            {"role": "assistant", "content": "src/a.py\ntests/test_a.py"},
        ]
        self.assertFalse(core.verify_agentless("localization", localization))
        localization[1]["content"] = "src/unseen.py"
        self.assertFalse(core.verify_agentless("localization", localization))

    def test_agentless_issue_identity_matches_real_template_variants(self) -> None:
        text = "The parser fails on empty input."
        bracketed = f"### GitHub Problem Description ###\n[ISSUE]{text}[/ISSUE]\n### Repository Structure ###"
        delimited = f"We are solving this.\n--- BEGIN ISSUE ---\n{text}\n--- END ISSUE ---\nRelevant files follow."
        xml = f"Task metadata\n<issue_description>{text}</issue_description>\nRepository follows."
        self.assertEqual(builder._issue_identity(bracketed), builder._issue_identity(delimited))
        self.assertEqual(builder._issue_identity(bracketed), builder._issue_identity(xml))

    def test_openhands_requires_detectable_inspect_edit_validation_sequence(self) -> None:
        row = {"messages": [
            {"role": "assistant", "tool_calls": [{"function": {"arguments": "rg symbol src"}}]},
            {"role": "assistant", "tool_calls": [{"function": {"arguments": "apply_patch src/a.py"}}]},
            {"role": "assistant", "tool_calls": [{"function": {"arguments": "pytest tests/test_a.py"}}]},
            {"role": "tool", "content": "1 failed"},
            {"role": "assistant", "content": "edit the implementation"},
            {"role": "tool", "content": "all tests passed"},
        ]}
        traits = core.openhands_traits(row)
        self.assertEqual(traits["action_count"], 3)
        self.assertTrue(traits["inspect_before_edit"])
        self.assertTrue(traits["validation_after_edit"])
        self.assertTrue(traits["failed_changed_passed"])
        row["messages"].insert(2, {"role": "assistant", "tool_calls": [{"function": {"arguments": "apply_patch src/a.py"}}]})
        self.assertTrue(core.openhands_traits(row)["repeated_identical_action"])

    def test_openhands_native_conversion_removes_scratch_and_converts_finish(self) -> None:
        tools = [
            {"type": "function", "function": {"name": "think", "parameters": {"type": "object", "properties": {"thought": {"type": "string"}}, "required": ["thought"]}}},
            {"type": "function", "function": {"name": "inspect", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "finish", "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}}},
        ]
        row = {"tools": tools, "messages": [
            {"role": "system", "content": "provider-only scaffold"},
            {"role": "user", "content": "Fix the parser."},
            {"role": "assistant", "content": "private scratch", "tool_calls": [{"id": "thought-1", "function": {"name": "think", "arguments": '{"thought":"inspect first"}'}}]},
            {"role": "tool", "name": "think", "tool_call_id": "thought-1", "content": "continue"},
            {"role": "assistant", "content": "private scratch", "tool_calls": [{"id": "inspect-1", "function": {"name": "inspect", "arguments": '{"path":"src/parser.py"}'}}]},
            {"role": "tool", "name": "inspect", "tool_call_id": "inspect-1", "content": "line 7 is incomplete"},
            {"role": "assistant", "content": "done", "tool_calls": [{"id": "finish-1", "function": {"name": "finish", "arguments": '{"message":"Fixed the parser and tests pass."}'}}]},
        ]}
        messages, traits, reason = core.openhands_trajectory(row)
        self.assertIsNone(reason)
        self.assertEqual([tool["function"]["name"] for tool in traits["native_tools"]], ["inspect"])
        self.assertEqual(traits["internal_think_call_count_removed"], 1)
        self.assertTrue(traits["finish_converted_to_final_answer"])
        self.assertEqual(traits["native_messages"][-1], {"role": "assistant", "content": "Fixed the parser and tests pass."})
        serialized = json.dumps(traits["native_messages"])
        self.assertNotIn("private scratch", serialized)
        self.assertNotIn("provider-only scaffold", serialized)
        self.assertEqual(messages[-1]["content"], "Fixed the parser and tests pass.")

    def test_openhands_native_conversion_rejects_bad_result_and_finish_order(self) -> None:
        tools = [
            {"type": "function", "function": {"name": "inspect", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "finish", "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}}},
        ]
        bad_name = {"tools": tools, "messages": [
            {"role": "user", "content": "Inspect it."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "one", "function": {"name": "inspect", "arguments": '{"path":"src/a.py"}'}}]},
            {"role": "tool", "name": "finish", "tool_call_id": "one", "content": "contents"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "done", "function": {"name": "finish", "arguments": '{"message":"Done."}'}}]},
        ]}
        self.assertEqual(core.openhands_trajectory(bad_name)[2], "invalid_tool_result_name")
        early_finish = json.loads(json.dumps(bad_name))
        early_finish["messages"] = [early_finish["messages"][0], early_finish["messages"][-1], {"role": "user", "content": "extra"}]
        self.assertEqual(core.openhands_trajectory(early_finish)[2], "invalid_finish_order")

    def test_agentic_tool_sequence_rejects_bad_order_and_detects_recovery(self) -> None:
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}]
        bad = {"tools": tools, "messages": [{"role": "user", "content": "Find it"}, {"role": "tool", "tool_call_id": "nope", "content": "x"}]}
        self.assertEqual(core.tool_trajectory(bad)[2], "invalid_tool_result_order")
        good = {"tools": tools, "messages": [
            {"role": "user", "content": "Find it"},
            {"role": "assistant", "content": "private scratch reasoning", "tool_calls": [{"id": "one", "function": {"name": "lookup", "arguments": '{"q":"old"}'}}]},
            {"role": "tool", "tool_call_id": "one", "content": "error: not found"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "two", "function": {"name": "lookup", "arguments": '{"q":"new"}'}}]},
            {"role": "tool", "tool_call_id": "two", "content": "found"},
            {"role": "assistant", "content": "The result is found."},
        ]}
        _, traits, reason = core.tool_trajectory(good)
        self.assertIsNone(reason)
        self.assertTrue(traits["error_recovered"])
        self.assertEqual(traits["recovered_error_count"], 1)
        self.assertIn("native_tools", traits)
        self.assertIn("native_messages", traits)
        native_call = next(message for message in traits["native_messages"] if message.get("tool_calls"))
        self.assertEqual(native_call["tool_calls"][0]["function"]["name"], "lookup")
        self.assertEqual(native_call["content"], "")
        self.assertNotIn("private scratch reasoning", json.dumps(traits["native_messages"]))

    def test_agentic_rejects_interleaved_messages_and_reused_call_ids(self) -> None:
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}]
        interleaved = {"tools": tools, "messages": [
            {"role": "user", "content": "Find it"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "one", "function": {"name": "lookup", "arguments": '{"q":"old"}'}}]},
            {"role": "assistant", "content": "I am still working."},
            {"role": "tool", "tool_call_id": "one", "content": "old result"},
            {"role": "assistant", "content": "old result"},
        ]}
        self.assertEqual(core.tool_trajectory(interleaved)[2], "invalid_tool_result_order")

        reused = {"tools": tools, "messages": [
            {"role": "user", "content": "Find both"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "one", "function": {"name": "lookup", "arguments": '{"q":"old"}'}}]},
            {"role": "tool", "tool_call_id": "one", "content": "old result"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "one", "function": {"name": "lookup", "arguments": '{"q":"new"}'}}]},
            {"role": "tool", "tool_call_id": "one", "content": "new result"},
            {"role": "assistant", "content": "old result and new result"},
        ]}
        self.assertEqual(core.tool_trajectory(reused)[2], "duplicate_tool_call_id")

    def test_agentic_rejects_mismatched_tool_result_name(self) -> None:
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}]
        mismatched = {"tools": tools, "messages": [
            {"role": "user", "content": "Find it"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "one", "function": {"name": "lookup", "arguments": '{"q":"item"}'}}]},
            {"role": "tool", "tool_call_id": "one", "name": "search", "content": "found item"},
            {"role": "assistant", "content": "The result is found item."},
        ]}
        self.assertEqual(core.tool_trajectory(mismatched)[2], "invalid_tool_result_name")

    def test_agentic_recovery_requires_changed_successful_call_after_observation(self) -> None:
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}]
        unrecovered = {"tools": tools, "messages": [
            {"role": "user", "content": "Find it"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "one", "function": {"name": "lookup", "arguments": '{"q":"old"}'}}]},
            {"role": "tool", "tool_call_id": "one", "content": '{"success": false, "error": "not found"}'},
            {"role": "assistant", "content": "No answer was produced."},
        ]}
        self.assertEqual(core.tool_trajectory(unrecovered)[2], "unrecovered_tool_error")

        same_message = {"tools": tools, "messages": [
            {"role": "user", "content": "Find both"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "one", "function": {"name": "lookup", "arguments": '{"q":"old"}'}},
                {"id": "two", "function": {"name": "lookup", "arguments": '{"q":"new"}'}},
            ]},
        ]}
        self.assertEqual(core.tool_trajectory(same_message)[2], "multiple_tool_calls_in_assistant_message")

    def test_agentic_requires_dependency_and_use_evidence(self) -> None:
        tools = [
            {"type": "function", "function": {"name": "find_order", "parameters": {"type": "object", "properties": {"customer": {"type": "string"}}, "required": ["customer"]}}},
            {"type": "function", "function": {"name": "order_price", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
        ]
        dependent = {"tools": tools, "messages": [
            {"role": "user", "content": "Find Alice's order price."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "one", "function": {"name": "find_order", "arguments": '{"customer":"Alice"}'}}]},
            {"role": "tool", "tool_call_id": "one", "content": '{"order_id":"abc123"}'},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "two", "function": {"name": "order_price", "arguments": '{"order_id":"abc123"}'}}]},
            {"role": "tool", "tool_call_id": "two", "content": '{"price":42,"currency":"AUD"}'},
            {"role": "assistant", "content": "Order abc123 costs 42 AUD."},
        ]}
        _, traits, reason = core.tool_trajectory(dependent)
        self.assertIsNone(reason)
        self.assertGreaterEqual(traits["dependency_link_count"], 1)
        self.assertEqual(traits["used_tool_result_count"], 2)

        unused = json.loads(json.dumps(dependent))
        unused["messages"][-1]["content"] = "The task is complete."
        self.assertEqual(core.tool_trajectory(unused)[2], "unused_tool_result")

        key_only = json.loads(json.dumps(dependent))
        key_only["messages"][3]["tool_calls"][0]["function"]["arguments"] = '{"order_id":"different"}'
        key_only["messages"][4]["content"] = '{"price":7,"currency":"AUD"}'
        key_only["messages"][-1]["content"] = "The different order costs 7 AUD."
        self.assertEqual(core.tool_trajectory(key_only)[2], "unused_tool_result")

    def test_agentic_rejects_terminal_inability_even_after_success(self) -> None:
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}]
        row = {"tools": tools, "messages": [
            {"role": "user", "content": "Find it"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "one", "function": {"name": "lookup", "arguments": '{"q":"item"}'}}]},
            {"role": "tool", "tool_call_id": "one", "content": "item 42"},
            {"role": "assistant", "content": "I am unable to provide the answer without your API key."},
        ]}
        self.assertEqual(core.tool_trajectory(row)[2], "terminal_failure")

    def test_opencode_requires_parseable_high_judgement_or_strong_tests(self) -> None:
        good = json.dumps({"logical_correctness": {"score": 5}, "requirement_conformance": {"score": 4}})
        self.assertIsNone(core.opencode_judgement(good, 3)[1])
        concern = json.dumps({"logical_correctness": {"score": 3}})
        self.assertEqual(core.opencode_judgement(concern, 10)[1], "llm_judgement_concern")
        self.assertIsNone(core.opencode_judgement("not json", 5)[1])
        self.assertEqual(core.opencode_judgement("not json", 4)[1], "unparseable_judgement")

    def test_opencode_loader_enforces_execution_and_judgement_gates(self) -> None:
        tests = ["edge input zero case", "boundary input one case", "state transition case"]
        base = {
            "id": "good",
            "input": "Implement a stateful parser with boundary handling.",
            "output": "```python\ndef parse(value):\n    return value.strip()\n```",
            "domain": "generic",
            "generation_algorithm": "evol-instruct",
            "llm_judgement": {"logical_correctness": {"score": 5}},
            "unit_tests": tests,
            "tests_execution_status": ["pass"] * 3,
            "average_test_score": 1.0,
        }
        rows = [base]
        bad_score = {**base, "id": "bad-score", "average_test_score": 0.99}
        too_few = {**base, "id": "too-few", "unit_tests": tests[:2], "tests_execution_status": ["pass"] * 2}
        concern = {**base, "id": "concern", "llm_judgement": {"logical_correctness": {"score": 3}}}
        strong = {
            **base,
            "id": "strong-unparseable",
            "generation_algorithm": "self-instruct",
            "llm_judgement": "not json",
            "unit_tests": tests + ["nested object input case", "multi function behavior case"],
            "tests_execution_status": ["pass"] * 5,
        }
        rows.extend([bad_score, too_few, concern, strong])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "OpenCodeInstruct/data"
            data.mkdir(parents=True)
            for index in range(50):
                (data / f"train-{index:05d}-of-00050.parquet").touch()
            synthetic = [(data / "train-00000-of-00050.parquet", index, row) for index, row in enumerate(rows)]
            stats = builder.Stats()
            with mock.patch.object(builder, "_parquet_rows", return_value=iter(synthetic)):
                pools, _ = builder.load_opencode(root, stats)
        self.assertEqual([item.original_id for item in pools["generic:evol-instruct"]], ["good"])
        self.assertEqual([item.original_id for item in pools["generic:self-instruct"]], ["strong-unparseable"])
        self.assertEqual(stats.rejections["open_code_instruct"]["test_score_not_perfect"], 1)
        self.assertEqual(stats.rejections["open_code_instruct"]["insufficient_verified_tests"], 1)
        self.assertEqual(stats.rejections["open_code_instruct"]["llm_judgement_concern"], 1)

    def test_codeact_rejects_repeated_actions_and_terminal_failure(self) -> None:
        repeated = {"content": [
            {"class_": "text_observation", "content": "Solve it"},
            {"class_": "code_action", "content": "print(1)"},
            {"class_": "text_observation", "content": "result one"},
            {"class_": "code_action", "content": "print(1)"},
            {"class_": "text_observation", "content": "result one"},
            {"class_": "message_action", "content": "Done"},
        ]}
        self.assertEqual(core.codeact_messages(repeated)[2], "repeated_identical_action")
        no_observation = {"content": [
            {"class_": "text_observation", "content": "Solve it"},
            {"class_": "code_action", "content": "print(1)"},
            {"class_": "message_action", "content": "Done"},
        ]}
        self.assertEqual(core.codeact_messages(no_observation)[2], "no_execution_evidence")
        recovered = {"content": [
            {"class_": "text_observation", "content": "Calculate the result"},
            {"class_": "code_action", "content": "print(1 / 0)"},
            {"class_": "text_observation", "content": "ZeroDivisionError: division failed"},
            {"class_": "code_action", "content": "print(1 / 2)"},
            {"class_": "text_observation", "content": "0.5 returned successfully"},
            {"class_": "message_action", "content": "The result is 0.5."},
        ]}
        _, traits, reason = core.codeact_messages(recovered)
        self.assertIsNone(reason)
        self.assertEqual(traits["action_observation_cycles"], 2)
        self.assertTrue(traits["error_recovered"])
        self.assertEqual(traits["native_tools"][0]["function"]["name"], "execute")
        native_calls = [message for message in traits["native_messages"] if message.get("tool_calls")]
        native_results = [message for message in traits["native_messages"] if message["role"] == "tool"]
        self.assertEqual(len(native_calls), 2)
        self.assertEqual(len(native_results), 2)
        self.assertEqual(native_calls[0]["tool_calls"][0]["id"], native_results[0]["tool_call_id"])
        self.assertEqual(traits["native_messages"][-1]["role"], "assistant")

    def test_codeact_constituents_are_immutable_and_ambiguous_wtq_is_excluded(self) -> None:
        self.assertNotIn("tabular/wiki_table_questions", core.CODEACT_CONSTITUENTS)
        for name, details in core.CODEACT_CONSTITUENTS.items():
            self.assertRegex(details["revision"], r"^[0-9a-f]{40}$", name)
            self.assertTrue(details["license"])
            self.assertIn(details["revision"], details["license_reference"])

    def test_codeact_math_subjects_share_one_upstream_family(self) -> None:
        algebra = candidate("algebra")
        geometry = candidate("geometry")
        algebra.metadata = {"constituent": "reasoning/algebra", "upstream_source_family": "MATH"}
        geometry.metadata = {"constituent": "reasoning/geometry", "upstream_source_family": "MATH"}
        self.assertEqual(builder._global_source_family(algebra), builder._global_source_family(geometry))

    def test_swecare_uses_review_patch_and_never_merged_patch_as_content(self) -> None:
        row = {
            "title": "Fix state race", "problem_statement": "State can race.",
            "commit_to_review": {"patch_to_review": "diff --git a/a.py b/a.py\n@@ -1 +1 @@\n-old\n+new\nUNRELATED_FULL_PATCH"},
            "reference_review_comments": [{"path": "a.py", "line": 1, "diff_hunk": "@@ -1 +1 @@\n-old\n+new", "text": "This can still race under concurrent access."}],
            "merged_patch": "diff --git a/a.py b/a.py\nSECRET_VERIFIER_PATCH",
        }
        messages, traits, reason = core.swecare_messages(row)
        self.assertIsNone(reason)
        self.assertTrue(traits["merged_patch_used_as_verifier_only"])
        self.assertNotIn("SECRET_VERIFIER_PATCH", json.dumps(messages))
        self.assertIn("Reviewed patch excerpts", messages[0]["content"])
        self.assertNotIn("UNRELATED_FULL_PATCH", messages[0]["content"])

    def test_swecare_caps_preserve_difficulty_and_review_effort_mix(self) -> None:
        item = candidate("review")
        item.dataset = "swe_care"
        item.repository = "owner/repo"
        counters = defaultdict(Counter)
        counters["swecare_difficulty"].update({"hard": 10, "medium": 7, "easy": 3})
        counters["swecare_effort"].update({"high": 7, "low": 3})
        item.metadata = {"source_difficulty": "medium", "estimated_review_effort": 4}
        self.assertEqual(builder._cap_reason(item, [], counters), "difficulty_mix_cap")
        counters["swecare_difficulty"]["medium"] = 6
        self.assertIsNone(builder._cap_reason(item, [], counters))
        item.metadata["estimated_review_effort"] = 3
        self.assertEqual(builder._cap_reason(item, [], counters), "review_effort_mix_cap")

    def test_swecare_repository_cap_uses_final_tranche_size(self) -> None:
        values = []
        for index in range(100):
            item = candidate(f"review-{index}", prompt=f"Review request {index}", target=f"Review response {index}")
            item.dataset = "swe_care"
            item.repository = "owner/repo" if index < 10 else f"owner-{index}/repo"
            values.append((item, 100))
        stats = builder.Stats()
        kept = builder._prune_swecare_repositories(values, stats)
        repositories = Counter(item.repository for item, _ in kept)
        cap = max(1, int(len(kept) * 0.03))
        self.assertLessEqual(max(repositories.values()), cap)
        self.assertEqual(stats.rejections["swe_care"]["source_repository_cap"], len(values) - len(kept))

    def test_interacting_final_caps_reach_a_fixed_point(self) -> None:
        values = []
        for index in range(100):
            item = candidate(f"review-final-{index}", prompt=f"Review {index}", target=f"Finding {index}")
            item.dataset = "swe_care"
            item.repository = "owner/shared" if index < 12 else f"owner/{index}"
            item.metadata = {"source_difficulty": "hard", "estimated_review_effort": 5}
            values.append((item, 40))
        for index in range(100):
            item = candidate(f"api-final-{index}", prompt=f"API {index}", target=f"Call {index}")
            item.dataset = "xlam_60k"
            category = "weather_lookup" if index < 12 else f"unique_api_{index}"
            item.metadata = {"api_categories": [category]}
            values.append((item, 40))
        kept = builder._apply_final_constraints(values, builder.Stats())
        swe = [item for item, _ in kept if item.dataset == "swe_care"]
        xlam = [item for item, _ in kept if item.dataset == "xlam_60k"]
        swe_counts = Counter(item.repository for item in swe)
        api_counts = Counter(category for item in xlam for category in item.metadata["api_categories"])
        self.assertLessEqual(max(swe_counts.values()), max(1, int(len(swe) * 0.03)))
        self.assertLessEqual(max(api_counts.values()), max(1, int(len(xlam) * 0.02)))
        self.assertEqual(kept, builder._apply_final_constraints(kept, builder.Stats()))

    def test_swecare_loader_is_dev_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "SWE-CARE"
            directory.mkdir()
            (directory / "test-00000-of-00001.parquet").write_bytes(b"not parquet")
            with self.assertRaisesRegex(RuntimeError, "dev split is missing"):
                builder.load_swecare(root, builder.Stats())

    def test_swecare_loader_records_the_locked_data_path(self) -> None:
        row = {
            "instance_id": "owner__repo-17",
            "title": "Fix state race",
            "problem_statement": "State can race.",
            "pull_number": 17,
            "repo": "owner/repo",
            "commit_to_review": {
                "patch_to_review": "diff --git a/a.py b/a.py\n@@ -1 +1 @@\n-old\n+new",
            },
            "reference_review_comments": [{
                "path": "a.py",
                "line": 1,
                "diff_hunk": "@@ -1 +1 @@\n-old\n+new",
                "text": "This can still race under concurrent access.",
            }],
            "merged_patch": "diff --git a/a.py b/a.py\n-old\n+fixed",
            "metadata": {"difficulty": "hard", "estimated_review_effort": 5},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "SWE-CARE" / "data"
            directory.mkdir(parents=True)
            path = directory / "dev-00000-of-00001.parquet"
            path.write_bytes(b"mock parquet")
            with mock.patch.object(builder, "_parquet_rows", return_value=iter([(path, 0, row)])):
                pools, _ = builder.load_swecare(root, builder.Stats())
        selected = [item for values in pools.values() for item in values]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].source_path, "data/dev-00000-of-00001.parquet")

    def test_acquisition_patterns_permanently_exclude_test_and_parent_collection(self) -> None:
        self.assertFalse(any("test" in pattern for pattern in acquirer.ACQUISITION["swe_care"]["patterns"]))
        codeact = acquirer.ACQUISITION["codeact_instruct"]["patterns"]
        self.assertEqual(codeact, ["codeactinstruct/full_std.jsonl", "codeactinstruct/README.md", "codeactinstruct/LICENSE"])
        agentic = acquirer.ACQUISITION["nemotron_agentic_v2"]["patterns"]
        self.assertNotIn("data/interactive_agent.jsonl", agentic)
        self.assertFalse(any("customer" in pattern for pattern in agentic))


class SelectionIntegrationTests(unittest.TestCase):
    def test_artifact_writers_use_checkout_stable_lf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "manifest.json"
            jsonl_path = root / "corpus.jsonl"
            builder._write_json_lf(json_path, {"value": "one\ntwo"})
            builder._write_jsonl_lf(jsonl_path, [{"value": "one\ntwo"}])
            self.assertNotIn(b"\r\n", json_path.read_bytes())
            self.assertNotIn(b"\r\n", jsonl_path.read_bytes())
            self.assertTrue(json_path.read_bytes().endswith(b"\n"))
            self.assertTrue(jsonl_path.read_bytes().endswith(b"\n"))

    def test_source_inventory_is_derived_from_verified_acquisition_lock(self) -> None:
        locked_source = {
            "repo_id": "example/source",
            "revision": "a" * 40,
            "local_directory": "Example",
            "files": [{
                "remote_path": "data/source.bin",
                "local_path": "local.bin",
                "size": 3,
                "sha256": "b" * 64,
            }],
        }
        verification = {
            "files": {"local.bin": {"size": 3, "sha256": "b" * 64, "verified": True}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Example").mkdir()
            with (
                mock.patch.object(builder, "DATASETS", {"example": {}}),
                mock.patch.dict(builder.ACQUISITION_LOCK, {"datasets": {"example": locked_source}}),
                mock.patch.object(builder, "verify_dataset", return_value=verification) as verify,
            ):
                inventory = builder._source_inventory(root)
        verify.assert_called_once_with("example", root, require_all=True)
        self.assertEqual(inventory, [{
            "dataset": "example",
            "repo_id": "example/source",
            "revision": "a" * 40,
            "remote_path": "data/source.bin",
            "path": "Example/local.bin",
            "bytes": 3,
            "sha256": "b" * 64,
            "identity": "sha256",
        }])

    def test_training_example_manifest_summary_includes_total_once(self) -> None:
        records = [
            {"split": "train", "metadata": {"training_example_count": 3}},
            {"split": "train", "metadata": {"training_example_count": 2}},
            {"split": "validation", "metadata": {"training_example_count": 4}},
        ]
        counts, total = builder._training_example_summary(records)
        self.assertEqual(counts, {"train": 5, "validation": 4, "total": 9})
        self.assertEqual(total, 9)

    def test_eval_manifest_leaks_are_rejected_before_selection(self) -> None:
        item = candidate("leak")
        pools = {"codeact_instruct": {"multi": [item]}}
        quotas = {name: {} for name in core.DATASETS}
        quotas["codeact_instruct"] = {"multi": 1}
        stats = builder.Stats()
        with mock.patch.object(builder, "QUOTAS", quotas), mock.patch.object(builder, "find_manifest_leaks", return_value=[{"kind": "normalized_content"}]):
            selected = builder.select(pools, FakeTokenizer(), {"synthetic": True}, stats)
        self.assertEqual(selected, [])
        self.assertEqual(stats.rejections["codeact_instruct"]["eval_v1_leakage"], 1)

    def test_original_upstream_id_is_checked_against_eval_manifest(self) -> None:
        item = candidate("held-out-id")
        pools = {"codeact_instruct": {"multi": [item]}}
        quotas = {name: {} for name in core.DATASETS}
        quotas["codeact_instruct"] = {"multi": 1}
        manifest = {"records": [{"id": "held-out-id", "id_sha256": core.sha256_text("held-out-id")}]}
        stats = builder.Stats()
        with mock.patch.object(builder, "QUOTAS", quotas):
            selected = builder.select(pools, FakeTokenizer(), manifest, stats)
        self.assertEqual(selected, [])
        self.assertEqual(stats.rejections["codeact_instruct"]["eval_v1_id_leakage"], 1)

    def test_cross_bucket_dataset_duplicate_keeps_highest_quality(self) -> None:
        low = candidate("low", prompt="Low quality request", target="Low quality answer")
        high = candidate("high", prompt="High quality request", target="High quality answer")
        for item in (low, high):
            item.dataset = "xlam_60k"
            item.metadata["dataset_duplicate_key_sha256"] = "same-upstream-key"
        low.scores["weighted_total"], high.scores["weighted_total"] = 0.1, 0.9
        pools = {"xlam_60k": {"earlier": [low], "later": [high]}}
        quotas = {name: {} for name in core.DATASETS}
        quotas["xlam_60k"] = {"earlier": 1, "later": 1}
        stats = builder.Stats()
        with mock.patch.object(builder, "QUOTAS", quotas), mock.patch.object(builder, "find_manifest_leaks", return_value=[]):
            selected = builder.select(pools, FakeTokenizer(), {"records": []}, stats)
        self.assertEqual([item.original_id for item, _ in selected], ["high"])

    def test_actual_tokenizer_overlength_is_rejected(self) -> None:
        item = candidate("long", target="word " * 1100)
        pools = {"codeact_instruct": {"multi": [item]}}
        quotas = {name: {} for name in core.DATASETS}
        quotas["codeact_instruct"] = {"multi": 1}
        stats = builder.Stats()
        with mock.patch.object(builder, "QUOTAS", quotas):
            selected = builder.select(pools, FakeTokenizer(), {"records": []}, stats)
        self.assertEqual(selected, [])
        self.assertEqual(stats.rejections["codeact_instruct"]["actual_tokenizer_overlength"], 1)

    def test_selection_validates_and_counts_each_native_assistant_target(self) -> None:
        item = candidate("native-length")
        item.metadata.update({
            "native_tools": [{"type": "function", "function": {
                "name": "execute", "description": "Run code",
                "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
            }}],
            "native_messages": [
                {"role": "user", "content": "Calculate it."},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "one", "type": "function", "function": {"name": "execute", "arguments": '{"code":"print(2)"}'}}]},
                {"role": "tool", "content": "2", "tool_call_id": "one", "name": "execute"},
                {"role": "assistant", "content": "The answer is 2."},
            ],
        })
        pools = {"codeact_instruct": {"multi": [item]}}
        quotas = {name: {} for name in core.DATASETS}
        quotas["codeact_instruct"] = {"multi": 1}
        with mock.patch.object(builder, "QUOTAS", quotas), mock.patch.object(builder, "find_manifest_leaks", return_value=[]):
            selected = builder.select(pools, FakeTokenizer(), {"records": []}, builder.Stats())
        self.assertEqual(len(selected), 1)
        self.assertEqual(item.metadata["training_example_count"], 2)

    def test_record_is_tier_b_with_complete_provenance(self) -> None:
        item = candidate("record")
        record = core.candidate_record(item, "train", 42)
        self.assertEqual(record["quality_tier"], "B")
        self.assertEqual(record["verification_status"], "source_verified")
        for key in ("dataset", "revision", "original_id", "source_path", "transformation_version", "acquisition_date"):
            self.assertIn(key, record["provenance"])

    def test_codeact_constituent_provenance_flows_to_both_ledgers(self) -> None:
        item = candidate("record-provenance")
        details = {"name": "reasoning/algebra", **core.CODEACT_CONSTITUENTS["reasoning/algebra"]}
        item.metadata["constituent_provenance"] = details
        record = core.candidate_record(item, "train", 42)
        source = core.source_record(record, item)
        self.assertEqual(record["provenance"]["constituent"], details)
        self.assertEqual(source["constituent_provenance"], details)


if __name__ == "__main__":
    unittest.main()
