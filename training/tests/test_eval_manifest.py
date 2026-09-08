from __future__ import annotations

import json
import copy
import tempfile
import unittest
from pathlib import Path

from training.scripts.build_eval_manifest import build_manifest
from training.scripts.training_common import find_manifest_leaks, validate_eval_manifest


def make_manifest(prompt: str, rubric: str = "Require concrete independent evidence.") -> dict:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        record = {
            "id": "eval-001",
            "messages": [{"role": "user", "content": prompt}],
            "task_type": "epistemics",
            "source": "project_owned_eval",
            "license": "project_owned",
            "quality_tier": "A",
            "split": "held_out_eval",
            "verification_status": "human_verified",
            "provenance": {"reference": "test:private"},
            "expected_behavior": [rubric],
        }
        (root / "seed.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
        return build_manifest(root)


class EvalManifestTests(unittest.TestCase):
    def test_manifest_contains_hashes_but_no_plaintext(self) -> None:
        secret_prompt = "Explain the private canary behavior without guessing."
        secret_rubric = "Require an explicit uncertainty statement."
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            category = root / "epistemics"
            category.mkdir()
            record = {
                "id": "eval-secret-001",
                "messages": [{"role": "user", "content": secret_prompt}],
                "task_type": "epistemics",
                "source": "project_owned_eval",
                "license": "project_owned",
                "quality_tier": "A",
                "split": "held_out_eval",
                "verification_status": "human_verified",
                "provenance": {"reference": "test:private"},
                "expected_behavior": [secret_rubric],
            }
            (category / "seed.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

            manifest = build_manifest(root)
            serialized = json.dumps(manifest)

        self.assertEqual(manifest["record_count"], 1)
        self.assertNotIn(secret_prompt, serialized)
        self.assertNotIn(secret_rubric, serialized)
        self.assertFalse(manifest["contains_plaintext_eval_content"])

    def test_normalized_prompt_and_content_shingles_are_rejected(self) -> None:
        manifest = make_manifest("Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho.")
        copied = {
            "id": "train-copy",
            "messages": [
                {"role": "user", "content": "  ALPHA beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho.  "},
                {"role": "assistant", "content": "A target."},
            ],
        }
        leaks = find_manifest_leaks([copied], manifest)
        self.assertTrue(any(item["kind"] == "normalized_prompt" for item in leaks))
        self.assertTrue(any(item["kind"] == "content_16_token_shingle" for item in leaks))

    def test_short_prompt_in_assistant_or_metadata_is_rejected(self) -> None:
        manifest = make_manifest("Ｈｉｄｄｅｎ canary prompt", "Unique rubric instruction")
        rows = [
            {"id": "train-assistant", "messages": [{"role": "assistant", "content": "Preamble HIDDEN   CANARY prompt afterword"}]},
            {"id": "train-metadata", "metadata": {"notes": "Prefix Unique rubric instruction suffix"}},
        ]
        collisions = find_manifest_leaks(rows, manifest)
        self.assertEqual({item["record_id"] for item in collisions}, {"train-assistant", "train-metadata"})
        self.assertTrue(all(item["kind"] == "normalized_content_fragment" for item in collisions))

    def test_held_out_id_cannot_be_reused_with_different_text(self) -> None:
        manifest = make_manifest("An independent private question")
        collisions = find_manifest_leaks([{"id": "eval-001", "messages": []}], manifest)
        self.assertEqual(collisions, [{"record_id": "eval-001", "kind": "held_out_id", "eval_id": "eval-001"}])

    def test_empty_and_tampered_manifests_fail_closed(self) -> None:
        manifest = make_manifest("Independent private question")
        for field, value in (("records", []), ("record_count", 0), ("records_sha256", "0" * 64), ("contains_prompts", True)):
            bad = copy.deepcopy(manifest)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                find_manifest_leaks([], bad)
        bad = copy.deepcopy(manifest)
        bad["records"][0]["normalized_prompt_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "integrity"):
            validate_eval_manifest(bad)
        bad = copy.deepcopy(manifest)
        bad["records"][0]["plaintext"] = "Must never be accepted"
        with self.assertRaisesRegex(RuntimeError, "Malformed"):
            validate_eval_manifest(bad)

    def test_freeze_is_reproducible_and_rejects_empty_roots(self) -> None:
        first = make_manifest("Stable source content")
        second = make_manifest("Stable source content")
        first.pop("created_at")
        second.pop("created_at")
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "No held-out"):
                build_manifest(Path(temporary))

    def test_assistant_answer_in_held_out_file_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = {
                "id": "eval-bad-001",
                "split": "held_out_eval",
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Leaked answer"},
                ],
            }
            (root / "bad.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "assistant answer"):
                build_manifest(root)


if __name__ == "__main__":
    unittest.main()
