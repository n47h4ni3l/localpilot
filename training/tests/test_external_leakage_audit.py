from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.scripts.audit_external_corpus_v1_leakage import (
    audit,
    screen_records,
    similarity_metrics,
    suspicion_reasons,
)


class ExternalLeakageAuditTests(unittest.TestCase):
    def test_flags_close_paraphrase_and_ignores_unrelated_task(self) -> None:
        held_out = {
            "id": "eval-1",
            "messages": [{"role": "user", "content": "Diagnose repeated Windows pytest permission failures in a temporary directory before blaming a refactor."}],
            "expected_behavior": ["Check background processes and rerun with a controlled temporary base directory."],
        }
        close = {
            "id": "candidate-close",
            "source": "example",
            "messages": [
                {"role": "user", "content": "Before blaming the refactor for repeated pytest permission failures on Windows, diagnose the temporary directory and background processes."},
                {"role": "assistant", "content": "Rerun using a controlled temporary base directory."},
            ],
            "provenance": {"original_id": "42"},
        }
        unrelated = {
            "id": "candidate-other",
            "source": "example",
            "messages": [
                {"role": "user", "content": "Implement a balanced binary tree iterator."},
                {"role": "assistant", "content": "Use an explicit stack for in-order traversal."},
            ],
        }
        findings = screen_records([close, unrelated], [held_out])
        self.assertEqual([item["record_id"] for item in findings], ["candidate-close"])
        self.assertEqual(findings[0]["upstream_id"], "42")

    def test_similarity_threshold_requires_substantial_shared_material(self) -> None:
        metrics = similarity_metrics("write a parser for csv rows", "write a parser for json objects")
        self.assertEqual(suspicion_reasons(metrics), [])

    def test_report_contains_only_ids_counts_and_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            eval_root = root / "evals" / "debugging"
            eval_root.mkdir(parents=True)
            report = root / "report.json"
            corpus_text = "Diagnose repeated Windows pytest permission failures in the temporary directory before blaming a refactor. Check background processes."
            eval_text = "Before blaming a refactor, diagnose repeated Windows pytest permission failures in the temporary directory and check background processes."
            corpus.write_text(json.dumps({"id": "candidate-1", "source": "sample", "messages": [{"role": "user", "content": corpus_text}]}) + "\n", encoding="utf-8")
            (eval_root / "eval_v1_seed.jsonl").write_text(json.dumps({"id": "eval-1", "messages": [{"role": "user", "content": eval_text}]}) + "\n", encoding="utf-8")
            value = audit(corpus, root / "evals", report)
            serialized = report.read_text(encoding="utf-8")
            self.assertEqual(value["finding_count"], 1)
            self.assertFalse(value["contains_plaintext_eval_content"])
            self.assertNotIn(corpus_text, serialized)
            self.assertNotIn(eval_text, serialized)


if __name__ == "__main__":
    unittest.main()
