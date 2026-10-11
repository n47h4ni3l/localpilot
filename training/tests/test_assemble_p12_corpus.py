"""P1.2 curated corpus must remain immutable, reproducible and evaluation-isolated."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "training/scripts/assemble_p12_corpus.py"
SPEC = importlib.util.spec_from_file_location("assemble_p12_corpus", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
assembler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assembler)


class P12CorpusTests(unittest.TestCase):
    def test_shards_and_manifest_valid_and_eval_v1_isolated(self) -> None:
        # A read-only end-to-end check of exact training inputs; no optimizer or GPU.
        result = assembler.build(check_only=True)
        self.assertEqual(result["records"], 1000)
        self.assertEqual(result["split_counts"], {"train": 898, "validation": 102})
        self.assertEqual(
            result["training_example_counts"],
            {"train": 2696, "validation": 307, "total": 3003},
        )
        self.assertEqual(result["eval_v1_leaks"], 0)
        self.assertEqual(
            result["sha256"],
            "8eb2f9300eda78137fc6f1204d34beb0ec2d65059e4e374ec5d42ffa29655801",
        )

    def test_wrong_digest_refused_without_writing(self) -> None:
        manifest = json.loads(assembler.MANIFEST.read_text(encoding="utf-8"))
        manifest["corpus_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            test_manifest = Path(temporary) / "corpus_manifest.json"
            test_manifest.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.object(assembler, "MANIFEST", test_manifest):
                with self.assertRaisesRegex(RuntimeError, "digest differs"):
                    assembler.build(check_only=True)


if __name__ == "__main__":
    unittest.main()
