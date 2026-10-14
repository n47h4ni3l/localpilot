import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class LineageReceiptTests(unittest.TestCase):
    def test_published_receipt_binds_unchanged_parent_and_completed_config(self):
        path = next((ROOT / "training/lineage/receipts").glob("p11-p12-*.json"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(path.name, f"p11-p12-{digest}.json")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        manifests = {}
        for label, entry in receipt["packages"].items():
            source = ROOT / entry["accepted_manifest"]["path"]
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), entry["accepted_manifest"]["sha256"])
            manifests[label] = json.loads(source.read_text(encoding="utf-8"))
            self.assertEqual(manifests[label]["training"], entry["training"])
            self.assertFalse(manifests[label]["benchmark_promoted"])
        parent = manifests["p1.2"]["parent"]
        self.assertEqual(parent["sha256"], receipt["packages"]["p1.1"]["accepted_manifest"]["sha256"])
        self.assertEqual(receipt["actual_trained_parent"]["adapter_files"], manifests["p1.1"]["training"]["adapter_files"])
        config = receipt["completed_config"]
        self.assertEqual(hashlib.sha256((ROOT / config["path"]).read_bytes()).hexdigest(), config["file_sha256"])
        self.assertEqual(receipt["promotion"]["rollback_digest"], manifests["p1.1"]["deployment"]["digest"])
        self.assertEqual(receipt["promotion"]["digest"], manifests["p1.2"]["deployment"]["digest"])
