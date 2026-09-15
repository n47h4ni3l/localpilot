from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_external_corpus_v1.py"
SPEC = importlib.util.spec_from_file_location("localpilot_validate_external_corpus_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


DATASETS = (
    "nemotron_swe_v2",
    "open_code_instruct",
    "nemotron_agentic_v2",
    "xlam_60k",
    "codeact_instruct",
    "swe_care",
)


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _native_tools() -> list[dict]:
    return [{
        "type": "function",
        "function": {
            "name": "inspect",
            "description": "Inspect a repository path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }]


def _native_trajectory() -> list[dict]:
    return [
        {"role": "user", "content": "Inspect the failing file."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "inspect", "arguments": "{\"path\":\"src/a.py\"}"},
            }],
        },
        {"role": "tool", "content": "line 7 fails", "tool_call_id": "call-1", "name": "inspect"},
        {"role": "assistant", "content": "The failure starts at line 7."},
    ]


class ExternalCorpusArtifactValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix=".external-artifacts-", dir=Path(__file__).parent)
        self.root = Path(self.temp.name)
        self.corpus_path = self.root / "training/datasets/external_corpus_v1.jsonl"
        self.sources_path = self.root / "training/sources/external_corpus_v1_sources.jsonl"
        self.manifest_path = self.root / "training/manifests/external_corpus_v1_manifest.json"
        self.lock_path = self.root / "training/manifests/external_corpus_v1_acquisition_lock.json"
        self.runtime_path = self.root / "training/requirements-external-corpus-v1.txt"
        self.eval_manifest_path = self.root / "training/manifests/eval_v1_manifest.json"
        self.records = [self._record(dataset, index) for index, dataset in enumerate(DATASETS)]
        self.sources = [self._source(record) for record in self.records]

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _record(dataset: str, index: int) -> dict:
        record_id = f"ext-v1-{dataset}-{index}"
        split = "validation" if dataset == "swe_care" else "train"
        task_type = {
            "nemotron_swe_v2": "software_engineering_agent",
            "nemotron_agentic_v2": "multi_tool_use",
            "xlam_60k": "tool_use",
            "codeact_instruct": "agentic_code_execution",
            "swe_care": "code_review",
        }.get(dataset, "code_generation")
        source_path = "data/dev-00000-of-00001.parquet" if dataset == "swe_care" else f"data/{dataset}.jsonl"
        metadata: dict = {
            "family_id": f"family-{index}",
            "problem_identity_sha256": f"problem-{index}",
            "training_example_count": 1,
        }
        if dataset in {"nemotron_swe_v2", "nemotron_agentic_v2", "codeact_instruct"}:
            metadata.update({
                "native_tools": _native_tools(),
                "native_messages": _native_trajectory(),
                "training_example_count": 2,
            })
        elif dataset == "xlam_60k":
            metadata.update({
                "native_tools": _native_tools(),
                "native_messages": [{"role": "user", "content": "Inspect src/a.py"}],
                "native_call_targets": [_native_trajectory()[1]],
            })
        if dataset == "swe_care":
            metadata["merged_patch_used_as_verifier_only"] = True
        return {
            "id": record_id,
            "messages": [
                {"role": "user", "content": f"Prompt for {dataset}"},
                {"role": "assistant", "content": f"Answer for {dataset}"},
            ],
            "task_type": task_type,
            "source": f"example/{dataset}",
            "license": "Apache-2.0",
            "quality_tier": "B",
            "split": split,
            "verification_status": "source_verified",
            "provenance": {
                "dataset": dataset,
                "revision": f"revision-{index}",
                "original_id": f"original-{index}",
                "source_path": source_path,
                "transformation_version": "external-corpus-v1.0.1",
                "acquisition_date": "2026-09-08",
                "reference": f"hf:example/{dataset}@revision-{index}#original-{index}",
                "license_reference": f"https://example.invalid/{dataset}/README.md",
            },
            "metadata": metadata,
        }

    @staticmethod
    def _source(record: dict) -> dict:
        provenance = record["provenance"]
        return {
            "id": record["id"],
            "dataset": provenance["dataset"],
            "dataset_name": record["source"],
            "pinned_revision": provenance["revision"],
            "original_id": provenance["original_id"],
            "license": record["license"],
            "source_path": provenance["source_path"],
            "transformation_version": provenance["transformation_version"],
            "acquisition_date": provenance["acquisition_date"],
            "family_id": record["metadata"]["family_id"],
            "raw_record_sha256": hashlib.sha256(record["id"].encode()).hexdigest(),
            "reference": provenance["reference"],
            "license_reference": provenance["license_reference"],
        }

    def _write(self) -> dict:
        for path in (
            self.corpus_path, self.sources_path, self.manifest_path, self.lock_path,
            self.runtime_path, self.eval_manifest_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.corpus_path.write_bytes(_jsonl_bytes(self.records))
        self.sources_path.write_bytes(_jsonl_bytes(self.sources))
        self.runtime_path.write_text("example-package==1.2.3\n", encoding="utf-8")
        eval_digest = hashlib.sha256(b"eval-v1").hexdigest()
        self.eval_manifest_path.write_text(
            json.dumps({"artifact_type": "held_out_eval_manifest", "records_sha256": eval_digest}),
            encoding="utf-8",
        )

        locked_datasets = {}
        source_files = []
        for index, dataset in enumerate(DATASETS):
            remote_path = f"data/source-{index}.jsonl"
            digest = hashlib.sha256(dataset.encode()).hexdigest()
            locked_datasets[dataset] = {
                "repo_id": f"example/{dataset}",
                "revision": f"revision-{index}",
                "local_directory": dataset,
                "files": [{
                    "remote_path": remote_path,
                    "local_path": remote_path,
                    "size": index + 10,
                    "sha256": digest,
                }],
            }
            source_files.append({
                "dataset": dataset,
                "repo_id": f"example/{dataset}",
                "revision": f"revision-{index}",
                "remote_path": remote_path,
                "path": f"{dataset}/{remote_path}",
                "bytes": index + 10,
                "sha256": digest,
                "identity": "sha256",
            })
        lock = {"format_version": 1, "datasets": locked_datasets}
        self.lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")

        accepted = dict(sorted(Counter(record["provenance"]["dataset"] for record in self.records).items()))
        splits = dict(sorted(Counter(record["split"] for record in self.records).items()))
        training_by_split = {
            split: sum(
                record["metadata"]["training_example_count"]
                for record in self.records if record["split"] == split
            )
            for split in sorted({record["split"] for record in self.records})
        }
        training_counts = {**training_by_split, "total": sum(training_by_split.values())}
        manifest = {
            "schema_version": 1,
            "artifact_type": "external_corpus_manifest",
            "maximum_records": 20_000,
            "records": len(self.records),
            "training_examples": training_counts["total"],
            "accepted_counts": accepted,
            "split_counts": splits,
            "training_example_counts": training_counts,
            "corpus_sha256": _sha256(self.corpus_path),
            "sources_sha256": _sha256(self.sources_path),
            "datasets": {dataset: {"ceiling": 6_000} for dataset in DATASETS},
            "source_files": source_files,
            "eval_manifest_records_sha256": eval_digest,
            "build_runtime": {"exact_package_versions_verified": True},
            "runtime_requirements": {
                "path": self.runtime_path.relative_to(self.root).as_posix(),
                "sha256": _sha256(self.runtime_path),
            },
            "acquisition_lock": {
                "path": self.lock_path.relative_to(self.root).as_posix(),
                "sha256": _sha256(self.lock_path),
                "format_version": 1,
                "file_count": len(DATASETS),
            },
            "rules": {
                "swe_care_test_split_used": False,
                "merged_patch_in_training_content": False,
            },
        }
        self.manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        return manifest

    def _validate(self):
        return validator.validate_artifacts(
            corpus_path=self.corpus_path,
            sources_path=self.sources_path,
            manifest_path=self.manifest_path,
            repo_root=self.root,
        )

    def test_valid_artifacts_pass_without_optional_dependencies(self) -> None:
        self._write()
        report = self._validate()
        self.assertTrue(report.valid, report.errors)
        self.assertEqual(report.summary["records"], 6)
        self.assertEqual(report.summary["training_example_counts"]["total"], 9)

    def test_artifact_and_lock_digests_are_enforced(self) -> None:
        manifest = self._write()
        manifest["corpus_sha256"] = "0" * 64
        manifest["runtime_requirements"]["sha256"] = "1" * 64
        manifest["acquisition_lock"]["sha256"] = "2" * 64
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        report = self._validate()
        self.assertFalse(report.valid)
        self.assertTrue(any("corpus_sha256 digest mismatch" in error for error in report.errors))
        self.assertTrue(any("runtime_requirements digest mismatch" in error for error in report.errors))
        self.assertTrue(any("acquisition_lock digest mismatch" in error for error in report.errors))

    def test_manifest_counts_training_counts_and_ceilings_are_enforced(self) -> None:
        manifest = self._write()
        manifest["records"] += 1
        manifest["split_counts"]["train"] += 1
        manifest["training_example_counts"]["total"] += 1
        manifest["training_examples"] += 1
        manifest["datasets"]["xlam_60k"]["ceiling"] = 0
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        report = self._validate()
        self.assertTrue(any("manifest records=" in error for error in report.errors))
        self.assertTrue(any("manifest split_counts=" in error for error in report.errors))
        self.assertTrue(any("manifest training_example_counts=" in error for error in report.errors))
        self.assertTrue(any("above ceiling=0" in error for error in report.errors))

    def test_source_ledger_is_one_to_one_and_consistent(self) -> None:
        self.sources.pop()
        self.sources[0]["family_id"] = "wrong-family"
        self._write()
        report = self._validate()
        self.assertTrue(any("not one-to-one" in error for error in report.errors))
        self.assertTrue(any("source ledger family_id disagrees" in error for error in report.errors))

    def test_family_and_problem_identity_cannot_cross_splits(self) -> None:
        self.records[-1]["metadata"]["family_id"] = self.records[0]["metadata"]["family_id"]
        self.records[-1]["metadata"]["problem_identity_sha256"] = self.records[0]["metadata"]["problem_identity_sha256"]
        self.sources[-1] = self._source(self.records[-1])
        self._write()
        report = self._validate()
        self.assertTrue(any("family_id" in error and "crosses" in error for error in report.errors))
        self.assertTrue(any("problem_identity_sha256" in error and "crosses" in error for error in report.errors))

    def test_swecare_is_dev_only_and_never_contains_merged_patch(self) -> None:
        swe = self.records[-1]
        swe["provenance"]["source_path"] = "data/test-00000-of-00001.parquet"
        swe["metadata"]["merged_patch"] = "secret verifier patch"
        self.sources[-1] = self._source(swe)
        self._write()
        report = self._validate()
        self.assertTrue(any("SWE-CARE source_path must be the dev" in error for error in report.errors))
        self.assertTrue(any("forbidden merged_patch" in error for error in report.errors))

    def test_provider_shaped_credentials_are_rejected_without_echoing_them(self) -> None:
        credential = "ghp_" + "A" * 36
        self.records[0]["messages"][-1]["content"] = "Never retain " + credential
        self._write()
        report = self._validate()
        self.assertTrue(any("sensitive credential pattern" in error for error in report.errors))
        self.assertFalse(any(credential in error for error in report.errors))

    def test_all_tool_datasets_require_complete_native_metadata(self) -> None:
        del self.records[2]["metadata"]["native_tools"]
        del self.records[3]["metadata"]["native_call_targets"]
        del self.records[4]["metadata"]["native_messages"]
        self._write()
        report = self._validate()
        self.assertTrue(any("native_tools must be a nonempty list" in error for error in report.errors))
        self.assertTrue(any("native_call_targets must be a nonempty list" in error for error in report.errors))
        self.assertTrue(any("native_messages must be a nonempty list" in error for error in report.errors))

    def test_native_tool_results_must_be_nonempty(self) -> None:
        self.records[0]["metadata"]["native_messages"][2]["content"] = ""
        self._write()
        report = self._validate()
        self.assertTrue(any("content must be nonempty text" in error for error in report.errors))

    def test_declared_training_count_must_match_native_expansion(self) -> None:
        self.records[0]["metadata"]["training_example_count"] = 1
        self._write()
        report = self._validate()
        self.assertTrue(any("native expansion yields 2" in error for error in report.errors))

    def test_acquisition_inventory_must_match_immutable_lock(self) -> None:
        manifest = self._write()
        manifest["source_files"][0]["bytes"] += 1
        manifest["source_files"][1]["path"] = "wrong/source.bin"
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        report = self._validate()
        self.assertTrue(any("incorrect bytes" in error for error in report.errors))
        self.assertTrue(any("incorrect path" in error for error in report.errors))

    def test_eval_manifest_identity_is_frozen(self) -> None:
        manifest = self._write()
        manifest["eval_manifest_records_sha256"] = "0" * 64
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        report = self._validate()
        self.assertTrue(any("different Eval v1 manifest" in error for error in report.errors))


if __name__ == "__main__":
    unittest.main()
