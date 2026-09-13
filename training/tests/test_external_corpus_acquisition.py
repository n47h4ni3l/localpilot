from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location("external_corpus_acquirer_tests", SCRIPTS / "acquire_external_corpus_v1.py")
assert SPEC and SPEC.loader
acquirer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acquirer)


def locked_entry(remote: str, local: str, payload: bytes) -> dict[str, object]:
    return {
        "remote_path": remote,
        "local_path": local,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


class AcquisitionLockTests(unittest.TestCase):
    def test_lock_contains_only_exact_approved_paths(self) -> None:
        self.assertEqual(len(acquirer.ACQUISITION), 6)
        self.assertEqual(
            {dataset: len(source["files"]) for dataset, source in acquirer.ACQUISITION.items()},
            {
                "nemotron_swe_v2": 3,
                "open_code_instruct": 51,
                "nemotron_agentic_v2": 3,
                "xlam_60k": 2,
                "codeact_instruct": 3,
                "swe_care": 2,
            },
        )
        self.assertEqual(sum(len(source["files"]) for source in acquirer.ACQUISITION.values()), 64)
        open_code = acquirer.ACQUISITION["open_code_instruct"]["patterns"]
        self.assertEqual(len(open_code), 51)
        self.assertEqual(sum(path.endswith(".parquet") for path in open_code), 50)
        for source in acquirer.ACQUISITION.values():
            self.assertFalse(any(any(mark in path for mark in "*?[") for path in source["patterns"]))
        self.assertEqual(
            acquirer.ACQUISITION["swe_care"]["permanently_excluded"],
            ["data/test-00000-of-00001.parquet"],
        )
        self.assertNotIn(
            "data/interactive_agent.jsonl",
            acquirer.ACQUISITION["nemotron_agentic_v2"]["patterns"],
        )
        self.assertEqual(
            acquirer.ACQUISITION["codeact_instruct"]["patterns"],
            ["codeactinstruct/full_std.jsonl", "codeactinstruct/README.md", "codeactinstruct/LICENSE"],
        )
        self.assertEqual(
            [entry["local_path"] for entry in acquirer.ACQUISITION["codeact_instruct"]["files"]],
            ["full_std.jsonl", "README.md", "LICENSE"],
        )
        self.assertEqual(
            acquirer.ACQUISITION["swe_care"]["files"][1],
            {
                "remote_path": "data/dev-00000-of-00001.parquet",
                "local_path": "dev-00000-of-00001.parquet",
                "size": 125079708,
                "sha256": "39bcd8ff2f833e5aae381a26d110f52999d1ac60a7715c0d5604e19b558f5e18",
            },
        )

    def test_every_lock_entry_has_one_unambiguous_identity(self) -> None:
        for dataset, source in acquirer.ACQUISITION.items():
            for entry in source["files"]:
                with self.subTest(dataset=dataset, path=entry["remote_path"]):
                    self.assertNotEqual("sha256" in entry, "git_oid" in entry)
                    if "sha256" in entry:
                        self.assertRegex(entry["sha256"], r"\A[0-9a-f]{64}\Z")
                    else:
                        self.assertRegex(entry["git_oid"], r"\A[0-9a-f]{40}\Z")

    def test_lock_revisions_match_the_six_corpus_specs(self) -> None:
        for dataset, source in acquirer.ACQUISITION_LOCK["datasets"].items():
            with self.subTest(dataset=dataset):
                self.assertEqual(source["repo_id"], acquirer.DATASETS[dataset]["repo_id"])
                self.assertEqual(source["revision"], acquirer.DATASETS[dataset]["revision"])

    def test_loader_rejects_wildcards_and_ambiguous_identities(self) -> None:
        lock = json.loads(acquirer.LOCK_PATH.read_text(encoding="utf-8"))
        entry = lock["datasets"]["xlam_60k"]["files"][0]
        entry["remote_path"] = "*.md"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unsafe remote_path"):
                acquirer.load_acquisition_lock(path)

        lock = json.loads(acquirer.LOCK_PATH.read_text(encoding="utf-8"))
        entry = lock["datasets"]["xlam_60k"]["files"][0]
        entry["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "ambiguous content identities"):
                acquirer.load_acquisition_lock(path)

    def test_git_blob_identity_is_checked_for_non_lfs_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "README.md"
            path.write_bytes(b"hello\n")
            entry = {"size": 6, "git_oid": "ce013625030ba8dba906f756967f9e9ca394464a"}
            result = acquirer.verify_locked_file(path, entry)
            self.assertTrue(result["verified"])
            path.write_bytes(b"jello\n")
            with self.assertRaisesRegex(RuntimeError, "Git object mismatch"):
                acquirer.verify_locked_file(path, entry)

    def test_unsafe_lock_paths_are_rejected(self) -> None:
        for value in (
            "../secret", "/absolute", "data\\file.jsonl", "data/./file.jsonl",
            "data/*.jsonl", "data/[01].jsonl", "data/line\nfeed.jsonl",
        ):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                acquirer._safe_relative_path(value, field="test")


class ImmutableAcquisitionTests(unittest.TestCase):
    def _patched_source(self, entries: list[dict[str, object]]):
        source = {
            "repo_id": "example/source",
            "revision": "a" * 40,
            "local_directory": "Source",
            "files": entries,
        }
        acquisition = {
            "directory": "Source",
            "patterns": [entry["remote_path"] for entry in entries],
            "files": entries,
            "permanently_excluded": [],
        }
        return (
            mock.patch.dict(acquirer.ACQUISITION_LOCK["datasets"], {"xlam_60k": source}),
            mock.patch.dict(acquirer.ACQUISITION, {"xlam_60k": acquisition}),
        )

    def test_only_missing_exact_file_is_staged_and_atomically_placed(self) -> None:
        retained = b"owner-provided bytes"
        missing = b"downloaded bytes"
        entries = [
            locked_entry("remote/retained.bin", "retained.bin", retained),
            locked_entry("remote/missing.bin", "nested/missing.bin", missing),
        ]
        seen: dict[str, object] = {}

        def fake_download(**kwargs: object) -> str:
            seen.update(kwargs)
            stage = Path(str(kwargs["local_dir"]))
            path = stage / "remote" / "missing.bin"
            path.parent.mkdir(parents=True)
            path.write_bytes(missing)
            return str(stage)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "Source"
            destination.mkdir()
            retained_path = destination / "retained.bin"
            retained_path.write_bytes(retained)
            patches = self._patched_source(entries)
            with patches[0], patches[1]:
                result = acquirer.acquire("xlam_60k", root, 2, snapshot_download_fn=fake_download)
            self.assertEqual(seen["repo_id"], "example/source")
            self.assertEqual(seen["revision"], "a" * 40)
            self.assertEqual(seen["allow_patterns"], ["remote/missing.bin"])
            self.assertEqual(retained_path.read_bytes(), retained)
            self.assertEqual((destination / "nested" / "missing.bin").read_bytes(), missing)
            self.assertEqual(result["downloaded"], ["nested/missing.bin"])

    def test_invalid_existing_file_is_never_replaced_or_downloaded(self) -> None:
        expected = b"correct bytes"
        entry = locked_entry("remote/file.bin", "file.bin", expected)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "Source"
            destination.mkdir()
            path = destination / "file.bin"
            path.write_bytes(b"wrong bytes!!")
            downloader = mock.Mock()
            patches = self._patched_source([entry])
            with patches[0], patches[1], self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                acquirer.acquire("xlam_60k", root, 1, snapshot_download_fn=downloader)
            downloader.assert_not_called()
            self.assertEqual(path.read_bytes(), b"wrong bytes!!")

    def test_all_staged_files_are_verified_before_any_is_placed(self) -> None:
        first = b"first"
        second = b"second"
        entries = [
            locked_entry("remote/first.bin", "first.bin", first),
            locked_entry("remote/second.bin", "second.bin", second),
        ]

        def fake_download(**kwargs: object) -> str:
            stage = Path(str(kwargs["local_dir"]))
            remote = stage / "remote"
            remote.mkdir()
            (remote / "first.bin").write_bytes(first)
            (remote / "second.bin").write_bytes(b"broken")
            return str(stage)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patches = self._patched_source(entries)
            with patches[0], patches[1], self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                acquirer.acquire("xlam_60k", root, 1, snapshot_download_fn=fake_download)
            self.assertFalse((root / "Source" / "first.bin").exists())
            self.assertFalse((root / "Source" / "second.bin").exists())

    def test_interrupted_download_reuses_revision_keyed_staging(self) -> None:
        payload = b"resumed bytes"
        entry = locked_entry("remote/file.bin", "file.bin", payload)
        stages: list[Path] = []

        def interrupted(**kwargs: object) -> str:
            stage = Path(str(kwargs["local_dir"]))
            stages.append(stage)
            marker = stage / ".cache" / "partial.marker"
            marker.parent.mkdir(parents=True)
            marker.write_bytes(b"partial")
            raise RuntimeError("simulated interruption")

        def resumed(**kwargs: object) -> str:
            stage = Path(str(kwargs["local_dir"]))
            stages.append(stage)
            self.assertEqual((stage / ".cache" / "partial.marker").read_bytes(), b"partial")
            source = stage / "remote" / "file.bin"
            source.parent.mkdir(parents=True)
            source.write_bytes(payload)
            return str(stage)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patches = self._patched_source([entry])
            with patches[0], patches[1], self.assertRaisesRegex(RuntimeError, "interruption"):
                acquirer.acquire("xlam_60k", root, 1, snapshot_download_fn=interrupted)
            patches = self._patched_source([entry])
            with patches[0], patches[1]:
                result = acquirer.acquire("xlam_60k", root, 1, snapshot_download_fn=resumed)
            self.assertEqual(stages[0], stages[1])
            self.assertIn(".external-corpus-v1-staging", str(stages[0]))
            self.assertEqual((root / "Source" / "file.bin").read_bytes(), payload)
            self.assertEqual(result["persistent_staging"], str(stages[1]))

    def test_existing_destination_wins_race_without_overwrite(self) -> None:
        payload = b"expected"
        entry = locked_entry("remote/file.bin", "file.bin", payload)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            staged = directory / "staged.bin"
            target = directory / "target.bin"
            staged.write_bytes(payload)
            target.write_bytes(payload)
            result = acquirer._place_without_overwrite(staged, target, entry)
            self.assertEqual(result, "reused_after_race")
            self.assertEqual(target.read_bytes(), payload)

    def test_verify_only_cli_does_not_create_a_missing_source_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "does-not-exist"
            with self.assertRaisesRegex(RuntimeError, "missing locked files"):
                acquirer.main([
                    "--source-root", str(root), "--dataset", "xlam_60k", "--verify-only",
                ])
            self.assertFalse(root.exists())

    def test_non_file_source_is_rejected_before_downloading(self) -> None:
        entry = locked_entry("remote/file.bin", "file.bin", b"expected")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "Source" / "file.bin"
            path.mkdir(parents=True)
            downloader = mock.Mock()
            patches = self._patched_source([entry])
            with patches[0], patches[1], self.assertRaisesRegex(RuntimeError, "regular file"):
                acquirer.acquire("xlam_60k", root, 1, snapshot_download_fn=downloader)
            downloader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
