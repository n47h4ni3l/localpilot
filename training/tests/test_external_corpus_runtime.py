from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from training.scripts.external_corpus_runtime import (
    EXPECTED_EXECUTABLE_BITS,
    EXPECTED_PACKAGE_VERSIONS,
    EXPECTED_PLATFORM_SYSTEM,
    EXPECTED_PYTHON_IMPLEMENTATION,
    EXPECTED_PYTHON_VERSION,
    build_runtime,
)


REPOSITORY = Path(__file__).resolve().parents[2]


def normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def exact_requirement(value: str) -> tuple[str, str, str | None]:
    name, separator, version_and_marker = value.partition("==")
    if separator != "==":
        raise AssertionError(f"requirement is not exactly pinned: {value}")
    version, marker_separator, marker = version_and_marker.partition(";")
    normalized_marker = None
    if marker_separator:
        normalized_marker = re.sub(r"[\"']", "", marker.strip())
        normalized_marker = re.sub(r"\s+", " ", normalized_marker)
    return normalized_name(name.strip()), version.strip(), normalized_marker


class ExternalCorpusRuntimeTests(unittest.TestCase):
    def test_exact_runtime_is_recorded(self) -> None:
        with patch(
            "training.scripts.external_corpus_runtime.importlib.metadata.version",
            side_effect=lambda name: EXPECTED_PACKAGE_VERSIONS[name],
        ), patch(
            "training.scripts.external_corpus_runtime.platform.python_version",
            return_value=EXPECTED_PYTHON_VERSION,
        ), patch(
            "training.scripts.external_corpus_runtime.platform.python_implementation",
            return_value=EXPECTED_PYTHON_IMPLEMENTATION,
        ), patch(
            "training.scripts.external_corpus_runtime.platform.system",
            return_value=EXPECTED_PLATFORM_SYSTEM,
        ), patch(
            "training.scripts.external_corpus_runtime.sys.maxsize",
            (2 ** (EXPECTED_EXECUTABLE_BITS - 1)) - 1,
        ):
            value = build_runtime(require_exact=True)
        self.assertTrue(value["exact_package_versions_verified"])
        self.assertEqual(value["packages"], EXPECTED_PACKAGE_VERSIONS)
        self.assertTrue(value["python"])

    def test_mismatch_fails_closed(self) -> None:
        with patch(
            "training.scripts.external_corpus_runtime.importlib.metadata.version",
            return_value="0.0.0",
        ):
            with self.assertRaisesRegex(RuntimeError, "exact locked build dependencies"):
                build_runtime(require_exact=True)

    def test_python_mismatch_fails_closed(self) -> None:
        with patch(
            "training.scripts.external_corpus_runtime.importlib.metadata.version",
            side_effect=lambda name: EXPECTED_PACKAGE_VERSIONS[name],
        ), patch(
            "training.scripts.external_corpus_runtime.platform.python_version",
            return_value="3.13.2",
        ), patch(
            "training.scripts.external_corpus_runtime.platform.python_implementation",
            return_value=EXPECTED_PYTHON_IMPLEMENTATION,
        ), patch(
            "training.scripts.external_corpus_runtime.platform.system",
            return_value=EXPECTED_PLATFORM_SYSTEM,
        ), patch(
            "training.scripts.external_corpus_runtime.sys.maxsize",
            (2 ** (EXPECTED_EXECUTABLE_BITS - 1)) - 1,
        ):
            with self.assertRaisesRegex(RuntimeError, "python: expected 3.13.3"):
                build_runtime(require_exact=True)

    def test_platform_mismatch_fails_closed(self) -> None:
        with patch(
            "training.scripts.external_corpus_runtime.importlib.metadata.version",
            side_effect=lambda name: EXPECTED_PACKAGE_VERSIONS[name],
        ), patch(
            "training.scripts.external_corpus_runtime.platform.python_version",
            return_value=EXPECTED_PYTHON_VERSION,
        ), patch(
            "training.scripts.external_corpus_runtime.platform.python_implementation",
            return_value=EXPECTED_PYTHON_IMPLEMENTATION,
        ), patch(
            "training.scripts.external_corpus_runtime.platform.system",
            return_value="Linux",
        ), patch(
            "training.scripts.external_corpus_runtime.sys.maxsize",
            (2 ** (EXPECTED_EXECUTABLE_BITS - 1)) - 1,
        ):
            with self.assertRaisesRegex(RuntimeError, "platform: expected Windows"):
                build_runtime(require_exact=True)

    def test_requirements_and_corpus_extra_match_runtime_lock(self) -> None:
        requirements: dict[str, str] = {}
        requirement_markers: dict[str, str] = {}
        for raw_line in (REPOSITORY / "training/requirements-external-corpus-v1.txt").read_text(
            encoding="utf-8"
        ).splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            name, version, marker = exact_requirement(line)
            requirements[name] = version
            if marker:
                requirement_markers[name] = marker

        with (REPOSITORY / "pyproject.toml").open("rb") as handle:
            corpus_extra = tomllib.load(handle)["project"]["optional-dependencies"]["corpus"]
        pyproject_versions = {}
        pyproject_markers: dict[str, str] = {}
        for requirement in corpus_extra:
            name, version, marker = exact_requirement(requirement)
            pyproject_versions[name] = version
            if marker:
                pyproject_markers[name] = marker

        self.assertEqual(requirements, EXPECTED_PACKAGE_VERSIONS)
        self.assertEqual(pyproject_versions, EXPECTED_PACKAGE_VERSIONS)
        self.assertEqual(requirement_markers, {"colorama": "sys_platform == win32"})
        self.assertEqual(pyproject_markers, requirement_markers)


if __name__ == "__main__":
    unittest.main()
