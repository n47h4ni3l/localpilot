"""Exact runtime contract for reproducible External Corpus v1 builds."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from typing import Any


EXPECTED_PACKAGE_VERSIONS = {
    "certifi": "2026.7.22",
    "charset-normalizer": "3.5.1",
    "colorama": "0.4.6",
    "filelock": "3.32.5",
    "fsspec": "2026.7.0",
    "huggingface-hub": "0.36.2",
    "idna": "3.19",
    "jinja2": "3.1.6",
    "markupsafe": "3.0.3",
    "numpy": "2.5.3",
    "packaging": "26.3",
    "pyarrow": "25.0.1",
    "pyyaml": "6.0.3",
    "regex": "2026.9.3",
    "requests": "2.34.2",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "tqdm": "4.70.0",
    "transformers": "4.56.2",
    "typing-extensions": "4.16.0",
    "urllib3": "2.7.0",
}
EXPECTED_PYTHON_VERSION = "3.13.3"
EXPECTED_PYTHON_IMPLEMENTATION = "CPython"
EXPECTED_PLATFORM_SYSTEM = "Windows"
EXPECTED_EXECUTABLE_BITS = 64


def build_runtime(*, require_exact: bool) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    mismatches: list[str] = []
    for package, expected in EXPECTED_PACKAGE_VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        packages[package] = actual
        if actual != expected:
            mismatches.append(f"{package}: expected {expected}, found {actual or 'not installed'}")
    python_version = platform.python_version()
    python_implementation = platform.python_implementation()
    platform_system = platform.system()
    executable_bits = 64 if sys.maxsize > 2**32 else 32
    if python_version != EXPECTED_PYTHON_VERSION:
        mismatches.append(f"python: expected {EXPECTED_PYTHON_VERSION}, found {python_version}")
    if python_implementation != EXPECTED_PYTHON_IMPLEMENTATION:
        mismatches.append(
            "python implementation: expected "
            f"{EXPECTED_PYTHON_IMPLEMENTATION}, found {python_implementation}"
        )
    if platform_system != EXPECTED_PLATFORM_SYSTEM:
        mismatches.append(
            f"platform: expected {EXPECTED_PLATFORM_SYSTEM}, found {platform_system}"
        )
    if executable_bits != EXPECTED_EXECUTABLE_BITS:
        mismatches.append(
            f"python executable: expected {EXPECTED_EXECUTABLE_BITS}-bit, found {executable_bits}-bit"
        )
    if require_exact and mismatches:
        raise RuntimeError(
            "External Corpus v1 requires its exact locked build dependencies; "
            "install training/requirements-external-corpus-v1.txt with Python "
            f"{EXPECTED_PYTHON_VERSION} before rebuilding (" + "; ".join(mismatches) + ")"
        )
    return {
        "python": python_version,
        "python_implementation": python_implementation,
        "python_executable_bits": executable_bits,
        "platform_system": platform_system,
        "platform": platform.platform(),
        "packages": packages,
        "exact_package_versions_verified": not mismatches,
    }
