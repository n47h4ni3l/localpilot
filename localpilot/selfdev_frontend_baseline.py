"""Identify unchanged approved frontend source without executing candidate code."""
from pathlib import Path
import re
import subprocess

from localpilot.process import hidden_process_creation_flags


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", "--no-replace-objects", *args], cwd=root, capture_output=True, text=True,
                                encoding="utf-8", errors="strict", timeout=10, check=False,
                                shell=False, creationflags=hidden_process_creation_flags())
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def approved_main_revision(root: Path, *, remote: str, branch: str) -> str | None:
    """Pin only a clean source checkout matching its trusted remote main ref."""
    top = _git(root, "rev-parse", "--show-toplevel")
    if (top is None or Path(top).resolve() != root.resolve()
            or _git(root, "branch", "--show-current") != branch
            or _git(root, "status", "--porcelain", "--untracked-files=all") != ""):
        return None
    head = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
    upstream = _git(root, "rev-parse", "--verify", f"refs/remotes/{remote}/{branch}^{{commit}}")
    return head if head and head == upstream and re.fullmatch(r"[0-9a-f]{40,64}", head) else None


class TrustedFrontendBaseline:
    def __init__(self, workspace: Path, approved_revision: str | None):
        self.workspace = workspace
        self.revision = None
        self._contents: dict[str, str | None] = {}
        if approved_revision and re.fullmatch(r"[0-9a-f]{40,64}", approved_revision):
            # Candidate-authored commits are never their own baseline. Only
            # the common ancestor with the independently pinned main is used.
            base = _git(workspace, "merge-base", approved_revision, "HEAD")
            if base and re.fullmatch(r"[0-9a-f]{40,64}", base):
                self.revision = base

    def matches(self, relative: Path, content: str) -> bool:
        if self.revision is None or relative.parts[:2] != ("localpilot", "webview"):
            return False
        name = relative.as_posix()
        if name not in self._contents:
            spec = f"{self.revision}:{name}"
            size = _git(self.workspace, "cat-file", "-s", spec)
            if size is None or not size.isdigit() or int(size) > 1_000_000:
                self._contents[name] = None
            else:
                # The blob is data; no diff driver, text conversion or candidate
                # JavaScript is executed. Preserve whitespace except newlines.
                try:
                    result = subprocess.run(
                        ["git", "--no-replace-objects", "show", "--no-ext-diff", "--no-textconv", spec],
                        cwd=self.workspace, capture_output=True, text=True,
                        encoding="utf-8", errors="strict", timeout=10, check=False,
                        shell=False, creationflags=hidden_process_creation_flags(),
                    )
                    self._contents[name] = result.stdout if result.returncode == 0 else None
                except (OSError, subprocess.SubprocessError, UnicodeError):
                    self._contents[name] = None
        return self._contents[name] is not None and content == self._contents[name]
