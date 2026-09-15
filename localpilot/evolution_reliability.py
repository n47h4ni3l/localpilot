from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Any

from localpilot.evolution_orchestrator import (
    EvolutionRunAlreadyActive,
    EvolutionRunLease,
)
from localpilot.selfdev import (
    CandidateRejectionError,
    CandidateRetryError,
    SelfDeveloper as _BaseSelfDeveloper,
)
from localpilot.selfdev_results import EvolutionResult


class SelfDeveloper(_BaseSelfDeveloper):
    """Production self-development entry point with lifecycle reliability guards.

    The base pipeline remains the single owner of research, grounding,
    implementation, safety checks, delivery, and durable learning. This wrapper
    only closes lifecycle failure modes that sit around those stages: concurrent
    invocations, retry/checkpoint continuity, durable framework-failure evidence,
    and cleanup of clean terminal worktrees.
    """

    def run_once(self, *, force: bool = False) -> EvolutionResult:
        invocation_id = uuid.uuid4().hex
        lease = EvolutionRunLease(
            self.data_dir / "evolution-run.lock",
            invocation_id,
        )
        try:
            lease.acquire()
        except EvolutionRunAlreadyActive as exc:
            self.audit.write(
                "evolve_run_deferred",
                invocation_id=invocation_id,
                force=force,
                reason=str(exc)[:1000],
                concurrency_guard=True,
            )
            return EvolutionResult(
                "deferred",
                None,
                None,
                f"Evolution deferred: {exc}",
            )
        try:
            return super().run_once(force=force)
        finally:
            lease.release()

    def retry_candidate(self, identifier: str, *, reason: str):
        """Preserve a valid checkpoint whenever retry resumes the same worktree."""
        try:
            checkpoint = self.checkpoints.load()
        except Exception:
            checkpoint = None

        result = super().retry_candidate(identifier, reason=reason)
        if checkpoint is None or result.resume_mode != "resume_existing_worktree":
            return result
        if checkpoint.branch != result.branch:
            return result

        rebound = checkpoint.rebind_cycle(result.retry_cycle_id)
        if rebound.milestone == "recovery" and rebound.files_changed:
            if rebound.static_check_status == "failed":
                rebound = replace(
                    rebound,
                    milestone="local_static_repair",
                    next_action=(
                        "Resume the retained candidate at static repair using the "
                        "recorded failure evidence; do not repeat completed research "
                        "or grounding."
                    ),
                )
            elif rebound.static_check_status == "passed":
                rebound = replace(
                    rebound,
                    milestone="static_checks",
                    next_action=(
                        "Revalidate the retained candidate and continue delivery; do "
                        "not repeat completed research or grounding."
                    ),
                )
        try:
            self.checkpoints.save(rebound)
            verified = self.checkpoints.load()
            if verified is None or verified.cycle_id != result.retry_cycle_id:
                raise RuntimeError(
                    "checkpoint rebind did not persist the authorized retry cycle identity"
                )
        except Exception as exc:
            self.audit.write(
                "candidate_policy_retry_checkpoint_rebind",
                status="failed",
                prior_cycle_id=result.prior_cycle_id,
                retry_cycle_id=result.retry_cycle_id,
                branch=result.branch,
                error=f"{type(exc).__name__}: {exc}"[:1000],
            )
            raise CandidateRetryError(
                "Retry was authorized, but its same-worktree checkpoint could not be "
                f"preserved safely: {type(exc).__name__}: {exc}"
            ) from exc
        self.audit.write(
            "candidate_policy_retry_checkpoint_rebind",
            status="preserved",
            prior_cycle_id=result.prior_cycle_id,
            retry_cycle_id=result.retry_cycle_id,
            branch=result.branch,
            milestone=rebound.milestone,
            git_state_digest=rebound.git_state_digest,
        )
        return result

    def _repair_static_failures(self, *args: Any, **kwargs: Any):
        """Retain structured-repair framework failures outside mutable summary text."""
        result = super()._repair_static_failures(*args, **kwargs)
        marker = "Structured static repair failed:"
        if marker not in result.final_text:
            return result

        cycle_id = kwargs.get("cycle_id")
        if cycle_id is None:
            return result
        detail = result.final_text[result.final_text.rfind(marker) :].strip()[:3000]
        evidence = (
            "Framework policy/orchestration failure: the structured static-repair "
            f"contract failed before a valid repair could be applied. {detail}"
        )
        self.memory.record_write_integrity_failure(int(cycle_id), evidence)
        self.audit.write(
            "selfdev_framework_repair_failure",
            cycle_id=int(cycle_id),
            branch=str(kwargs.get("branch") or ""),
            failure_attribution="framework_policy",
            evidence=evidence[:2000],
        )
        return result

    def _continue_candidate(self, **kwargs: Any) -> EvolutionResult:
        """Run the base pipeline, then remove only provably clean terminal worktrees."""
        result = super()._continue_candidate(**kwargs)
        if result.status not in {"failed", "no_changes"}:
            return result
        if not bool(kwargs.get("is_worktree")):
            return result

        branch = str(kwargs.get("branch") or result.branch or "")
        workspace_value = kwargs.get("workspace") or result.workspace
        if not branch or workspace_value is None:
            return result
        workspace = workspace_value.resolve()
        try:
            registered = self.github.worktree_for_branch(branch)
            if registered is None or registered.resolve() != workspace:
                return result
            changed = self.github.candidate_changed_paths(workspace)
            has_candidate_commit = self.github.branch_has_candidate_commit(workspace)
        except Exception as exc:
            self.audit.write(
                "selfdev_clean_terminal_cleanup",
                status="skipped",
                branch=branch,
                error=f"{type(exc).__name__}: {exc}"[:1000],
            )
            return result
        if changed or has_candidate_commit:
            return result

        try:
            cleanup = self.github.remove_candidate_worktree(
                branch,
                expected_workspace=workspace,
            )
        except Exception as exc:
            self.audit.write(
                "selfdev_clean_terminal_cleanup",
                status="failed",
                branch=branch,
                workspace=str(workspace),
                error=f"{type(exc).__name__}: {exc}"[:1000],
            )
            return result

        detail = (cleanup.stdout or cleanup.stderr or "").strip()
        self.audit.write(
            "selfdev_clean_terminal_cleanup",
            status="removed" if cleanup.ok else "failed",
            branch=branch,
            workspace=str(workspace),
            detail=detail[:1000],
            branch_history_retained=True,
        )
        if not cleanup.ok:
            return result
        suffix = "Clean terminal candidate worktree removed; branch/history retained."
        return EvolutionResult(
            result.status,
            result.branch,
            result.workspace,
            f"{result.summary}\n\n{suffix}",
            result.tests_passed,
        )


__all__ = [
    "CandidateRejectionError",
    "CandidateRetryError",
    "SelfDeveloper",
]
