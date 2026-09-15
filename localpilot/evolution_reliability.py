from __future__ import annotations

import json
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
    closes lifecycle failure modes around those stages and keeps configured
    Claude Code ownership intact for implementation repair work.
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

    @staticmethod
    def _normalize_recovery_checkpoint(checkpoint):
        """Resume an existing candidate diff at validation, never pre-write grounding.

        A ``recovery`` checkpoint can be recorded after any exception. If the
        candidate already has changed files, implementation has already happened
        and re-running research/grounding can only discard progress or fail on a
        stage that is no longer relevant. Revalidate the retained diff first;
        if the prior static status is known-failed, resume directly at repair.
        """
        if checkpoint.milestone != "recovery" or not checkpoint.files_changed:
            return checkpoint
        if checkpoint.static_check_status == "failed":
            return replace(
                checkpoint,
                milestone="local_static_repair",
                next_action=(
                    "Resume the retained candidate at static repair using the "
                    "recorded failure evidence; do not repeat completed research, "
                    "grounding, or implementation."
                ),
            )
        return replace(
            checkpoint,
            milestone="static_checks",
            next_action=(
                "Revalidate the retained candidate before delivery; do not repeat "
                "completed research, grounding, or implementation."
            ),
        )

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

        rebound = self._normalize_recovery_checkpoint(
            checkpoint.rebind_cycle(result.retry_cycle_id)
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

    def _resume_checkpoint_candidate(self, validated, *, force: bool) -> EvolutionResult:
        """Normalize recovery checkpoints with an existing diff before base resume."""
        checkpoint, candidate, task, workspace = validated
        normalized = self._normalize_recovery_checkpoint(checkpoint)
        if normalized is not checkpoint:
            try:
                self.checkpoints.save(normalized)
                verified = self.checkpoints.load()
                if (
                    verified is None
                    or verified.cycle_id != normalized.cycle_id
                    or verified.branch != normalized.branch
                    or verified.milestone != normalized.milestone
                ):
                    raise RuntimeError(
                        "normalized recovery checkpoint did not persist safely"
                    )
            except Exception as exc:
                self.audit.write(
                    "selfdev_recovery_checkpoint_normalization",
                    status="failed",
                    branch=checkpoint.branch,
                    cycle_id=checkpoint.cycle_id,
                    error=f"{type(exc).__name__}: {exc}"[:1000],
                )
                return EvolutionResult(
                    "failed",
                    checkpoint.branch,
                    workspace,
                    "Recovered candidate could not be rebound safely to its "
                    f"validation stage: {type(exc).__name__}: {exc}",
                    False,
                )
            self.audit.write(
                "selfdev_recovery_checkpoint_normalization",
                status="preserved",
                branch=normalized.branch,
                cycle_id=normalized.cycle_id,
                prior_milestone=checkpoint.milestone,
                milestone=normalized.milestone,
                files_changed=len(normalized.files_changed),
            )
        return super()._resume_checkpoint_candidate(
            (normalized, candidate, task, workspace),
            force=force,
        )

    def _claude_repair_scope(self, workspace, tools) -> tuple[str, ...]:
        """Return the existing candidate-owned paths Claude may repair.

        A pushed candidate is normally clean, so ``git status`` alone is not
        enough. The repair scope is the committed candidate diff from the merge
        base with trusted main plus any current uncommitted candidate changes.
        Reviewer-protected tests are always removed from the writable scope.
        """
        main_branch = str(self.config.github.main_branch).strip()
        base = self.github._run(
            ["git", "merge-base", main_branch, "HEAD"],
            cwd=workspace,
        )
        if not base.ok or not base.stdout:
            raise RuntimeError(
                "Claude Code repair could not establish the candidate merge base with trusted main."
            )
        committed = self.github._run(
            [
                "git",
                "diff",
                "--name-only",
                "--diff-filter=ACMRT",
                f"{base.stdout}..HEAD",
                "--",
                ".",
            ],
            cwd=workspace,
        )
        if not committed.ok:
            raise RuntimeError(
                "Claude Code repair could not establish the committed candidate path scope: "
                + (committed.stderr or committed.stdout or "git diff failed")
            )

        candidates = {
            line.strip().replace("\\", "/")
            for line in committed.stdout.splitlines()
            if line.strip()
        }
        candidates.update(
            str(item).strip().replace("\\", "/")
            for item in self.github.candidate_changed_paths(workspace)
            if str(item).strip()
        )
        scope: list[str] = []
        for relative in sorted(candidates):
            if relative in tools.protected_paths:
                continue
            try:
                resolved = tools._resolve(relative)
            except Exception:
                continue
            if resolved.is_file():
                scope.append(relative)
        return tuple(scope)

    def _tool_stage(
        self,
        *,
        chat,
        model: str,
        messages: list[dict[str, Any]],
        functions,
        rounds: int,
        force: bool,
        branch: str,
        stage: str,
    ) -> str:
        """Keep LocalPilot as reviewer while Claude Code owns repair edits.

        The base implementation predates the Claude Code backend and still uses
        LocalPilot's direct file-tool loop for ``ci_repair`` and
        ``local_static_repair``. When Claude Code is configured, route those
        write stages through the same confined implementation backend and its
        independent LocalPilot acceptance-review loop instead.
        """
        if (
            stage not in {"ci_repair", "local_static_repair"}
            or self.config.selfdev.implementation_backend != "claude_code"
        ):
            return super()._tool_stage(
                chat=chat,
                model=model,
                messages=messages,
                functions=functions,
                rounds=rounds,
                force=force,
                branch=branch,
                stage=stage,
            )

        context = self._active_checkpoint
        if not isinstance(context, dict):
            raise RuntimeError(
                f"Claude Code {stage} requires an active candidate checkpoint."
            )
        task = context.get("task")
        tools = context.get("tools")
        workspace = context.get("workspace")
        cycle_id = context.get("cycle_id")
        if not isinstance(task, dict) or tools is None or workspace is None or cycle_id is None:
            raise RuntimeError(
                f"Claude Code {stage} could not resolve the active candidate context."
            )
        evaluation = task.get("evaluation")
        if not isinstance(evaluation, dict):
            raise RuntimeError(
                f"Claude Code {stage} requires the candidate evaluation contract."
            )

        allowed_paths = self._claude_repair_scope(workspace, tools)
        if not allowed_paths:
            raise RuntimeError(
                f"Claude Code {stage} has no safe candidate-owned paths to repair."
            )

        stage_label = "CI failure" if stage == "ci_repair" else "static-check failure"
        system_context = "\n".join(
            str(item.get("content") or "")
            for item in messages
            if isinstance(item, dict) and item.get("role") == "system"
        )
        research = (
            f"REPAIR DIRECTIVE: Repair the existing candidate's {stage_label}. Preserve correct "
            "existing work, make the smallest concrete fix, and do not broaden scope. LocalPilot "
            "remains the independent reviewer and will send rejected work back for another bounded "
            "Claude Code pass.\n\n"
            f"Recorded repair evidence:\n{system_context[:18000]}"
        )
        lessons = self.memory.reusable_lessons(self.config.selfdev.lesson_limit)
        self._emit(
            f"Delegating {stage} edits to Claude Code across {len(allowed_paths)} confined path(s)"
        )
        self.audit.write(
            "selfdev_claude_repair_handoff",
            branch=branch,
            cycle_id=int(cycle_id),
            task_id=str(task.get("id") or ""),
            stage=stage,
            allowed_paths=list(allowed_paths),
            reviewer_protected_paths=sorted(tools.protected_paths),
        )
        return self._run_claude_code_implementation(
            chat=chat,
            developer_model=model,
            task=task,
            branch=branch,
            workspace=workspace,
            tools=tools,
            cycle_id=int(cycle_id),
            research=research,
            grounding_plan={
                "referenced_paths": list(allowed_paths),
                "new_runtime_paths": [],
            },
            grounding_evidence=[
                "Repair scope is confined to existing candidate-owned paths relative to trusted main."
            ],
            evolution_context=self._evolution_context(task),
            lessons=lessons,
            force=force,
        )

    def _repair_static_failures(self, *args: Any, **kwargs: Any):
        """Preserve repair evidence and the capability-evaluation handoff."""
        result = super()._repair_static_failures(*args, **kwargs)
        original_final_text = result.final_text

        # A successful static repair returns a small summary/lesson object. In a
        # resumed candidate that used to replace the pre-existing pending-CI
        # evaluation handoff, making the capability evidence gate report the
        # candidate as unmeasured even though the configured measurement plan was
        # still valid. Restore only the conservative pending-CI contract; never
        # overwrite an explicit measured/regressed/inconclusive result.
        task = kwargs.get("task")
        if result.passed and isinstance(task, dict):
            try:
                report = self._evaluation_report(result.final_text, task)
            except Exception:
                report = {}
            if report.get("result") == "unmeasured":
                evaluation = task.get("evaluation")
                if isinstance(evaluation, dict):
                    measurement_artifact = str(
                        evaluation.get("measurement_method") or ""
                    ).strip()
                    if measurement_artifact:
                        summary, lesson = self._outcome(
                            result.final_text,
                            "Preserve the capability evaluation contract across bounded repair.",
                        )
                        result = replace(
                            result,
                            final_text=json.dumps(
                                {
                                    "summary": summary,
                                    "reusable_lesson": lesson,
                                    "evaluation_evidence": {
                                        "metric": str(evaluation.get("metric") or ""),
                                        "baseline_evidence": str(
                                            evaluation.get("baseline") or ""
                                        ),
                                        "candidate_evidence": (
                                            "Static repair completed and static checks passed; "
                                            "capability measurement remains pending the configured "
                                            "GitHub CI evaluation artifact."
                                        ),
                                        "result": "pending_ci",
                                        "measurement_artifact": measurement_artifact,
                                    },
                                },
                                ensure_ascii=False,
                            ),
                        )
                        self.audit.write(
                            "selfdev_repair_evaluation_handoff",
                            cycle_id=int(kwargs.get("cycle_id") or 0),
                            branch=str(kwargs.get("branch") or ""),
                            status="preserved",
                            result="pending_ci",
                            measurement_artifact=measurement_artifact[:1000],
                        )

        marker = "Structured static repair failed:"
        if marker not in original_final_text:
            return result

        cycle_id = kwargs.get("cycle_id")
        if cycle_id is None:
            return result
        detail = original_final_text[original_final_text.rfind(marker) :].strip()[:3000]
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
