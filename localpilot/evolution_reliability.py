from __future__ import annotations

import json
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from localpilot import selfdev_model_selection
from localpilot.evolution_orchestrator import (
    EvolutionRunAlreadyActive,
    EvolutionRunLease,
)
from localpilot.implementation_backend import (
    ClaudeCodeBackend,
    ImplementationRequest,
    ImplementationResult,
    ImplementationStatus,
)
from localpilot.selfdev import (
    CandidateRejectionError,
    CandidateRetryError,
    SelfDeveloper as _BaseSelfDeveloper,
)
from localpilot.selfdev_results import CyclePaused, EvolutionResult


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

    def _claude_progress_path(self) -> Path:
        return self.data_dir / "claude-repair-progress.json"

    def _load_claude_repair_progress(
        self,
        *,
        branch: str,
        cycle_id: int,
        stage: str,
    ) -> dict[str, Any] | None:
        path = self._claude_progress_path()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        try:
            repair_pass = int(payload.get("repair_pass") or 0)
            saved_cycle = int(payload.get("cycle_id") or 0)
        except (TypeError, ValueError):
            return None
        if (
            saved_cycle != int(cycle_id)
            or str(payload.get("branch") or "") != str(branch)
            or str(payload.get("stage") or "") != str(stage)
            or repair_pass < 1
        ):
            return None
        return {
            "repair_pass": repair_pass,
            "review_feedback": str(payload.get("review_feedback") or "")[:12000],
            "review_summary": str(payload.get("review_summary") or "")[:2000],
        }

    def _save_claude_repair_progress(
        self,
        *,
        branch: str,
        cycle_id: int,
        task_id: str,
        stage: str,
        repair_pass: int,
        review_feedback: str,
        review_summary: str,
    ) -> None:
        path = self._claude_progress_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "branch": str(branch)[:300],
            "cycle_id": int(cycle_id),
            "task_id": str(task_id)[:200],
            "stage": str(stage)[:100],
            "repair_pass": max(1, int(repair_pass)),
            "review_feedback": str(review_feedback or "")[:12000],
            "review_summary": str(review_summary or "")[:2000],
        }
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        self.audit.write(
            "selfdev_claude_repair_progress_saved",
            branch=branch,
            cycle_id=int(cycle_id),
            task_id=str(task_id),
            stage=stage,
            repair_pass=payload["repair_pass"],
        )

    def _clear_claude_repair_progress(
        self,
        *,
        branch: str,
        cycle_id: int,
        stage: str,
        reason: str,
    ) -> None:
        path = self._claude_progress_path()
        try:
            path.unlink()
        except FileNotFoundError:
            return
        self.audit.write(
            "selfdev_claude_repair_progress_cleared",
            branch=branch,
            cycle_id=int(cycle_id),
            stage=stage,
            reason=str(reason)[:500],
        )

    def _check_resources(
        self,
        force: bool,
        branch: str,
        *,
        during_inference: bool = False,
    ) -> None:
        """Do not interrupt an already-admitted Claude/reviewer unit for wall time.

        The whole-cycle budget is an admission boundary between Claude repair
        passes. Once one pass has been admitted, its own 600-second backend
        timeout plus the following bounded LocalPilot review are allowed to
        finish. Foreground and memory/resource gates remain active because only
        the run-budget object is temporarily detached while the base resource
        checks execute.
        """
        if not bool(getattr(self, "_claude_unit_admitted", False)):
            return super()._check_resources(
                force,
                branch,
                during_inference=during_inference,
            )
        budget = self._budget
        self._budget = None
        try:
            return super()._check_resources(
                force,
                branch,
                during_inference=during_inference,
            )
        finally:
            self._budget = budget

    def _admit_claude_unit(self, *, force: bool, branch: str, stage: str) -> None:
        """Enforce the whole-cycle budget before starting another bounded pass."""
        self._claude_unit_admitted = False
        selfdev_model_selection._run_inference_guard(
            lambda: self._check_resources(
                force,
                branch,
                during_inference=True,
            )
        )
        self._claude_unit_admitted = True
        self.audit.write(
            "selfdev_claude_unit_admitted",
            branch=branch,
            stage=stage,
            budget_boundary_checked=True,
        )

    def _claude_code_backend(self, *, force: bool, branch: str):
        """Reuse one verified live preflight throughout one Claude repair loop.

        ``ClaudeCodeBackend.run`` defensively preflights before each invocation.
        During a multi-pass LocalPilot review loop that used to reload/probe
        Ollama repeatedly. Ollama can briefly report an active model without its
        ``context_length`` field while a runner is transitioning, which turned a
        successful first Claude pass into a false backend-unavailable failure on
        the next repair pass. A backend instance is scoped to one implementation
        loop, so once its full live preflight succeeds, subsequent passes reuse
        that proof. Runtime process failures and the resource guard still fail
        independently and are not cached.
        """
        backend = super()._claude_code_backend(force=force, branch=branch)
        if not isinstance(backend, ClaudeCodeBackend):
            return backend

        original_preflight = backend.preflight
        verified_preflight = None

        def stable_preflight():
            nonlocal verified_preflight
            if verified_preflight is not None and verified_preflight.healthy:
                return verified_preflight
            result = original_preflight()
            if result.healthy:
                verified_preflight = result
            return result

        backend.preflight = stable_preflight
        backend.resource_guard = lambda: selfdev_model_selection._run_inference_guard(
            lambda: self._check_resources(
                force,
                branch,
                during_inference=True,
            )
        )
        return backend

    def _run_claude_code_implementation(
        self,
        *,
        chat,
        developer_model: str,
        task: dict[str, Any],
        branch: str,
        workspace: Path,
        tools,
        cycle_id: int,
        research: str,
        grounding_plan: dict[str, list[Any]],
        grounding_evidence: list[str],
        evolution_context: str,
        lessons: list[Any],
        force: bool,
    ) -> str:
        """Run a durable Claude/reviewer loop whose repair passes survive pauses."""
        allowed_paths = tuple(
            sorted(
                {
                    Path(str(item)).as_posix()
                    for field in ("referenced_paths", "new_runtime_paths")
                    for item in grounding_plan.get(field, [])
                    if str(item).strip()
                }
            )
        )
        if not allowed_paths:
            raise RuntimeError("Claude Code implementation requires grounded allowed paths")

        context = self._active_checkpoint if isinstance(self._active_checkpoint, dict) else {}
        explicit_stage = str(getattr(self, "_active_claude_repair_stage", "") or "")
        checkpoint_stage = str(context.get("milestone") or "")
        stage = explicit_stage or (
            checkpoint_stage
            if checkpoint_stage in {"implementation", "ci_repair", "local_static_repair"}
            else "implementation"
        )
        max_repairs = int(self.config.selfdev.implementation_review_repair_passes)
        progress = self._load_claude_repair_progress(
            branch=branch,
            cycle_id=cycle_id,
            stage=stage,
        )
        if progress is not None and int(progress["repair_pass"]) > max_repairs:
            self._clear_claude_repair_progress(
                branch=branch,
                cycle_id=cycle_id,
                stage=stage,
                reason="configured repair limit is now lower than durable progress",
            )
            raise RuntimeError(
                "implementation backend review_rejected: durable Claude repair progress "
                "exceeds the current configured repair-pass limit"
            )

        prompt = (
            "Work only inside the current isolated candidate Git workspace. Implement the focused contract below. "
            "Perform the complete read, edit, repository-test, and repair loop before returning. Do not access parent "
            "directories, localpilot-data, training/evals, training/evolution_execution/acceptance, secrets, the web, "
            "GitHub, package managers, or any network endpoint. Do not commit, push, checkout, switch, branch, reset, "
            "clean, delete files, spawn another shell, or weaken tests. Use only the explicitly available file tools, "
            "narrow git inspection, Python compile checks, and existing repository tests. Change only the exact grounded "
            f"paths listed here: {json.dumps(allowed_paths)}. Reviewer-protected paths are read-only: "
            f"{json.dumps(sorted(tools.protected_paths))}. Finish with one strict JSON object containing summary and "
            "tests. tests must be a non-empty list of {command, passed, exit_code, output_digest}; output_digest is a "
            "SHA-256 digest of bounded command output, never raw logs or reasoning. Do not return chain-of-thought.\n"
            f"Task: {task['title']}\nAcceptance: {json.dumps(task.get('acceptance', []), ensure_ascii=False)}\n"
            f"Capability experiment contract:\n{evolution_context}\n"
            f"LocalPilot research brief:\n{research[:12000]}\n"
            f"Verified grounding plan:\n{json.dumps(grounding_plan, ensure_ascii=False)}\n"
            f"Grounding evidence:\n{json.dumps(grounding_evidence, ensure_ascii=False)}\n"
            f"Earlier reusable lessons:\n{json.dumps(lessons, ensure_ascii=False)}"
        )

        self._admit_claude_unit(force=force, branch=branch, stage=stage)
        try:
            backend = self._claude_code_backend(force=force, branch=branch)
            preflight = backend.preflight()
            self.audit.write(
                "selfdev_implementation_preflight",
                cycle_id=cycle_id,
                backend=preflight.backend,
                model=preflight.model,
                executable=preflight.executable,
                version=preflight.version,
                context_tokens=preflight.context_tokens,
                healthy=preflight.healthy,
                messages=list(preflight.messages),
            )
            if not preflight.healthy:
                raise RuntimeError(
                    "implementation backend unavailable: " + "; ".join(preflight.messages)
                )

            if (workspace / "tests").is_dir():
                uses_pytest = any(
                    (workspace / name).is_file()
                    for name in ("pytest.ini", "conftest.py")
                )
                pyproject = workspace / "pyproject.toml"
                if pyproject.is_file() and "[tool.pytest" in pyproject.read_text(
                    encoding="utf-8", errors="replace"
                ):
                    uses_pytest = True
                test_command = (
                    (sys.executable, "-m", "pytest", "-q")
                    if uses_pytest
                    else (
                        sys.executable,
                        "-m",
                        "unittest",
                        "discover",
                        "-s",
                        "tests",
                        "-p",
                        "test_*.py",
                    )
                )
                test_commands = (test_command,)
            else:
                test_commands = ()

            request = ImplementationRequest(
                workspace=workspace,
                prompt=prompt,
                allowed_paths=allowed_paths,
                protected_paths=tuple(sorted(tools.protected_paths)),
                test_commands=test_commands,
            )
            if progress is not None:
                repair_count = int(progress["repair_pass"])
                self._emit(
                    f"Resuming Claude Code {stage} repair pass {repair_count}/{max_repairs}"
                )
                result = backend.run(
                    replace(
                        request,
                        review_feedback=(
                            str(progress.get("review_feedback") or "")
                            or str(progress.get("review_summary") or "")
                        ),
                    ),
                    repair_pass=repair_count,
                )
            else:
                repair_count = 0
                result = backend.run(request)

            review_status = "not_reviewed"
            terminal_without_review = {
                ImplementationStatus.BACKEND_UNAVAILABLE,
                ImplementationStatus.TIMEOUT,
                ImplementationStatus.RESOURCE_PRESSURE,
                ImplementationStatus.CLI_ERROR,
                ImplementationStatus.CONFINEMENT_VIOLATION,
            }
            if result.status in terminal_without_review:
                if result.status == ImplementationStatus.CONFINEMENT_VIOLATION:
                    self.memory.record_write_integrity_failure(cycle_id, result.summary)
                self._record_backend_evidence(
                    cycle_id=cycle_id,
                    task=task,
                    result=result,
                    review_passes=repair_count,
                    review_status=review_status,
                )
                raise RuntimeError(
                    f"implementation backend {result.status.value}: {result.summary}"
                )

            while True:
                static_result = tools.run_candidate_static_checks()
                diff = tools.show_candidate_diff()
                review_response = self._developer_chat(
                    chat,
                    force=force,
                    branch=branch,
                    model=developer_model,
                    _request_think="low",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are LocalPilot's independent acceptance reviewer. You did not implement this change. "
                                "Review the bounded diff, test evidence, static checks, task contract, and scope. Reject missing "
                                "tests, failed tests, incomplete acceptance, unsafe scope, or unsupported evidence. Return one "
                                "strict JSON object with approved (boolean), feedback (list of concrete repair instructions), "
                                "and summary. Do not reveal hidden reasoning.\n"
                                f"Task: {task['title']}\nAcceptance: {json.dumps(task.get('acceptance', []))}\n"
                                f"Backend status: {result.status.value}\nChanged paths: {json.dumps(result.changed_paths)}\n"
                                f"Test evidence: {json.dumps(result.tests, ensure_ascii=False)}\n"
                                f"Static checks:\n{static_result[:6000]}\nDiff:\n{diff[-24000:]}"
                            ),
                        },
                        {
                            "role": "user",
                            "content": "Independently accept or reject this candidate implementation.",
                        },
                    ],
                    options={"temperature": 0.0, "num_predict": 1024},
                )
                try:
                    approved, feedback, review_summary = self._implementation_review_payload(
                        self._content(review_response)
                    )
                except ValueError as exc:
                    approved, feedback, review_summary = (
                        False,
                        str(exc),
                        "malformed LocalPilot review",
                    )
                if result.status != ImplementationStatus.COMPLETED:
                    approved = False
                    feedback = (
                        f"Backend reported {result.status.value}: {result.summary}. "
                        + feedback
                    )[:12000]
                if not static_result.startswith("static_checks=passed"):
                    approved = False
                    feedback = (
                        f"LocalPilot static checks failed:\n{static_result}\n" + feedback
                    )[:12000]
                if approved:
                    review_status = "approved"
                    break
                if repair_count >= max_repairs:
                    review_status = "rejected"
                    result = ImplementationResult(
                        ImplementationStatus.REVIEW_REJECTED,
                        result.backend,
                        result.model,
                        review_summary
                        or feedback
                        or "LocalPilot rejected the candidate implementation.",
                        changed_paths=result.changed_paths,
                        diff_digest=result.diff_digest,
                        tests=result.tests,
                        session_id=result.session_id,
                        exit_code=result.exit_code,
                        usage=result.usage,
                        duration_seconds=result.duration_seconds,
                        repair_pass=repair_count,
                        sanitized_output=result.sanitized_output,
                    )
                    self._clear_claude_repair_progress(
                        branch=branch,
                        cycle_id=cycle_id,
                        stage=stage,
                        reason="independent review exhausted the configured repair limit",
                    )
                    break

                next_repair = repair_count + 1
                self._save_claude_repair_progress(
                    branch=branch,
                    cycle_id=cycle_id,
                    task_id=str(task.get("id") or ""),
                    stage=stage,
                    repair_pass=next_repair,
                    review_feedback=feedback,
                    review_summary=review_summary,
                )
                self._claude_unit_admitted = False
                self._admit_claude_unit(force=force, branch=branch, stage=stage)
                repair_count = next_repair
                self._emit(
                    f"LocalPilot review requested Claude Code repair pass {repair_count}/{max_repairs}"
                )
                result = backend.run(
                    replace(
                        request,
                        review_feedback=feedback or review_summary,
                    ),
                    repair_pass=repair_count,
                )
                if result.status in terminal_without_review:
                    review_status = "repair_failed"
                    break

            for relative in result.changed_paths:
                path = tools.validate_project_write(
                    relative,
                    (workspace / relative).read_text(encoding="utf-8"),
                )
                tools.files_written.add(path)
                tools.write_count += 1
            self._record_backend_evidence(
                cycle_id=cycle_id,
                task=task,
                result=result,
                review_passes=repair_count,
                review_status=review_status,
            )
            if result.status != ImplementationStatus.COMPLETED or review_status != "approved":
                raise RuntimeError(
                    f"implementation backend {result.status.value}: {result.summary}"
                )
            self._clear_claude_repair_progress(
                branch=branch,
                cycle_id=cycle_id,
                stage=stage,
                reason="independent review approved the Claude implementation",
            )
            return json.dumps(
                {
                    "summary": result.summary,
                    "reusable_lesson": (
                        "Use Claude Code inside the candidate boundary and retain "
                        "independent LocalPilot acceptance review."
                    ),
                    "evaluation_evidence": {
                        "metric": task["evaluation"]["metric"],
                        "baseline_evidence": task["evaluation"]["baseline"],
                        "candidate_evidence": (
                            "Claude Code tests passed; "
                            f"diff sha256={result.diff_digest}; LocalPilot review approved."
                        ),
                        "result": "pending_ci",
                        "measurement_artifact": task["evaluation"]["measurement_method"],
                    },
                }
            )
        finally:
            self._claude_unit_admitted = False

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
        previous_stage = getattr(self, "_active_claude_repair_stage", None)
        self._active_claude_repair_stage = stage
        try:
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
        except RuntimeError as exc:
            detail = str(exc)
            normalized = detail.lower()
            transient = (
                "implementation backend unavailable:",
                "implementation backend backend_unavailable:",
                "implementation backend timeout:",
                "implementation backend resource_pressure:",
            )
            if any(marker in normalized for marker in transient):
                self.audit.write(
                    "selfdev_claude_repair_deferred",
                    branch=branch,
                    cycle_id=int(cycle_id),
                    task_id=str(task.get("id") or ""),
                    stage=stage,
                    reason=detail[:2000],
                    candidate_preserved=True,
                )
                raise CyclePaused(
                    f"Claude Code {stage} temporarily unavailable; candidate preserved: {detail}"
                ) from exc
            raise
        finally:
            if previous_stage is None:
                try:
                    delattr(self, "_active_claude_repair_stage")
                except AttributeError:
                    pass
            else:
                self._active_claude_repair_stage = previous_stage

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
