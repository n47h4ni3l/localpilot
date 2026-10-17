"""Optional Claude Code delegation inside Nestra's native candidate loop."""
from __future__ import annotations

from typing import Any, Callable

from localpilot.implementation_backend import candidate_changed_paths
from localpilot.selfdev_results import _ALLOWED_SUFFIXES


def candidate_claude_code_tools(
    developer: Any,
    *,
    chat: Callable[..., Any],
    model: str,
    tools: Any,
    task: dict[str, Any],
    branch: str,
    cycle_id: int,
    force: bool,
    grounding_plan: dict[str, list[Any]] | None = None,
) -> list[Callable[..., str]]:
    """Offer delegation only after grounding, or for existing repair-owned paths.

    Discovery and research never receive this capability. The ordinary Claude
    implementation/reviewer loop owns permission, resource, static and test
    checks; this adapter gives the native planner the choice to invoke it.
    """
    if developer.config.selfdev.implementation_backend != "local_tools":
        return []
    if grounding_plan is None:
        repair_scope = getattr(developer, "_claude_repair_scope", None)
        try:
            scope = (
                repair_scope(tools.workspace, tools) if repair_scope is not None
                else tuple(path.relative_to(tools.workspace).as_posix() for path in tools.files_written)
            )
        except (OSError, ValueError, RuntimeError):
            # Failing to establish an optional delegation scope must neither
            # grant broader permissions nor disable the existing native tools.
            developer.audit.write("selfdev_claude_tool_unavailable", branch=branch,
                                  reason="candidate repair scope could not be verified")
            return []
    else:
        scope = tuple(
            str(item) for field in ("referenced_paths", "new_runtime_paths")
            for item in grounding_plan.get(field, [])
        )
    grounded: set[str] = set()
    for item in scope:
        try:
            resolved = tools._resolve(str(item))
        except (ValueError, PermissionError):
            # A grounded read reference can be outside the editable policy.
            continue
        relative = resolved.relative_to(tools.workspace).as_posix()
        suffix = resolved.suffix.lower() if resolved.name != ".gitignore" else ".gitignore"
        if relative not in tools.protected_paths and suffix in _ALLOWED_SUFFIXES:
            grounded.add(relative)
    if not grounded:
        return []

    def claude_code(task: str, allowed_paths: list[str] | None = None) -> str:
        """Optionally delegate a focused coding task to local Claude Code using the same Nestra model.

        Only exact grounded candidate paths may be selected; omit allowed_paths
        to use the verified plan. The original experiment contract, independent
        Nestra review, resource checks and CI still apply. Never edits stable or
        merges. This tool may run the existing bounded repository tests.
        """
        from localpilot.tools.claude_code import claude_code_run_lock

        if not isinstance(task, str) or not task.strip() or len(task) > 6000:
            raise ValueError("Claude Code task must contain 1 to 6000 characters")
        cfg = developer.config
        if cfg.model.provider != "ollama" or not (model == cfg.model.name == cfg.selfdev.developer_model == cfg.selfdev.implementation_model):
            raise ValueError("Claude Code tool requires the same accepted Nestra model for every role")
        if allowed_paths is not None and (
            not isinstance(allowed_paths, list)
            or not all(isinstance(item, str) for item in allowed_paths)
        ):
            raise ValueError("allowed_paths must be a list of exact grounded file paths")
        selected = grounded if allowed_paths is None else set(allowed_paths)
        if not selected or not selected <= grounded:
            raise PermissionError("Claude Code tool paths must be a nonempty subset of the verified candidate scope")
        for item in selected:
            if tools._resolve(item).relative_to(tools.workspace).as_posix() != item:
                raise PermissionError("Claude Code tool requires canonical candidate-relative paths")
        # Backend validation examines the complete Git diff. Do not silently
        # widen permissions to accommodate edits from an earlier native call.
        before = set(candidate_changed_paths(tools.workspace))
        if not before <= selected:
            raise PermissionError("Existing candidate changes fall outside the selected Claude Code scope")
        metadata_before = {
            item: (path.stat().st_mtime_ns, path.stat().st_size) if path.is_file() else None
            for item in selected for path in [tools._resolve(item)]
        }
        writes_before = tools.write_count
        developer._check_resources(force, branch)
        developer.audit.write(
            "selfdev_claude_tool_invoked", branch=branch, cycle_id=cycle_id,
            model=model, allowed_paths=sorted(selected), task_chars=len(task),
        )
        with claude_code_run_lock(developer.data_dir):
            try:
                return developer._run_claude_code_implementation(
                    chat=chat, developer_model=model, task=task_contract,
                    branch=branch, workspace=tools.workspace, tools=tools,
                    cycle_id=cycle_id,
                    research="Nestra requested this optional tool task:\n" + task,
                    grounding_plan={"referenced_paths": sorted(selected), "new_runtime_paths": []},
                    grounding_evidence=["Optional tool scope is a subset of the verified candidate plan or repair diff."],
                    evolution_context=developer._evolution_context(task_contract),
                    lessons=developer.memory.reusable_lessons(cfg.selfdev.lesson_limit), force=force,
                )
            finally:
                # A stopped CLI may leave useful edits. Register valid partial
                # work and retain invalid-write evidence so fallback cannot hide
                # a confinement failure or deliver an unvalidated file.
                reserved = set(tools.files_written)
                backend_registered_writes = tools.write_count != writes_before
                for relative in candidate_changed_paths(tools.workspace):
                    try:
                        path = tools._resolve(relative)
                        if relative not in selected:
                            raise PermissionError("Claude Code changed a path outside its selected scope")
                        content = path.read_text(encoding="utf-8", errors="strict")
                        validated = tools.validate_project_write(relative, content, reserved_paths=reserved)
                    except Exception as exc:
                        tools._record_failed_write(relative, exc)
                        raise
                    reserved.add(validated)
                    if validated not in tools.files_written:
                        tools.files_written.add(validated)
                        tools.write_count += 1
                    elif not backend_registered_writes and metadata_before[relative] != (
                        validated.stat().st_mtime_ns, validated.stat().st_size
                    ):
                        tools.write_count += 1

    task_contract = task
    return [claude_code]
