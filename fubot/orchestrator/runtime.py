"""Coordinator runtime that dispatches routed executor work."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from loguru import logger

from fubot.config.schema import AgentProfile, Config
from fubot.orchestrator.models import (
    AssignmentRecord,
    ExecutionLogRecord,
    RouteDecision,
    TaskRecord,
    WorkflowRecord,
)
from fubot.orchestrator.router import ProviderExecutionError, RoutePlanner, classify_provider_error
from fubot.orchestrator.store import WorkflowStore


@dataclass
class ExecutorResult:
    """Result returned by an executor runtime."""

    profile: AgentProfile
    task: TaskRecord
    content: str
    route: dict[str, Any]
    all_messages: list[dict[str, Any]] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    sent_direct_message: bool = False


class CoordinatorRuntime:
    """Plans tasks, dispatches executors, and persists workflow state."""

    def __init__(self, config: Config, store: WorkflowStore):
        self.config = config
        self.store = store
        self.router = RoutePlanner(config, self.store.load_health())

    async def run(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        content: str,
        media: list[str] | None,
        execute_task: Callable[
            [AgentProfile, TaskRecord, RouteDecision, str, str, str, str, list[str] | None, dict[str, Any], Callable[[str, bool], Awaitable[None]] | None],
            Awaitable[ExecutorResult],
        ],
        emit_update: Callable[[ExecutionLogRecord], Awaitable[None]] | None = None,
        cleanup_task_lineage: Callable[[str, str], None] | None = None,
    ) -> tuple[WorkflowRecord, list[TaskRecord], list[AssignmentRecord], list[ExecutorResult], str]:
        workflow = WorkflowRecord(
            id=f"wf_{uuid.uuid4().hex[:10]}",
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            user_message=content,
            shared_board={"user_message": content, "media": media or []},
        )
        task_type = self.router.classify(content, media)
        profiles = self.router.choose_executors(task_type, content)
        tasks: list[TaskRecord] = []
        assignments: list[AssignmentRecord] = []

        for profile in profiles:
            task = TaskRecord(
                id=f"task_{uuid.uuid4().hex[:8]}",
                workflow_id=workflow.id,
                title=f"{profile.name or profile.id} {task_type}",
                kind=task_type,
                status="queued",
                metadata={"input": content, "media": media or []},
            )
            assignment = AssignmentRecord(
                id=f"assign_{uuid.uuid4().hex[:8]}",
                workflow_id=workflow.id,
                task_id=task.id,
                agent_id=profile.id,
                agent_name=profile.name or profile.id,
                agent_role=profile.role,
            )
            workflow.task_ids.append(task.id)
            workflow.assignment_ids.append(assignment.id)
            tasks.append(task)
            assignments.append(assignment)

        self.store.save_workflow(workflow, tasks, assignments)

        def _progress_factory(profile: AgentProfile, task: TaskRecord) -> Callable[[str, bool], Awaitable[None]]:
            async def _emit(message: str, tool_hint: bool = False) -> None:
                if not self.config.observability.show_executor_progress:
                    return
                kind = "tool" if tool_hint else "progress"
                record = ExecutionLogRecord(
                    workflow_id=workflow.id,
                    task_id=task.id,
                    agent_id=profile.id,
                    agent_name=profile.name or profile.id,
                    agent_role=profile.role,
                    message_kind=kind,
                    content=message,
                    is_final=False,
                )
                self.store.append_log(workflow, tasks, assignments, record)
                if emit_update:
                    await emit_update(record)

            return _emit

        async def _run_one(profile: AgentProfile, task: TaskRecord, assignment: AssignmentRecord) -> ExecutorResult:
            self.router.replace_health(self.store.load_health())
            decision = self.router.choose_model(profile, task.kind, task.metadata.get("input", ""))
            task.route = decision.to_dict()
            self.router.begin_load(profile.id)
            assignment.status = "running"
            task.status = "running"
            self.store.save_workflow(workflow, tasks, assignments)
            current_decision = decision
            attempted_candidates: list[str] = []
            try:
                while True:
                    if self.config.observability.show_route_decisions:
                        route_fields = current_decision.log_fields()
                        reason = current_decision.reason if self.config.observability.include_route_reason else "suppressed"
                        logger.debug(
                            "Route decision route_kind={} trace_id={} parent_trace_id={} previous_attempt_trace_id={} agent={} task={} provider={} model={} attempt={} fallback_chain={} reason={}",
                            route_fields["route_kind"],
                            route_fields["trace_id"],
                            route_fields["parent_trace_id"],
                            route_fields["previous_attempt_trace_id"],
                            profile.id,
                            task.id,
                            route_fields["provider"],
                            route_fields["model"],
                            route_fields["attempt_index"],
                            ",".join(current_decision.fallback_chain),
                            reason,
                        )
                    if (
                        current_decision.provider is None
                        and current_decision.fallback_chain != [current_decision.model]
                    ):
                        message = self.router.fallback_failure_message(
                            current_decision,
                            attempted_candidates=attempted_candidates,
                            stop_reason="all configured providers unavailable before execution",
                        )
                        raise ProviderExecutionError(message, error_kind="provider_unavailable")
                    try:
                        result = await execute_task(
                            profile,
                            task,
                            current_decision,
                            workflow.id,
                            session_key,
                            channel,
                            chat_id,
                            media,
                            workflow.shared_board,
                            _progress_factory(profile, task),
                        )
                        break
                    except Exception as exc:
                        error_kind = classify_provider_error(exc)
                        fallback_resolution = None
                        if error_kind and current_decision.provider:
                            self.router.mark_failure(profile.id, current_decision.provider, error_kind)
                            self.store.save_health(self.router.export_health())
                            fallback_resolution = self.router.next_fallback_route(
                                current_decision,
                                error_kind=error_kind,
                                error_detail=str(exc),
                                attempted_candidates=attempted_candidates,
                            )
                        if fallback_resolution is not None and fallback_resolution.next_route is not None:
                            logger.warning(
                                "Provider fallback agent={} task={} from={} to={} error_kind={} attempt={} stop_reason={}",
                                profile.id,
                                task.id,
                                f"{current_decision.provider or 'default'}:{current_decision.model}",
                                f"{fallback_resolution.next_route.provider or 'default'}:{fallback_resolution.next_route.model}",
                                error_kind,
                                fallback_resolution.next_route.attempt_index,
                                fallback_resolution.stop_reason,
                            )
                            attempted_candidates = fallback_resolution.attempted_candidates
                            current_decision = fallback_resolution.next_route
                            task.route = current_decision.to_dict()
                            self.store.save_workflow(workflow, tasks, assignments)
                            continue
                        task.status = "failed"
                        assignment.status = "failed"
                        self.store.save_workflow(workflow, tasks, assignments)
                        self.store.save_health(self.router.export_health())
                        if fallback_resolution is not None:
                            message = self.router.fallback_failure_message(
                                current_decision,
                                attempted_candidates=fallback_resolution.attempted_candidates,
                                stop_reason=fallback_resolution.stop_reason,
                            )
                            raise ProviderExecutionError(message, error_kind=error_kind or "provider_error") from exc
                        raise

                task.status = "completed"
                task.assigned_agent_id = profile.id
                task.assigned_agent_name = profile.name or profile.id
                task.route = result.route
                task.result = result.content
                task.summary = result.content[:400]
                assignment.status = "completed"
                self.router.mark_success(profile.id, result.route.get("provider"))
                self.store.update_shared_board(
                    workflow,
                    tasks,
                    assignments,
                    profile.id,
                    {"content": result.content, "route": result.route, "tools_used": result.tools_used},
                )
                final_log = ExecutionLogRecord(
                    workflow_id=workflow.id,
                    task_id=task.id,
                    agent_id=profile.id,
                    agent_name=profile.name or profile.id,
                    agent_role=profile.role,
                    message_kind="result",
                    content=result.content,
                    is_final=False,
                )
                self.store.append_log(workflow, tasks, assignments, final_log)
                if emit_update:
                    await emit_update(final_log)
                return result
            except asyncio.CancelledError:
                task.status = "failed"
                assignment.status = "failed"
                self.store.save_workflow(workflow, tasks, assignments)
                raise
            finally:
                self.router.end_load(profile.id)
                self.store.save_health(self.router.export_health())
                if cleanup_task_lineage is not None:
                    cleanup_task_lineage(workflow.id, task.id)

        results = await asyncio.gather(
            *[_run_one(profile, task, assignment) for profile, task, assignment in zip(profiles, tasks, assignments)],
        )
        final_response = self._synthesize_final(results)
        workflow.status = "completed"
        workflow.final_response = final_response
        self.store.save_workflow(workflow, tasks, assignments)
        self.store.save_health(self.router.export_health())
        return workflow, tasks, assignments, results, final_response

    def _synthesize_final(self, results: list[ExecutorResult]) -> str:
        if not results:
            return "No executor produced a result."
        if len(results) == 1:
            return results[0].content
        lines = []
        for result in results:
            header = result.profile.name or result.profile.id
            lines.append(f"[{header}] {result.content}")
        return "\n\n".join(lines)
