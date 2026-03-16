"""Subagent manager for background task execution."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from fubot.agent.tools.base import ToolExecutionLineage, build_tool_idempotency_key
from fubot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from fubot.agent.tools.registry import ToolRegistry
from fubot.agent.tools.shell import ExecTool
from fubot.agent.tools.web import WebFetchTool, WebSearchTool
from fubot.bus.events import InboundMessage
from fubot.bus.queue import MessageBus
from fubot.config.schema import ExecToolConfig
from fubot.orchestrator.models import RouteDecision
from fubot.orchestrator.router import ProviderExecutionError, RoutePlanner, classify_provider_error
from fubot.orchestrator.store import WorkflowStore
from fubot.providers.base import LLMProvider
from fubot.providers.factory import build_provider_for_route
from fubot.utils.helpers import build_assistant_message

if TYPE_CHECKING:
    from fubot.agent.tools.base import ToolExecutionContext
    from fubot.config.schema import Config, WebSearchConfig


class SubagentManager:
    """Manages background subagent execution."""

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        web_search_config: "WebSearchConfig | None" = None,
        web_proxy: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
        config: Config | None = None,
        default_provider_name: str | None = None,
        allow_legacy_route_fallback: bool = True,
        workflow_store: WorkflowStore | None = None,
    ):
        from fubot.config.schema import ExecToolConfig, WebSearchConfig

        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.config = config
        self.default_provider_name = (
            default_provider_name
            if default_provider_name is not None
            else (config.resolve_provider(model=self.model).provider_name if config is not None else None)
        )
        self.allow_legacy_route_fallback = allow_legacy_route_fallback
        self.web_search_config = web_search_config or WebSearchConfig()
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self.workflow_store = workflow_store
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._session_tasks: dict[str, set[str]] = {}  # session_key -> {task_id, ...}
        self._tool_execution_scopes: dict[str, dict[str, Any]] = {}

    def _legacy_route_decision(self) -> RouteDecision:
        """Build the only compatibility route that may use the manager default provider."""
        if not self.allow_legacy_route_fallback:
            raise RuntimeError(
                "Subagent spawn requires an explicit parent RouteDecision; refusing to fall back to the "
                "SubagentManager initialization provider.",
            )
        fallback_chain = [f"{self.default_provider_name}:{self.model}"] if self.default_provider_name else [self.model]
        return RouteDecision(
            agent_id="subagent",
            agent_name="Subagent",
            agent_role="subagent",
            task_type="subagent",
            model=self.model,
            provider=self.default_provider_name,
            reason="legacy single-provider fallback to subagent manager initialization route",
            fallback_chain=fallback_chain,
        )

    def _resolve_child_route_decision(
        self,
        *,
        parent_route_decision: RouteDecision | None,
        child_route_decision: RouteDecision | None,
        allow_child_reroute: bool,
    ) -> RouteDecision:
        """Resolve the explicit child route for one spawned subagent."""
        if child_route_decision is not None:
            if not allow_child_reroute:
                raise RuntimeError(
                    "Child route decision supplied without allow_child_reroute=True.",
                )
            if (
                parent_route_decision is not None
                and child_route_decision.parent_trace_id
                and child_route_decision.parent_trace_id != parent_route_decision.trace_id
            ):
                raise RuntimeError(
                    "Child route decision parent_trace_id does not match the supplied parent route decision.",
                )
            if parent_route_decision is not None and child_route_decision.parent_trace_id is None:
                return parent_route_decision.derive_child(
                    provider=child_route_decision.provider,
                    model=child_route_decision.model,
                    trace_id=child_route_decision.trace_id,
                    fallback_chain=child_route_decision.fallback_chain,
                    attempt_index=child_route_decision.attempt_index,
                    inherited_from_parent=child_route_decision.inherited_from_parent,
                    reason=child_route_decision.reason,
                )
            return child_route_decision
        if parent_route_decision is not None:
            return parent_route_decision.derive_child(
                reason=f"inherited from parent route {parent_route_decision.trace_id}",
            )
        return self._legacy_route_decision()

    def _provider_for_route(self, route_decision: RouteDecision) -> LLMProvider:
        """Resolve the concrete provider instance for a child route decision."""
        if (
            self.allow_legacy_route_fallback
            and route_decision.parent_trace_id is None
            and route_decision.agent_id == "subagent"
            and route_decision.reason == "legacy single-provider fallback to subagent manager initialization route"
            and route_decision.model == self.model
            and route_decision.provider == self.default_provider_name
        ):
            return self.provider
        if self.config is not None:
            return build_provider_for_route(
                self.config,
                route_decision,
                default_provider=self.provider,
                default_model=self.model,
                allow_default_provider=False,
            )
        if route_decision.provider is None and route_decision.model == self.model:
            return self.provider
        raise RuntimeError(
            f"Subagent route {route_decision.trace_id} requires runtime config to build provider "
            f"{route_decision.provider or 'default'}:{route_decision.model}.",
        )

    def _get_tool_execution_scope(
        self,
        task_id: str,
        *,
        route_decision: RouteDecision,
        parent_execution_context: "ToolExecutionContext | None" = None,
    ) -> dict[str, Any]:
        """Keep child replay protection scoped to one subagent task across its fallback attempts."""
        scope = self._tool_execution_scopes.get(task_id)
        if scope is None:
            scope = {
                "lineage": ToolExecutionLineage.create_root(
                    route_trace_id=route_decision.trace_id,
                    attempt_index=route_decision.attempt_index,
                    parent_execution_id=(
                        parent_execution_context.execution_id if parent_execution_context is not None else None
                    ),
                ),
                "replay_cache": {},
            }
            self._tool_execution_scopes[task_id] = scope
        return scope

    def _cleanup_tool_execution_scope(self, task_id: str) -> None:
        self._tool_execution_scopes.pop(task_id, None)

    @staticmethod
    def _make_tool_context_builder(
        lineage: ToolExecutionLineage,
    ) -> Callable[[str, dict[str, Any], int], "ToolExecutionContext"]:
        def _build(tool_name: str, params: dict[str, Any], occurrence_index: int) -> "ToolExecutionContext":
            return lineage.derive_execution(
                idempotency_key=build_tool_idempotency_key(
                    lineage.trace_id,
                    tool_name,
                    occurrence_index,
                    params,
                )
            )

        return _build

    @staticmethod
    def _tool_call_occurrence(
        tool_occurrences: dict[str, int],
        *,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> int:
        fingerprint = json.dumps(
            {"name": tool_name, "arguments": arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        occurrence_index = tool_occurrences.get(fingerprint, 0)
        tool_occurrences[fingerprint] = occurrence_index + 1
        return occurrence_index

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
        parent_route_decision: RouteDecision | None = None,
        parent_execution_context: "ToolExecutionContext | None" = None,
        child_route_decision: RouteDecision | None = None,
        allow_child_reroute: bool = False,
    ) -> str:
        """Spawn a subagent to execute a task in the background."""
        route_decision = self._resolve_child_route_decision(
            parent_route_decision=parent_route_decision,
            child_route_decision=child_route_decision,
            allow_child_reroute=allow_child_reroute,
        )
        routed_provider = self._provider_for_route(route_decision)
        task_id = str(uuid.uuid4())[:8]
        tool_scope = self._get_tool_execution_scope(
            task_id,
            route_decision=route_decision,
            parent_execution_context=parent_execution_context,
        )
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        origin = {"channel": origin_channel, "chat_id": origin_chat_id}

        bg_task = asyncio.create_task(
            self._run_subagent(
                task_id,
                task,
                display_label,
                origin,
                route_decision=route_decision,
                provider=routed_provider,
                tool_lineage=tool_scope["lineage"],
                replay_cache=tool_scope["replay_cache"],
            )
        )
        self._running_tasks[task_id] = bg_task
        if session_key:
            self._session_tasks.setdefault(session_key, set()).add(task_id)

        def _cleanup(_: asyncio.Task) -> None:
            self._running_tasks.pop(task_id, None)
            self._cleanup_tool_execution_scope(task_id)
            if session_key and (ids := self._session_tasks.get(session_key)):
                ids.discard(task_id)
                if not ids:
                    del self._session_tasks[session_key]

        bg_task.add_done_callback(_cleanup)

        route_fields = route_decision.log_fields()
        lineage_fields = tool_scope["lineage"].log_fields()
        logger.info(
            "Spawned subagent [{}]: {} route_kind={} trace_id={} parent_trace_id={} previous_attempt_trace_id={} provider={} model={} attempt={} lineage_trace_id={} lineage_root_execution_id={} lineage_parent_execution_id={}",
            task_id,
            display_label,
            route_fields["route_kind"],
            route_fields["trace_id"],
            route_fields["parent_trace_id"],
            route_fields["previous_attempt_trace_id"],
            route_fields["provider"],
            route_fields["model"],
            route_fields["attempt_index"],
            lineage_fields["trace_id"],
            lineage_fields["root_execution_id"],
            lineage_fields["parent_execution_id"],
        )
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
        route_decision: RouteDecision | None = None,
        provider: LLMProvider | None = None,
        tool_lineage: ToolExecutionLineage | None = None,
        replay_cache: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Execute the subagent task and announce the result."""
        child_route = route_decision or self._legacy_route_decision()
        scope = {
            "lineage": tool_lineage or ToolExecutionLineage.create_root(
                route_trace_id=child_route.trace_id,
                attempt_index=child_route.attempt_index,
            ),
            "replay_cache": replay_cache if replay_cache is not None else {},
        }
        planner = RoutePlanner(
            self.config,
            self.workflow_store.load_health() if self.workflow_store is not None else {},
        ) if self.config is not None else None
        current_route = child_route
        current_provider = provider
        attempted_candidates: list[str] = []
        route_fields = current_route.log_fields()
        lineage_fields = scope["lineage"].log_fields()
        logger.info(
            "Subagent [{}] starting task: {} route_kind={} trace_id={} parent_trace_id={} previous_attempt_trace_id={} provider={} model={} attempt={} lineage_trace_id={} lineage_root_execution_id={} lineage_parent_execution_id={}",
            task_id,
            label,
            route_fields["route_kind"],
            route_fields["trace_id"],
            route_fields["parent_trace_id"],
            route_fields["previous_attempt_trace_id"],
            route_fields["provider"],
            route_fields["model"],
            route_fields["attempt_index"],
            lineage_fields["trace_id"],
            lineage_fields["root_execution_id"],
            lineage_fields["parent_execution_id"],
        )

        try:
            while True:
                if planner is not None and self.workflow_store is not None:
                    planner.replace_health(self.workflow_store.load_health())
                if current_route.provider is None and current_route.fallback_chain != [current_route.model]:
                    stop_reason = (
                        planner.fallback_failure_message(
                            current_route,
                            attempted_candidates=attempted_candidates,
                            stop_reason="all configured providers unavailable before child execution",
                        )
                        if planner is not None
                        else "all configured providers unavailable before child execution"
                    )
                    raise ProviderExecutionError(stop_reason, error_kind="provider_unavailable")

                active_lineage = scope["lineage"].for_route(
                    route_trace_id=current_route.trace_id,
                    attempt_index=current_route.attempt_index,
                )
                tool_context_builder = self._make_tool_context_builder(active_lineage)

                # Build subagent tools (no message tool, no spawn tool)
                tools = ToolRegistry(replay_cache=scope["replay_cache"])
                allowed_dir = self.workspace if self.restrict_to_workspace else None
                tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
                tools.register(WriteFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
                tools.register(EditFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
                tools.register(ListDirTool(workspace=self.workspace, allowed_dir=allowed_dir))
                tools.register(ExecTool(
                    working_dir=str(self.workspace),
                    timeout=self.exec_config.timeout,
                    restrict_to_workspace=self.restrict_to_workspace,
                    path_append=self.exec_config.path_append,
                ))
                tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
                tools.register(WebFetchTool(proxy=self.web_proxy))

                system_prompt = self._build_subagent_prompt()
                messages: list[dict[str, Any]] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task},
                ]

                max_iterations = 15
                iteration = 0
                final_result: str | None = None
                tool_occurrences: dict[str, int] = {}
                current_provider = current_provider or self._provider_for_route(current_route)

                while iteration < max_iterations:
                    iteration += 1

                    response = await current_provider.chat_with_retry(
                        messages=messages,
                        tools=tools.get_definitions(),
                        model=current_route.model,
                    )

                    if response.has_tool_calls:
                        tool_call_dicts = [
                            tc.to_openai_tool_call()
                            for tc in response.tool_calls
                        ]
                        messages.append(build_assistant_message(
                            response.content or "",
                            tool_calls=tool_call_dicts,
                            reasoning_content=response.reasoning_content,
                            thinking_blocks=response.thinking_blocks,
                        ))

                        for tool_call in response.tool_calls:
                            occurrence_index = self._tool_call_occurrence(
                                tool_occurrences,
                                tool_name=tool_call.name,
                                arguments=tool_call.arguments,
                            )
                            tool_context = tool_context_builder(
                                tool_call.name,
                                tool_call.arguments,
                                occurrence_index,
                            )
                            args_preview = json.dumps(tool_call.arguments, ensure_ascii=False)[:200]
                            route_fields = current_route.log_fields()
                            tool_fields = tool_context.log_fields()
                            logger.debug(
                                "Subagent [{}] tool route_kind={} trace_id={} parent_trace_id={} previous_attempt_trace_id={} provider={} model={} attempt={} execution_id={} route_trace_id={} parent_execution_id={} tool={} args={}",
                                task_id,
                                route_fields["route_kind"],
                                route_fields["trace_id"],
                                route_fields["parent_trace_id"],
                                route_fields["previous_attempt_trace_id"],
                                route_fields["provider"],
                                route_fields["model"],
                                route_fields["attempt_index"],
                                tool_fields["execution_id"],
                                tool_fields["route_trace_id"],
                                tool_fields["parent_execution_id"],
                                tool_call.name,
                                args_preview,
                            )
                            result = await tools.execute(
                                tool_call.name,
                                tool_call.arguments,
                                execution_context=tool_context,
                            )
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.name,
                                "content": result,
                            })
                    else:
                        if response.finish_reason == "error":
                            raise ProviderExecutionError(
                                response.content or "Subagent provider execution failed.",
                                error_kind=classify_provider_error(response.content or "") or "provider_error",
                            )
                        final_result = response.content
                        break

                if final_result is None:
                    final_result = "Task completed but no final response was generated."

                route_fields = current_route.log_fields()
                if planner is not None and current_route.provider is not None:
                    planner.mark_success("subagent", current_route.provider)
                    if self.workflow_store is not None:
                        self.workflow_store.save_health(planner.export_health())
                logger.info(
                    "Subagent [{}] completed successfully route_kind={} trace_id={} parent_trace_id={} previous_attempt_trace_id={} provider={} model={} attempt={}",
                    task_id,
                    route_fields["route_kind"],
                    route_fields["trace_id"],
                    route_fields["parent_trace_id"],
                    route_fields["previous_attempt_trace_id"],
                    route_fields["provider"],
                    route_fields["model"],
                    route_fields["attempt_index"],
                )
                await self._announce_result(
                    task_id,
                    label,
                    task,
                    final_result,
                    origin,
                    "ok",
                    route_decision=current_route,
                    tool_execution_lineage=active_lineage.derive_child(
                        route_trace_id=current_route.trace_id,
                        attempt_index=current_route.attempt_index,
                    ),
                )
                return

        except Exception as e:
            error_kind = classify_provider_error(e)
            if error_kind and current_route.provider and planner is not None:
                planner.mark_failure("subagent", current_route.provider, error_kind)
                if self.workflow_store is not None:
                    self.workflow_store.save_health(planner.export_health())
                resolution = planner.next_fallback_route(
                    current_route,
                    error_kind=error_kind,
                    error_detail=str(e),
                    attempted_candidates=attempted_candidates,
                )
                if resolution.next_route is not None:
                    logger.warning(
                        "Subagent [{}] provider fallback from={} to={} error_kind={} attempt={} stop_reason={}",
                        task_id,
                        f"{current_route.provider or 'default'}:{current_route.model}",
                        f"{resolution.next_route.provider or 'default'}:{resolution.next_route.model}",
                        error_kind,
                        resolution.next_route.attempt_index,
                        resolution.stop_reason,
                    )
                    attempted_candidates = resolution.attempted_candidates
                    current_route = resolution.next_route
                    current_provider = None
                    await self._run_subagent(
                        task_id,
                        task,
                        label,
                        origin,
                        route_decision=current_route,
                        provider=current_provider,
                        tool_lineage=scope["lineage"],
                        replay_cache=scope["replay_cache"],
                    )
                    return
                error_msg = f"Error: {planner.fallback_failure_message(current_route, attempted_candidates=resolution.attempted_candidates, stop_reason=resolution.stop_reason)}"
            else:
                error_msg = f"Error: {str(e)}"
            route_fields = current_route.log_fields()
            logger.error(
                "Subagent [{}] failed route_kind={} trace_id={} parent_trace_id={} previous_attempt_trace_id={} provider={} model={} attempt={}: {}",
                task_id,
                route_fields["route_kind"],
                route_fields["trace_id"],
                route_fields["parent_trace_id"],
                route_fields["previous_attempt_trace_id"],
                route_fields["provider"],
                route_fields["model"],
                route_fields["attempt_index"],
                e,
            )
            await self._announce_result(
                task_id,
                label,
                task,
                error_msg,
                origin,
                "error",
                route_decision=current_route,
                tool_execution_lineage=scope["lineage"].derive_child(
                    route_trace_id=current_route.trace_id,
                    attempt_index=current_route.attempt_index,
                ),
            )

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
        route_decision: RouteDecision | None = None,
        tool_execution_lineage: ToolExecutionLineage | None = None,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        status_text = "completed successfully" if status == "ok" else "failed"

        announce_content = f"""[Subagent '{label}' {status_text}]

Task: {task}

Result:
{result}

Summarize this naturally for the user. Keep it brief (1-2 sentences). Do not mention technical details like "subagent" or task IDs."""

        # Inject as system message to trigger the derived main-agent execution chain.
        # This carries route + tool lineage metadata, but replay protection remains
        # in-process only and does not persist across restarts.
        metadata: dict[str, Any] = {}
        if route_decision is not None:
            metadata["route_decision"] = route_decision.to_dict()
        if tool_execution_lineage is not None:
            metadata["tool_execution_lineage"] = tool_execution_lineage.to_dict()
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
            metadata=metadata,
        )

        await self.bus.publish_inbound(msg)
        route_fields = route_decision.log_fields() if route_decision is not None else {
            "route_kind": "unknown",
            "trace_id": "-",
            "parent_trace_id": "-",
            "previous_attempt_trace_id": "-",
            "provider": "default",
            "model": self.model,
            "attempt_index": 0,
        }
        lineage_fields = (
            tool_execution_lineage.log_fields()
            if tool_execution_lineage is not None
            else {
                "trace_id": "-",
                "root_execution_id": "-",
                "parent_execution_id": "-",
                "attempt_index": 0,
                "route_trace_id": route_fields["trace_id"],
            }
        )
        logger.debug(
            "Subagent [{}] announced result route_kind={} trace_id={} parent_trace_id={} previous_attempt_trace_id={} provider={} model={} attempt={} lineage_trace_id={} lineage_root_execution_id={} lineage_parent_execution_id={} to {}:{}",
            task_id,
            route_fields["route_kind"],
            route_fields["trace_id"],
            route_fields["parent_trace_id"],
            route_fields["previous_attempt_trace_id"],
            route_fields["provider"],
            route_fields["model"],
            route_fields["attempt_index"],
            lineage_fields["trace_id"],
            lineage_fields["root_execution_id"],
            lineage_fields["parent_execution_id"],
            origin["channel"],
            origin["chat_id"],
        )

    def _build_subagent_prompt(self) -> str:
        """Build a focused system prompt for the subagent."""
        from fubot.agent.context import ContextBuilder
        from fubot.agent.skills import SkillsLoader

        time_ctx = ContextBuilder._build_runtime_context(None, None)
        parts = [f"""# Subagent

{time_ctx}

You are a subagent spawned by the main agent to complete a specific task.
Stay focused on the assigned task. Your final response will be reported back to the main agent.

## Workspace
{self.workspace}"""]

        skills_summary = SkillsLoader(self.workspace).build_skills_summary()
        if skills_summary:
            parts.append(f"## Skills\n\nRead SKILL.md with read_file to use a skill.\n\n{skills_summary}")

        return "\n\n".join(parts)

    async def cancel_by_session(self, session_key: str) -> int:
        """Cancel all subagents for the given session. Returns count cancelled."""
        tasks = [self._running_tasks[tid] for tid in self._session_tasks.get(session_key, [])
                 if tid in self._running_tasks and not self._running_tasks[tid].done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)
