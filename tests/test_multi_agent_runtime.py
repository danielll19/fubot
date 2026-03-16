from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from fubot.agent.loop import AgentLoop
from fubot.agent.tools.filesystem import ReadFileTool, WriteFileTool
from fubot.bus.events import InboundMessage
from fubot.bus.queue import MessageBus
from fubot.config.schema import AgentProfile, Config
from fubot.orchestrator.models import ExecutionLogRecord, RouteDecision, TaskRecord, WorkflowRecord
from fubot.orchestrator.router import ProviderExecutionError
from fubot.orchestrator.runtime import CoordinatorRuntime, ExecutorResult
from fubot.orchestrator.store import WorkflowStore
from fubot.providers.base import LLMResponse, ToolCallRequest


def _config() -> Config:
    config = Config()
    config.orchestration.routing.max_parallel_executors = 2
    config.observability.show_executor_progress = True
    return config


@pytest.mark.asyncio
async def test_coordinator_dispatches_executor_and_persists_shared_board(tmp_path: Path) -> None:
    config = _config()
    store = WorkflowStore(tmp_path)
    runtime = CoordinatorRuntime(config, store)
    seen_profiles: list[str] = []

    async def _execute_task(
        profile,
        task,
        route_decision,
        workflow_id,
        session_key,
        channel,
        chat_id,
        media,
        shared_board,
        on_progress,
    ) -> ExecutorResult:
        seen_profiles.append(profile.id)
        assert route_decision.provider is None or isinstance(route_decision.trace_id, str)
        await on_progress("working", False)
        return ExecutorResult(
            profile=profile,
            task=task,
            content=f"{profile.id}:{task.kind}",
            route={"model": "test-model", "provider": "dashscope"},
        )

    workflow, tasks, assignments, results, final = await runtime.run(
        session_key="cli:test",
        channel="cli",
        chat_id="test",
        content="say hello",
        media=None,
        execute_task=_execute_task,
    )

    assert seen_profiles == ["generalist"]
    assert workflow.status == "completed"
    assert len(tasks) == 1
    assert len(assignments) == 1
    assert len(results) == 1
    assert final == "generalist:communication"

    saved = store.load_workflow(workflow.id)
    assert saved is not None
    workflow_payload = saved["workflow"]
    assert workflow_payload["shared_board"]["generalist"]["content"] == "generalist:communication"
    assert workflow_payload["execution_logs"][0]["agent_name"] == "Generalist"


@pytest.mark.asyncio
async def test_coordinator_runs_multiple_executors_concurrently_for_coding(tmp_path: Path) -> None:
    config = _config()
    store = WorkflowStore(tmp_path)
    runtime = CoordinatorRuntime(config, store)
    active = 0
    max_active = 0

    async def _execute_task(
        profile,
        task,
        route_decision,
        workflow_id,
        session_key,
        channel,
        chat_id,
        media,
        shared_board,
        on_progress,
    ) -> ExecutorResult:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await on_progress(f"{profile.id} start", False)
        await asyncio.sleep(0.05)
        active -= 1
        return ExecutorResult(
            profile=profile,
            task=task,
            content=f"{profile.id} done",
            route={"model": "test-model", "provider": "dashscope"},
        )

    workflow, tasks, assignments, results, final = await runtime.run(
        session_key="cli:test",
        channel="cli",
        chat_id="test",
        content="implement a new feature and refactor the module for production use",
        media=None,
        execute_task=_execute_task,
    )

    assert workflow.status == "completed"
    assert len(tasks) == 2
    assert len(assignments) == 2
    assert len(results) == 2
    assert max_active >= 2
    assert "[Builder] builder done" in final
    assert "[Verifier] verifier done" in final


def test_workflow_store_recovers_incomplete_workflows(tmp_path: Path) -> None:
    store = WorkflowStore(tmp_path)
    workflow = WorkflowRecord(
        id="wf_incomplete",
        session_key="cli:test",
        channel="cli",
        chat_id="test",
        user_message="hello",
        status="running",
    )
    task = TaskRecord(id="task1", workflow_id=workflow.id, title="t", kind="communication")
    store.save_workflow(workflow, [task], [])

    recovered = store.recover_incomplete()

    assert len(recovered) == 1
    assert recovered[0]["workflow"]["id"] == "wf_incomplete"


@pytest.mark.asyncio
async def test_agent_identity_metadata_is_emitted_on_progress(tmp_path: Path) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, runtime_config=_config())
    record = ExecutionLogRecord(
        workflow_id="wf1",
        task_id="task1",
        agent_id="builder",
        agent_name="Builder",
        agent_role="coding",
        message_kind="progress",
        content="working",
        is_final=False,
    )

    await loop._emit_workflow_update(record, "cli", "direct")

    outbound = await loop.bus.consume_outbound()
    assert outbound.content == "[Builder] working"
    assert outbound.agent_id == "builder"
    assert outbound.agent_name == "Builder"
    assert outbound.agent_role == "coding"
    assert outbound.workflow_id == "wf1"
    assert outbound.task_id == "task1"
    assert outbound.message_kind == "progress"
    assert outbound.is_final is False


@pytest.mark.asyncio
async def test_executor_tool_allowlist_is_enforced(tmp_path: Path, monkeypatch) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="call1", name="message", arguments={"content": "hi"})],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        runtime_config=_config(),
    )
    monkeypatch.setattr("fubot.agent.loop.build_provider_for_route", lambda *_args, **_kwargs: provider)
    task = TaskRecord(
        id="task1",
        workflow_id="wf1",
        title="limited",
        kind="research",
        metadata={"input": "send a message"},
    )
    profile = AgentProfile(
        id="limited",
        name="Limited",
        role="research",
        tool_allowlist=["read_file"],
    )

    result = await loop._execute_profile_task(
        profile=profile,
        task=task,
        route_decision=RouteDecision(
            agent_id=profile.id,
            agent_name=profile.name,
            agent_role=profile.role,
            task_type=task.kind,
            model="test-model",
            provider="anthropic",
            reason="test",
            fallback_chain=["anthropic:test-model"],
        ),
        workflow_id="wf1",
        session_key="cli:test",
        channel="cli",
        chat_id="test",
        media=None,
        shared_board={},
        on_progress=None,
    )

    tool_messages = [message for message in result.all_messages if message.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "Tool 'message' not found" in tool_messages[0]["content"]


@pytest.mark.asyncio
async def test_executor_uses_route_decision_provider(tmp_path: Path, monkeypatch) -> None:
    default_provider = MagicMock()
    default_provider.get_default_model.return_value = "anthropic/claude-opus-4-5"
    default_provider.chat_with_retry = AsyncMock(side_effect=AssertionError("default provider should not be used"))

    routed_provider = MagicMock()
    routed_provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="done", tool_calls=[]))
    routed_provider.get_default_model.return_value = "qwen-max"

    loop = AgentLoop(
        bus=MessageBus(),
        provider=default_provider,
        workspace=tmp_path,
        model="anthropic/claude-opus-4-5",
        runtime_config=_config(),
    )
    seen: dict[str, str | None] = {}

    def _build_provider_for_route(
        _config,
        route_decision,
        *,
        default_provider=None,
        default_model=None,
        allow_default_provider=False,
    ):
        seen["model"] = route_decision.model
        seen["provider"] = route_decision.provider
        return routed_provider

    monkeypatch.setattr("fubot.agent.loop.build_provider_for_route", _build_provider_for_route)

    task = TaskRecord(
        id="task-route",
        workflow_id="wf-route",
        title="route-test",
        kind="coding",
        metadata={"input": "use dashscope"},
    )
    profile = AgentProfile(
        id="builder",
        name="Builder",
        role="coding",
        tool_allowlist=["read_file"],
    )
    decision = RouteDecision(
        agent_id=profile.id,
        agent_name=profile.name,
        agent_role=profile.role,
        task_type=task.kind,
        model="qwen-max",
        provider="dashscope",
        reason="test route",
        fallback_chain=["dashscope:qwen-max"],
    )

    result = await loop._execute_profile_task(
        profile=profile,
        task=task,
        route_decision=decision,
        workflow_id="wf-route",
        session_key="cli:test",
        channel="cli",
        chat_id="test",
        media=None,
        shared_board={},
        on_progress=None,
    )

    assert seen == {"model": "qwen-max", "provider": "dashscope"}
    routed_provider.chat_with_retry.assert_awaited()
    default_provider.chat_with_retry.assert_not_called()
    assert result.route["provider"] == "dashscope"
    assert result.route["model"] == "qwen-max"


@pytest.mark.asyncio
async def test_subagent_inherits_parent_route_provider(tmp_path: Path, monkeypatch) -> None:
    default_provider = MagicMock()
    default_provider.get_default_model.return_value = "anthropic/claude-opus-4-5"
    default_provider.chat_with_retry = AsyncMock(side_effect=AssertionError("default provider should not be used"))

    inherited_provider = MagicMock()
    inherited_provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="done", tool_calls=[]))
    inherited_provider.get_default_model.return_value = "qwen-max"

    loop = AgentLoop(
        bus=MessageBus(),
        provider=default_provider,
        workspace=tmp_path,
        model="anthropic/claude-opus-4-5",
        runtime_config=_config(),
    )
    loop.subagents._announce_result = AsyncMock()

    seen: dict[str, object] = {}

    def _build_provider_for_route(
        _config,
        route_decision,
        *,
        default_provider=None,
        default_model=None,
        allow_default_provider=False,
    ):
        seen["provider"] = route_decision.provider
        seen["model"] = route_decision.model
        seen["parent_trace_id"] = route_decision.parent_trace_id
        seen["inherited_from_parent"] = route_decision.inherited_from_parent
        seen["attempt_index"] = route_decision.attempt_index
        return inherited_provider

    monkeypatch.setattr("fubot.agent.subagent.build_provider_for_route", _build_provider_for_route)

    parent = RouteDecision(
        agent_id="builder",
        agent_name="Builder",
        agent_role="coding",
        task_type="coding",
        model="qwen-max",
        provider="dashscope",
        reason="parent explicit route",
        fallback_chain=["dashscope:qwen-max"],
    )

    await loop.subagents.spawn(task="child task", parent_route_decision=parent)
    running = list(loop.subagents._running_tasks.values())
    assert len(running) == 1
    await running[0]

    assert seen == {
        "provider": "dashscope",
        "model": "qwen-max",
        "parent_trace_id": parent.trace_id,
        "inherited_from_parent": True,
        "attempt_index": 0,
    }
    inherited_provider.chat_with_retry.assert_awaited_once()
    _, kwargs = inherited_provider.chat_with_retry.await_args
    assert kwargs["model"] == "qwen-max"
    default_provider.chat_with_retry.assert_not_called()


@pytest.mark.asyncio
async def test_coordinator_fallback_attempt_updates_route_and_child_inherits(tmp_path: Path) -> None:
    config = _config()
    profile = config.orchestration.executors["researcher"]
    profile.preferred_providers = ["dashscope", "openrouter"]
    store = WorkflowStore(tmp_path)
    runtime = CoordinatorRuntime(config, store)
    seen_routes: list[RouteDecision] = []
    child_routes: list[RouteDecision] = []

    async def _execute_task(
        profile,
        task,
        route_decision,
        workflow_id,
        session_key,
        channel,
        chat_id,
        media,
        shared_board,
        on_progress,
    ) -> ExecutorResult:
        seen_routes.append(route_decision)
        if route_decision.attempt_index == 0:
            raise ProviderExecutionError("429 rate limit", error_kind="rate_limit")
        child_routes.append(route_decision.derive_child(reason=f"inherited from parent route {route_decision.trace_id}"))
        return ExecutorResult(
            profile=profile,
            task=task,
            content="done",
            route=route_decision.to_dict(),
        )

    workflow, tasks, assignments, results, final = await runtime.run(
        session_key="cli:test",
        channel="cli",
        chat_id="test",
        content="research the incident and analyze provider behavior",
        media=None,
        execute_task=_execute_task,
    )

    assert workflow.status == "completed"
    assert len(tasks) == 1
    assert len(assignments) == 1
    assert len(results) == 1
    assert final == "done"
    assert [decision.provider for decision in seen_routes] == ["dashscope", "openrouter"]
    assert seen_routes[1].attempt_index == 1
    assert seen_routes[1].previous_attempt_trace_id == seen_routes[0].trace_id
    assert "fallback attempt 1" in seen_routes[1].reason
    assert "rate_limit" in seen_routes[1].reason
    assert "dashscope" in seen_routes[1].reason
    assert "openrouter" in seen_routes[1].reason
    assert results[0].route["provider"] == "openrouter"
    assert results[0].route["attempt_index"] == 1
    assert child_routes[0].provider == "openrouter"
    assert child_routes[0].attempt_index == 1
    assert child_routes[0].parent_trace_id == seen_routes[1].trace_id
    health = store.load_health()
    assert health["dashscope"]["status"] == "cooldown"
    assert health["dashscope"]["last_error_kind"] == "rate_limit"


def test_subagent_inherits_parent_fallback_route_decision(tmp_path: Path) -> None:
    from fubot.agent.subagent import SubagentManager

    provider = MagicMock()
    provider.get_default_model.return_value = "anthropic/claude-opus-4-5"
    manager = SubagentManager(
        provider=provider,
        workspace=tmp_path,
        bus=MessageBus(),
        allow_legacy_route_fallback=False,
    )

    parent = RouteDecision(
        agent_id="builder",
        agent_name="Builder",
        agent_role="coding",
        task_type="coding",
        model="gpt-4o",
        provider="openrouter",
        reason="fallback after dashscope failure",
        fallback_chain=["dashscope:qwen-max", "openrouter:gpt-4o"],
        attempt_index=1,
    )

    child = manager._resolve_child_route_decision(
        parent_route_decision=parent,
        child_route_decision=None,
        allow_child_reroute=False,
    )

    assert child.provider == "openrouter"
    assert child.model == "gpt-4o"
    assert child.trace_id != parent.trace_id
    assert child.parent_trace_id == parent.trace_id
    assert child.fallback_chain == ["dashscope:qwen-max", "openrouter:gpt-4o"]
    assert child.attempt_index == 1
    assert child.inherited_from_parent is True


@pytest.mark.asyncio
async def test_subagent_allows_explicit_child_reroute_and_logs_it(tmp_path: Path, monkeypatch) -> None:
    from fubot.agent.subagent import SubagentManager

    provider = MagicMock()
    provider.get_default_model.return_value = "anthropic/claude-opus-4-5"

    rerouted_provider = MagicMock()
    rerouted_provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="done", tool_calls=[]))
    rerouted_provider.get_default_model.return_value = "gpt-4o-mini"

    manager = SubagentManager(
        provider=provider,
        workspace=tmp_path,
        bus=MessageBus(),
        model="anthropic/claude-opus-4-5",
        config=_config(),
        default_provider_name="anthropic",
        allow_legacy_route_fallback=False,
    )
    manager._announce_result = AsyncMock()

    events: list[str] = []

    def _capture(message, *args):
        events.append(message.format(*args))

    monkeypatch.setattr("fubot.agent.subagent.logger.info", _capture)
    monkeypatch.setattr("fubot.agent.subagent.logger.debug", _capture)
    monkeypatch.setattr("fubot.agent.subagent.build_provider_for_route", lambda *_args, **_kwargs: rerouted_provider)

    parent = RouteDecision(
        agent_id="builder",
        agent_name="Builder",
        agent_role="coding",
        task_type="coding",
        model="qwen-max",
        provider="dashscope",
        reason="parent explicit route",
        fallback_chain=["dashscope:qwen-max"],
    )
    child = parent.derive_child(
        provider="openrouter",
        model="gpt-4o-mini",
        fallback_chain=["openrouter:gpt-4o-mini"],
        inherited_from_parent=False,
        reason="child reroute for web-heavy follow-up",
    )

    await manager.spawn(
        task="child task",
        parent_route_decision=parent,
        child_route_decision=child,
        allow_child_reroute=True,
    )
    running = list(manager._running_tasks.values())
    assert len(running) == 1
    await running[0]

    assert any(
        "route_kind=child_reroute" in event
        and f"trace_id={child.trace_id}" in event
        and f"parent_trace_id={parent.trace_id}" in event
        and "provider=openrouter" in event
        and "model=gpt-4o-mini" in event
        and "attempt=0" in event
        for event in events
    )


@pytest.mark.asyncio
async def test_subagent_rejects_implicit_fallback_to_manager_provider(tmp_path: Path) -> None:
    from fubot.agent.subagent import SubagentManager

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    manager = SubagentManager(
        provider=provider,
        workspace=tmp_path,
        bus=MessageBus(),
        allow_legacy_route_fallback=False,
    )

    with pytest.raises(RuntimeError, match="requires an explicit parent RouteDecision"):
        await manager.spawn(task="child task")


@pytest.mark.asyncio
async def test_subagent_legacy_single_provider_path_reuses_initialized_provider(tmp_path: Path, monkeypatch) -> None:
    from fubot.agent.subagent import SubagentManager

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="done", tool_calls=[]))

    manager = SubagentManager(
        provider=provider,
        workspace=tmp_path,
        bus=MessageBus(),
        model="test-model",
        config=_config(),
        default_provider_name="anthropic",
        allow_legacy_route_fallback=True,
    )
    manager._announce_result = AsyncMock()

    monkeypatch.setattr(
        "fubot.agent.subagent.build_provider_for_route",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy path should reuse initialized provider")),
    )

    await manager.spawn(task="child task")
    running = list(manager._running_tasks.values())
    assert len(running) == 1
    await running[0]

    provider.chat_with_retry.assert_awaited_once()


@pytest.mark.asyncio
async def test_system_message_uses_route_decision_provider(tmp_path: Path, monkeypatch) -> None:
    from fubot.bus.events import InboundMessage

    default_provider = MagicMock()
    default_provider.get_default_model.return_value = "anthropic/claude-opus-4-5"
    default_provider.chat_with_retry = AsyncMock(side_effect=AssertionError("default provider should not be used"))

    routed_provider = MagicMock()
    routed_provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="done", tool_calls=[]))
    routed_provider.get_default_model.return_value = "qwen-max"

    loop = AgentLoop(
        bus=MessageBus(),
        provider=default_provider,
        workspace=tmp_path,
        model="anthropic/claude-opus-4-5",
        runtime_config=_config(),
    )

    decision = RouteDecision(
        agent_id="builder",
        agent_name="Builder",
        agent_role="coding",
        task_type="coding",
        model="qwen-max",
        provider="dashscope",
        reason="inherit child result route",
        fallback_chain=["dashscope:qwen-max"],
    )

    monkeypatch.setattr("fubot.agent.loop.build_provider_for_route", lambda *_args, **_kwargs: routed_provider)

    response = await loop._process_message(
        InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id="cli:test",
            content="summarize child result",
            metadata={"route_decision": decision.to_dict()},
        )
    )

    assert response is not None
    assert response.content == "done"
    routed_provider.chat_with_retry.assert_awaited_once()
    default_provider.chat_with_retry.assert_not_called()


def test_workflow_store_recovery_ignores_health_cache_files(tmp_path: Path) -> None:
    store = WorkflowStore(tmp_path)
    workflow = WorkflowRecord(
        id="wf_valid",
        session_key="cli:test",
        channel="cli",
        chat_id="test",
        user_message="hello",
        status="running",
    )
    task = TaskRecord(id="task1", workflow_id=workflow.id, title="t", kind="communication")
    store.save_workflow(workflow, [task], [])
    store.save_health({"dashscope": {"failures": 3}})

    recovered = store.recover_incomplete()

    assert [item["workflow"]["id"] for item in recovered] == ["wf_valid"]


@pytest.mark.asyncio
async def test_fallback_attempt_does_not_replay_message_side_effect(tmp_path: Path, monkeypatch) -> None:
    config = _config()
    config.orchestration.executors["generalist"].preferred_providers = ["dashscope", "openrouter"]

    default_provider = MagicMock()
    default_provider.get_default_model.return_value = "anthropic/claude-opus-4-5"
    default_provider.chat_with_retry = AsyncMock(
        side_effect=AssertionError("default provider should not be used")
    )

    first_provider = MagicMock()
    first_provider.get_default_model.return_value = "qwen-max"
    first_provider.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="call1", name="message", arguments={"content": "hello once"})],
            ),
            LLMResponse(content="503 server error", finish_reason="error"),
        ]
    )

    fallback_provider = MagicMock()
    fallback_provider.get_default_model.return_value = "gpt-4o"
    fallback_provider.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="call2", name="message", arguments={"content": "hello once"})],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )

    loop = AgentLoop(
        bus=MessageBus(),
        provider=default_provider,
        workspace=tmp_path,
        model="anthropic/claude-opus-4-5",
        runtime_config=config,
    )
    sent_messages: list[str] = []
    message_tool = loop.tools.get("message")
    if isinstance(message_tool, MagicMock):
        raise AssertionError("unexpected message tool mock")
    if message_tool is not None:
        message_tool.set_context("cli", "test")
        message_tool.set_send_callback(
            AsyncMock(side_effect=lambda outbound: sent_messages.append(outbound.content))
        )

    providers = {
        "dashscope": first_provider,
        "openrouter": fallback_provider,
    }

    def _build_provider_for_route(
        _config,
        route_decision,
        *,
        default_provider=None,
        default_model=None,
        allow_default_provider=False,
    ):
        return providers[route_decision.provider]

    monkeypatch.setattr("fubot.agent.loop.build_provider_for_route", _build_provider_for_route)

    workflow, tasks, assignments, results, final = await loop.coordinator.run(
        session_key="cli:test",
        channel="cli",
        chat_id="test",
        content="say hello",
        media=None,
        execute_task=loop._execute_profile_task,
    )

    assert workflow.status == "completed"
    assert len(tasks) == 1
    assert len(assignments) == 1
    assert len(results) == 1
    assert final == "done"
    assert sent_messages == ["hello once"]
    assert results[0].route["provider"] == "openrouter"
    replay_messages = [
        message["content"]
        for message in results[0].all_messages
        if message.get("role") == "tool"
    ]
    assert any("Replay prevented for side-effect tool 'message'" in content for content in replay_messages)


@pytest.mark.asyncio
async def test_read_only_tool_retries_across_fallback_with_stable_execution_lineage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config()
    config.orchestration.executors["generalist"].preferred_providers = ["dashscope", "openrouter"]

    default_provider = MagicMock()
    default_provider.get_default_model.return_value = "anthropic/claude-opus-4-5"
    default_provider.chat_with_retry = AsyncMock(
        side_effect=AssertionError("default provider should not be used")
    )

    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("hello\nworld\n", encoding="utf-8")

    first_provider = MagicMock()
    first_provider.get_default_model.return_value = "qwen-max"
    first_provider.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="call1",
                        name="read_file",
                        arguments={"path": str(sample_file)},
                    )
                ],
            ),
            LLMResponse(content="503 server error", finish_reason="error"),
        ]
    )

    fallback_provider = MagicMock()
    fallback_provider.get_default_model.return_value = "gpt-4o"
    fallback_provider.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="call2",
                        name="read_file",
                        arguments={"path": str(sample_file)},
                    )
                ],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )

    loop = AgentLoop(
        bus=MessageBus(),
        provider=default_provider,
        workspace=tmp_path,
        model="anthropic/claude-opus-4-5",
        runtime_config=config,
    )

    providers = {
        "dashscope": first_provider,
        "openrouter": fallback_provider,
    }

    def _build_provider_for_route(
        _config,
        route_decision,
        *,
        default_provider=None,
        default_model=None,
        allow_default_provider=False,
    ):
        return providers[route_decision.provider]

    monkeypatch.setattr("fubot.agent.loop.build_provider_for_route", _build_provider_for_route)

    contexts = []
    original_execute = ReadFileTool.execute

    async def _capture_execute(self, path: str, offset: int = 1, limit: int | None = None, **kwargs):
        contexts.append(kwargs.get("_tool_execution_context"))
        return await original_execute(self, path=path, offset=offset, limit=limit, **kwargs)

    monkeypatch.setattr(ReadFileTool, "execute", _capture_execute)

    workflow, _tasks, _assignments, results, final = await loop.coordinator.run(
        session_key="cli:test",
        channel="cli",
        chat_id="test",
        content="say hello",
        media=None,
        execute_task=loop._execute_profile_task,
    )

    assert workflow.status == "completed"
    assert final == "done"
    assert len(contexts) == 2
    assert contexts[0] is not None
    assert contexts[1] is not None
    assert contexts[0].trace_id == contexts[1].trace_id
    assert contexts[0].route_trace_id != contexts[1].route_trace_id
    assert [context.attempt_index for context in contexts] == [0, 1]
    assert contexts[0].idempotency_key == contexts[1].idempotency_key
    assert contexts[0].is_replay is False
    assert contexts[1].is_replay is False
    tool_messages = [
        message["content"]
        for message in results[0].all_messages
        if message.get("role") == "tool"
    ]
    assert all("Replay prevented" not in content for content in tool_messages)


@pytest.mark.asyncio
async def test_subagent_side_effect_tool_replay_guard_across_child_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from fubot.agent.subagent import SubagentManager

    config = _config()
    store = WorkflowStore(tmp_path)
    default_provider = MagicMock()
    default_provider.get_default_model.return_value = "anthropic/claude-opus-4-5"

    first_provider = MagicMock()
    first_provider.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="call1",
                        name="write_file",
                        arguments={"path": str(tmp_path / "note.txt"), "content": "hello"},
                    )
                ],
            ),
            LLMResponse(content="503 server error", finish_reason="error"),
        ]
    )

    fallback_provider = MagicMock()
    fallback_provider.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="call2",
                        name="write_file",
                        arguments={"path": str(tmp_path / "note.txt"), "content": "hello"},
                    )
                ],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )

    manager = SubagentManager(
        provider=default_provider,
        workspace=tmp_path,
        bus=MessageBus(),
        model="anthropic/claude-opus-4-5",
        config=config,
        default_provider_name="anthropic",
        allow_legacy_route_fallback=False,
        workflow_store=store,
    )
    manager._announce_result = AsyncMock()

    providers = {
        "dashscope": first_provider,
        "openrouter": fallback_provider,
    }

    monkeypatch.setattr(
        "fubot.agent.subagent.build_provider_for_route",
        lambda _config, route_decision, **_kwargs: providers[route_decision.provider],
    )

    write_contexts = []
    original_execute = WriteFileTool.execute

    async def _capture_execute(self, path: str, content: str, **kwargs):
        write_contexts.append(kwargs.get("_tool_execution_context"))
        return await original_execute(self, path=path, content=content, **kwargs)

    monkeypatch.setattr(WriteFileTool, "execute", _capture_execute)

    parent = RouteDecision(
        agent_id="builder",
        agent_name="Builder",
        agent_role="coding",
        task_type="coding",
        model="qwen-max",
        provider="dashscope",
        reason="parent route",
        fallback_chain=["dashscope:qwen-max", "openrouter:gpt-4o"],
    )

    await manager.spawn(task="child task", parent_route_decision=parent)
    running = list(manager._running_tasks.values())
    assert len(running) == 1
    await running[0]

    assert len(write_contexts) == 1
    assert write_contexts[0] is not None
    assert manager._announce_result.await_args.kwargs["route_decision"].provider == "openrouter"


@pytest.mark.asyncio
async def test_subagent_read_only_tool_retries_across_child_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from fubot.agent.subagent import SubagentManager

    config = _config()
    store = WorkflowStore(tmp_path)
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("hello\nworld\n", encoding="utf-8")

    default_provider = MagicMock()
    default_provider.get_default_model.return_value = "anthropic/claude-opus-4-5"

    first_provider = MagicMock()
    first_provider.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="call1",
                        name="read_file",
                        arguments={"path": str(sample_file)},
                    )
                ],
            ),
            LLMResponse(content="503 server error", finish_reason="error"),
        ]
    )

    fallback_provider = MagicMock()
    fallback_provider.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="call2",
                        name="read_file",
                        arguments={"path": str(sample_file)},
                    )
                ],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )

    manager = SubagentManager(
        provider=default_provider,
        workspace=tmp_path,
        bus=MessageBus(),
        model="anthropic/claude-opus-4-5",
        config=config,
        default_provider_name="anthropic",
        allow_legacy_route_fallback=False,
        workflow_store=store,
    )
    manager._announce_result = AsyncMock()

    providers = {
        "dashscope": first_provider,
        "openrouter": fallback_provider,
    }

    monkeypatch.setattr(
        "fubot.agent.subagent.build_provider_for_route",
        lambda _config, route_decision, **_kwargs: providers[route_decision.provider],
    )

    contexts = []
    original_execute = ReadFileTool.execute

    async def _capture_execute(self, path: str, offset: int = 1, limit: int | None = None, **kwargs):
        contexts.append(kwargs.get("_tool_execution_context"))
        return await original_execute(self, path=path, offset=offset, limit=limit, **kwargs)

    monkeypatch.setattr(ReadFileTool, "execute", _capture_execute)

    parent = RouteDecision(
        agent_id="builder",
        agent_name="Builder",
        agent_role="coding",
        task_type="coding",
        model="qwen-max",
        provider="dashscope",
        reason="parent route",
        fallback_chain=["dashscope:qwen-max", "openrouter:gpt-4o"],
    )

    await manager.spawn(task="child task", parent_route_decision=parent)
    running = list(manager._running_tasks.values())
    assert len(running) == 1
    await running[0]

    assert len(contexts) == 2
    assert contexts[0].trace_id == contexts[1].trace_id
    assert contexts[0].idempotency_key == contexts[1].idempotency_key
    assert [context.attempt_index for context in contexts] == [0, 1]
    assert manager._announce_result.await_args.kwargs["route_decision"].provider == "openrouter"


@pytest.mark.asyncio
async def test_parent_child_and_system_tool_execution_lineage_chain(
    tmp_path: Path,
    monkeypatch,
) -> None:
    default_provider = MagicMock()
    default_provider.get_default_model.return_value = "anthropic/claude-opus-4-5"

    parent_provider = MagicMock()
    parent_provider.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="call1", name="spawn", arguments={"task": "child task"})],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )

    system_provider = MagicMock()
    sample_file = tmp_path / "system.txt"
    sample_file.write_text("payload", encoding="utf-8")
    system_provider.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="call2", name="read_file", arguments={"path": str(sample_file)})],
            ),
            LLMResponse(content="system done", tool_calls=[]),
        ]
    )

    child_provider = MagicMock()
    child_provider.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="call3", name="read_file", arguments={"path": str(sample_file)})],
            ),
            LLMResponse(content="child done", tool_calls=[]),
        ]
    )

    loop = AgentLoop(
        bus=MessageBus(),
        provider=default_provider,
        workspace=tmp_path,
        model="anthropic/claude-opus-4-5",
        runtime_config=_config(),
    )

    captured_spawn_kwargs: list[dict[str, object]] = []

    async def _capture_spawn(*args, **kwargs):
        captured_spawn_kwargs.append(kwargs)
        return "spawned"

    loop.subagents.spawn = AsyncMock(side_effect=_capture_spawn)
    provider_queue = iter([parent_provider, system_provider])
    monkeypatch.setattr("fubot.agent.loop.build_provider_for_route", lambda *_args, **_kwargs: next(provider_queue))

    profile = AgentProfile(
        id="builder",
        name="Builder",
        role="coding",
        tool_allowlist=["spawn"],
    )
    task = TaskRecord(
        id="task-parent",
        workflow_id="wf-parent",
        title="parent task",
        kind="coding",
        metadata={"input": "spawn child"},
    )
    decision = RouteDecision(
        agent_id="builder",
        agent_name="Builder",
        agent_role="coding",
        task_type="coding",
        model="qwen-max",
        provider="dashscope",
        reason="parent route",
        fallback_chain=["dashscope:qwen-max"],
    )

    await loop._execute_profile_task(
        profile=profile,
        task=task,
        route_decision=decision,
        workflow_id="wf-parent",
        session_key="cli:test",
        channel="cli",
        chat_id="test",
        media=None,
        shared_board={},
        on_progress=None,
    )

    parent_execution_context = captured_spawn_kwargs[0]["parent_execution_context"]
    assert parent_execution_context is not None

    from fubot.agent.subagent import SubagentManager

    manager = SubagentManager(
        provider=default_provider,
        workspace=tmp_path,
        bus=MessageBus(),
        model="anthropic/claude-opus-4-5",
        config=_config(),
        default_provider_name="anthropic",
        allow_legacy_route_fallback=False,
        workflow_store=WorkflowStore(tmp_path),
    )
    manager._announce_result = AsyncMock()
    monkeypatch.setattr("fubot.agent.subagent.build_provider_for_route", lambda *_args, **_kwargs: child_provider)

    observed_contexts = []
    original_read = ReadFileTool.execute

    async def _capture_read(self, path: str, offset: int = 1, limit: int | None = None, **kwargs):
        observed_contexts.append(kwargs.get("_tool_execution_context"))
        return await original_read(self, path=path, offset=offset, limit=limit, **kwargs)

    monkeypatch.setattr(ReadFileTool, "execute", _capture_read)

    child_parent_route = RouteDecision(
        agent_id="builder",
        agent_name="Builder",
        agent_role="coding",
        task_type="coding",
        model="qwen-max",
        provider="dashscope",
        reason="parent route",
        fallback_chain=["dashscope:qwen-max"],
    )

    spawn_result = await manager.spawn(
        task="child task",
        parent_route_decision=child_parent_route,
        parent_execution_context=parent_execution_context,
    )
    child_task_id = spawn_result.split("(id: ", 1)[1].split(")", 1)[0]
    child_scope = manager._tool_execution_scopes[child_task_id]["lineage"]
    running = list(manager._running_tasks.values())
    assert len(running) == 1
    await running[0]

    child_announce_lineage = manager._announce_result.await_args.kwargs["tool_execution_lineage"]
    child_route_decision = manager._announce_result.await_args.kwargs["route_decision"]
    child_tool_context = observed_contexts[0]
    assert child_scope.parent_execution_id == parent_execution_context.execution_id
    assert child_tool_context.parent_execution_id == child_scope.root_execution_id
    assert child_announce_lineage.parent_execution_id == child_scope.root_execution_id

    before_system = len(observed_contexts)
    response = await loop._process_message(
        InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id="cli:test",
            content="summarize child result",
            metadata={
                "route_decision": child_route_decision.to_dict(),
                "tool_execution_lineage": child_announce_lineage.to_dict(),
            },
        )
    )
    assert response is not None
    system_tool_context = observed_contexts[before_system]
    assert system_tool_context.trace_id == child_announce_lineage.trace_id
    assert system_tool_context.parent_execution_id == child_announce_lineage.root_execution_id
    assert system_tool_context.route_trace_id == child_announce_lineage.route_trace_id


@pytest.mark.asyncio
async def test_runtime_reports_all_candidates_unavailable_with_diagnostics(tmp_path: Path) -> None:
    config = _config()
    profile = config.orchestration.executors["researcher"]
    profile.preferred_providers = ["dashscope", "openrouter"]
    store = WorkflowStore(tmp_path)
    store.save_health(
        {
            "dashscope": {"status": "cooldown", "cooldown_until": "9999-01-01T00:00:00+00:00"},
            "openrouter": {"status": "disabled", "last_error_kind": "manual_disabled"},
        }
    )
    runtime = CoordinatorRuntime(config, store)

    async def _execute_task(*args, **kwargs):
        raise AssertionError("execute_task should not be called when all candidates are unavailable")

    with pytest.raises(ProviderExecutionError, match="Provider fallback exhausted") as exc_info:
        await runtime.run(
            session_key="cli:test",
            channel="cli",
            chat_id="test",
            content="research the incident and analyze provider behavior",
            media=None,
            execute_task=_execute_task,
        )

    message = str(exc_info.value)
    assert "trace_id=" in message
    assert "dashscope:" in message and "[cooldown]" in message
    assert "openrouter:" in message and "[disabled]" in message
    assert "all configured providers unavailable before execution" in message


@pytest.mark.asyncio
async def test_root_tool_execution_cache_cleans_up_after_success(tmp_path: Path, monkeypatch) -> None:
    default_provider = MagicMock()
    default_provider.get_default_model.return_value = "anthropic/claude-opus-4-5"
    routed_provider = MagicMock()
    routed_provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="done", tool_calls=[]))

    loop = AgentLoop(
        bus=MessageBus(),
        provider=default_provider,
        workspace=tmp_path,
        model="anthropic/claude-opus-4-5",
        runtime_config=_config(),
    )
    monkeypatch.setattr("fubot.agent.loop.build_provider_for_route", lambda *_args, **_kwargs: routed_provider)

    await loop.coordinator.run(
        session_key="cli:test",
        channel="cli",
        chat_id="test",
        content="say hello",
        media=None,
        execute_task=loop._execute_profile_task,
        cleanup_task_lineage=loop._cleanup_tool_execution_lineage,
    )

    assert loop._tool_execution_lineages == {}


@pytest.mark.asyncio
async def test_root_tool_execution_cache_cleans_up_after_failure_and_cancel(
    tmp_path: Path,
    monkeypatch,
) -> None:
    default_provider = MagicMock()
    default_provider.get_default_model.return_value = "anthropic/claude-opus-4-5"
    loop = AgentLoop(
        bus=MessageBus(),
        provider=default_provider,
        workspace=tmp_path,
        model="anthropic/claude-opus-4-5",
        runtime_config=_config(),
    )

    failing_provider = MagicMock()
    failing_provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="503 server error", finish_reason="error"))

    async def _canceling_execute_task(
        profile,
        task,
        route_decision,
        workflow_id,
        session_key,
        channel,
        chat_id,
        media,
        shared_board,
        on_progress,
    ):
        _ = (profile, session_key, channel, chat_id, media, shared_board, on_progress)
        loop._get_tool_execution_lineage(workflow_id, task.id, route_decision=route_decision)
        await asyncio.sleep(60)
        return ExecutorResult(profile=profile, task=task, content="never", route=route_decision.to_dict())

    monkeypatch.setattr("fubot.agent.loop.build_provider_for_route", lambda *_args, **_kwargs: failing_provider)

    with pytest.raises(ProviderExecutionError):
        await loop.coordinator.run(
            session_key="cli:test",
            channel="cli",
            chat_id="test",
            content="say hello",
            media=None,
            execute_task=loop._execute_profile_task,
            cleanup_task_lineage=loop._cleanup_tool_execution_lineage,
        )
    assert loop._tool_execution_lineages == {}

    cancel_task = asyncio.create_task(
        loop.coordinator.run(
            session_key="cli:test",
            channel="cli",
            chat_id="test",
            content="say hello",
            media=None,
            execute_task=_canceling_execute_task,
            cleanup_task_lineage=loop._cleanup_tool_execution_lineage,
        )
    )
    await asyncio.sleep(0.05)
    assert loop._tool_execution_lineages
    cancel_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancel_task
    assert loop._tool_execution_lineages == {}


@pytest.mark.asyncio
async def test_new_task_and_child_scope_do_not_inherit_parent_replay_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    default_provider = MagicMock()
    default_provider.get_default_model.return_value = "anthropic/claude-opus-4-5"
    loop = AgentLoop(
        bus=MessageBus(),
        provider=default_provider,
        workspace=tmp_path,
        model="anthropic/claude-opus-4-5",
        runtime_config=_config(),
    )

    write_calls = []
    original_execute = WriteFileTool.execute

    async def _capture_execute(self, path: str, content: str, **kwargs):
        write_calls.append(kwargs.get("_tool_execution_context"))
        return await original_execute(self, path=path, content=content, **kwargs)

    monkeypatch.setattr(WriteFileTool, "execute", _capture_execute)

    root_provider_first = MagicMock()
    root_provider_first.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="call1",
                        name="write_file",
                        arguments={"path": str(tmp_path / "shared.txt"), "content": "same"},
                    )
                ],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    monkeypatch.setattr("fubot.agent.loop.build_provider_for_route", lambda *_args, **_kwargs: root_provider_first)

    profile = AgentProfile(
        id="generalist",
        name="Generalist",
        role="assistant",
        tool_allowlist=["write_file"],
    )
    decision = RouteDecision(
        agent_id="generalist",
        agent_name="Generalist",
        agent_role="assistant",
        task_type="communication",
        model="qwen-max",
        provider="dashscope",
        reason="root route",
        fallback_chain=["dashscope:qwen-max"],
    )
    task = TaskRecord(
        id="task-root",
        workflow_id="wf-root",
        title="root",
        kind="communication",
        metadata={"input": "write"},
    )

    await loop._execute_profile_task(
        profile=profile,
        task=task,
        route_decision=decision,
        workflow_id="wf-root",
        session_key="cli:test",
        channel="cli",
        chat_id="test",
        media=None,
        shared_board={},
        on_progress=None,
    )
    loop._cleanup_tool_execution_lineage("wf-root", "task-root")
    assert len(write_calls) == 1

    root_provider_second = MagicMock()
    root_provider_second.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="call1b",
                        name="write_file",
                        arguments={"path": str(tmp_path / "shared.txt"), "content": "same"},
                    )
                ],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    monkeypatch.setattr("fubot.agent.loop.build_provider_for_route", lambda *_args, **_kwargs: root_provider_second)

    await loop._execute_profile_task(
        profile=profile,
        task=TaskRecord(
            id="task-root-2",
            workflow_id="wf-root-2",
            title="root-2",
            kind="communication",
            metadata={"input": "write"},
        ),
        route_decision=RouteDecision(
            agent_id="generalist",
            agent_name="Generalist",
            agent_role="assistant",
            task_type="communication",
            model="qwen-max",
            provider="dashscope",
            reason="root route second task",
            fallback_chain=["dashscope:qwen-max"],
        ),
        workflow_id="wf-root-2",
        session_key="cli:test",
        channel="cli",
        chat_id="test",
        media=None,
        shared_board={},
        on_progress=None,
    )
    loop._cleanup_tool_execution_lineage("wf-root-2", "task-root-2")
    assert len(write_calls) == 2

    from fubot.agent.subagent import SubagentManager

    child_provider = MagicMock()
    child_provider.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="call2",
                        name="write_file",
                        arguments={"path": str(tmp_path / "shared.txt"), "content": "same"},
                    )
                ],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )

    manager = SubagentManager(
        provider=default_provider,
        workspace=tmp_path,
        bus=MessageBus(),
        model="anthropic/claude-opus-4-5",
        config=_config(),
        default_provider_name="anthropic",
        allow_legacy_route_fallback=False,
        workflow_store=WorkflowStore(tmp_path),
    )
    manager._announce_result = AsyncMock()
    monkeypatch.setattr("fubot.agent.subagent.build_provider_for_route", lambda *_args, **_kwargs: child_provider)

    parent = RouteDecision(
        agent_id="builder",
        agent_name="Builder",
        agent_role="coding",
        task_type="coding",
        model="qwen-max",
        provider="dashscope",
        reason="parent route",
        fallback_chain=["dashscope:qwen-max"],
    )
    await manager.spawn(task="child task", parent_route_decision=parent)
    running = list(manager._running_tasks.values())
    assert len(running) == 1
    await running[0]

    assert len(write_calls) == 3
    assert write_calls[0].trace_id != write_calls[1].trace_id
    assert write_calls[2].trace_id not in {write_calls[0].trace_id, write_calls[1].trace_id}
