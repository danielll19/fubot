from unittest.mock import AsyncMock

import pytest

from fubot.agent.tools.base import ToolExecutionContext
from fubot.agent.tools.message import MessageTool
from fubot.agent.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_duplicate_idempotency_key_hits_replay_guard_and_logs(monkeypatch) -> None:
    sent_messages: list[str] = []
    tool = MessageTool(
        send_callback=AsyncMock(side_effect=lambda outbound: sent_messages.append(outbound.content))
    )
    tool.set_context("cli", "direct")
    registry = ToolRegistry()
    registry.register(tool)

    events: list[tuple[str, str]] = []

    def _capture(level: str):
        def _log(message, *args):
            events.append((level, message.format(*args)))

        return _log

    monkeypatch.setattr("fubot.agent.tools.registry.logger.info", _capture("info"))
    monkeypatch.setattr("fubot.agent.tools.registry.logger.warning", _capture("warning"))

    first = ToolExecutionContext(
        execution_id="exec_1",
        trace_id="tooltrace1",
        route_trace_id="route_a",
        parent_execution_id=None,
        attempt_index=0,
        idempotency_key="same-key-123",
        is_replay=False,
    )
    second = ToolExecutionContext(
        execution_id="exec_2",
        trace_id="tooltrace1",
        route_trace_id="route_b",
        parent_execution_id=None,
        attempt_index=1,
        idempotency_key="same-key-123",
        is_replay=False,
    )

    first_result = await registry.execute("message", {"content": "hello"}, execution_context=first)
    second_result = await registry.execute("message", {"content": "hello"}, execution_context=second)

    assert first_result == "Message sent to cli:direct"
    assert "Replay prevented for side-effect tool 'message'" in second_result
    assert sent_messages == ["hello"]
    assert any(
        level == "info" and "Tool event=side_effect_execute tool=message" in message
        for level, message in events
    )
    assert any(
        level == "warning" and "Tool event=replay_prevented tool=message" in message
        for level, message in events
    )
