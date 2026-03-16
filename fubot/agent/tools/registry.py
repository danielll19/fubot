"""Tool registry for dynamic tool management."""

from datetime import datetime, timezone
from typing import Any

from loguru import logger

from fubot.agent.tools.base import Tool, ToolExecutionContext


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    def __init__(self, replay_cache: dict[str, dict[str, Any]] | None = None):
        self._tools: dict[str, Tool] = {}
        self._replay_cache = replay_cache if replay_cache is not None else {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(
        self,
        name: str,
        params: dict[str, Any],
        execution_context: ToolExecutionContext | None = None,
    ) -> str:
        """Execute a tool by name with given parameters."""
        hint = "\n\n[Analyze the error above and try a different approach.]"

        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}"

        try:
            # Attempt to cast parameters to match schema types
            params = tool.cast_params(params)

            # Validate parameters
            errors = tool.validate_params(params)
            if errors:
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors) + hint

            # MVP replay guard: read-only tools opt out, side-effect tools share
            # an in-memory cache across fallback attempts within the same task lineage.
            execution_mode = tool.execution_mode(params)
            active_context = execution_context
            if execution_mode == "side_effect" and execution_context is not None:
                replay_record = self._replay_cache.get(execution_context.idempotency_key)
                if replay_record is not None:
                    active_context = execution_context.as_replay(
                        parent_execution_id=replay_record.get("execution_id"),
                    )
                    fields = active_context.log_fields()
                    logger.warning(
                        "Tool event=replay_prevented tool={} execution_id={} parent_execution_id={} trace_id={} route_trace_id={} attempt={} idempotency_key={}",
                        name,
                        fields["execution_id"],
                        fields["parent_execution_id"],
                        fields["trace_id"],
                        fields["route_trace_id"],
                        fields["attempt_index"],
                        fields["idempotency_key"],
                    )
                    return (
                        f"Replay prevented for side-effect tool '{name}' "
                        f"(idempotency_key={active_context.idempotency_key_prefix}, "
                        f"previous_execution_id={active_context.parent_execution_id or '-'})"
                    )
                fields = execution_context.log_fields()
                logger.info(
                    "Tool event=side_effect_execute tool={} execution_id={} parent_execution_id={} trace_id={} route_trace_id={} attempt={} idempotency_key={} replay={}",
                    name,
                    fields["execution_id"],
                    fields["parent_execution_id"],
                    fields["trace_id"],
                    fields["route_trace_id"],
                    fields["attempt_index"],
                    fields["idempotency_key"],
                    fields["is_replay"],
                )
            elif execution_mode == "read_only" and execution_context is not None:
                fields = execution_context.log_fields()
                event = "read_only_retry" if execution_context.attempt_index > 0 else "read_only_execute"
                logger.info(
                    "Tool event={} tool={} execution_id={} parent_execution_id={} trace_id={} route_trace_id={} attempt={} idempotency_key={}",
                    event,
                    name,
                    fields["execution_id"],
                    fields["parent_execution_id"],
                    fields["trace_id"],
                    fields["route_trace_id"],
                    fields["attempt_index"],
                    fields["idempotency_key"],
                )

            result = await tool.execute(**params, _tool_execution_context=active_context)
            if (
                execution_mode == "side_effect"
                and execution_context is not None
                and isinstance(result, str)
                and not result.startswith("Error")
            ):
                self._replay_cache[execution_context.idempotency_key] = {
                    "tool_name": name,
                    "execution_id": execution_context.execution_id,
                    "trace_id": execution_context.trace_id,
                    "route_trace_id": execution_context.route_trace_id,
                    "attempt_index": execution_context.attempt_index,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            if isinstance(result, str) and result.startswith("Error"):
                return result + hint
            return result
        except Exception as e:
            return f"Error executing {name}: {str(e)}" + hint

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
