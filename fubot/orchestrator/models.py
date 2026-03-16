"""Structured records for workflow orchestration."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


def utc_now() -> str:
    """Return an audit-friendly UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def parse_utc_timestamp(value: str | None) -> datetime:
    """Parse an ISO timestamp, falling back to the Unix epoch on invalid input."""
    if not value:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def _require_non_empty_str(value: Any, *, field_name: str, error_prefix: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{error_prefix}: {field_name} must be a non-empty string")
    return value


def _optional_str(value: Any, *, field_name: str, error_prefix: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{error_prefix}: {field_name} must be a string when provided")
    return value


def _require_non_negative_int(value: Any, *, field_name: str, error_prefix: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{error_prefix}: {field_name} must be a non-negative integer")
    return value


@dataclass
class ProviderHealthState:
    """Persisted health state for one provider."""

    provider: str
    status: str = "healthy"
    cooldown_until: str | None = None
    last_error_kind: str | None = None
    failure_count: int = 0
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, provider: str, payload: Mapping[str, Any] | None) -> "ProviderHealthState":
        """Rehydrate health state, including legacy payloads."""
        data = payload or {}
        failure_count = int(data.get("failure_count", data.get("failures", 0)) or 0)
        cooldown_until = data.get("cooldown_until") or None
        status = str(data.get("status") or "").strip().lower()
        if status not in {"healthy", "cooldown", "degraded", "disabled"}:
            if data.get("disabled"):
                status = "disabled"
            elif cooldown_until:
                status = "cooldown"
            elif failure_count > 0:
                status = "degraded"
            else:
                status = "healthy"
        return cls(
            provider=provider,
            status=status,
            cooldown_until=cooldown_until,
            last_error_kind=data.get("last_error_kind"),
            failure_count=failure_count,
            updated_at=data.get("updated_at") or utc_now(),
        )

    def normalized(self, now: str | None = None) -> "ProviderHealthState":
        """Auto-heal expired cooldown states to healthy."""
        if self.status != "cooldown" or not self.cooldown_until:
            return self
        current = parse_utc_timestamp(now or utc_now())
        if parse_utc_timestamp(self.cooldown_until) > current:
            return self
        return ProviderHealthState(
            provider=self.provider,
            status="healthy",
            cooldown_until=None,
            last_error_kind=self.last_error_kind,
            failure_count=0,
            updated_at=now or utc_now(),
        )

    def merged_with(self, other: "ProviderHealthState") -> "ProviderHealthState":
        """Keep the newer state, breaking ties by severity and failure count."""
        current_ts = parse_utc_timestamp(self.updated_at)
        other_ts = parse_utc_timestamp(other.updated_at)
        if other_ts > current_ts:
            return other
        if current_ts > other_ts:
            return self
        severity = {"healthy": 0, "degraded": 1, "cooldown": 2, "disabled": 3}
        current_rank = (severity.get(self.status, -1), self.failure_count)
        other_rank = (severity.get(other.status, -1), other.failure_count)
        return other if other_rank >= current_rank else self


@dataclass
class RouteDecision:
    """Auditable routing decision for one executor turn."""

    agent_id: str
    agent_name: str
    agent_role: str
    task_type: str
    model: str
    provider: str | None
    reason: str
    fallback_chain: list[str] = field(default_factory=list)
    attempt_index: int = 0
    trace_id: str = field(default_factory=_new_trace_id)
    parent_trace_id: str | None = None
    previous_attempt_trace_id: str | None = None
    inherited_from_parent: bool = False
    health_score: float = 1.0
    current_load: int = 0
    created_at: str = field(default_factory=utc_now)

    _SERIALIZED_FIELDS = (
        "agent_id",
        "agent_name",
        "agent_role",
        "task_type",
        "model",
        "provider",
        "reason",
        "fallback_chain",
        "attempt_index",
        "trace_id",
        "parent_trace_id",
        "previous_attempt_trace_id",
        "inherited_from_parent",
        "health_score",
        "current_load",
        "created_at",
    )

    def to_dict(self) -> dict[str, Any]:
        return {field_name: getattr(self, field_name) for field_name in self._SERIALIZED_FIELDS}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RouteDecision":
        """Rehydrate a route decision from serialized metadata.

        Unknown fields are ignored on purpose; route_kind is always derived internally.
        """
        error_prefix = "Invalid route_decision metadata"
        if not isinstance(payload, Mapping):
            raise ValueError(f"{error_prefix}: payload must be a mapping")
        model = _require_non_empty_str(payload.get("model"), field_name="model", error_prefix=error_prefix)
        provider = _require_non_empty_str(
            payload.get("provider"),
            field_name="provider",
            error_prefix=error_prefix,
        )
        trace_id = _require_non_empty_str(
            payload.get("trace_id"),
            field_name="trace_id",
            error_prefix=error_prefix,
        )
        attempt_index = _require_non_negative_int(
            payload.get("attempt_index"),
            field_name="attempt_index",
            error_prefix=error_prefix,
        )
        fallback_chain = payload.get("fallback_chain")
        if not isinstance(fallback_chain, list):
            raise ValueError(f"{error_prefix}: fallback_chain must be a list")
        if any(not isinstance(candidate, str) or not candidate.strip() for candidate in fallback_chain):
            raise ValueError(f"{error_prefix}: fallback_chain entries must be non-empty strings")
        inherited_from_parent = bool(payload.get("inherited_from_parent", False))
        parent_trace_id = _optional_str(
            payload.get("parent_trace_id"),
            field_name="parent_trace_id",
            error_prefix=error_prefix,
        )
        if inherited_from_parent and parent_trace_id is None:
            raise ValueError(
                f"{error_prefix}: parent_trace_id is required when inherited_from_parent is true",
            )
        previous_attempt_trace_id = _optional_str(
            payload.get("previous_attempt_trace_id"),
            field_name="previous_attempt_trace_id",
            error_prefix=error_prefix,
        )
        health_score = payload.get("health_score", 1.0)
        if not isinstance(health_score, (int, float)) or isinstance(health_score, bool):
            raise ValueError(f"{error_prefix}: health_score must be numeric")
        current_load = payload.get("current_load", 0)
        if not isinstance(current_load, int) or current_load < 0:
            raise ValueError(f"{error_prefix}: current_load must be a non-negative integer")
        return cls(
            agent_id=_require_non_empty_str(
                payload.get("agent_id"),
                field_name="agent_id",
                error_prefix=error_prefix,
            ),
            agent_name=_require_non_empty_str(
                payload.get("agent_name"),
                field_name="agent_name",
                error_prefix=error_prefix,
            ),
            agent_role=_require_non_empty_str(
                payload.get("agent_role"),
                field_name="agent_role",
                error_prefix=error_prefix,
            ),
            task_type=_require_non_empty_str(
                payload.get("task_type"),
                field_name="task_type",
                error_prefix=error_prefix,
            ),
            model=model,
            provider=provider,
            reason=_require_non_empty_str(
                payload.get("reason"),
                field_name="reason",
                error_prefix=error_prefix,
            ),
            fallback_chain=list(fallback_chain),
            attempt_index=attempt_index,
            trace_id=trace_id,
            parent_trace_id=parent_trace_id,
            previous_attempt_trace_id=previous_attempt_trace_id,
            inherited_from_parent=inherited_from_parent,
            health_score=float(health_score),
            current_load=current_load,
            created_at=str(payload.get("created_at") or utc_now()),
        )

    def derive_child(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        trace_id: str | None = None,
        fallback_chain: list[str] | None = None,
        attempt_index: int | None = None,
        inherited_from_parent: bool = True,
        reason: str | None = None,
    ) -> "RouteDecision":
        """Create a child route decision linked to this parent execution."""
        child_provider = self.provider if provider is None else provider
        child_model = self.model if model is None else model
        child_fallback_chain = list(self.fallback_chain) if fallback_chain is None else list(fallback_chain)
        return RouteDecision(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            agent_role=self.agent_role,
            task_type=self.task_type,
            model=child_model,
            provider=child_provider,
            reason=reason or ("inherit parent route" if inherited_from_parent else "child reroute"),
            fallback_chain=child_fallback_chain,
            attempt_index=self.attempt_index if attempt_index is None else attempt_index,
            trace_id=trace_id or _new_trace_id(),
            parent_trace_id=self.trace_id,
            inherited_from_parent=inherited_from_parent,
            health_score=self.health_score,
            current_load=self.current_load,
        )

    @property
    def route_kind(self) -> str:
        """Return the route lineage classification used in logs."""
        if not self.parent_trace_id:
            return "root"
        return "child_inherit" if self.inherited_from_parent else "child_reroute"

    def derive_attempt(
        self,
        *,
        provider: str | None,
        model: str | None = None,
        trace_id: str | None = None,
        reason: str | None = None,
        fallback_chain: list[str] | None = None,
        attempt_index: int | None = None,
        health_score: float | None = None,
    ) -> "RouteDecision":
        """Create the next execution attempt for the same task lineage."""
        return RouteDecision(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            agent_role=self.agent_role,
            task_type=self.task_type,
            model=model or self.model,
            provider=provider,
            reason=reason or self.reason,
            fallback_chain=list(self.fallback_chain) if fallback_chain is None else list(fallback_chain),
            attempt_index=self.attempt_index + 1 if attempt_index is None else attempt_index,
            trace_id=trace_id or _new_trace_id(),
            parent_trace_id=self.parent_trace_id,
            previous_attempt_trace_id=self.trace_id,
            inherited_from_parent=self.inherited_from_parent,
            health_score=self.health_score if health_score is None else health_score,
            current_load=self.current_load,
        )

    def log_fields(self) -> dict[str, str | int]:
        return {
            "route_kind": self.route_kind,
            "trace_id": self.trace_id,
            "parent_trace_id": self.parent_trace_id or "-",
            "previous_attempt_trace_id": self.previous_attempt_trace_id or "-",
            "provider": self.provider or "default",
            "model": self.model,
            "attempt_index": self.attempt_index,
        }


AgentRouteDecision = RouteDecision


@dataclass
class TaskRecord:
    """A unit of work scheduled by the coordinator."""

    id: str
    workflow_id: str
    title: str
    kind: str
    status: str = "pending"
    assigned_agent_id: str | None = None
    assigned_agent_name: str | None = None
    route: dict[str, Any] | None = None
    summary: str = ""
    result: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AssignmentRecord:
    """Executor assignment for a task."""

    id: str
    workflow_id: str
    task_id: str
    agent_id: str
    agent_name: str
    agent_role: str
    status: str = "assigned"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionLogRecord:
    """Append-only workflow execution log."""

    workflow_id: str
    task_id: str | None
    agent_id: str
    agent_name: str
    agent_role: str
    message_kind: str
    content: str
    is_final: bool = False
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowRecord:
    """Workflow state persisted across restarts."""

    id: str
    session_key: str
    channel: str
    chat_id: str
    user_message: str
    status: str = "running"
    task_ids: list[str] = field(default_factory=list)
    assignment_ids: list[str] = field(default_factory=list)
    shared_board: dict[str, Any] = field(default_factory=dict)
    execution_logs: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    final_response: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
