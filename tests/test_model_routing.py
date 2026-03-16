from __future__ import annotations

import pytest

from fubot.config.schema import Config
from fubot.orchestrator.models import ProviderHealthState, RouteDecision
from fubot.orchestrator.router import RoutePlanner
from fubot.orchestrator.store import WorkflowStore


def test_route_planner_classifies_debugging_and_multimodal() -> None:
    planner = RoutePlanner(Config())

    assert planner.classify("Please debug this traceback") == "debugging"
    assert planner.classify("What is in this image?", media=["/tmp/a.png"]) == "multimodal"


def test_route_planner_uses_parallel_profiles_for_coding() -> None:
    config = Config()
    config.orchestration.routing.max_parallel_executors = 2
    planner = RoutePlanner(config)

    profiles = planner.choose_executors("coding", "implement a new feature with tests and refactor the module")

    assert [profile.id for profile in profiles] == ["builder", "verifier"]


def test_route_planner_prefers_healthy_provider_after_cooldown() -> None:
    config = Config()
    profile = config.orchestration.executors["researcher"]
    profile.preferred_providers = ["dashscope", "openrouter"]
    planner = RoutePlanner(
        config,
        {
            "dashscope": {
                "status": "cooldown",
                "failure_count": 5,
                "updated_at": "2026-03-14T00:00:00+00:00",
                "cooldown_until": "9999-01-01T00:00:00+00:00",
            }
        },
    )

    decision = planner.choose_model(profile, "research", "analyze the problem")

    assert decision.provider == "openrouter"
    assert decision.model == config.agents.defaults.model


def test_route_planner_records_failures_in_health_cache() -> None:
    planner = RoutePlanner(Config())

    planner.mark_failure("builder", "dashscope", "timeout")
    planner.mark_failure("builder", "dashscope", "timeout")
    planner.mark_failure("builder", "dashscope", "timeout")

    health = planner.export_health()
    assert health["dashscope"]["failure_count"] == 3
    assert health["dashscope"]["status"] == "cooldown"
    assert "cooldown_until" in health["dashscope"]
    assert planner.health_score("dashscope") == 0.0


def test_route_planner_cooldown_expiry_returns_provider_to_healthy_selection() -> None:
    config = Config()
    profile = config.orchestration.executors["researcher"]
    profile.preferred_providers = ["dashscope", "openrouter"]
    planner = RoutePlanner(
        config,
        {
            "dashscope": {
                "status": "cooldown",
                "failure_count": 2,
                "last_error_kind": "rate_limit",
                "updated_at": "2026-03-14T00:00:00+00:00",
                "cooldown_until": "2026-03-14T00:01:00+00:00",
            }
        },
    )

    decision = planner.choose_model(profile, "research", "analyze the problem")

    assert decision.provider == "dashscope"
    assert planner.export_health()["dashscope"]["status"] == "healthy"


def test_route_planner_avoids_provider_in_cooldown_after_consecutive_failures(tmp_path) -> None:
    config = Config()
    profile = config.orchestration.executors["researcher"]
    profile.preferred_providers = ["dashscope", "openrouter"]
    planner = RoutePlanner(config)
    store = WorkflowStore(tmp_path)

    planner.mark_failure("researcher", "dashscope", "timeout")
    planner.mark_failure("researcher", "dashscope", "timeout")
    planner.mark_failure("researcher", "dashscope", "timeout")
    store.save_health(planner.export_health())
    planner = RoutePlanner(config, store.load_health())

    decision = planner.choose_model(profile, "research", "analyze the problem")

    assert planner.export_health()["dashscope"]["status"] == "cooldown"
    assert decision.provider == "openrouter"


def test_disabled_provider_is_never_selected() -> None:
    config = Config()
    profile = config.orchestration.executors["researcher"]
    profile.preferred_providers = ["dashscope", "openrouter"]
    planner = RoutePlanner(config)

    planner.set_provider_disabled("dashscope")
    decision = planner.choose_model(profile, "research", "analyze the problem")

    assert planner.export_health()["dashscope"]["status"] == "disabled"
    assert decision.provider == "openrouter"


def test_workflow_store_merges_health_without_state_regression(tmp_path) -> None:
    store = WorkflowStore(tmp_path)
    newer = ProviderHealthState(
        provider="dashscope",
        status="cooldown",
        cooldown_until="2026-03-16T10:05:00+00:00",
        last_error_kind="rate_limit",
        failure_count=3,
        updated_at="2026-03-16T10:00:00+00:00",
    )
    older = ProviderHealthState(
        provider="dashscope",
        status="healthy",
        cooldown_until=None,
        last_error_kind=None,
        failure_count=0,
        updated_at="2026-03-16T09:00:00+00:00",
    )

    store.save_health({"dashscope": newer.to_dict()})
    store.save_health({"dashscope": older.to_dict()})
    merged = store.load_health()

    assert merged["dashscope"]["status"] == "cooldown"
    assert merged["dashscope"]["failure_count"] == 3
    assert merged["dashscope"]["last_error_kind"] == "rate_limit"


def test_route_decision_from_dict_rejects_missing_provider() -> None:
    with pytest.raises(ValueError, match="provider must be a non-empty string"):
        RouteDecision.from_dict(
            {
                "agent_id": "builder",
                "agent_name": "Builder",
                "agent_role": "coding",
                "task_type": "coding",
                "model": "qwen-max",
                "reason": "test",
                "fallback_chain": ["dashscope:qwen-max"],
                "attempt_index": 0,
                "trace_id": "trace12345678",
            }
        )


def test_route_decision_from_dict_rejects_missing_trace_id() -> None:
    with pytest.raises(ValueError, match="trace_id must be a non-empty string"):
        RouteDecision.from_dict(
            {
                "agent_id": "builder",
                "agent_name": "Builder",
                "agent_role": "coding",
                "task_type": "coding",
                "model": "qwen-max",
                "provider": "dashscope",
                "reason": "test",
                "fallback_chain": ["dashscope:qwen-max"],
                "attempt_index": 0,
            }
        )


def test_route_decision_from_dict_rejects_invalid_attempt_index() -> None:
    with pytest.raises(ValueError, match="attempt_index must be a non-negative integer"):
        RouteDecision.from_dict(
            {
                "agent_id": "builder",
                "agent_name": "Builder",
                "agent_role": "coding",
                "task_type": "coding",
                "model": "qwen-max",
                "provider": "dashscope",
                "reason": "test",
                "fallback_chain": ["dashscope:qwen-max"],
                "attempt_index": -1,
                "trace_id": "trace12345678",
            }
        )


def test_route_decision_from_dict_rejects_non_list_fallback_chain() -> None:
    with pytest.raises(ValueError, match="fallback_chain must be a list"):
        RouteDecision.from_dict(
            {
                "agent_id": "builder",
                "agent_name": "Builder",
                "agent_role": "coding",
                "task_type": "coding",
                "model": "qwen-max",
                "provider": "dashscope",
                "reason": "test",
                "fallback_chain": "dashscope:qwen-max",
                "attempt_index": 0,
                "trace_id": "trace12345678",
            }
        )


def test_route_decision_from_dict_rejects_inherited_without_parent_trace_id() -> None:
    with pytest.raises(ValueError, match="parent_trace_id is required"):
        RouteDecision.from_dict(
            {
                "agent_id": "builder",
                "agent_name": "Builder",
                "agent_role": "coding",
                "task_type": "coding",
                "model": "qwen-max",
                "provider": "dashscope",
                "reason": "test",
                "fallback_chain": ["dashscope:qwen-max"],
                "attempt_index": 0,
                "trace_id": "trace12345678",
                "inherited_from_parent": True,
            }
        )


def test_route_planner_prevents_fallback_candidate_loops() -> None:
    config = Config()
    planner = RoutePlanner(config)
    decision = RouteDecision(
        agent_id="builder",
        agent_name="Builder",
        agent_role="coding",
        task_type="coding",
        model="qwen-max",
        provider="dashscope",
        reason="test",
        fallback_chain=["dashscope:qwen-max", "openrouter:qwen-max", "dashscope:qwen-max"],
    )

    first = planner.next_fallback_route(
        decision,
        error_kind="rate_limit",
        attempted_candidates=[],
    )
    assert first.next_route is not None
    assert first.next_route.provider == "openrouter"

    second = planner.next_fallback_route(
        first.next_route,
        error_kind="rate_limit",
        attempted_candidates=first.attempted_candidates,
    )
    assert second.next_route is None
    assert "already_attempted" in second.stop_reason


def test_route_planner_honors_max_provider_fallback_attempts() -> None:
    config = Config()
    config.orchestration.routing.max_provider_fallback_attempts = 1
    planner = RoutePlanner(config)
    decision = RouteDecision(
        agent_id="builder",
        agent_name="Builder",
        agent_role="coding",
        task_type="coding",
        model="qwen-max",
        provider="dashscope",
        reason="test",
        fallback_chain=["dashscope:qwen-max", "openrouter:qwen-max"],
    )

    resolution = planner.next_fallback_route(
        decision,
        error_kind="rate_limit",
        attempted_candidates=[],
    )

    assert resolution.next_route is None
    assert "maximum provider fallback attempts reached" in resolution.stop_reason
