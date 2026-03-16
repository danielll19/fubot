"""Task classification and executor/model selection."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from fubot.config.schema import AgentProfile, Config
from fubot.orchestrator.models import ProviderHealthState, RouteDecision, utc_now
from fubot.providers.factory import ProviderConfigurationError

_TASK_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("debugging", ("traceback", "stack trace", "报错", "error", "exception", "bug", "debug")),
    ("review", ("review", "code review", "审查")),
    ("testing", ("test", "pytest", "unit test", "集成测试")),
    ("refactor", ("refactor", "重构", "cleanup")),
    ("coding", ("implement", "write code", "build", "feature", "代码", "开发")),
    ("research", ("research", "investigate", "analyze", "调研", "分析")),
    ("search", ("search", "look up", "find online", "查一下")),
    ("writing", ("write", "draft", "rewrite", "润色")),
    ("ops", ("deploy", "docker", "service", "ops", "运维")),
    ("scheduling", ("schedule", "cron", "remind", "定时")),
    ("communication", ("reply", "email", "message", "沟通")),
)


class ProviderExecutionError(RuntimeError):
    """Raised when a routed provider execution fails and may trigger fallback."""

    def __init__(self, message: str, *, error_kind: str = "provider_error"):
        super().__init__(message)
        self.error_kind = error_kind


@dataclass(frozen=True)
class FallbackResolution:
    """Result of evaluating whether a routed execution may fallback."""

    next_route: RouteDecision | None
    stop_reason: str
    attempted_candidates: list[str]
    candidate_states: list[dict[str, str]]


def classify_provider_error(error: BaseException | str | None) -> str | None:
    """Normalize provider failures into health-state error kinds."""
    if error is None:
        return None
    if isinstance(error, ProviderExecutionError):
        return error.error_kind
    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return "timeout"
    if isinstance(error, ProviderConfigurationError):
        return "configuration"
    text = str(error).lower()
    if not text:
        return None
    if "manual_disabled" in text:
        return "manual_disabled"
    if "quota" in text:
        return "quota"
    if "429" in text or "rate limit" in text or "too many requests" in text:
        return "rate_limit"
    if "tls" in text or "ssl" in text or "certificate" in text:
        return "tls"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if "connection" in text or "connect" in text or "dns" in text or "network" in text:
        return "connection"
    if any(marker in text for marker in ("500", "502", "503", "504", "server error", "bad gateway", "gateway timeout", "service unavailable")):
        return "server_error"
    if "api key" in text or "providerconfigurationerror" in text or "requires both api_key and api_base" in text:
        return "configuration"
    return None


class RoutePlanner:
    """Heuristic planner that stays config-driven and auditable."""

    def __init__(self, config: Config, health_cache: dict[str, Any] | None = None):
        self.config = config
        self._health_cache: dict[str, ProviderHealthState] = {}
        self._executor_load = defaultdict(int)
        self._executor_failures = defaultdict(int)
        self.replace_health(health_cache or {})

    def classify(self, content: str, media: list[str] | None = None) -> str:
        if media:
            return "multimodal"
        lower = content.lower()
        for task_type, patterns in _TASK_PATTERNS:
            if any(pattern in lower for pattern in patterns):
                return task_type
        return "communication"

    def choose_executors(self, task_type: str, content: str) -> list[AgentProfile]:
        overrides = self.config.orchestration.routing.task_executor_overrides.get(task_type, [])
        profiles = [self.config.get_profile(profile_id) for profile_id in overrides]
        chosen = [profile for profile in profiles if profile is not None]
        if not chosen:
            chosen = self.config.get_executor_profiles()[:1]
        if task_type not in self.config.orchestration.routing.prefer_parallel_for or len(content) < 32:
            return chosen[:1]
        limit = max(1, self.config.orchestration.routing.max_parallel_executors)
        return chosen[:limit]

    def choose_model(self, profile: AgentProfile, task_type: str, content: str) -> RouteDecision:
        candidates = [profile.default_model, *profile.candidate_models, self.config.agents.defaults.model]
        model = ""
        seen: set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            if profile.allowed_models and candidate not in profile.allowed_models:
                continue
            seen.add(candidate)
            model = candidate
            break
        if not model:
            model = self.config.agents.defaults.model

        provider_candidates = self._provider_candidates(profile, model)
        provider = self._choose_provider(provider_candidates)
        provider_state = self.get_provider_state(provider) if provider else None
        reason_bits = [
            f"task={task_type}",
            f"role={profile.role}",
            f"tool_allowlist={len(profile.tool_allowlist)}",
        ]
        if profile.preferred_providers:
            reason_bits.append(f"preferred={','.join(profile.preferred_providers)}")
        if provider_state is not None:
            reason_bits.append(f"provider_status={provider_state.status}")
        elif provider_candidates:
            reason_bits.append("provider_status=unavailable")
        fallback_chain = [f"{candidate}:{model}" for candidate in provider_candidates] or [model]
        return RouteDecision(
            agent_id=profile.id,
            agent_name=profile.name or profile.id,
            agent_role=profile.role,
            task_type=task_type,
            model=model,
            provider=provider,
            reason="; ".join(reason_bits),
            fallback_chain=fallback_chain,
            health_score=self.health_score(provider) if provider else (0.0 if provider_candidates else 1.0),
            current_load=self._executor_load[profile.id],
        )

    def _provider_candidates(self, profile: AgentProfile, model: str) -> list[str]:
        providers = [
            *profile.preferred_providers,
            *(profile.allowed_providers or []),
        ]
        detected = self.config.resolve_provider(model=model).provider_name
        if detected:
            providers.append(detected)
        providers = [provider for provider in providers if provider]
        seen: set[str] = set()
        ordered: list[str] = []
        for provider in providers:
            if provider in seen:
                continue
            seen.add(provider)
            ordered.append(provider)
        available: list[tuple[float, int, str]] = []
        blocked: list[str] = []
        for index, provider in enumerate(ordered):
            state = self.get_provider_state(provider)
            if state.status == "disabled":
                blocked.append(provider)
                continue
            score = self.health_score(provider)
            if score > 0:
                available.append((score, index, provider))
            else:
                blocked.append(provider)
        available.sort(key=lambda item: (-item[0], item[1]))
        return [provider for _, _, provider in available] + blocked

    def _choose_provider(self, providers: list[str]) -> str | None:
        for provider in providers:
            if self.health_score(provider) > 0:
                return provider
        return None

    def candidate_states(self, fallback_chain: list[str], *, default_model: str) -> list[dict[str, str]]:
        """Return auditable provider/model availability for one fallback chain."""
        states: list[dict[str, str]] = []
        seen: set[str] = set()
        for candidate in fallback_chain:
            provider_name, model = self._parse_candidate(candidate, default_model=default_model)
            candidate_key = self._candidate_key(provider_name, model)
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            provider_state = self.get_provider_state(provider_name) if provider_name else None
            status = provider_state.status if provider_state is not None else "legacy_default"
            states.append(
                {
                    "candidate": candidate_key,
                    "provider": provider_name or "default",
                    "model": model,
                    "status": status,
                    "cooldown_until": provider_state.cooldown_until or "-" if provider_state is not None else "-",
                    "last_error_kind": provider_state.last_error_kind or "-" if provider_state is not None else "-",
                }
            )
        return states

    def fallback_failure_message(
        self,
        decision: RouteDecision,
        *,
        attempted_candidates: list[str],
        stop_reason: str,
    ) -> str:
        candidate_states = self.candidate_states(decision.fallback_chain, default_model=decision.model)
        attempted = ", ".join(attempted_candidates) if attempted_candidates else "-"
        state_bits = ", ".join(
            f"{state['candidate']}[{state['status']}]"
            for state in candidate_states
        ) or "-"
        return (
            f"Provider fallback exhausted trace_id={decision.trace_id} "
            f"attempted={attempted} "
            f"candidates={state_bits} "
            f"reason={stop_reason}"
        )

    def get_provider_state(self, provider_name: str | None) -> ProviderHealthState | None:
        """Return the normalized health state for one provider."""
        if not provider_name:
            return None
        current = self._health_cache.get(provider_name, ProviderHealthState(provider=provider_name))
        normalized = current.normalized()
        if normalized != current:
            self._set_provider_state(normalized, transition_reason="cooldown_expired")
        return self._health_cache.get(provider_name, normalized)

    def health_score(self, provider_name: str | None) -> float:
        if not provider_name:
            return 1.0
        state = self.get_provider_state(provider_name)
        if state is None:
            return 1.0
        if state.status == "healthy":
            return 1.0
        if state.status == "degraded":
            return max(0.2, 0.7 - min(state.failure_count, 4) * 0.1)
        if state.status == "cooldown":
            return 0.0
        return -1.0

    def mark_success(self, profile_id: str, provider_name: str | None) -> ProviderHealthState | None:
        self._executor_failures[profile_id] = 0
        if not provider_name:
            return None
        current = self.get_provider_state(provider_name)
        if current and current.status == "disabled" and current.last_error_kind == "manual_disabled":
            return current
        next_state = ProviderHealthState(
            provider=provider_name,
            status="healthy",
            cooldown_until=None,
            last_error_kind=None,
            failure_count=0,
            updated_at=utc_now(),
        )
        return self._set_provider_state(next_state, transition_reason="success")

    def mark_failure(
        self,
        profile_id: str,
        provider_name: str | None,
        error_kind: str = "provider_error",
    ) -> ProviderHealthState | None:
        self._executor_failures[profile_id] += 1
        if not provider_name:
            return None
        current = self.get_provider_state(provider_name) or ProviderHealthState(provider=provider_name)
        next_state = self._transition_failure_state(current, error_kind)
        return self._set_provider_state(next_state, transition_reason=f"failure:{error_kind}")

    def set_provider_disabled(self, provider_name: str, disabled: bool = True) -> ProviderHealthState:
        """Manually disable or re-enable a provider."""
        if disabled:
            state = ProviderHealthState(
                provider=provider_name,
                status="disabled",
                cooldown_until=None,
                last_error_kind="manual_disabled",
                failure_count=max((self.get_provider_state(provider_name) or ProviderHealthState(provider=provider_name)).failure_count, 1),
                updated_at=utc_now(),
            )
            return self._set_provider_state(state, transition_reason="manual_disable")
        state = ProviderHealthState(
            provider=provider_name,
            status="healthy",
            cooldown_until=None,
            last_error_kind=None,
            failure_count=0,
            updated_at=utc_now(),
        )
        return self._set_provider_state(state, transition_reason="manual_enable")

    def next_fallback_route(
        self,
        decision: RouteDecision,
        *,
        error_kind: str,
        error_detail: str = "",
        attempted_candidates: list[str] | None = None,
    ) -> FallbackResolution:
        """Pick the next provider/model from the routed fallback chain."""
        attempted = list(dict.fromkeys(attempted_candidates or []))
        current_key = self._candidate_key(decision.provider, decision.model)
        if current_key and current_key not in attempted:
            attempted.append(current_key)
        candidate_states = self.candidate_states(decision.fallback_chain, default_model=decision.model)
        if not self.config.orchestration.routing.enable_provider_fallback:
            return FallbackResolution(
                next_route=None,
                stop_reason="provider fallback disabled",
                attempted_candidates=attempted,
                candidate_states=candidate_states,
            )
        if decision.attempt_index + 1 >= self.config.orchestration.routing.max_provider_fallback_attempts:
            return FallbackResolution(
                next_route=None,
                stop_reason=(
                    "maximum provider fallback attempts reached "
                    f"({self.config.orchestration.routing.max_provider_fallback_attempts})"
                ),
                attempted_candidates=attempted,
                candidate_states=candidate_states,
            )
        current_index = -1
        for index, candidate in enumerate(decision.fallback_chain):
            if candidate == current_key:
                current_index = index
                break
        skipped_candidates: list[str] = []
        for candidate in decision.fallback_chain[current_index + 1:]:
            provider_name, model = self._parse_candidate(candidate, default_model=decision.model)
            candidate_key = self._candidate_key(provider_name, model)
            if candidate_key in attempted:
                skipped_candidates.append(f"{candidate_key}[already_attempted]")
                continue
            if provider_name is not None and self.health_score(provider_name) <= 0:
                state = self.get_provider_state(provider_name)
                skipped_candidates.append(
                    f"{candidate_key}[{state.status if state is not None else 'unavailable'}]"
                )
                continue
            from_ref = current_key or decision.model
            to_ref = candidate_key or model
            trimmed_detail = error_detail.strip()
            if len(trimmed_detail) > 120:
                trimmed_detail = trimmed_detail[:117] + "..."
            reason = (
                f"{decision.reason}; fallback attempt {decision.attempt_index + 1} due to {error_kind}: "
                f"{from_ref} -> {to_ref}"
            )
            if trimmed_detail:
                reason += f" ({trimmed_detail})"
            return FallbackResolution(
                next_route=decision.derive_attempt(
                    provider=provider_name,
                    model=model,
                    reason=reason,
                    fallback_chain=decision.fallback_chain,
                    health_score=self.health_score(provider_name),
                ),
                stop_reason="fallback available",
                attempted_candidates=attempted,
                candidate_states=candidate_states,
            )
        stop_reason = "all fallback candidates unavailable"
        if skipped_candidates:
            stop_reason += f" ({', '.join(skipped_candidates)})"
        return FallbackResolution(
            next_route=None,
            stop_reason=stop_reason,
            attempted_candidates=attempted,
            candidate_states=candidate_states,
        )

    def _transition_failure_state(self, current: ProviderHealthState, error_kind: str) -> ProviderHealthState:
        """Apply the unified health-state machine for one provider failure."""
        now = utc_now()
        failure_count = current.failure_count + 1
        if error_kind in {"manual_disabled"}:
            return ProviderHealthState(
                provider=current.provider,
                status="disabled",
                cooldown_until=None,
                last_error_kind=error_kind,
                failure_count=failure_count,
                updated_at=now,
            )
        if error_kind in {"rate_limit", "quota"}:
            cooldown_until = (
                datetime.now(timezone.utc)
                + timedelta(seconds=self.config.orchestration.routing.provider_cooldown_seconds)
            ).isoformat()
            return ProviderHealthState(
                provider=current.provider,
                status="cooldown",
                cooldown_until=cooldown_until,
                last_error_kind=error_kind,
                failure_count=failure_count,
                updated_at=now,
            )
        if error_kind in {"configuration"}:
            return ProviderHealthState(
                provider=current.provider,
                status="disabled",
                cooldown_until=None,
                last_error_kind=error_kind,
                failure_count=failure_count,
                updated_at=now,
            )
        status = "degraded"
        cooldown_until = None
        if failure_count >= 3:
            status = "cooldown"
            cooldown_until = (
                datetime.now(timezone.utc)
                + timedelta(seconds=self.config.orchestration.routing.provider_cooldown_seconds)
            ).isoformat()
        return ProviderHealthState(
            provider=current.provider,
            status=status,
            cooldown_until=cooldown_until,
            last_error_kind=error_kind,
            failure_count=failure_count,
            updated_at=now,
        )

    def _set_provider_state(self, state: ProviderHealthState, *, transition_reason: str) -> ProviderHealthState:
        """Persist a normalized provider state and log any transition."""
        previous = self._health_cache.get(state.provider)
        self._health_cache[state.provider] = state
        if previous != state:
            logger.info(
                "Provider health transition provider={} from_status={} to_status={} failures={} cooldown_until={} error_kind={} reason={}",
                state.provider,
                previous.status if previous is not None else "unknown",
                state.status,
                state.failure_count,
                state.cooldown_until or "-",
                state.last_error_kind or "-",
                transition_reason,
            )
        return state

    @staticmethod
    def _candidate_key(provider_name: str | None, model: str) -> str:
        return f"{provider_name}:{model}" if provider_name else model

    @staticmethod
    def _parse_candidate(candidate: str, *, default_model: str) -> tuple[str | None, str]:
        if ":" not in candidate:
            return None, candidate or default_model
        provider_name, model = candidate.split(":", 1)
        return provider_name or None, model or default_model

    def begin_load(self, profile_id: str) -> None:
        self._executor_load[profile_id] += 1

    def end_load(self, profile_id: str) -> None:
        self._executor_load[profile_id] = max(0, self._executor_load[profile_id] - 1)

    def export_health(self) -> dict[str, Any]:
        for provider_name in list(self._health_cache):
            self.get_provider_state(provider_name)
        return {provider: state.to_dict() for provider, state in self._health_cache.items()}

    @staticmethod
    def decision_to_dict(decision: RouteDecision) -> dict[str, Any]:
        return decision.to_dict()

    def replace_health(self, health_cache: dict[str, Any]) -> None:
        """Refresh the in-memory health cache from persisted state."""
        self._health_cache = {
            provider: ProviderHealthState.from_dict(provider, payload).normalized()
            for provider, payload in (health_cache or {}).items()
            if provider
        }
