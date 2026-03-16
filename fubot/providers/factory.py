"""Runtime provider factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fubot.config.schema import Config
from fubot.providers.base import GenerationSettings, LLMProvider

if TYPE_CHECKING:
    from fubot.orchestrator.models import RouteDecision


class ProviderConfigurationError(RuntimeError):
    """Raised when the requested provider cannot be constructed safely."""


def build_provider(
    config: Config,
    *,
    model: str | None = None,
    provider_name: str | None = None,
) -> LLMProvider:
    """Build a provider from an explicit routing decision or config defaults."""
    from fubot.providers.azure_openai_provider import AzureOpenAIProvider
    from fubot.providers.custom_provider import CustomProvider
    from fubot.providers.litellm_provider import LiteLLMProvider
    from fubot.providers.openai_codex_provider import OpenAICodexProvider
    from fubot.providers.registry import find_by_name

    resolution = config.resolve_provider(model=model, provider_name=provider_name)
    resolved_model = resolution.model
    resolved_provider_name = resolution.provider_name
    provider_config = resolution.provider_config

    if resolved_provider_name == "openai_codex" or resolved_model.startswith(("openai-codex/", "openai_codex/")):
        provider = OpenAICodexProvider(default_model=resolved_model)
    elif resolved_provider_name == "custom":
        provider = CustomProvider(
            api_key=provider_config.api_key if provider_config else "no-key",
            api_base=resolution.api_base or "http://localhost:8000/v1",
            default_model=resolved_model,
        )
    elif resolved_provider_name == "azure_openai":
        if not provider_config or not provider_config.api_key or not resolution.api_base:
            raise ProviderConfigurationError(
                "Azure OpenAI requires both api_key and api_base in providers.azure_openai.",
            )
        provider = AzureOpenAIProvider(
            api_key=provider_config.api_key,
            api_base=resolution.api_base,
            default_model=resolved_model,
        )
    else:
        spec = find_by_name(resolved_provider_name) if resolved_provider_name else None
        if (
            not resolved_model.startswith("bedrock/")
            and not (provider_config and provider_config.api_key)
            and not (spec and (spec.is_oauth or spec.is_local))
        ):
            requested = resolved_provider_name or "resolved provider"
            raise ProviderConfigurationError(
                f"No API key configured for provider '{requested}'.",
            )
        provider = LiteLLMProvider(
            api_key=provider_config.api_key if provider_config else None,
            api_base=resolution.api_base,
            default_model=resolved_model,
            extra_headers=provider_config.extra_headers if provider_config else None,
            provider_name=resolved_provider_name,
        )

    defaults = config.agents.defaults
    provider.generation = GenerationSettings(
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
        reasoning_effort=defaults.reasoning_effort,
    )
    return provider


def build_provider_for_route(
    config: Config,
    route_decision: "RouteDecision",
    *,
    default_provider: LLMProvider | None = None,
    default_model: str | None = None,
    allow_default_provider: bool = False,
) -> LLMProvider:
    """Resolve the concrete provider instance for one explicit route decision."""
    if route_decision.provider:
        return build_provider(
            config,
            model=route_decision.model,
            provider_name=route_decision.provider,
        )
    if allow_default_provider and default_provider is not None and route_decision.model == default_model:
        return default_provider
    raise ProviderConfigurationError(
        f"Route decision {route_decision.trace_id} did not include a provider for model {route_decision.model}",
    )
