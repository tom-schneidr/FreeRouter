"""Route selection policy shared by streaming and non-streaming execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.capability_tags import route_satisfies_capabilities
from app.model_catalog import ModelCatalog, ModelRoute
from app.providers.base import ProviderAdapter


@dataclass(frozen=True)
class RouteCandidate:
    route: ModelRoute
    provider: ProviderAdapter | None


def enabled_routes_for_request(
    catalog: ModelCatalog,
    *,
    requested_model: Any,
    required_capabilities: frozenset[str] | None = None,
) -> list[ModelRoute]:
    required = required_capabilities or frozenset()
    return [
        route
        for route in catalog.enabled_routes(requested_model if isinstance(requested_model, str) else None)
        if route_satisfies_capabilities(route, required)
    ]


def configured_provider_names(
    routes: list[ModelRoute],
    providers_by_name: dict[str, ProviderAdapter],
) -> list[str]:
    return sorted(
        {
            route.provider_name
            for route in routes
            if route.provider_name in providers_by_name
            and providers_by_name[route.provider_name].is_configured
        }
    )


def static_route_skip_reason(
    provider: ProviderAdapter | None,
    route: ModelRoute,
    *,
    estimated_prompt_tokens: int,
) -> str | None:
    if provider is None:
        return "unknown_provider"
    if not provider.is_configured:
        return "missing_api_key"
    max_context_tokens = route.context_window or provider.max_context_tokens
    if max_context_tokens is not None and estimated_prompt_tokens > max_context_tokens:
        return "context_window_exceeded"
    return None
