"""Route selection policy shared by streaming and non-streaming execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent_profiles import filter_routes_for_profile, is_agent_profile
from app.capability_tags import route_satisfies_capabilities
from app.model_catalog import ModelCatalog, ModelRoute
from app.providers.base import ProviderAdapter
from app.sentinel_types import SentinelEvaluation


@dataclass(frozen=True)
class RouteCandidate:
    route: ModelRoute
    provider: ProviderAdapter | None


def enabled_routes_for_request(
    catalog: ModelCatalog,
    *,
    requested_model: Any,
    required_capabilities: frozenset[str] | None = None,
    avoid_capabilities: frozenset[str] | None = None,
    allow_unconfirmed_tool_use_fallback: bool = False,
    sentinel_evaluations: dict[str, SentinelEvaluation] | None = None,
) -> list[ModelRoute]:
    required = required_capabilities or frozenset()
    requested_model_value = requested_model if isinstance(requested_model, str) else None
    base_routes = list(catalog.enabled_routes(requested_model_value))
    routes = [
        route
        for route in base_routes
        if route_satisfies_capabilities(route, required)
    ]
    if is_agent_profile(requested_model_value):
        routes = filter_routes_for_profile(
            routes,
            sentinel_evaluations or {},
            requested_model_value,
        )
    if allow_unconfirmed_tool_use_fallback and "tool-use" in required:
        confirmed_ids = {route.route_id for route in routes}
        routes.extend(
            route
            for route in base_routes
            if route.route_id not in confirmed_ids
            and route_satisfies_capabilities(
                route,
                required,
                allow_unconfirmed_tool_use=True,
            )
        )
    avoided = avoid_capabilities or frozenset()
    if (
        not avoided
        or is_agent_profile(requested_model_value)
        or (requested_model_value and requested_model_value != "auto")
    ):
        return routes
    preferred = [route for route in routes if not avoided.intersection(route.tags)]
    if not preferred:
        return routes
    fallback = [route for route in routes if route not in preferred]
    return preferred + fallback


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


def effective_context_limit(
    route: ModelRoute,
    provider: ProviderAdapter | None,
) -> int | None:
    limits = [
        value
        for value in (
            route.context_window,
            provider.max_context_tokens if provider is not None else None,
        )
        if value is not None
    ]
    if not limits:
        return None
    return min(limits)


def static_route_skip_reason(
    provider: ProviderAdapter | None,
    route: ModelRoute,
    *,
    estimated_prompt_tokens: int,
    estimated_total_tokens: int | None = None,
) -> str | None:
    if provider is None:
        return "unknown_provider"
    if not provider.is_configured:
        return "missing_api_key"
    max_context_tokens = effective_context_limit(route, provider)
    budget_tokens = estimated_total_tokens or estimated_prompt_tokens
    if max_context_tokens is not None and budget_tokens > max_context_tokens:
        return "context_window_exceeded"
    return None
