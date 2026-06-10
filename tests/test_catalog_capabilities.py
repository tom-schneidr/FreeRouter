from __future__ import annotations

from app.capability_registry import registry_claims_for
from app.capability_tags import (
    CapabilityClaim,
    apply_probe_claims,
    route_satisfies_capabilities,
    should_probe_tool_use,
    tags_to_capabilities,
)
from app.model_catalog import DEFAULT_MODEL_ROUTES


def _registry_tool_use_hint(route) -> bool:
    claims = dict((tag, status) for tag, status, _ in registry_claims_for(route.provider_name, route.model_id))
    return claims.get("tool-use") == "supported"


def test_top_ranked_routes_are_candidates_for_tool_use_probes():
    """OpenClaw tool-use requests should probe high-ranked registry-backed routes first."""
    top_10 = sorted(DEFAULT_MODEL_ROUTES, key=lambda route: route.rank)[:10]
    probe_candidates = [route for route in top_10 if should_probe_tool_use(route)]
    assert len(probe_candidates) >= 6, (
        f"Expected most top-10 routes to be tool-use probe candidates, got "
        f"{[route.route_id for route in top_10 if route not in probe_candidates]}"
    )


def test_gemini_deepseek_and_groq_chat_routes_have_registry_tool_use_hints():
    must_hint_tool_use = [
        ("gemini", "gemini-3.1-pro-preview"),
        ("nvidia", "deepseek-ai/deepseek-v4-pro"),
        ("groq", "openai/gpt-oss-120b"),
        ("cerebras", "llama3.1-8b"),
        ("openrouter", "openrouter/free"),
    ]
    by_target = {(route.provider_name, route.model_id): route for route in DEFAULT_MODEL_ROUTES}
    for target in must_hint_tool_use:
        route = by_target[target]
        assert _registry_tool_use_hint(route), f"{target} missing registry tool-use hint"
        assert "tool-use" not in route.tags, f"{target} should not ship with unconfirmed tool-use tag"


def test_groq_compound_never_gets_tool_use_from_registry():
    for model_id in ("groq/compound", "groq/compound-mini"):
        claims = dict((tag, status) for tag, status, _ in registry_claims_for("groq", model_id))
        assert claims.get("tool-use") == "unsupported"
        route = next(
            route
            for route in DEFAULT_MODEL_ROUTES
            if route.provider_name == "groq" and route.model_id == model_id
        )
        assert "tool-use" not in route.tags
        assert "web-search" in route.tags
        assert not should_probe_tool_use(route)


def test_openrouter_bulk_tool_use_claims_are_registry_hints_not_tags():
    bulk = [
        route
        for route in DEFAULT_MODEL_ROUTES
        if route.provider_name == "openrouter" and route.model_id != "openrouter/free"
    ]
    assert bulk
    hinted = [route for route in bulk if _registry_tool_use_hint(route)]
    assert hinted
    for route in hinted:
        assert "tool-use" not in route.tags
        claim = route.capabilities.get("tool-use")
        assert claim is not None
        assert claim.source == "registry"
        assert claim.status == "supported"


def test_tool_use_routing_requires_probe_or_runtime_confirmation():
    route = next(route for route in DEFAULT_MODEL_ROUTES if route.route_id.startswith("gemini-"))
    assert not route_satisfies_capabilities(route, frozenset({"text", "tool-use"}))

    confirmed = apply_probe_claims(
        route,
        {
            "tool-use": CapabilityClaim(
                tag="tool-use",
                status="supported",
                source="probe",
                confidence="high",
                checked_at=1,
                evidence="echo tool call",
            )
        },
    )
    assert route_satisfies_capabilities(confirmed, frozenset({"text", "tool-use"}))
