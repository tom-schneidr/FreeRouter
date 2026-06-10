from __future__ import annotations

from dataclasses import dataclass

from app.capability_registry import registry_claims_for
from app.capability_tags import apply_capability_pipeline, derive_tags_from_capabilities, tags_to_capabilities
from app.model_catalog import ModelRoute, route_id_for


def test_registry_marks_groq_compound_web_search_without_tool_use():
    claims = dict(
        (tag, status) for tag, status, _ in registry_claims_for("groq", "groq/compound")
    )
    assert claims["web-search"] == "supported"
    assert claims["tool-use"] == "unsupported"


def test_apply_capability_pipeline_records_registry_tool_use_hint_without_tag():
    route = ModelRoute(
        route_id=route_id_for("gemini", "gemini-3.1-pro-preview"),
        provider_name="gemini",
        model_id="gemini-3.1-pro-preview",
        display_name="Gemini 3.1 Pro Preview",
        rank=1,
        tags=["text", "reasoning", "vision"],
        capabilities=tags_to_capabilities(["text", "reasoning", "vision"], source="manual"),
        tag_locks=frozenset({"text", "reasoning", "vision"}),
    )
    updated = apply_capability_pipeline(route)
    assert "tool-use" not in updated.tags
    assert updated.capabilities["tool-use"].source == "registry"
    assert updated.capabilities["tool-use"].status == "supported"


def test_probe_confirmed_tool_use_appears_in_tags():
    from app.capability_tags import apply_probe_claims, CapabilityClaim

    route = ModelRoute(
        route_id=route_id_for("gemini", "gemini-3.1-pro-preview"),
        provider_name="gemini",
        model_id="gemini-3.1-pro-preview",
        display_name="Gemini 3.1 Pro Preview",
        rank=1,
        tags=["text"],
        capabilities=tags_to_capabilities(["text"], source="manual"),
    )
    updated = apply_probe_claims(
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
    assert "tool-use" in updated.tags


def test_route_satisfies_capabilities_rejects_unsupported_hard_tag():
    from app.capability_tags import route_satisfies_capabilities
    from app.model_catalog import ModelRoute, route_id_for

    route = ModelRoute(
        route_id=route_id_for("groq", "groq/compound"),
        provider_name="groq",
        model_id="groq/compound",
        display_name="Groq Compound",
        rank=1,
        tags=["text", "web-search"],
        capabilities={
            "text": tags_to_capabilities(["text"])["text"],
            "web-search": tags_to_capabilities(["web-search"])["web-search"],
            "tool-use": tags_to_capabilities(["tool-use"])["tool-use"].__class__(
                tag="tool-use",
                status="unsupported",
                source="registry",
                confidence="high",
                evidence="native search only",
            ),
        },
    )
    assert not route_satisfies_capabilities(route, frozenset({"text", "tool-use"}))


def test_unsupported_capability_claim_is_not_derived_as_tag():
    caps = tags_to_capabilities(["text", "tool-use"], source="manual")
    caps["tool-use"] = caps["tool-use"].__class__(
        tag="tool-use",
        status="unsupported",
        source="registry",
        confidence="high",
        evidence="native search only",
    )
    assert derive_tags_from_capabilities(caps) == ["text"]


@dataclass
class FakeOpenRouterProvider:
    name: str = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"


def test_normalize_route_tool_use_policy_strips_manual_tool_use_tag():
    from app.capability_tags import normalize_route_tool_use_policy

    route = ModelRoute(
        route_id=route_id_for("openrouter", "fake/model:free"),
        provider_name="openrouter",
        model_id="fake/model:free",
        display_name="Fake",
        rank=99,
        tags=["text", "tool-use"],
        capabilities=tags_to_capabilities(["text", "tool-use"], source="manual"),
        tag_locks=frozenset({"text", "tool-use"}),
    )
    updated = normalize_route_tool_use_policy(route)
    assert "tool-use" not in updated.tags
    assert "tool-use" not in updated.tag_locks


def test_openrouter_discovered_route_does_not_blanket_tag_tools():
    from app.model_discovery import route_from_catalog_item

    route = route_from_catalog_item(
        FakeOpenRouterProvider(),
        {
            "id": "unknown/small-model:free",
            "name": "Unknown Small Model",
            "architecture": {"modality": "text->text"},
            "pricing": {"prompt": "0", "completion": "0"},
        },
    )
    assert route is not None
    assert "text" in route.tags
    assert "tool-use" not in route.tags
    assert "web-search" not in route.tags
