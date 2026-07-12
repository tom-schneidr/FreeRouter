"""Tests for pure routing policy helpers."""

from __future__ import annotations

from dataclasses import dataclass

from app.capability_tags import CapabilityClaim, tags_to_capabilities
from app.model_catalog import ModelCatalog, ModelRoute
from app.routing_policy import (
    configured_provider_names,
    enabled_routes_for_request,
    static_route_skip_reason,
)


@dataclass
class _FakeProvider:
    name: str
    configured: bool
    max_context_tokens: int | None = 8192

    @property
    def is_configured(self) -> bool:
        return self.configured


def test_static_route_skip_reason_flags_missing_api_key():
    route = ModelRoute(
        route_id="groq-test",
        provider_name="groq",
        model_id="test",
        display_name="Test",
        rank=1,
    )
    assert static_route_skip_reason(
        _FakeProvider(name="groq", configured=False), route, estimated_prompt_tokens=10
    ) == "missing_api_key"


def test_static_route_skip_reason_uses_tightest_context_limit():
    route = ModelRoute(
        route_id="groq-test",
        provider_name="groq",
        model_id="test",
        display_name="Test",
        rank=1,
        context_window=131_072,
    )
    provider = _FakeProvider(name="groq", configured=True, max_context_tokens=8192)
    assert (
        static_route_skip_reason(
            provider,
            route,
            estimated_prompt_tokens=1000,
            estimated_total_tokens=9000,
        )
        == "context_window_exceeded"
    )


def test_enabled_routes_for_request_filters_by_single_capability(tmp_path):
    catalog = ModelCatalog(str(tmp_path / "catalog.json"))
    catalog._routes = [
        ModelRoute(
            route_id="a",
            provider_name="groq",
            model_id="m1",
            display_name="A",
            rank=1,
            tags=["text", "web-search"],
        ),
        ModelRoute(
            route_id="b",
            provider_name="groq",
            model_id="m2",
            display_name="B",
            rank=2,
            tags=["text"],
        ),
    ]
    routes = enabled_routes_for_request(
        catalog,
        requested_model="auto",
        required_capabilities=frozenset({"text", "web-search"}),
    )
    assert [route.route_id for route in routes] == ["a"]


def test_enabled_routes_for_request_filters_by_multiple_capabilities(tmp_path):
    catalog = ModelCatalog(str(tmp_path / "catalog.json"))
    catalog._routes = [
        ModelRoute(
            route_id="a",
            provider_name="groq",
            model_id="m1",
            display_name="A",
            rank=1,
            tags=["text", "vision", "tool-use"],
        ),
        ModelRoute(
            route_id="b",
            provider_name="groq",
            model_id="m2",
            display_name="B",
            rank=2,
            tags=["text", "vision"],
        ),
        ModelRoute(
            route_id="c",
            provider_name="groq",
            model_id="m3",
            display_name="C",
            rank=3,
            tags=["text", "tool-use"],
        ),
    ]
    routes = enabled_routes_for_request(
        catalog,
        requested_model="auto",
        required_capabilities=frozenset({"text", "vision", "tool-use"}),
    )
    assert [route.route_id for route in routes] == ["a"]


def test_enabled_routes_can_append_unconfirmed_tool_use_fallbacks_after_confirmed(tmp_path):
    catalog = ModelCatalog(str(tmp_path / "catalog.json"))
    catalog._routes = [
        ModelRoute(
            route_id="hint",
            provider_name="groq",
            model_id="m1",
            display_name="Hint",
            rank=1,
            tags=["text"],
            capabilities={
                **tags_to_capabilities(["text"], source="manual"),
                "tool-use": CapabilityClaim(
                    tag="tool-use",
                    status="supported",
                    source="registry",
                    confidence="medium",
                    checked_at=1,
                    evidence="registry hint",
                ),
            },
        ),
        ModelRoute(
            route_id="confirmed",
            provider_name="groq",
            model_id="m2",
            display_name="Confirmed",
            rank=2,
            tags=["text", "tool-use"],
            capabilities={
                **tags_to_capabilities(["text"], source="manual"),
                "tool-use": CapabilityClaim(
                    tag="tool-use",
                    status="supported",
                    source="probe",
                    confidence="high",
                    checked_at=1,
                    evidence="probe passed",
                ),
            },
        ),
    ]

    strict = enabled_routes_for_request(
        catalog,
        requested_model="auto",
        required_capabilities=frozenset({"text", "tool-use"}),
    )
    fallback = enabled_routes_for_request(
        catalog,
        requested_model="auto",
        required_capabilities=frozenset({"text", "tool-use"}),
        allow_unconfirmed_tool_use_fallback=True,
    )

    assert [route.route_id for route in strict] == ["confirmed"]
    assert [route.route_id for route in fallback] == ["confirmed", "hint"]


def test_enabled_routes_for_request_with_no_required_capabilities_returns_all_enabled(tmp_path):
    catalog = ModelCatalog(str(tmp_path / "catalog.json"))
    catalog._routes = [
        ModelRoute(
            route_id="a",
            provider_name="groq",
            model_id="m1",
            display_name="A",
            rank=1,
            tags=["text"],
        ),
        ModelRoute(
            route_id="b",
            provider_name="groq",
            model_id="m2",
            display_name="B",
            rank=2,
            tags=["text", "vision"],
        ),
    ]
    routes = enabled_routes_for_request(catalog, requested_model="auto")
    assert [route.route_id for route in routes] == ["a", "b"]


def test_enabled_routes_for_request_avoids_capabilities_but_keeps_fallback(tmp_path):
    catalog = ModelCatalog(str(tmp_path / "catalog.json"))
    catalog._routes = [
        ModelRoute(
            route_id="tool",
            provider_name="groq",
            model_id="m1",
            display_name="Tool",
            rank=1,
            tags=["text", "tool-use"],
        ),
        ModelRoute(
            route_id="normal",
            provider_name="groq",
            model_id="m2",
            display_name="Normal",
            rank=2,
            tags=["text"],
        ),
    ]
    routes = enabled_routes_for_request(
        catalog,
        requested_model="auto",
        required_capabilities=frozenset({"text"}),
        avoid_capabilities=frozenset({"tool-use"}),
    )
    assert [route.route_id for route in routes] == ["normal", "tool"]


def test_enabled_routes_for_request_keeps_avoided_routes_when_no_normal_fallback(tmp_path):
    catalog = ModelCatalog(str(tmp_path / "catalog.json"))
    catalog._routes = [
        ModelRoute(
            route_id="tool",
            provider_name="groq",
            model_id="m1",
            display_name="Tool",
            rank=1,
            tags=["text", "tool-use"],
        ),
    ]
    routes = enabled_routes_for_request(
        catalog,
        requested_model="auto",
        required_capabilities=frozenset({"text"}),
        avoid_capabilities=frozenset({"tool-use"}),
    )
    assert [route.route_id for route in routes] == ["tool"]


def test_enabled_routes_for_request_respects_explicit_model_selection(tmp_path):
    catalog = ModelCatalog(str(tmp_path / "catalog.json"))
    catalog._routes = [
        ModelRoute(
            route_id="a",
            provider_name="groq",
            model_id="m1",
            display_name="A",
            rank=1,
            tags=["text", "json-schema"],
        ),
        ModelRoute(
            route_id="b",
            provider_name="groq",
            model_id="m2",
            display_name="B",
            rank=2,
            tags=["text"],
        ),
    ]
    routes = enabled_routes_for_request(
        catalog,
        requested_model="a",
        required_capabilities=frozenset({"text", "json-schema"}),
    )
    assert [route.route_id for route in routes] == ["a"]


def test_configured_provider_names_only_includes_configured(tmp_path):
    route = ModelRoute(
        route_id="groq-test",
        provider_name="groq",
        model_id="test",
        display_name="Test",
        rank=1,
    )
    providers = {
        "groq": _FakeProvider(name="groq", configured=True),
        "openrouter": _FakeProvider(name="openrouter", configured=False),
    }
    assert configured_provider_names([route], providers) == ["groq"]
