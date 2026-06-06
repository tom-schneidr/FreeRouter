"""Tests for pure routing policy helpers."""

from __future__ import annotations

from dataclasses import dataclass

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


def test_enabled_routes_for_request_filters_by_tag(tmp_path):
    catalog = ModelCatalog(str(tmp_path / "catalog.json"))
    catalog._routes = [
        ModelRoute(
            route_id="a",
            provider_name="groq",
            model_id="m1",
            display_name="A",
            rank=1,
            tags=["web-search"],
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
    routes = enabled_routes_for_request(catalog, requested_model="auto", required_tag="web-search")
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
