"""Tests for provider and route availability domain rules."""

from __future__ import annotations

from app.state_rules import provider_availability, route_availability
from app.state_types import ProviderQuota, ProviderState, RouteState


def test_provider_availability_respects_daily_token_limit():
    quota = ProviderQuota("groq", tokens_per_day=100, requests_per_day=None, requests_per_minute=None)
    state = ProviderState("groq", 95, 0, 0, 0, 0, "2026-06-06")
    result = provider_availability(quota, state, estimated_tokens=10, now=1)
    assert result.available is False
    assert result.reason == "daily_token_limit"


def test_route_availability_marks_potentially_outdated():
    route_state = RouteState(
        route_id="r1",
        provider_name="groq",
        model_id="m",
        status="potentially_outdated",
        status_reason=None,
        consecutive_failures=0,
        rate_limited_until=0,
        next_probe_at=0,
        updated_at=0,
    )
    result = route_availability(route_state, now=100)
    assert result.available is False
    assert result.reason == "potentially_outdated"
