from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderQuota:
    """Static rate-limit configuration for a single provider."""

    name: str
    tokens_per_day: int | None
    requests_per_day: int | None
    requests_per_minute: int | None


@dataclass(frozen=True)
class ProviderState:
    """Snapshot of a provider's current usage counters and cooldown status."""

    provider_name: str
    tokens_used_today: int
    requests_today: int
    requests_this_minute: int
    minute_window_start: int
    cooldown_until: int
    day: str


@dataclass(frozen=True)
class Availability:
    """Result of a quota availability check."""

    available: bool
    reason: str | None = None
    retry_after_seconds: int | None = None


@dataclass(frozen=True)
class RouteState:
    """Health state for one provider/model route."""

    route_id: str
    provider_name: str
    model_id: str
    status: str
    status_reason: str | None
    consecutive_failures: int
    rate_limited_until: int
    next_probe_at: int
    updated_at: int
