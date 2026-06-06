"""Pure domain rules for provider and route availability (no I/O)."""

from __future__ import annotations

from app.state_types import Availability, ProviderQuota, ProviderState, RouteState


def provider_availability(
    quota: ProviderQuota,
    state: ProviderState,
    *,
    estimated_tokens: int = 0,
    now: int,
) -> Availability:
    if state.cooldown_until > now:
        return Availability(
            available=False,
            reason="cooldown",
            retry_after_seconds=state.cooldown_until - now,
        )

    if quota.tokens_per_day is not None:
        if state.tokens_used_today + max(estimated_tokens, 0) > quota.tokens_per_day:
            return Availability(available=False, reason="daily_token_limit")

    if quota.requests_per_day is not None and state.requests_today >= quota.requests_per_day:
        return Availability(available=False, reason="daily_request_limit")

    if (
        quota.requests_per_minute is not None
        and state.requests_this_minute >= quota.requests_per_minute
    ):
        retry_after = max(1, 60 - (now - state.minute_window_start))
        return Availability(
            available=False,
            reason="rpm_limit",
            retry_after_seconds=retry_after,
        )

    return Availability(available=True)


def route_availability(
    route_state: RouteState,
    *,
    allow_rate_limit_probe: bool = False,
    now: int,
) -> Availability:
    if route_state.status == "potentially_outdated":
        return Availability(available=False, reason="potentially_outdated")

    if route_state.status in {"rate_limited", "too_slow"}:
        retry_after = max(
            1, max(route_state.rate_limited_until, route_state.next_probe_at) - now
        )
        if not allow_rate_limit_probe or route_state.next_probe_at > now:
            reason = (
                "route_rate_limited"
                if route_state.status == "rate_limited"
                else "route_too_slow"
            )
            return Availability(
                available=False,
                reason=reason,
                retry_after_seconds=retry_after,
            )
        reason = (
            "rate_limit_probe" if route_state.status == "rate_limited" else "too_slow_probe"
        )
        return Availability(available=True, reason=reason)

    return Availability(available=True)
