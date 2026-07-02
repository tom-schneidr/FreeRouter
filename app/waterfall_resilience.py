"""Helpers to keep waterfall routing from failing on transient provider pressure."""

from __future__ import annotations

from typing import Protocol

MAX_WATERFALL_PASSES = 2
WATERFALL_RETRY_DELAY_SECONDS = 2.5

# Per-request: do not try the same route again on a later waterfall pass.
NON_RETRYABLE_ROUTE_REASONS = frozenset(
    {
        "auth_error",
        "context_window_exceeded",
        "invalid_tool_response",
        "missing_api_key",
        "model_not_found",
        "request_too_large",
        "unknown_provider",
    }
)

# Whole-waterfall retry is worthwhile when pressure is likely to clear quickly.
_WATERFALL_RETRY_TRIGGER_REASONS = frozenset(
    {
        "cooldown",
        "provider_429",
        "provider_5xx",
        "route_rate_limited",
        "route_too_slow",
        "rpm_limit",
        "timeout",
    }
)


class _AttemptLike(Protocol):
    status: str
    reason: str | None


def should_retry_waterfall(attempts: list[_AttemptLike]) -> bool:
    if any(attempt.status == "selected" for attempt in attempts):
        return False
    return any(
        attempt.status == "rate_limited" or attempt.reason in _WATERFALL_RETRY_TRIGGER_REASONS
        for attempt in attempts
        if attempt.reason is not None or attempt.status == "rate_limited"
    )


def waterfall_retry_delay_seconds() -> float:
    return WATERFALL_RETRY_DELAY_SECONDS


def route_exhausted_for_request(reason: str | None) -> bool:
    return reason in NON_RETRYABLE_ROUTE_REASONS
