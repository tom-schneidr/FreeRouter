from __future__ import annotations

import asyncio

from app.state import ProviderQuota, StateManager


async def test_try_reserve_request_enforces_rpm_atomically(tmp_path):
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("groq", tokens_per_day=None, requests_per_day=None, requests_per_minute=1)],
    )
    await state.initialize()

    first, second = await asyncio.gather(
        state.try_reserve_request("groq"),
        state.try_reserve_request("groq"),
    )

    assert [first.available, second.available].count(True) == 1
    assert [first.reason, second.reason].count("rpm_limit") == 1


async def test_token_limit_allows_exact_boundary(tmp_path):
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("cerebras", tokens_per_day=10, requests_per_day=None, requests_per_minute=None)],
    )
    await state.initialize()

    availability = await state.try_reserve_request("cerebras", estimated_tokens=10)

    assert availability.available is True


async def test_success_headers_can_set_cooldown(tmp_path):
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("cerebras", tokens_per_day=10, requests_per_day=None, requests_per_minute=None)],
    )
    await state.initialize()

    await state.record_success(
        "cerebras",
        usage={"total_tokens": 1},
        headers={"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset": "60"},
        status_code=200,
    )

    availability = await state.check_available("cerebras")

    assert availability.available is False
    assert availability.reason == "cooldown"


# ── New tests ────────────────────────────────────────────────────────────────


async def test_token_limit_rejects_over_boundary(tmp_path):
    """Requesting more tokens than the daily limit should be rejected."""
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("test", tokens_per_day=100, requests_per_day=None, requests_per_minute=None)],
    )
    await state.initialize()

    availability = await state.try_reserve_request("test", estimated_tokens=101)

    assert availability.available is False
    assert availability.reason == "daily_token_limit"


async def test_daily_request_limit(tmp_path):
    """When requests_per_day is exhausted, the provider should be unavailable."""
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("test", tokens_per_day=None, requests_per_day=2, requests_per_minute=None)],
    )
    await state.initialize()

    r1 = await state.try_reserve_request("test")
    r2 = await state.try_reserve_request("test")
    r3 = await state.try_reserve_request("test")

    assert r1.available is True
    assert r2.available is True
    assert r3.available is False
    assert r3.reason == "daily_request_limit"


async def test_mark_exhausted_sets_cooldown(tmp_path):
    """mark_exhausted should put the provider in cooldown."""
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("test", tokens_per_day=None, requests_per_day=None, requests_per_minute=None)],
    )
    await state.initialize()

    await state.mark_exhausted("test", cooldown_seconds=120)

    availability = await state.check_available("test")
    assert availability.available is False
    assert availability.reason == "cooldown"
    assert availability.retry_after_seconds is not None
    assert availability.retry_after_seconds > 0


async def test_mark_exhausted_respects_retry_after_header(tmp_path):
    """mark_exhausted should use the Retry-After header value if present."""
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("test", tokens_per_day=None, requests_per_day=None, requests_per_minute=None)],
    )
    await state.initialize()

    await state.mark_exhausted("test", headers={"retry-after": "300"})

    availability = await state.check_available("test")
    assert availability.available is False
    assert availability.retry_after_seconds is not None
    assert availability.retry_after_seconds > 200


async def test_record_success_accumulates_tokens(tmp_path):
    """Successful responses should accumulate token usage."""
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("test", tokens_per_day=1000, requests_per_day=None, requests_per_minute=None)],
    )
    await state.initialize()

    await state.record_success("test", usage={"total_tokens": 100}, headers={}, status_code=200)
    await state.record_success("test", usage={"total_tokens": 200}, headers={}, status_code=200)

    provider_state = await state.get_state("test")
    assert provider_state.tokens_used_today == 300


async def test_record_success_with_none_usage(tmp_path):
    """record_success should handle None usage gracefully."""
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("test", tokens_per_day=None, requests_per_day=None, requests_per_minute=None)],
    )
    await state.initialize()

    # Should not raise
    await state.record_success("test", usage=None, headers={}, status_code=200)

    provider_state = await state.get_state("test")
    assert provider_state.tokens_used_today == 0


async def test_get_state_raises_for_unknown_provider(tmp_path):
    """Querying an unknown provider should raise KeyError."""
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("known", tokens_per_day=None, requests_per_day=None, requests_per_minute=None)],
    )
    await state.initialize()

    import pytest
    with pytest.raises(KeyError, match="Unknown provider"):
        await state.get_state("nonexistent")


async def test_no_quota_limits_always_available(tmp_path):
    """A provider with no limits at all should always be available."""
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("unlimited", tokens_per_day=None, requests_per_day=None, requests_per_minute=None)],
    )
    await state.initialize()

    # Reserve many requests
    for _ in range(100):
        result = await state.try_reserve_request("unlimited")
        assert result.available is True


async def test_remaining_requests_zero_sets_cooldown(tmp_path):
    """When x-ratelimit-remaining-requests is 0, provider should enter cooldown."""
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [ProviderQuota("test", tokens_per_day=None, requests_per_day=None, requests_per_minute=None)],
    )
    await state.initialize()

    await state.record_success(
        "test",
        usage={"total_tokens": 1},
        headers={
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-reset": "30",
        },
        status_code=200,
    )

    availability = await state.check_available("test")
    assert availability.available is False
    assert availability.reason == "cooldown"
