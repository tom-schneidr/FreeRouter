from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.model_catalog import ModelCatalog
from app.providers.base import ProviderError, ProviderRateLimited, ProviderResponse
from app.router import NoProviderAvailable, WaterfallRouter
from app.state import ProviderQuota, StateManager


class FakeProvider:
    def __init__(
        self,
        name: str,
        *,
        response: dict[str, Any] | None = None,
        error: Exception | None = None,
        max_context_tokens: int | None = 8192,
        configured: bool = True,
    ) -> None:
        self.name = name
        self.api_key = "test-key" if configured else None
        self.max_context_tokens = max_context_tokens
        self.response = response or {
            "id": f"chatcmpl-{name}",
            "object": "chat.completion",
            "choices": [],
            "usage": {"total_tokens": 2},
        }
        self.error = error
        self.calls = 0
        self._configured = configured

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def chat_completion(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, Any],
        target_model: str | None = None,
    ) -> ProviderResponse:
        self.calls += 1
        self.target_model = target_model
        if self.error:
            raise self.error
        return ProviderResponse(self.name, 200, {}, dict(self.response))


def _payload() -> dict[str, Any]:
    return {"model": "auto", "messages": [{"role": "user", "content": "hello"}]}


async def _state(tmp_path, providers=None) -> StateManager:
    providers = providers or [
        ProviderQuota("primary", tokens_per_day=None, requests_per_day=None, requests_per_minute=30),
        ProviderQuota("fallback", tokens_per_day=None, requests_per_day=None, requests_per_minute=30),
    ]
    state = StateManager(str(tmp_path / "state.sqlite3"), providers)
    await state.initialize()
    return state


def _catalog(tmp_path) -> ModelCatalog:
    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.replace_routes(
        [
            {
                "route_id": "primary-test",
                "provider_name": "primary",
                "model_id": "primary/model",
                "display_name": "Primary Model",
                "rank": 1,
                "enabled": True,
                "context_window": 8192,
            },
            {
                "route_id": "fallback-test",
                "provider_name": "fallback",
                "model_id": "fallback/model",
                "display_name": "Fallback Model",
                "rank": 2,
                "enabled": True,
                "context_window": 8192,
            },
        ]
    )
    return catalog


# ── Existing tests (preserved) ──────────────────────────────────────────────


async def test_router_skips_local_cooldown_and_preserves_provider_body(tmp_path):
    state = await _state(tmp_path)
    await state.mark_exhausted("primary", cooldown_seconds=60, status_code=429)
    primary = FakeProvider("primary")
    fallback_body = {
        "id": "chatcmpl-fallback",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"total_tokens": 3},
    }
    fallback = FakeProvider("fallback", response=fallback_body)
    router = WaterfallRouter([primary, fallback], _catalog(tmp_path), state, request_timeout_seconds=5)

    result = await router.route_chat_completion(_payload())

    assert primary.calls == 0
    assert fallback.calls == 1
    assert result.body == fallback_body
    assert result.provider_name == "fallback"
    assert result.model_id == "fallback/model"
    assert [attempt.status for attempt in result.attempts] == ["skipped", "selected"]


async def test_router_falls_back_on_429_and_marks_cooldown(tmp_path):
    state = await _state(tmp_path)
    primary = FakeProvider(
        "primary",
        error=ProviderRateLimited(
            "rate limited",
            status_code=429,
            headers={"retry-after": "120"},
        ),
    )
    fallback = FakeProvider("fallback")
    router = WaterfallRouter([primary, fallback], _catalog(tmp_path), state, request_timeout_seconds=5)

    result = await router.route_chat_completion(_payload())
    primary_state = await state.get_state("primary")

    assert result.provider_name == "fallback"
    assert primary.calls == 1
    assert fallback.calls == 1
    assert primary_state.cooldown_until > 0
    assert [attempt.status for attempt in result.attempts] == ["rate_limited", "selected"]


async def test_router_marks_repeated_404_as_potentially_outdated(tmp_path):
    state = await _state(tmp_path)
    primary = FakeProvider("primary", error=ProviderError("not found", status_code=404))
    fallback = FakeProvider("fallback")
    router = WaterfallRouter([primary, fallback], _catalog(tmp_path), state, request_timeout_seconds=5)

    await router.route_chat_completion(_payload())
    await router.route_chat_completion(_payload())
    result = await router.route_chat_completion(_payload())
    route_state = await state.get_route_state("primary-test", "primary", "primary/model")

    assert primary.calls == 2
    assert fallback.calls == 3
    assert route_state.status == "potentially_outdated"
    assert result.attempts[0].reason == "potentially_outdated"


async def test_router_marks_repeated_timeouts_as_too_slow(tmp_path):
    state = await _state(tmp_path)
    primary = FakeProvider("primary", error=httpx.ReadTimeout("timed out"))
    fallback = FakeProvider("fallback")
    router = WaterfallRouter([primary, fallback], _catalog(tmp_path), state, request_timeout_seconds=5)

    await router.route_chat_completion(_payload())
    await router.route_chat_completion(_payload())
    result = await router.route_chat_completion(_payload())
    route_state = await state.get_route_state("primary-test", "primary", "primary/model")

    assert primary.calls == 2
    assert fallback.calls == 3
    assert route_state.status == "too_slow"
    assert result.attempts[0].reason == "route_too_slow"


# ── New coverage: NoProviderAvailable ────────────────────────────────────────


async def test_router_raises_no_provider_when_all_exhausted(tmp_path):
    """When every provider is in cooldown, NoProviderAvailable should be raised."""
    state = await _state(tmp_path)
    await state.mark_exhausted("primary", cooldown_seconds=300)
    await state.mark_exhausted("fallback", cooldown_seconds=300)

    primary = FakeProvider("primary")
    fallback = FakeProvider("fallback")
    router = WaterfallRouter([primary, fallback], _catalog(tmp_path), state, request_timeout_seconds=5)

    with pytest.raises(NoProviderAvailable) as exc_info:
        await router.route_chat_completion(_payload())

    assert len(exc_info.value.attempts) == 2
    assert all(a.status == "skipped" for a in exc_info.value.attempts)


# ── New coverage: unconfigured provider ──────────────────────────────────────


async def test_router_skips_unconfigured_provider(tmp_path):
    """Providers without API keys should be skipped with 'missing_api_key'."""
    state = await _state(tmp_path)
    primary = FakeProvider("primary", configured=False)
    fallback = FakeProvider("fallback")
    router = WaterfallRouter([primary, fallback], _catalog(tmp_path), state, request_timeout_seconds=5)

    result = await router.route_chat_completion(_payload())

    assert primary.calls == 0
    assert result.provider_name == "fallback"
    assert result.attempts[0].reason == "missing_api_key"


# ── New coverage: context window exceeded ────────────────────────────────────


async def test_router_skips_context_window_exceeded(tmp_path):
    """Routes with small context windows should be skipped for large prompts."""
    state = await _state(tmp_path)

    # Create a catalog with a tiny context window for primary
    catalog = ModelCatalog(str(tmp_path / "models.json"))
    catalog.replace_routes(
        [
            {
                "route_id": "primary-test",
                "provider_name": "primary",
                "model_id": "primary/model",
                "display_name": "Primary Model",
                "rank": 1,
                "enabled": True,
                "context_window": 10,  # Tiny
            },
            {
                "route_id": "fallback-test",
                "provider_name": "fallback",
                "model_id": "fallback/model",
                "display_name": "Fallback Model",
                "rank": 2,
                "enabled": True,
                "context_window": 999999,
            },
        ]
    )

    primary = FakeProvider("primary", max_context_tokens=10)
    fallback = FakeProvider("fallback", max_context_tokens=999999)
    router = WaterfallRouter([primary, fallback], catalog, state, request_timeout_seconds=5)

    # Make a payload with a big message
    big_payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "x" * 1000}],
    }
    result = await router.route_chat_completion(big_payload)

    assert primary.calls == 0
    assert result.provider_name == "fallback"
    assert result.attempts[0].reason == "context_window_exceeded"


# ── New coverage: timeout fallback ───────────────────────────────────────────


async def test_router_falls_back_on_timeout(tmp_path):
    """Timeout on primary should cascade to fallback."""
    state = await _state(tmp_path)
    primary = FakeProvider("primary", error=httpx.ReadTimeout("timed out"))
    fallback = FakeProvider("fallback")
    router = WaterfallRouter([primary, fallback], _catalog(tmp_path), state, request_timeout_seconds=5)

    result = await router.route_chat_completion(_payload())

    assert result.provider_name == "fallback"
    assert result.attempts[0].status == "failed"
    assert result.attempts[0].reason == "timeout"


# ── New coverage: 5xx fallback ───────────────────────────────────────────────


async def test_router_falls_back_on_5xx(tmp_path):
    """5xx errors from a provider should cascade to the next."""
    state = await _state(tmp_path)
    primary = FakeProvider(
        "primary",
        error=ProviderError("server error", status_code=500),
    )
    fallback = FakeProvider("fallback")
    router = WaterfallRouter([primary, fallback], _catalog(tmp_path), state, request_timeout_seconds=5)

    result = await router.route_chat_completion(_payload())

    assert result.provider_name == "fallback"
    assert result.attempts[0].status == "failed"
    assert result.attempts[0].reason == "provider_5xx"


# ── New coverage: auth error fallback ────────────────────────────────────────


async def test_router_falls_back_on_auth_error(tmp_path):
    """401/403 errors should cascade to the next provider."""
    state = await _state(tmp_path)
    primary = FakeProvider(
        "primary",
        error=ProviderError("unauthorized", status_code=401),
    )
    fallback = FakeProvider("fallback")
    router = WaterfallRouter([primary, fallback], _catalog(tmp_path), state, request_timeout_seconds=5)

    result = await router.route_chat_completion(_payload())

    assert result.provider_name == "fallback"
    assert result.attempts[0].reason == "auth_error"


# ── New coverage: non-recoverable provider error ─────────────────────────────


async def test_router_raises_non_recoverable_error(tmp_path):
    """A 400-class error that isn't auth should bubble up."""
    state = await _state(tmp_path)
    primary = FakeProvider(
        "primary",
        error=ProviderError("bad request", status_code=400),
    )
    fallback = FakeProvider("fallback")
    router = WaterfallRouter([primary, fallback], _catalog(tmp_path), state, request_timeout_seconds=5)

    with pytest.raises(ProviderError):
        await router.route_chat_completion(_payload())


# ── New coverage: request connection error fallback ──────────────────────────


async def test_router_falls_back_on_connection_error(tmp_path):
    """Network connection errors should cascade to the next provider."""
    state = await _state(tmp_path)
    primary = FakeProvider(
        "primary",
        error=httpx.ConnectError("connection refused"),
    )
    fallback = FakeProvider("fallback")
    router = WaterfallRouter([primary, fallback], _catalog(tmp_path), state, request_timeout_seconds=5)

    result = await router.route_chat_completion(_payload())

    assert result.provider_name == "fallback"
    assert result.attempts[0].status == "failed"
    assert result.attempts[0].reason == "ConnectError"


# ── New coverage: payload validation ─────────────────────────────────────────


async def test_router_rejects_empty_messages(tmp_path):
    """Requests with empty messages array should be rejected."""
    state = await _state(tmp_path)
    router = WaterfallRouter(
        [FakeProvider("primary")], _catalog(tmp_path), state, request_timeout_seconds=5
    )

    with pytest.raises(ValueError, match="non-empty"):
        await router.route_chat_completion({"model": "auto", "messages": []})


async def test_router_rejects_missing_content(tmp_path):
    """Messages without a content field should be rejected."""
    state = await _state(tmp_path)
    router = WaterfallRouter(
        [FakeProvider("primary")], _catalog(tmp_path), state, request_timeout_seconds=5
    )

    with pytest.raises(ValueError, match="content"):
        await router.route_chat_completion(
            {"model": "auto", "messages": [{"role": "user"}]}
        )


async def test_router_rejects_missing_role(tmp_path):
    """Messages without a role field should be rejected."""
    state = await _state(tmp_path)
    router = WaterfallRouter(
        [FakeProvider("primary")], _catalog(tmp_path), state, request_timeout_seconds=5
    )

    with pytest.raises(ValueError, match="role"):
        await router.route_chat_completion(
            {"model": "auto", "messages": [{"content": "hi"}]}
        )


# ── New coverage: streaming rejection ────────────────────────────────────────


async def test_router_rejects_streaming(tmp_path):
    """Streaming requests should be rejected with a clear error."""
    state = await _state(tmp_path)
    router = WaterfallRouter(
        [FakeProvider("primary")], _catalog(tmp_path), state, request_timeout_seconds=5
    )

    with pytest.raises(ValueError, match="[Ss]treaming"):
        await router.route_chat_completion(
            {"model": "auto", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        )
