from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest

from app.model_catalog import ModelCatalog
from app.providers.base import ProviderError, ProviderRateLimited, ProviderResponse
from app.request_requirements import (
    RequestRequirements,
)
from app.router import (
    NoProviderAvailable,
    UnsupportedCapabilities,
    WaterfallRouter,
    validate_chat_completion_payload,
)
from app.state import ProviderQuota, StateManager


class FakeAdversarialProvider:
    def __init__(
        self,
        name: str,
        *,
        response: dict[str, Any] | None = None,
        error: Exception | None = None,
        max_context_tokens: int | None = 8192,
        configured: bool = True,
        yields_before_error: list[str] | None = None,
    ) -> None:
        self.name = name
        self.api_key = "test-key" if configured else None
        self.max_context_tokens = max_context_tokens
        self.response = response or {
            "id": f"chatcmpl-{name}",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"total_tokens": 2},
        }
        self.error = error
        self.yields_before_error = yields_before_error
        self.calls = 0
        self.stream_calls = 0
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
        if self.error:
            raise self.error
        return ProviderResponse(self.name, 200, {}, dict(self.response))

    async def chat_completion_stream(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, Any],
        target_model: str | None = None,
    ) -> AsyncGenerator[str, None]:
        self.stream_calls += 1
        if self.yields_before_error:
            for item in self.yields_before_error:
                yield item
            if self.error:
                raise self.error
            return
        if self.error:
            raise self.error
        
        body = dict(self.response)
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        chunk = json.dumps(
            {
                "id": body.get("id", f"chatcmpl-{self.name}"),
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": content}}],
            }
        )
        yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"


def _payload(messages: Any = None) -> dict[str, Any]:
    if messages is None:
        messages = [{"role": "user", "content": "hello"}]
    return {"model": "auto", "messages": messages}


async def _state(tmp_path) -> StateManager:
    providers = [
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
                "tags": ["text", "web-search"],
            },
        ]
    )
    return catalog


# ─── 1. Malformed Payloads & Empty/Missing Fields ───

def test_validation_non_dict_payload():
    with pytest.raises(ValueError, match="Request body must be a JSON object"):
        validate_chat_completion_payload(["not a dict"])  # type: ignore


def test_validation_missing_messages():
    with pytest.raises(ValueError, match="Request body must include a non-empty 'messages' array"):
        validate_chat_completion_payload({"model": "auto"})


def test_validation_empty_messages():
    with pytest.raises(ValueError, match="Request body must include a non-empty 'messages' array"):
        validate_chat_completion_payload({"model": "auto", "messages": []})


def test_validation_message_not_object():
    with pytest.raises(ValueError, match="messages\\[0\\] must be an object"):
        validate_chat_completion_payload({"model": "auto", "messages": ["hello"]})


def test_validation_missing_role():
    with pytest.raises(ValueError, match="messages\\[0\\].role must be a string"):
        validate_chat_completion_payload({"model": "auto", "messages": [{"content": "hello"}]})


def test_validation_missing_content():
    with pytest.raises(ValueError, match="messages\\[0\\].content is required"):
        validate_chat_completion_payload({"model": "auto", "messages": [{"role": "user"}]})


# ─── 2. Invalid Roles & Unknown Blocks ───

def test_invalid_role_rejected():
    """Invalid role strings are rejected before any provider request is attempted."""
    payload = {"model": "auto", "messages": [{"role": "invalid_role", "content": "hello"}]}
    with pytest.raises(ValueError, match="messages\\[0\\].role must be one of"):
        validate_chat_completion_payload(payload)


async def test_unknown_blocks_forwarded_safely(tmp_path):
    """Extra fields/unknown blocks at root and within message dictionary should be tolerated and forwarded."""
    state = await _state(tmp_path)
    primary = FakeAdversarialProvider("primary")
    router = WaterfallRouter([primary], _catalog(tmp_path), state, request_timeout_seconds=5)
    
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "hello", "extra_message_key": 123}],
        "extra_root_key": "ignored_or_forwarded"
    }
    validate_chat_completion_payload(payload)
    
    result = await router.route_chat_completion(payload)
    assert result.provider_name == "primary"
    assert primary.calls == 1


# ─── 3. Unsupported Capabilities & No Capable Route ───

async def test_unsupported_capabilities_raises(tmp_path):
    """If the client requests capabilities not supported by any routes, UnsupportedCapabilities is raised."""
    state = await _state(tmp_path)
    primary = FakeAdversarialProvider("primary")
    router = WaterfallRouter([primary], _catalog(tmp_path), state, request_timeout_seconds=5)
    
    # "json-schema" capability is not defined in any route of _catalog (they only have text and fallback has web-search)
    reqs = RequestRequirements(required_capabilities=frozenset({"text", "json-schema"}))
    with pytest.raises(UnsupportedCapabilities) as exc_info:
        await router.route_chat_completion(_payload(), requirements=reqs)
    
    assert "json-schema" in exc_info.value.required


async def test_no_capable_route_raised(tmp_path):
    """When all providers are exhausted or no route can satisfy request (e.g., all disabled), NoProviderAvailable is raised."""
    state = await _state(tmp_path)
    # Mark both primary and fallback exhausted
    await state.mark_exhausted("primary", cooldown_seconds=60)
    await state.mark_exhausted("fallback", cooldown_seconds=60)
    
    primary = FakeAdversarialProvider("primary")
    fallback = FakeAdversarialProvider("fallback")
    router = WaterfallRouter([primary, fallback], _catalog(tmp_path), state, request_timeout_seconds=5)
    
    with pytest.raises(NoProviderAvailable):
        await router.route_chat_completion(_payload())


# ─── 4. Provider Errors ───

async def test_provider_500_cascades_to_fallback(tmp_path):
    state = await _state(tmp_path)
    primary = FakeAdversarialProvider("primary", error=ProviderError("internal error", status_code=500))
    fallback = FakeAdversarialProvider("fallback")
    router = WaterfallRouter([primary, fallback], _catalog(tmp_path), state, request_timeout_seconds=5)
    
    result = await router.route_chat_completion(_payload())
    assert result.provider_name == "fallback"
    assert primary.calls == 1
    assert fallback.calls == 1
    assert result.attempts[0].status == "failed"
    assert result.attempts[0].reason == "provider_5xx"


async def test_provider_401_auth_cascades_to_fallback(tmp_path):
    state = await _state(tmp_path)
    primary = FakeAdversarialProvider("primary", error=ProviderError("unauthorized", status_code=401))
    fallback = FakeAdversarialProvider("fallback")
    router = WaterfallRouter([primary, fallback], _catalog(tmp_path), state, request_timeout_seconds=5)
    
    result = await router.route_chat_completion(_payload())
    assert result.provider_name == "fallback"
    assert primary.calls == 1
    assert fallback.calls == 1
    assert result.attempts[0].status == "failed"
    assert result.attempts[0].reason == "auth_error"


async def test_provider_429_rate_limit_cascades_and_cooldowns(tmp_path):
    state = await _state(tmp_path)
    primary = FakeAdversarialProvider("primary", error=ProviderRateLimited("limit", status_code=429))
    fallback = FakeAdversarialProvider("fallback")
    router = WaterfallRouter([primary, fallback], _catalog(tmp_path), state, request_timeout_seconds=5)
    
    result = await router.route_chat_completion(_payload())
    assert result.provider_name == "fallback"
    assert primary.calls == 1
    assert fallback.calls == 1
    assert result.attempts[0].status == "rate_limited"
    
    # Provider should be marked exhausted/cooldown
    p_state = await state.get_state("primary")
    assert p_state.cooldown_until > 0


# ─── 5. Disconnects & Stream Errors ───

async def test_disconnect_connection_refused_cascades(tmp_path):
    """Connection errors (like ConnectError) should cause cascade to fallback."""
    state = await _state(tmp_path)
    primary = FakeAdversarialProvider("primary", error=httpx.ConnectError("refused"))
    fallback = FakeAdversarialProvider("fallback")
    router = WaterfallRouter([primary, fallback], _catalog(tmp_path), state, request_timeout_seconds=5)
    
    result = await router.route_chat_completion(_payload())
    assert result.provider_name == "fallback"
    assert primary.calls == 1
    assert fallback.calls == 1
    assert result.attempts[0].status == "failed"
    assert result.attempts[0].reason == "ConnectError"


async def test_stream_error_before_commit_cascades(tmp_path):
    """Mid-stream failure BEFORE any content is written (before stream commit) cascades to fallback."""
    state = await _state(tmp_path)
    primary = FakeAdversarialProvider("primary", error=httpx.ReadTimeout("read timeout"))
    fallback = FakeAdversarialProvider("fallback")
    router = WaterfallRouter([primary, fallback], _catalog(tmp_path), state, request_timeout_seconds=5)
    
    events = [event async for event in router.iter_chat_completion_openai_stream(_payload())]
    # Verify that stream results came from fallback
    fallback_received = False
    for chunk in events:
        if isinstance(chunk, str) and "chatcmpl-fallback" in chunk:
            fallback_received = True
    assert fallback_received
    assert primary.stream_calls == 1
    assert fallback.stream_calls == 1


async def test_stream_error_after_commit_does_not_cascade(tmp_path):
    """Mid-stream failure AFTER content is committed does NOT cascade to fallback and raises/done."""
    state = await _state(tmp_path)
    primary = FakeAdversarialProvider(
        "primary",
        yields_before_error=[
            'data: {"id": "chatcmpl-primary", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "hello"}}]}\n\n'
        ],
        error=httpx.ReadTimeout("mid-stream timeout")
    )
    fallback = FakeAdversarialProvider("fallback")
    router = WaterfallRouter([primary, fallback], _catalog(tmp_path), state, request_timeout_seconds=5)
    
    events = [event async for event in router.iter_chat_completion_openai_stream(_payload())]
    
    # We should see content from primary
    primary_seen = any("hello" in chunk for chunk in events if isinstance(chunk, str))
    assert primary_seen
    
    # Fallback should NOT have been called
    assert fallback.stream_calls == 0
    
    # Done message should be appended to clean up
    assert any("[DONE]" in chunk for chunk in events if isinstance(chunk, str))
