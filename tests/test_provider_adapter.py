from __future__ import annotations

import json

import httpx
import pytest

from app.providers.base import ProviderAdapter, ProviderError, ProviderRateLimited


def _adapter() -> ProviderAdapter:
    return ProviderAdapter(
        name="unit",
        api_key="test-key",
        base_url="https://provider.test/v1",
        default_model="default-model",
        model_aliases={"friendly-model": "provider-model"},
    )


async def test_list_models_returns_json_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"object": "list", "data": [{"id": "m1"}]})

    adapter = _adapter()
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        payload = await adapter.list_models(client)
    assert payload["object"] == "list"
    assert payload["data"][0]["id"] == "m1"


async def test_list_models_raises_rate_limited_for_429():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text='{"error":"rate_limited"}')

    adapter = _adapter()
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ProviderRateLimited) as exc:
            await adapter.list_models(client)
    assert exc.value.status_code == 429


async def test_list_models_raises_for_non_json_success_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    adapter = _adapter()
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ProviderError, match="non-JSON"):
            await adapter.list_models(client)


async def test_chat_completion_uses_target_model_when_provided():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "explicit-target"
        return httpx.Response(200, json={"id": "chat", "choices": [], "usage": {"total_tokens": 1}})

    adapter = _adapter()
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        response = await adapter.chat_completion(
            client,
            {"model": "friendly-model", "messages": [{"role": "user", "content": "hello"}]},
            "explicit-target",
        )
    assert response.provider_name == "unit"
    assert response.body["id"] == "chat"


async def test_chat_completion_uses_alias_model_mapping():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "provider-model"
        return httpx.Response(200, json={"id": "chat", "choices": [], "usage": {"total_tokens": 1}})

    adapter = _adapter()
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await adapter.chat_completion(
            client,
            {"model": "friendly-model", "messages": [{"role": "user", "content": "hello"}]},
        )


async def test_chat_completion_raises_provider_error_for_http_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text='{"error":"upstream failed"}')

    adapter = _adapter()
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ProviderError) as exc:
            await adapter.chat_completion(
                client,
                {"model": "friendly-model", "messages": [{"role": "user", "content": "hello"}]},
            )
    assert exc.value.status_code == 502
