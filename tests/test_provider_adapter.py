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


async def test_groq_web_search_preview_translates_to_compound_tool_config():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "groq/compound"
        assert "tool_choice" not in body
        assert "tools" not in body
        assert "max_tokens" not in body
        assert "max_completion_tokens" not in body
        assert body["compound_custom"]["tools"]["enabled_tools"] == ["web_search"]
        return httpx.Response(200, json={"id": "chat", "choices": [], "usage": {"total_tokens": 1}})

    adapter = ProviderAdapter(
        name="groq",
        api_key="test-key",
        base_url="https://api.groq.com/openai/v1",
        default_model="groq/compound",
    )
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await adapter.chat_completion(
            client,
            {
                "model": "auto",
                "messages": [{"role": "user", "content": "latest news"}],
                "tools": [{"type": "web_search_preview"}],
                "tool_choice": {"type": "web_search_preview"},
                "max_tokens": 100,
                "max_completion_tokens": 100,
            },
            "groq/compound",
        )


async def test_groq_web_search_translation_ignores_malformed_compound_custom():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["compound_custom"]["tools"]["enabled_tools"] == ["web_search"]
        return httpx.Response(200, json={"id": "chat", "choices": [], "usage": {"total_tokens": 1}})

    adapter = ProviderAdapter(
        name="groq",
        api_key="test-key",
        base_url="https://api.groq.com/openai/v1",
        default_model="groq/compound",
    )
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await adapter.chat_completion(
            client,
            {
                "model": "auto",
                "messages": [{"role": "user", "content": "latest news"}],
                "tools": [{"type": "web_search_preview"}],
                "tool_choice": {"type": "web_search_preview"},
                "compound_custom": "invalid",
            },
            "groq/compound",
        )


async def test_openrouter_web_search_preview_translates_to_server_tool():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "openrouter/free"
        assert "tool_choice" not in body
        assert body["tools"] == [{"type": "openrouter:web_search"}]
        return httpx.Response(200, json={"id": "chat", "choices": [], "usage": {"total_tokens": 1}})

    adapter = ProviderAdapter(
        name="openrouter",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        default_model="openrouter/free",
    )
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await adapter.chat_completion(
            client,
            {
                "model": "auto",
                "messages": [{"role": "user", "content": "latest news"}],
                "tools": [{"type": "web_search_preview"}],
                "tool_choice": {"type": "web_search_preview"},
            },
            "openrouter/free",
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


async def test_chat_completion_strips_stream_flag_for_non_streaming_requests():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert "stream" not in body
        return httpx.Response(200, json={"id": "chat", "choices": [], "usage": {"total_tokens": 1}})

    adapter = _adapter()
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await adapter.chat_completion(
            client,
            {
                "model": "friendly-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )


async def test_chat_completion_forwards_multiple_tools_and_parallel_tool_calls():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["parallel_tool_calls"] is False
        assert len(body["tools"]) == 2
        assert body["tools"][0]["function"]["name"] == "alpha"
        assert body["tools"][1]["function"]["name"] == "beta"
        return httpx.Response(200, json={"id": "chat", "choices": [], "usage": {"total_tokens": 1}})

    adapter = _adapter()
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await adapter.chat_completion(
            client,
            {
                "model": "friendly-model",
                "messages": [{"role": "user", "content": "hello"}],
                "parallel_tool_calls": False,
                "tools": [
                    {"type": "function", "function": {"name": "alpha", "parameters": {}}},
                    {"type": "function", "function": {"name": "beta", "parameters": {}}},
                ],
            },
        )


async def test_chat_completion_forwards_multimodal_message_array():
    image_part = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,abc"},
    }
    text_part = {"type": "text", "text": "describe"}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        content = body["messages"][0]["content"]
        assert content == [text_part, image_part]
        return httpx.Response(200, json={"id": "chat", "choices": [], "usage": {"total_tokens": 1}})

    adapter = _adapter()
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await adapter.chat_completion(
            client,
            {
                "model": "friendly-model",
                "messages": [{"role": "user", "content": [text_part, image_part]}],
            },
        )


async def test_chat_completion_forwards_response_format_and_reasoning_fields():
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": {"type": "object"}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["response_format"] == response_format
        assert body["reasoning_effort"] == "high"
        assert body["reasoning"] == {"effort": "medium"}
        return httpx.Response(200, json={"id": "chat", "choices": [], "usage": {"total_tokens": 1}})

    adapter = _adapter()
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await adapter.chat_completion(
            client,
            {
                "model": "friendly-model",
                "messages": [{"role": "user", "content": "hello"}],
                "response_format": response_format,
                "reasoning_effort": "high",
                "reasoning": {"effort": "medium"},
            },
        )


async def test_chat_completion_returns_multiple_tool_calls_without_dropping_indices():
    response_body = {
        "id": "chatcmpl-tools",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "alpha", "arguments": "{}"},
                        },
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {"name": "beta", "arguments": "{}"},
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"total_tokens": 4},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_body)

    adapter = _adapter()
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        response = await adapter.chat_completion(
            client,
            {"model": "friendly-model", "messages": [{"role": "user", "content": "hello"}]},
        )
    tool_calls = response.body["choices"][0]["message"]["tool_calls"]
    assert [call["id"] for call in tool_calls] == ["call_a", "call_b"]


async def test_chat_completion_forwards_json_schema_response_format_unchanged():
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": {"type": "object", "properties": {"x": {"type": "string"}}}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["response_format"] == response_format
        return httpx.Response(200, json={"id": "chat", "choices": [], "usage": {"total_tokens": 1}})

    adapter = _adapter()
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await adapter.chat_completion(
            client,
            {
                "model": "friendly-model",
                "messages": [{"role": "user", "content": "hello"}],
                "response_format": response_format,
            },
        )


async def test_chat_completion_stream_yields_lines_and_sets_stream_true():
    sse_body = (
        'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["stream"] is True
        assert body["model"] == "provider-model"
        return httpx.Response(200, text=sse_body)

    adapter = _adapter()
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        lines = [
            line async for line in adapter.chat_completion_stream(
                client,
                {"model": "friendly-model", "messages": [{"role": "user", "content": "hello"}]},
            )
        ]
    joined = "".join(lines)
    assert "delta" in joined
    assert "[DONE]" in joined
