from __future__ import annotations

import json

import httpx

from app.codex_compat import responses_payload_to_chat
from app.providers.base import ProviderAdapter
from app.router import validate_chat_completion_payload


def test_responses_with_instructions_maps_to_system_prompt():
    payload = {
        "model": "auto",
        "input": "User query",
        "instructions": "You are a helpful assistant.",
    }
    chat_payload = responses_payload_to_chat(payload)
    assert chat_payload["model"] == "auto"
    assert len(chat_payload["messages"]) == 2
    assert chat_payload["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    assert chat_payload["messages"][1] == {"role": "user", "content": "User query"}


def test_responses_with_max_output_tokens():
    payload1 = {
        "model": "auto",
        "input": "hello",
        "max_output_tokens": 150,
    }
    chat_payload1 = responses_payload_to_chat(payload1)
    assert chat_payload1["max_tokens"] == 150

    payload2 = {
        "model": "auto",
        "input": "hello",
        "max_tokens": 250,
    }
    chat_payload2 = responses_payload_to_chat(payload2)
    assert chat_payload2["max_tokens"] == 250


def test_validate_chat_completion_payload_accepts_extra_fields():
    # Verify that the validator accepts extra fields (e.g. logprobs, temperature)
    # without raising ValueError, ensuring compatibility.
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "hello"}],
        "logprobs": True,
        "top_logprobs": 5,
        "temperature": 0.7,
        "presence_penalty": 0.5,
    }
    # Should not raise any exceptions
    validate_chat_completion_payload(payload)


def test_validate_chat_completion_payload_rejects_invalid_role():
    payload = {
        "model": "auto",
        "messages": [{"role": "hacker", "content": "hello"}],
    }

    try:
        validate_chat_completion_payload(payload)
    except ValueError as exc:
        assert "role" in str(exc)
    else:
        raise AssertionError("invalid role should be rejected")


def test_validate_chat_completion_payload_rejects_invalid_content_shape():
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": {"text": "hello"}}],
    }

    try:
        validate_chat_completion_payload(payload)
    except ValueError as exc:
        assert "content" in str(exc)
    else:
        raise AssertionError("object content should be rejected")


def test_validate_chat_completion_payload_allows_assistant_tool_call_null_content():
    payload = {
        "model": "auto",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ],
    }

    validate_chat_completion_payload(payload)


def test_validate_chat_completion_payload_allows_assistant_tool_call_omitted_content():
    payload = {
        "model": "auto",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ],
    }

    validate_chat_completion_payload(payload)


async def test_provider_adapter_forwards_logprobs_and_extra_parameters():
    # Verify that the ProviderAdapter forwards logprobs and extra parameters untouched to the upstream provider
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "provider-model"
        assert body["logprobs"] is True
        assert body["top_logprobs"] == 5
        assert body["temperature"] == 0.7
        assert body["presence_penalty"] == 0.5
        return httpx.Response(200, json={"id": "chat", "choices": [], "usage": {"total_tokens": 1}})

    adapter = ProviderAdapter(
        name="unit",
        api_key="test-key",
        base_url="https://provider.test/v1",
        default_model="provider-model",
    )
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await adapter.chat_completion(
            client,
            {
                "model": "auto",
                "messages": [{"role": "user", "content": "hello"}],
                "logprobs": True,
                "top_logprobs": 5,
                "temperature": 0.7,
                "presence_penalty": 0.5,
            },
        )
