"""Table-driven tests for chat payload capability derivation."""

from __future__ import annotations

import pytest

from app.request_requirements import RequestRequirements, chat_request_requirements


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        pytest.param(
            {"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
            frozenset({"text"}),
            id="plain-text",
        ),
        pytest.param(
            {
                "model": "auto",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
            },
            frozenset({"text", "tool-use"}),
            id="function-tools",
        ),
        pytest.param(
            {
                "model": "auto",
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": {"type": "function", "function": {"name": "lookup"}},
            },
            frozenset({"text", "tool-use"}),
            id="function-tool-choice",
        ),
        pytest.param(
            {
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
                    {"role": "tool", "tool_call_id": "call_1", "content": "result"},
                ],
            },
            frozenset({"text", "tool-use"}),
            id="assistant-tool-calls-and-tool-role",
        ),
        pytest.param(
            {
                "model": "auto",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is this"},
                            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                        ],
                    }
                ],
            },
            frozenset({"text", "vision"}),
            id="multimodal-image-url",
        ),
        pytest.param(
            {
                "model": "auto",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {"type": "json_object"},
            },
            frozenset({"text"}),
            id="json-object-does-not-require-json-schema",
        ),
        pytest.param(
            {
                "model": "auto",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "schema": {"type": "object"}},
                },
            },
            frozenset({"text", "json-schema"}),
            id="json-schema-response-format",
        ),
        pytest.param(
            {
                "model": "auto",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "high",
            },
            frozenset({"text", "reasoning"}),
            id="reasoning-effort",
        ),
        pytest.param(
            {
                "model": "auto",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning": {"effort": "medium"},
            },
            frozenset({"text", "reasoning"}),
            id="reasoning-config-effort",
        ),
        pytest.param(
            {
                "model": "auto",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "web_search_preview"}],
            },
            frozenset({"text"}),
            id="web-search-preview-tool-does-not-imply-tool-use",
        ),
    ],
)
def test_chat_request_requirements(payload: dict, expected: frozenset[str]) -> None:
    result = chat_request_requirements(payload)
    assert isinstance(result, RequestRequirements)
    assert result.required_capabilities == expected


def test_chat_request_requirements_is_order_independent() -> None:
    payload_a = {
        "model": "auto",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "a", "parameters": {}}}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}},
    }
    payload_b = {
        "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}},
        "tools": [{"type": "function", "function": {"name": "a", "parameters": {}}}],
        "messages": [{"role": "user", "content": "hi"}],
        "model": "auto",
    }
    assert chat_request_requirements(payload_a) == chat_request_requirements(payload_b)
