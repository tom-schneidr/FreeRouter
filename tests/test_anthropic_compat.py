"""Pure translation tests for Anthropic Messages compatibility."""

from __future__ import annotations

import json

import pytest

from app.anthropic_compat import (
    AnthropicStreamMapper,
    anthropic_error_body,
    anthropic_stream_event,
    chat_body_to_anthropic_message,
    messages_payload_to_chat,
)


def test_messages_payload_to_chat_plain_user_text():
    chat = messages_payload_to_chat(
        {
            "model": "auto",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    assert chat["model"] == "auto"
    assert chat["max_tokens"] == 128
    assert chat["messages"] == [{"role": "user", "content": "hello"}]


def test_messages_payload_to_chat_system_string():
    chat = messages_payload_to_chat(
        {
            "model": "auto",
            "max_tokens": 64,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert chat["messages"][0] == {"role": "system", "content": "You are helpful."}


def test_messages_payload_to_chat_system_text_blocks():
    chat = messages_payload_to_chat(
        {
            "model": "auto",
            "max_tokens": 64,
            "system": [{"type": "text", "text": "Line one"}, {"type": "text", "text": "Line two"}],
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert chat["messages"][0]["role"] == "system"
    assert "Line one" in chat["messages"][0]["content"]
    assert "Line two" in chat["messages"][0]["content"]


def test_messages_payload_to_chat_tools_and_tool_choice():
    chat = messages_payload_to_chat(
        {
            "model": "auto",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                }
            ],
            "tool_choice": {"type": "tool", "name": "get_weather"},
        }
    )
    assert chat["tools"][0]["type"] == "function"
    assert chat["tools"][0]["function"]["name"] == "get_weather"
    assert chat["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def test_messages_payload_to_chat_assistant_tool_use_and_user_tool_result():
    chat = messages_payload_to_chat(
        {
            "model": "auto",
            "max_tokens": 128,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_01",
                            "name": "lookup",
                            "input": {"q": "x"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_01",
                            "content": "found",
                        }
                    ],
                },
            ],
        }
    )
    assert chat["messages"][0]["role"] == "assistant"
    assert chat["messages"][0]["tool_calls"][0]["id"] == "toolu_01"
    assert chat["messages"][1] == {
        "role": "tool",
        "tool_call_id": "toolu_01",
        "content": "found",
    }


def test_messages_payload_to_chat_image_block_maps_to_openai_vision_part():
    chat = messages_payload_to_chat(
        {
            "model": "auto",
            "max_tokens": 128,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc",
                            },
                        },
                    ],
                }
            ],
        }
    )
    parts = chat["messages"][-1]["content"]
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"] == "data:image/png;base64,abc"


def test_messages_payload_to_chat_preserves_temperature_top_p_stop_sequences():
    chat = messages_payload_to_chat(
        {
            "model": "auto",
            "max_tokens": 128,
            "temperature": 0.2,
            "top_p": 0.9,
            "stop_sequences": ["END"],
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert chat["temperature"] == 0.2
    assert chat["top_p"] == 0.9
    assert chat["stop"] == ["END"]


def test_messages_payload_rejects_non_object():
    with pytest.raises(ValueError, match="JSON object"):
        messages_payload_to_chat([])  # type: ignore[arg-type]


def test_messages_payload_rejects_empty_messages():
    with pytest.raises(ValueError, match="messages"):
        messages_payload_to_chat({"model": "auto", "max_tokens": 10, "messages": []})


def test_messages_payload_rejects_invalid_max_tokens():
    for invalid in (0, -1, True, 1.5, "10"):
        with pytest.raises(ValueError, match="max_tokens"):
            messages_payload_to_chat(
                {
                    "model": "auto",
                    "max_tokens": invalid,
                    "messages": [{"role": "user", "content": "hi"}],
                }
            )


def test_messages_payload_rejects_unsupported_top_level_field():
    with pytest.raises(ValueError, match="thinking"):
        messages_payload_to_chat(
            {
                "model": "auto",
                "max_tokens": 10,
                "thinking": {"type": "enabled", "budget_tokens": 1000},
                "messages": [{"role": "user", "content": "hi"}],
            }
        )


def test_messages_payload_accepts_metadata_without_forwarding_it():
    chat = messages_payload_to_chat(
        {
            "model": "auto",
            "max_tokens": 10,
            "metadata": {"user_id": "abc"},
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert "metadata" not in chat


def test_messages_payload_rejects_top_k():
    with pytest.raises(ValueError, match="top_k"):
        messages_payload_to_chat(
            {
                "model": "auto",
                "max_tokens": 10,
                "top_k": 5,
                "messages": [{"role": "user", "content": "hi"}],
            }
        )


def test_messages_payload_orders_tool_results_before_follow_up_user_text():
    chat = messages_payload_to_chat(
        {
            "model": "auto",
            "max_tokens": 64,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup",
                            "input": {"q": "x"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "found"},
                        {"type": "text", "text": "Now summarize it."},
                    ],
                },
            ],
        }
    )

    assert [message["role"] for message in chat["messages"]] == ["assistant", "tool", "user"]


def test_messages_payload_rejects_unsupported_content_block():
    with pytest.raises(ValueError, match="document"):
        messages_payload_to_chat(
            {
                "model": "auto",
                "max_tokens": 10,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "document", "source": {"type": "url", "url": "x"}}],
                    }
                ],
            }
        )


def test_chat_body_to_anthropic_message_text():
    message = chat_body_to_anthropic_message(
        {
            "id": "chatcmpl-1",
            "model": "upstream/model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello there"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
        requested_model="auto",
        message_id="msg_test123",
    )
    assert message["type"] == "message"
    assert message["id"] == "msg_test123"
    assert message["role"] == "assistant"
    assert message["content"] == [{"type": "text", "text": "Hello there"}]
    assert message["stop_reason"] == "end_turn"
    assert message["usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_chat_body_to_anthropic_message_tool_calls():
    message = chat_body_to_anthropic_message(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "alpha", "arguments": '{"x":1}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 7},
        },
        requested_model="auto",
    )
    assert message["stop_reason"] == "tool_use"
    assert message["content"][0] == {
        "type": "tool_use",
        "id": "call_a",
        "name": "alpha",
        "input": {"x": 1},
    }


def test_chat_body_to_anthropic_message_tool_calls_override_incorrect_stop_reason():
    message = chat_body_to_anthropic_message(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "alpha", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "stop",
                }
            ]
        },
        requested_model="auto",
    )
    assert message["stop_reason"] == "tool_use"


def test_anthropic_error_body_shape():
    body = anthropic_error_body("invalid_request_error", "bad request")
    assert body == {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "bad request"},
    }


def test_anthropic_stream_mapper_text_lifecycle():
    mapper = AnthropicStreamMapper(message_id="msg_stream", model="auto")
    events: list[str] = []
    chunk = json.dumps(
        {
            "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
        }
    )
    events.extend(mapper.events_from_openai_sse(f"data: {chunk}"))
    events.extend(mapper.events_from_openai_sse("data: [DONE]"))

    assert any("event: message_start" in e for e in events)
    assert any("content_block_start" in e and '"text"' in e for e in events)
    assert any("text_delta" in e for e in events)
    assert any("content_block_stop" in e for e in events)
    assert any("message_delta" in e for e in events)
    assert any("event: message_stop" in e for e in events)


def test_anthropic_stream_mapper_dual_tool_calls_with_fragmented_json():
    mapper = AnthropicStreamMapper(message_id="msg_tools", model="auto")
    events: list[str] = []
    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "alpha", "arguments": ""},
                            },
                            {
                                "index": 1,
                                "id": "call_b",
                                "type": "function",
                                "function": {"name": "beta", "arguments": ""},
                            },
                        ]
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"a":'}},
                            {"index": 1, "function": {"arguments": '{"b":'}},
                        ]
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": "1}"}},
                            {"index": 1, "function": {"arguments": "2}"}},
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    for chunk in chunks:
        events.extend(mapper.events_from_openai_sse(f"data: {json.dumps(chunk)}"))
    events.extend(mapper.events_from_openai_sse("data: [DONE]"))

    compact = "".join(events).replace(" ", "")
    assert "tool_use" in compact
    assert "input_json_delta" in compact
    assert '"type":"content_block_start","index":0' in compact
    assert '"type":"content_block_start","index":1' in compact
    assert '"type":"content_block_start","index":2' not in compact
    assert '"partial_json":"{\\"a\\":' in compact or '"partial_json":"{"a":' in compact
    assert '"partial_json":"1}"' in compact
    assert '"stop_reason":"tool_use"' in compact
    assert "message_stop" in compact


def test_anthropic_stream_mapper_waits_for_tool_metadata_before_start():
    mapper = AnthropicStreamMapper(message_id="msg_tools", model="auto")
    arguments_first = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [{"index": 0, "function": {"arguments": '{"q"'}}]
                }
            }
        ]
    }
    metadata_later = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_late",
                            "function": {"name": "lookup", "arguments": ':"x"}'},
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ]
    }

    first_events = mapper.events_from_openai_sse(f"data: {json.dumps(arguments_first)}")
    later_events = mapper.events_from_openai_sse(f"data: {json.dumps(metadata_later)}")

    assert not any("content_block_start" in event for event in first_events)
    combined = "".join(later_events)
    assert '"id":"call_late"' in combined
    assert '"name":"lookup"' in combined
    assert '"partial_json":"{\\"q\\"' in combined
    assert '"partial_json":":\\"x\\"}"' in combined


def test_anthropic_stream_mapper_ignores_incomplete_tool_call_for_stop_reason():
    mapper = AnthropicStreamMapper(message_id="msg_tools", model="auto")
    chunk = {"choices": [{"delta": {"tool_calls": [{"index": 0}]}}]}
    mapper.events_from_openai_sse(f"data: {json.dumps(chunk)}")

    completed = "".join(mapper.events_from_openai_sse("data: [DONE]"))

    assert '"stop_reason":"end_turn"' in completed
    assert "tool_use" not in completed


def test_anthropic_stream_event_format():
    event = anthropic_stream_event("ping", {"type": "ping"})
    assert event.startswith("event: ping\n")
    assert '"type":"ping"' in event or '"type": "ping"' in event
