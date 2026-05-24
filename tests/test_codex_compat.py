from __future__ import annotations

import json

from app.codex_compat import (
    ResponsesStreamMapper,
    chat_body_to_response,
    responses_payload_to_chat,
)


def test_responses_payload_to_chat_maps_codex_fields():
    payload = responses_payload_to_chat(
        {
            "model": "auto",
            "instructions": "Be direct.",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
            "max_output_tokens": 123,
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "name": "shell",
                    "description": "Run a command.",
                    "parameters": {"type": "object"},
                }
            ],
            "tool_choice": {"type": "function", "name": "shell"},
        }
    )

    assert payload["model"] == "auto"
    assert payload["messages"] == [
        {"role": "system", "content": "Be direct."},
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
    ]
    assert payload["max_tokens"] == 123
    assert payload["stream"] is True
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run a command.",
                "parameters": {"type": "object"},
            },
        }
    ]
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "shell"}}


def test_responses_payload_to_chat_maps_function_call_outputs():
    payload = responses_payload_to_chat(
        {
            "model": "auto",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "shell",
                    "arguments": '{"cmd":"pwd"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "C:/repo",
                },
            ],
        }
    )

    assert payload["messages"] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": '{"cmd":"pwd"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "C:/repo"},
    ]


def test_chat_body_to_response_exposes_output_text_and_usage():
    response = chat_body_to_response(
        {
            "choices": [{"message": {"role": "assistant", "content": "answer"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        },
        requested_model="auto",
        response_id="resp_test",
    )

    assert response["id"] == "resp_test"
    assert response["object"] == "response"
    assert response["output_text"] == "answer"
    assert response["output"][0]["content"][0]["text"] == "answer"
    assert response["usage"]["input_tokens"] == 3
    assert response["usage"]["output_tokens"] == 4


def test_chat_body_to_response_maps_tool_calls():
    response = chat_body_to_response(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "shell",
                                    "arguments": '{"cmd":"pwd"}',
                                },
                            }
                        ],
                    }
                }
            ],
        },
        requested_model="auto",
        response_id="resp_test",
    )

    assert response["output_text"] == ""
    assert response["output"] == [
        {
            "id": "call_1",
            "type": "function_call",
            "status": "completed",
            "call_id": "call_1",
            "name": "shell",
            "arguments": '{"cmd":"pwd"}',
        }
    ]


def test_responses_stream_delta_from_openai_sse_maps_text_delta():
    mapper = ResponsesStreamMapper(response_id="resp_test")
    block = "data: " + json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "hi"},
                }
            ]
        }
    )

    events = mapper.events_from_openai_sse(block)

    assert any("event: response.output_text.delta" in event for event in events)
    assert any('"delta": "hi"' in event for event in events)


def test_responses_stream_mapper_emits_tool_call_item_on_done():
    mapper = ResponsesStreamMapper(response_id="resp_test")
    block = "data: " + json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "shell",
                                    "arguments": '{"cmd":"pwd"}',
                                },
                            }
                        ]
                    },
                }
            ]
        }
    )

    assert mapper.events_from_openai_sse(block) == []
    done_events = "\n".join(mapper.events_from_openai_sse("data: [DONE]"))

    assert "response.output_item.done" in done_events
    assert '"type": "function_call"' in done_events
    assert '"call_id": "call_1"' in done_events
    assert '"arguments": "{\\"cmd\\":\\"pwd\\"}"' in done_events
