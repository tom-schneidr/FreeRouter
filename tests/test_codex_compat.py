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


def test_responses_payload_to_chat_maps_text_format_to_response_format():
    payload = responses_payload_to_chat(
        {
            "model": "auto",
            "input": "Return JSON.",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {"type": "object"},
                    "strict": True,
                }
            },
        }
    )

    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {"type": "object"},
            "strict": True,
        },
    }


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


def test_responses_stream_mapper_emits_tool_call_argument_deltas_and_done():
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

    stream_events = "\n".join(mapper.events_from_openai_sse(block))
    assert "response.output_item.added" in stream_events
    assert "response.function_call_arguments.delta" in stream_events
    assert '"delta": "{\\"cmd\\":\\"pwd\\"}"' in stream_events
    done_events = "\n".join(mapper.events_from_openai_sse("data: [DONE]"))

    assert "response.function_call_arguments.done" in done_events
    assert "response.output_item.done" in done_events
    assert '"type": "function_call"' in done_events
    assert '"call_id": "call_1"' in done_events
    assert '"arguments": "{\\"cmd\\":\\"pwd\\"}"' in done_events


def test_responses_payload_to_chat_empty_or_invalid_input():
    import pytest
    with pytest.raises(ValueError, match="Responses payload must include non-empty 'input'"):
        responses_payload_to_chat({"model": "auto", "input": []})

    with pytest.raises(ValueError, match="Responses payload must include non-empty 'input'"):
        responses_payload_to_chat({"model": "auto", "input": None})
    with pytest.raises(ValueError, match="input\\[0\\] must be a string or object"):
        responses_payload_to_chat({"model": "auto", "input": [123]})


def test_responses_payload_to_chat_complex_inputs():
    payload = responses_payload_to_chat(
        {
            "model": "gpt-4o",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "hello"},
                        {"type": "input_image", "image_url": "data:image/png;base64,abc"},
                        "just raw string in array",
                    ]
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_99",
                    "output": {"status": "success", "data": [1, 2, 3]},
                }
            ],
            "max_tokens": 500,
        }
    )

    assert payload["model"] == "gpt-4o"
    assert payload["max_tokens"] == 500
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        {"type": "text", "text": "just raw string in array"},
    ]
    assert payload["messages"][1]["role"] == "tool"
    assert payload["messages"][1]["tool_call_id"] == "call_99"
    assert json.loads(payload["messages"][1]["content"]) == {"status": "success", "data": [1, 2, 3]}


def test_chat_body_to_response_no_choices_or_empty():
    response = chat_body_to_response(
        {},
        requested_model="auto",
        response_id="resp_empty",
    )
    assert response["id"] == "resp_empty"
    assert response["output_text"] == ""
    assert len(response["output"]) == 1
    assert response["output"][0]["type"] == "message"
    assert response["output"][0]["content"] == []
    assert response["usage"] is None


def test_responses_stream_mapper_no_text_only_tools():
    mapper = ResponsesStreamMapper(response_id="resp_tool_only")
    block = "data: " + json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_x",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Sydney"}',
                                },
                            }
                        ]
                    },
                }
            ]
        }
    )

    events = mapper.events_from_openai_sse(block)
    assert any("response.function_call_arguments.delta" in event for event in events)
    assert any("response.output_item.added" in event for event in events)

    done_events = mapper.events_from_openai_sse("data: [DONE]")
    done_str = "\n".join(done_events)

    # Should not have message output items, only tool call item
    assert "response.output_item.done" in done_str
    assert "response.completed" in done_str
    assert "get_weather" in done_str
    assert "Sydney" in done_str
    assert "output_index: 0" in done_str or '"output_index": 0' in done_str
