from __future__ import annotations

from app.tool_call_stream import ToolCallStreamAccumulator


def test_accumulator_reassembles_interleaved_tool_calls():
    accumulator = ToolCallStreamAccumulator()
    accumulator.ingest(
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
                                "function": {"name": "alpha", "arguments": '{"x"'},
                            },
                            {
                                "index": 1,
                                "id": "call_b",
                                "type": "function",
                                "function": {"name": "beta", "arguments": '{"y"'},
                            },
                        ]
                    },
                }
            ]
        }
    )
    accumulator.ingest(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 1, "function": {"arguments": ":2}"}},
                            {"index": 0, "function": {"arguments": ":1}"}},
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )

    body = accumulator.to_chat_body()
    calls = body["choices"][0]["message"]["tool_calls"]

    assert [call["id"] for call in calls] == ["call_a", "call_b"]
    assert calls[0]["function"]["arguments"] == '{"x":1}'
    assert calls[1]["function"]["arguments"] == '{"y":2}'
    assert body["choices"][0]["finish_reason"] == "tool_calls"


def test_accumulator_preserves_incomplete_call_for_terminal_validation():
    accumulator = ToolCallStreamAccumulator()
    accumulator.ingest({"choices": [{"delta": {"tool_calls": [{"index": 0}]}}]})

    call = accumulator.to_chat_body()["choices"][0]["message"]["tool_calls"][0]

    assert accumulator.has_tool_calls is True
    assert call["id"] == ""
    assert call["function"] == {"name": "", "arguments": ""}


def test_accumulator_handles_cumulative_ids_and_preserves_stream_metadata():
    accumulator = ToolCallStreamAccumulator()
    accumulator.ingest(
        {
            "id": "chatcmpl-upstream",
            "model": "provider/model",
            "created": 123,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call_",
                                "function": {"name": "look", "arguments": '{"q":'},
                            }
                        ]
                    },
                }
            ],
        }
    )
    accumulator.ingest(
        {
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
    )

    body = accumulator.to_chat_body()
    call = body["choices"][0]["message"]["tool_calls"][0]

    assert call["id"] == "call_abc"
    assert call["function"] == {"name": "lookup", "arguments": '{"q":"x"}'}
    assert body["id"] == "chatcmpl-upstream"
    assert body["model"] == "provider/model"
    assert body["created"] == 123
    assert body["usage"]["total_tokens"] == 5
