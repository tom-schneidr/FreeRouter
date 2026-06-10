from __future__ import annotations

from app.capability_probes import evaluate_probe_response


def test_evaluate_tool_use_probe_accepts_valid_tool_call():
    body = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "echo", "arguments": '{"message":"ping"}'},
                        }
                    ]
                }
            }
        ]
    }
    assert evaluate_probe_response("tool-use", body) == "supported"


def test_evaluate_tool_use_probe_rejects_missing_tool_calls():
    body = {"choices": [{"message": {"content": "hello"}}]}
    assert evaluate_probe_response("tool-use", body) == "unsupported"
