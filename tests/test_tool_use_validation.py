from __future__ import annotations

from app.capability_probes import evaluate_tool_use_probe_response
from app.capability_runtime import adjust_capabilities_from_traffic
from app.model_catalog import ModelCatalog, route_id_for
from app.capability_tags import CapabilityClaim, apply_probe_claims, tags_to_capabilities
from app.tool_use_validation import (
    evaluate_tool_use_outcome,
    response_fakes_tool_use_in_text,
    should_abort_tool_stream_early,
    tool_use_response_mandatory,
)


def test_tool_use_response_mandatory_when_tool_choice_required():
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "x", "parameters": {}}}],
        "tool_choice": "required",
    }
    assert tool_use_response_mandatory(payload) is True


def test_tool_history_does_not_make_tool_response_mandatory():
    payload = {
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "sunny", "tool_call_id": "1"},
            {"role": "user", "content": "thanks"},
        ],
        "tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}],
        "tool_choice": "auto",
    }
    assert tool_use_response_mandatory(payload) is False


def test_text_reply_with_tools_and_auto_choice_is_neutral():
    payload = {
        "tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}],
        "tool_choice": "auto",
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "ok", "tool_call_id": "1"},
        ],
    }
    body = {"choices": [{"message": {"content": "Here is what I found for you."}}]}
    assert evaluate_tool_use_outcome(payload, body) == "neutral"


def test_detects_fake_tool_json_in_assistant_text():
    body = {
        "choices": [
            {
                "message": {
                    "content": (
                        'Sure! {"name": "search", "arguments": {"q": "weather"}}'
                    )
                }
            }
        ]
    }
    assert response_fakes_tool_use_in_text(body) is True
    assert evaluate_tool_use_outcome(
        {
            "tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}],
            "tool_choice": "required",
            "messages": [],
        },
        body,
    ) == "unsupported"


def test_valid_tool_calls_are_supported():
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
    payload = {
        "tools": [{"type": "function", "function": {"name": "echo", "parameters": {}}}],
        "tool_choice": "required",
        "messages": [],
    }
    assert evaluate_tool_use_outcome(payload, body) == "supported"


def test_runtime_demotes_fake_tool_text(tmp_path):
    catalog = ModelCatalog(str(tmp_path / "catalog.json"))
    route_id = route_id_for("nvidia", "moonshotai/kimi-k2.6")
    catalog.replace_routes(
        [
            {
                "route_id": route_id,
                "provider_name": "nvidia",
                "model_id": "moonshotai/kimi-k2.6",
                "display_name": "Kimi K2.6",
                "rank": 1,
                "enabled": True,
                "tags": ["text", "tool-use"],
                "capabilities": {
                    "text": {
                        "tag": "text",
                        "status": "supported",
                        "source": "probe",
                        "confidence": "high",
                        "checked_at": 1,
                        "evidence": "probe",
                    },
                    "tool-use": {
                        "tag": "tool-use",
                        "status": "supported",
                        "source": "probe",
                        "confidence": "high",
                        "checked_at": 1,
                        "evidence": "probe",
                    },
                },
            }
        ]
    )
    adjust_capabilities_from_traffic(
        catalog,
        route_id=route_id,
        required_capabilities=frozenset({"text", "tool-use"}),
        payload={
            "messages": [{"role": "user", "content": "run tool"}],
            "tools": [{"type": "function", "function": {"name": "alpha", "parameters": {}}}],
            "tool_choice": "required",
        },
        response_body={
            "choices": [
                {
                    "message": {
                        "content": 'I will call {"name":"alpha","arguments":{}} now.'
                    }
                }
            ]
        },
    )
    route = next(route for route in catalog.all_routes() if route.route_id == route_id)
    assert "tool-use" not in route.tags
    assert route.capabilities["tool-use"].status == "unsupported"
    assert route.capabilities["tool-use"].source == "runtime"


def test_should_abort_tool_stream_early_on_fake_text():
    payload = {
        "tools": [{"type": "function", "function": {"name": "alpha", "parameters": {}}}],
        "tool_choice": "required",
        "messages": [],
    }
    assert should_abort_tool_stream_early(
        payload,
        text='{"name":"alpha","arguments":{}}',
        saw_tool_calls=False,
    )


def test_should_not_abort_long_preamble_when_tool_choice_auto():
    payload = {
        "tools": [{"type": "function", "function": {"name": "alpha", "parameters": {}}}],
        "tool_choice": "auto",
        "messages": [
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "a", "arguments": "{}"}}]},
            {"role": "tool", "content": "ok", "tool_call_id": "1"},
        ],
    }
    assert should_abort_tool_stream_early(
        payload,
        text="x" * 500,
        saw_tool_calls=False,
    ) is False


def test_probe_rejects_text_only_fake_tool_response():
    body = {
        "choices": [
            {
                "message": {
                    "content": '[{"type":"function","function":{"name":"echo","arguments":"{}"}}]'
                }
            }
        ]
    }
    assert evaluate_tool_use_probe_response(body, expected_function="echo") == "unsupported"
