from __future__ import annotations

from app.capability_probes import evaluate_tool_use_probe_response
from app.capability_runtime import adjust_capabilities_from_traffic
from app.model_catalog import ModelCatalog, route_id_for
from app.tool_use_validation import (
    evaluate_tool_use_outcome,
    response_fakes_tool_use_in_text,
    response_promises_action_in_text,
    should_abort_tool_stream_early,
    tool_use_response_mandatory,
    validate_and_normalize_tool_response,
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


def test_initial_action_promise_can_be_rejected_for_auto_tool_choice():
    payload = {
        "tools": [{"type": "function", "function": {"name": "write_file", "parameters": {}}}],
        "tool_choice": "auto",
        "messages": [{"role": "user", "content": "build the file"}],
    }
    body = {
        "choices": [{"message": {"content": "Fair. Let me actually build the file right now."}}]
    }

    assert response_promises_action_in_text(body) is True
    assert evaluate_tool_use_outcome(payload, body) == "neutral"
    assert (
        evaluate_tool_use_outcome(
            payload,
            body,
            reject_initial_action_promise=True,
        )
        == "unsupported"
    )


def test_explanatory_build_text_is_not_action_promise():
    body = {"choices": [{"message": {"content": "I can explain how to build it."}}]}

    assert response_promises_action_in_text(body) is False


def test_action_promise_after_tool_loop_remains_neutral():
    payload = {
        "tools": [{"type": "function", "function": {"name": "write_file", "parameters": {}}}],
        "tool_choice": "auto",
        "messages": [
            {"role": "user", "content": "build the file"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "write_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "ok", "tool_call_id": "1"},
        ],
    }
    body = {"choices": [{"message": {"content": "I will verify it now."}}]}

    assert (
        evaluate_tool_use_outcome(
            payload,
            body,
            reject_initial_action_promise=True,
        )
        == "neutral"
    )


def test_detects_fake_tool_json_in_assistant_text():
    body = {
        "choices": [
            {"message": {"content": ('Sure! {"name": "search", "arguments": {"q": "weather"}}')}}
        ]
    }
    assert response_fakes_tool_use_in_text(body) is True
    assert (
        evaluate_tool_use_outcome(
            {
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}],
                "tool_choice": "required",
                "messages": [],
            },
            body,
        )
        == "unsupported"
    )


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


def test_runtime_records_fake_tool_text_without_demoting_transport_capability(tmp_path):
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
                {"message": {"content": 'I will call {"name":"alpha","arguments":{}} now.'}}
            ]
        },
    )
    route = next(route for route in catalog.all_routes() if route.route_id == route_id)
    assert "tool-use" in route.tags
    assert route.capabilities["tool-use"].status == "supported"


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


def test_synthesized_call_ids_are_stable_per_turn_but_distinct_across_turns():
    body = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "alpha", "arguments": "{}"},
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    base = {
        "tools": [{"type": "function", "function": {"name": "alpha", "parameters": {}}}],
        "tool_choice": "required",
    }
    first_payload = {**base, "messages": [{"role": "user", "content": "first"}]}
    second_payload = {**base, "messages": [{"role": "user", "content": "second"}]}

    first = validate_and_normalize_tool_response(first_payload, body)
    first_repeat = validate_and_normalize_tool_response(first_payload, body)
    second = validate_and_normalize_tool_response(second_payload, body)
    first_id = first.normalized_body["choices"][0]["message"]["tool_calls"][0]["id"]
    first_repeat_id = first_repeat.normalized_body["choices"][0]["message"]["tool_calls"][0]["id"]
    second_id = second.normalized_body["choices"][0]["message"]["tool_calls"][0]["id"]

    assert first_id == first_repeat_id
    assert first_id != second_id


def test_should_not_abort_long_preamble_when_tool_choice_auto():
    payload = {
        "tools": [{"type": "function", "function": {"name": "alpha", "parameters": {}}}],
        "tool_choice": "auto",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": {"name": "a", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "ok", "tool_call_id": "1"},
        ],
    }
    assert (
        should_abort_tool_stream_early(
            payload,
            text="x" * 500,
            saw_tool_calls=False,
        )
        is False
    )


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


def _tool_payload(
    *,
    tool_choice="auto",
    parallel_tool_calls=True,
):
    return {
        "messages": [{"role": "user", "content": "do the work"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "mode": {"type": "string", "enum": ["create", "append"]},
                            "lines": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["path", "mode"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
        ],
        "tool_choice": tool_choice,
        "parallel_tool_calls": parallel_tool_calls,
    }


def _tool_body(*calls, finish_reason="stop"):
    return {
        "choices": [
            {
                "message": {"content": None, "tool_calls": list(calls)},
                "finish_reason": finish_reason,
            }
        ]
    }


def test_request_aware_validation_normalizes_arguments_id_type_and_finish_reason():
    payload = _tool_payload()
    body = _tool_body(
        {
            "function": {
                "name": "write_file",
                "arguments": {"mode": "create", "path": "notes.txt", "lines": ["a"]},
            }
        }
    )

    first = validate_and_normalize_tool_response(payload, body)
    second = validate_and_normalize_tool_response(payload, body)

    assert first.is_valid is True
    assert first.outcome == "supported"
    call = first.normalized_body["choices"][0]["message"]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["id"].startswith("call_fr_")
    assert call["id"] == second.normalized_body["choices"][0]["message"]["tool_calls"][0]["id"]
    assert call["function"]["arguments"] == ('{"lines":["a"],"mode":"create","path":"notes.txt"}')
    assert first.normalized_body["choices"][0]["finish_reason"] == "tool_calls"
    assert "id" not in body["choices"][0]["message"]["tool_calls"][0]


def test_auto_choice_rejects_present_unknown_or_schema_invalid_calls():
    unknown = validate_and_normalize_tool_response(
        _tool_payload(),
        _tool_body(
            {
                "type": "function",
                "function": {"name": "delete_everything", "arguments": "{}"},
            }
        ),
    )
    invalid_schema = validate_and_normalize_tool_response(
        _tool_payload(),
        _tool_body(
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": '{"path":3,"mode":"overwrite","extra":true}',
                },
            }
        ),
    )

    assert unknown.outcome == "unsupported"
    assert {failure.category for failure in unknown.failures} == {"undeclared_function"}
    assert invalid_schema.outcome == "unsupported"
    assert invalid_schema.failures[0].category == "arguments_schema_mismatch"
    assert invalid_schema.failures[0].path.endswith(".path")


def test_schema_validation_enforces_required_enum_and_additional_properties():
    cases = [
        ({"path": "a"}, "mode"),
        ({"path": "a", "mode": "overwrite"}, "mode"),
        ({"path": "a", "mode": "create", "surprise": 1}, "surprise"),
    ]
    for arguments, expected_path in cases:
        result = validate_and_normalize_tool_response(
            _tool_payload(),
            _tool_body(
                {
                    "type": "function",
                    "function": {"name": "write_file", "arguments": arguments},
                }
            ),
        )
        assert result.failures[0].category == "arguments_schema_mismatch"
        assert result.failures[0].path.endswith(f".{expected_path}")


def test_tool_choice_none_required_and_named_are_enforced():
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path":"a"}'},
    }
    none_result = validate_and_normalize_tool_response(
        _tool_payload(tool_choice="none"),
        _tool_body(call),
    )
    required_result = validate_and_normalize_tool_response(
        _tool_payload(tool_choice="required"),
        {"choices": [{"message": {"content": "I cannot."}, "finish_reason": "stop"}]},
    )
    named_result = validate_and_normalize_tool_response(
        _tool_payload(tool_choice={"type": "function", "function": {"name": "write_file"}}),
        _tool_body(call),
    )

    assert none_result.failures[0].category == "tool_choice_none_violation"
    assert required_result.failures[0].category == "tool_choice_required_missing"
    assert named_result.failures[0].category == "tool_choice_named_mismatch"


def test_parallel_policy_and_unique_call_ids_are_enforced():
    calls = [
        {
            "id": "same",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"a"}'},
        },
        {
            "id": "same",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"b"}'},
        },
    ]
    result = validate_and_normalize_tool_response(
        _tool_payload(parallel_tool_calls=False),
        _tool_body(*calls),
    )

    assert {failure.category for failure in result.failures} == {
        "parallel_tool_calls_disabled",
        "duplicate_call_id",
    }


def test_non_object_and_malformed_json_arguments_have_distinct_categories():
    not_object = validate_and_normalize_tool_response(
        _tool_payload(),
        _tool_body(
            {
                "type": "function",
                "function": {"name": "write_file", "arguments": "[]"},
            }
        ),
    )
    malformed = validate_and_normalize_tool_response(
        _tool_payload(),
        _tool_body(
            {
                "type": "function",
                "function": {"name": "write_file", "arguments": '{"path":'},
            }
        ),
    )

    assert not_object.failures[0].category == "arguments_not_object"
    assert malformed.failures[0].category == "invalid_arguments_json"
