from __future__ import annotations

from app.capability_probes import (
    ADD_PROBE_ARGUMENTS,
    CONTINUATION_PROBE_REPLY,
    ECHO_PROBE_MESSAGE,
    ToolUseProbeProfile,
    _provider_error_indicates_unsupported,
    evaluate_multi_turn_stability_response,
    evaluate_probe_response,
    evaluate_tool_result_continuation_response,
    evaluate_tool_use_probe_response,
    multi_turn_stability_final_payload,
    probe_route_tag,
    tool_result_continuation_probe_payload,
    tool_use_probe_payloads,
)
from app.model_catalog import ModelRoute
from app.providers.base import ProviderError, ProviderResponse


def test_evaluate_tool_use_probe_accepts_valid_tool_call():
    body = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_probe_1",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": f'{{"message":"{ECHO_PROBE_MESSAGE}"}}',
                            },
                        }
                    ]
                }
            }
        ]
    }
    assert evaluate_probe_response("tool-use", body) == "supported"


def test_tool_error_classifier_distinguishes_bad_request_from_unsupported_transport():
    invalid_schema = ProviderError(
        "provider returned 400",
        status_code=400,
        body="Invalid function schema: required field is malformed",
    )
    unsupported = ProviderError(
        "provider returned 400",
        status_code=400,
        body="This model does not support tools",
    )

    assert _provider_error_indicates_unsupported("tool-use", invalid_schema) is False
    assert _provider_error_indicates_unsupported("tool-use", unsupported) is True


def test_evaluate_tool_use_probe_rejects_missing_tool_calls():
    body = {"choices": [{"message": {"content": "hello"}}]}
    assert evaluate_probe_response("tool-use", body) == "unsupported"


def test_tool_probe_requires_exact_values_and_nonempty_call_id():
    def body(call_id: str, arguments: str) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": "add", "arguments": arguments},
                            }
                        ]
                    }
                }
            ]
        }

    exact = '{"a":17,"b":25}'
    assert (
        evaluate_tool_use_probe_response(body("call_add", exact), expected_function="add")
        == "supported"
    )
    assert (
        evaluate_tool_use_probe_response(body("", exact), expected_function="add") == "inconclusive"
    )
    assert (
        evaluate_tool_use_probe_response(body("call_add", '{"a":2,"b":3}'), expected_function="add")
        == "inconclusive"
    )


def test_auto_selection_probe_has_distractor_and_exact_add_request():
    probes = dict(tool_use_probe_payloads("model"))
    add_payload = probes["add"]
    assert add_payload["tool_choice"] == "auto"
    assert [tool["function"]["name"] for tool in add_payload["tools"]] == ["echo", "add"]
    assert str(ADD_PROBE_ARGUMENTS["a"]) in add_payload["messages"][0]["content"]
    assert str(ADD_PROBE_ARGUMENTS["b"]) in add_payload["messages"][0]["content"]


def test_tool_result_continuation_reuses_provider_call_id_and_is_strict():
    add_call = {
        "id": "call_provider_42",
        "type": "function",
        "function": {"name": "add", "arguments": '{"a":17,"b":25}'},
    }
    payload = tool_result_continuation_probe_payload("model", add_call)
    assert payload["messages"][1]["tool_calls"] == [add_call]
    assert payload["messages"][2]["tool_call_id"] == "call_provider_42"
    assert CONTINUATION_PROBE_REPLY in payload["messages"][2]["content"]
    assert CONTINUATION_PROBE_REPLY not in payload["messages"][3]["content"]
    assert payload["tool_choice"] == "auto"
    assert (
        evaluate_tool_result_continuation_response(
            {"choices": [{"message": {"content": CONTINUATION_PROBE_REPLY}}]}
        )
        == "supported"
    )
    assert (
        evaluate_tool_result_continuation_response(
            {"choices": [{"message": {"content": f"Result: {CONTINUATION_PROBE_REPLY}"}}]}
        )
        == "unsupported"
    )
    assert (
        evaluate_tool_result_continuation_response(
            {"choices": [{"message": {"tool_calls": [add_call]}}]}
        )
        == "unsupported"
    )


def test_multi_turn_stability_requires_the_next_distinct_tool_call():
    echo = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "echo_1",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"message":"OPENCLAW_STABILITY_ECHO"}',
                            },
                        }
                    ]
                }
            }
        ]
    }
    repeated = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "echo_2",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"message":"OPENCLAW_STABILITY_ECHO"}',
                            },
                        }
                    ]
                }
            }
        ]
    }
    add = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "add_1",
                            "type": "function",
                            "function": {"name": "add", "arguments": '{"a":17,"b":25}'},
                        }
                    ]
                }
            }
        ]
    }

    assert evaluate_multi_turn_stability_response(echo, expected_function="echo") == "supported"
    assert evaluate_multi_turn_stability_response(repeated, expected_function="add") == "unsupported"
    assert evaluate_multi_turn_stability_response(add, expected_function="add") == "supported"
    final_payload = multi_turn_stability_final_payload("model", echo["choices"][0]["message"]["tool_calls"][0], add["choices"][0]["message"]["tool_calls"][0])
    assert "OPENCLAW_MULTI_TURN_OK_84" in final_payload["messages"][5]["content"]


def test_openclaw_probe_profile_separates_compatibility_from_behavior_quality():
    assert ToolUseProbeProfile("supported", "supported", "supported").status == "supported"
    assert ToolUseProbeProfile("supported", "inconclusive", "supported").status == "supported"
    assert ToolUseProbeProfile("supported", "unsupported", "supported").status == "supported"
    assert ToolUseProbeProfile("unsupported", "supported", "supported").status == "inconclusive"


async def test_tool_probe_runs_openclaw_profile_and_records_feature_evidence():
    class ProbeProvider:
        name = "probe-provider"
        is_configured = True

        def __init__(self) -> None:
            self.payloads: list[dict] = []

        async def chat_completion(self, client, payload, model_id):
            self.payloads.append(payload)
            if any(message.get("role") == "tool" for message in payload["messages"]):
                body = {"choices": [{"message": {"content": CONTINUATION_PROBE_REPLY}}]}
            elif len(payload["tools"]) == 2:
                body = {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "call_add_profile",
                                        "type": "function",
                                        "function": {
                                            "name": "add",
                                            "arguments": '{"a":17,"b":25}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            else:
                body = {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "call_echo_profile",
                                        "type": "function",
                                        "function": {
                                            "name": "echo",
                                            "arguments": (f'{{"message":"{ECHO_PROBE_MESSAGE}"}}'),
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            return ProviderResponse(self.name, 200, {}, body)

    provider = ProbeProvider()
    route = ModelRoute(
        route_id="probe-route",
        provider_name=provider.name,
        model_id="probe-model",
        display_name="Probe Model",
        rank=1,
        tags=["text"],
    )
    claim = await probe_route_tag(provider, None, route, "tool-use")

    assert claim.status == "supported"
    assert "required_exact_call=supported" in claim.evidence
    assert "auto_selection=supported" in claim.evidence
    assert "tool_result_continuation=supported" in claim.evidence
    assert len(provider.payloads) == 3
    assert provider.payloads[2]["messages"][2]["tool_call_id"] == "call_add_profile"
