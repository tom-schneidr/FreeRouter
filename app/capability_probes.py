from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from app.capability_tags import CapabilityClaim, CapabilityStatus
from app.model_catalog import ModelRoute
from app.providers.base import ProviderAdapter, ProviderError, ProviderRateLimited
from app.tool_use_validation import (
    function_tool_calls_from_body,
    parse_function_tool_arguments,
    response_fakes_tool_use_in_text,
    response_has_valid_function_tool_calls,
)

PROBE_TAGS = ("text", "tool-use", "vision", "json-schema")

ECHO_PROBE_MESSAGE = "openclaw-probe-7f3a"
ADD_PROBE_ARGUMENTS = {"a": 17, "b": 25}
CONTINUATION_PROBE_REPLY = "OPENCLAW_TOOL_RESULT_OK_42"

# 1x1 red PNG
_TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

ECHO_TOOL = {
    "type": "function",
    "function": {
        "name": "echo",
        "description": "Echo a message back",
        "parameters": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    },
}

ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    },
}


@dataclass(frozen=True)
class ToolUseProbeProfile:
    """Feature-level results collapsed into the legacy ``tool-use`` capability."""

    required_exact_call: CapabilityStatus = "inconclusive"
    auto_selection: CapabilityStatus = "inconclusive"
    tool_result_continuation: CapabilityStatus = "inconclusive"

    @property
    def status(self) -> CapabilityStatus:
        # The forced exact call establishes transport/schema compatibility. The
        # autonomous selection and continuation checks measure behavioral quality
        # and are retained as evidence for ranking, not used to erase capability.
        if self.required_exact_call == "supported":
            return "supported"
        return "inconclusive"

    def evidence(self) -> str:
        return (
            "OpenClaw tool profile: "
            f"required_exact_call={self.required_exact_call}; "
            f"auto_selection={self.auto_selection}; "
            f"tool_result_continuation={self.tool_result_continuation}"
        )


def probe_payload_for(tag: str, model_id: str) -> dict[str, Any] | None:
    if tag == "text":
        return {
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
            "max_tokens": 16,
            "temperature": 0,
        }
    if tag == "tool-use":
        return tool_use_probe_payloads(model_id)[0][1]
    if tag == "vision":
        return {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image? One word."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_B64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 16,
            "temperature": 0,
        }
    if tag == "json-schema":
        return {
            "model": model_id,
            "messages": [{"role": "user", "content": 'Return JSON {"x":"hi"}'}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "probe",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                        "required": ["x"],
                        "additionalProperties": False,
                    },
                },
            },
            "max_tokens": 32,
            "temperature": 0,
        }
    return None


def tool_use_probe_payloads(model_id: str) -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            "echo",
            {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Call the echo tool exactly once with message "
                            f"{ECHO_PROBE_MESSAGE}. Do not answer in text."
                        ),
                    }
                ],
                "tools": [ECHO_TOOL],
                "tool_choice": "required",
                "max_tokens": 96,
                "temperature": 0,
            },
        ),
        (
            "add",
            {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Use the appropriate tool to add 17 and 25. "
                            "Do not call echo and do not answer in text."
                        ),
                    }
                ],
                # The distractor and auto choice verify autonomous selection, which
                # is the path used by OpenClaw-style agents in normal operation.
                "tools": [ECHO_TOOL, ADD_TOOL],
                "tool_choice": "auto",
                "max_tokens": 96,
                "temperature": 0,
            },
        ),
    ]


def evaluate_tool_use_probe_response(
    body: dict[str, Any],
    *,
    expected_function: str,
) -> CapabilityStatus:
    if response_fakes_tool_use_in_text(body):
        return "unsupported"
    if not response_has_valid_function_tool_calls(body):
        return "unsupported"
    calls = function_tool_calls_from_body(body)
    if len(calls) != 1:
        return "inconclusive"
    first = calls[0]
    call_id = first.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        return "inconclusive"
    fn = first.get("function")
    if not isinstance(fn, dict) or fn.get("name") != expected_function:
        return "inconclusive"
    args = parse_function_tool_arguments(fn.get("arguments"))
    if args is None:
        return "inconclusive"
    if expected_function == "echo":
        return "supported" if args == {"message": ECHO_PROBE_MESSAGE} else "inconclusive"
    if expected_function == "add":
        return "supported" if args == ADD_PROBE_ARGUMENTS else "inconclusive"
    return "inconclusive"


def tool_result_continuation_probe_payload(
    model_id: str,
    add_call: dict[str, Any],
) -> dict[str, Any]:
    """Continue the exact tool call using its provider-generated call ID."""
    call_id = add_call.get("id")
    return {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Use the appropriate tool to add 17 and 25."},
            {"role": "assistant", "content": None, "tool_calls": [add_call]},
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps({"sum": 42, "receipt": CONTINUATION_PROBE_REPLY}),
            },
            {
                "role": "user",
                "content": "Reply with exactly the receipt string from the tool result.",
            },
        ],
        "tools": [ECHO_TOOL, ADD_TOOL],
        "tool_choice": "auto",
        "max_tokens": 48,
        "temperature": 0,
    }


def evaluate_tool_result_continuation_response(body: dict[str, Any]) -> CapabilityStatus:
    if function_tool_calls_from_body(body):
        return "unsupported"
    text = _assistant_text(body).strip()
    if text == CONTINUATION_PROBE_REPLY:
        return "supported"
    return "unsupported" if text else "inconclusive"


def evaluate_probe_response(tag: str, body: dict[str, Any]) -> CapabilityStatus:
    if tag == "text":
        return "supported" if _assistant_text(body) else "inconclusive"

    if tag == "tool-use":
        return evaluate_tool_use_probe_response(body, expected_function="echo")

    if tag == "vision":
        return "supported" if _assistant_text(body) else "inconclusive"

    if tag == "json-schema":
        text = _assistant_text(body)
        if not text:
            return "inconclusive"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return "unsupported"
        return (
            "supported"
            if isinstance(parsed, dict) and isinstance(parsed.get("x"), str)
            else "unsupported"
        )

    return "inconclusive"


def _assistant_text(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _provider_error_indicates_unsupported(tag: str, exc: ProviderError) -> bool:
    haystack = f"{exc} {exc.body or ''}".lower()
    if tag == "tool-use":
        feature_words = ("tool", "tools", "function", "function calling")
        unsupported_phrases = (
            "does not support",
            "doesn't support",
            "not supported",
            "unsupported",
            "not available for",
            "tools are disabled",
        )
        return any(word in haystack for word in feature_words) and any(
            phrase in haystack for phrase in unsupported_phrases
        )
    if tag == "vision":
        return any(term in haystack for term in ("image", "vision", "multimodal", "unsupported"))
    if tag == "json-schema":
        return any(
            term in haystack
            for term in ("response_format", "json_schema", "structured output", "schema")
        )
    return False


async def probe_route_tag(
    provider: ProviderAdapter,
    client: httpx.AsyncClient,
    route: ModelRoute,
    tag: str,
    *,
    timeout_seconds: float = 45.0,
) -> CapabilityClaim:
    checked_at = int(time.time())
    payload = probe_payload_for(tag, route.model_id)
    if payload is None:
        return CapabilityClaim(
            tag=tag,
            status="inconclusive",
            source="probe",
            confidence="low",
            checked_at=checked_at,
            evidence="No probe payload for tag",
        )
    if not provider.is_configured:
        return CapabilityClaim(
            tag=tag,
            status="inconclusive",
            source="probe",
            confidence="low",
            checked_at=checked_at,
            evidence="Provider missing API key",
        )
    try:
        async with asyncio.timeout(timeout_seconds):
            if tag == "tool-use":
                status, evidence = await _probe_tool_use_variants(
                    provider,
                    client,
                    route.model_id,
                )
            else:
                response = await provider.chat_completion(client, payload, route.model_id)
                status = evaluate_probe_response(tag, response.body)
                evidence = f"Probe HTTP {response.status_code}"
    except ProviderRateLimited as exc:
        return CapabilityClaim(
            tag=tag,
            status="inconclusive",
            source="probe",
            confidence="medium",
            checked_at=checked_at,
            evidence=f"Rate limited: {exc}",
        )
    except ProviderError as exc:
        if _provider_error_indicates_unsupported(tag, exc):
            status: CapabilityStatus = "unsupported"
            evidence = str(exc)[:240]
        else:
            status = "inconclusive"
            evidence = str(exc)[:240]
    except (httpx.TimeoutException, TimeoutError):
        return CapabilityClaim(
            tag=tag,
            status="inconclusive",
            source="probe",
            confidence="low",
            checked_at=checked_at,
            evidence="Probe timed out",
        )
    except Exception as exc:  # noqa: BLE001 — probe boundary
        return CapabilityClaim(
            tag=tag,
            status="inconclusive",
            source="probe",
            confidence="low",
            checked_at=checked_at,
            evidence=str(exc)[:240],
        )

    confidence: Literal["high", "medium", "low"] = (
        "high" if status == "supported" else "medium" if status == "unsupported" else "low"
    )
    return CapabilityClaim(
        tag=tag,
        status=status,
        source="probe",
        confidence=confidence,
        checked_at=checked_at,
        evidence=evidence,
    )


async def _probe_tool_use_variants(
    provider: ProviderAdapter,
    client: httpx.AsyncClient,
    model_id: str,
) -> tuple[CapabilityStatus, str]:
    payloads = tool_use_probe_payloads(model_id)

    echo_name, echo_payload = payloads[0]
    echo_response = await provider.chat_completion(client, echo_payload, model_id)
    echo_status = evaluate_tool_use_probe_response(
        echo_response.body,
        expected_function=echo_name,
    )
    profile = ToolUseProbeProfile(required_exact_call=echo_status)
    if echo_status != "supported":
        return profile.status, profile.evidence()

    add_name, add_payload = payloads[1]
    add_response = await provider.chat_completion(client, add_payload, model_id)
    add_status = evaluate_tool_use_probe_response(
        add_response.body,
        expected_function=add_name,
    )
    profile = ToolUseProbeProfile(
        required_exact_call=echo_status,
        auto_selection=add_status,
    )
    if add_status != "supported":
        return profile.status, profile.evidence()

    add_call = function_tool_calls_from_body(add_response.body)[0]
    continuation_payload = tool_result_continuation_probe_payload(model_id, add_call)
    continuation_response = await provider.chat_completion(
        client,
        continuation_payload,
        model_id,
    )
    continuation_status = evaluate_tool_result_continuation_response(continuation_response.body)
    profile = ToolUseProbeProfile(
        required_exact_call=echo_status,
        auto_selection=add_status,
        tool_result_continuation=continuation_status,
    )
    return profile.status, profile.evidence()


async def probe_route_capabilities(
    provider: ProviderAdapter,
    client: httpx.AsyncClient,
    route: ModelRoute,
    *,
    tags: tuple[str, ...] | None = None,
) -> dict[str, CapabilityClaim]:
    targets = tags or PROBE_TAGS
    results: dict[str, CapabilityClaim] = {}
    for tag in targets:
        results[tag] = await probe_route_tag(provider, client, route, tag)
    return results
