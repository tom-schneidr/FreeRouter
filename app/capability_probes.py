from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
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
AGENTIC_PROBE_TAG = "tool-use.multi-turn-stability"
AGENTIC_STABILITY_PROBE_MAX_RANK = 10

ECHO_PROBE_MESSAGE = "openclaw-probe-7f3a"
ADD_PROBE_ARGUMENTS = {"a": 17, "b": 25}
CONTINUATION_PROBE_REPLY = "OPENCLAW_TOOL_RESULT_OK_42"
STABILITY_PROBE_REPLY = "OPENCLAW_MULTI_TURN_OK_84"

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
    multi_turn_stability: CapabilityStatus = "inconclusive"

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
            f"tool_result_continuation={self.tool_result_continuation}; "
            f"multi_turn_stability={self.multi_turn_stability}"
        )

    def subclaim_statuses(self) -> dict[str, CapabilityStatus]:
        """Return the profile as machine-readable capability dimensions."""
        return {
            "tool-use.required-exact-call": self.required_exact_call,
            "tool-use.auto-selection": self.auto_selection,
            "tool-use.tool-result-continuation": self.tool_result_continuation,
            "tool-use.multi-turn-stability": self.multi_turn_stability,
            "tool-use.call-id-integrity": self.required_exact_call,
            "tool-use.argument-schema-integrity": self.required_exact_call,
        }


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


def multi_turn_stability_probe_payloads(model_id: str) -> list[dict[str, Any]]:
    """Build a bounded two-tool sequence used for agentic protocol probing.

    The probe is deliberately small.  It checks that a model advances after a
    tool result instead of replaying the first action, without pretending to be
    a general model-quality benchmark.
    """
    return [
        {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": "Call echo with message OPENCLAW_STABILITY_ECHO. Do not answer in text.",
                }
            ],
            "tools": [ECHO_TOOL, ADD_TOOL],
            "tool_choice": "auto",
            "max_tokens": 96,
            "temperature": 0,
        },
    ]


def multi_turn_stability_continuation_payload(
    model_id: str,
    first_call: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": "Call echo with message OPENCLAW_STABILITY_ECHO.",
            },
            {"role": "assistant", "content": None, "tool_calls": [first_call]},
            {
                "role": "tool",
                "tool_call_id": first_call.get("id"),
                "content": json.dumps({"echo": "OPENCLAW_STABILITY_ECHO"}),
            },
            {"role": "user", "content": "Now use add to add 17 and 25. Do not answer in text."},
        ],
        "tools": [ECHO_TOOL, ADD_TOOL],
        "tool_choice": "auto",
        "max_tokens": 96,
        "temperature": 0,
    }


def multi_turn_stability_final_payload(
    model_id: str,
    first_call: dict[str, Any],
    second_call: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": "Call echo with message OPENCLAW_STABILITY_ECHO.",
            },
            {"role": "assistant", "content": None, "tool_calls": [first_call]},
            {
                "role": "tool",
                "tool_call_id": first_call.get("id"),
                "content": json.dumps({"echo": "OPENCLAW_STABILITY_ECHO"}),
            },
            {"role": "user", "content": "Now use add to add 17 and 25."},
            {"role": "assistant", "content": None, "tool_calls": [second_call]},
            {
                "role": "tool",
                "tool_call_id": second_call.get("id"),
                "content": json.dumps({"sum": 42, "receipt": STABILITY_PROBE_REPLY}),
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


def evaluate_multi_turn_stability_response(
    body: dict[str, Any],
    *,
    expected_function: str,
) -> CapabilityStatus:
    if response_fakes_tool_use_in_text(body):
        return "unsupported"
    calls = function_tool_calls_from_body(body)
    if len(calls) != 1:
        return "unsupported" if calls else "inconclusive"
    call = calls[0]
    function = call.get("function")
    if not isinstance(function, dict) or function.get("name") != expected_function:
        return "unsupported"
    if not isinstance(call.get("id"), str) or not call["id"].strip():
        return "inconclusive"
    args = parse_function_tool_arguments(function.get("arguments"))
    if args is None:
        return "inconclusive"
    if expected_function == "echo":
        return (
            "supported"
            if args == {"message": "OPENCLAW_STABILITY_ECHO"}
            else "unsupported"
        )
    if expected_function == "add":
        return "supported" if args == ADD_PROBE_ARGUMENTS else "unsupported"
    return "inconclusive"


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


def _retry_after_seconds(headers: Any, *, default: int) -> int:
    if isinstance(headers, Mapping):
        raw = headers.get("retry-after") or headers.get("Retry-After")
        try:
            return max(1, min(24 * 3600, int(float(str(raw)))))
        except (TypeError, ValueError):
            pass
    return default


def _probe_claim(
    *,
    tag: str,
    status: CapabilityStatus,
    checked_at: int,
    evidence: str,
    confidence: Literal["high", "medium", "low"],
    reason: str = "",
    retry_after: int | None = None,
) -> CapabilityClaim:
    return CapabilityClaim(
        tag=tag,
        status=status,
        source="probe",
        confidence=confidence,
        checked_at=checked_at,
        evidence=evidence[:500],
        last_attempted_at=checked_at,
        next_probe_at=(checked_at + retry_after) if retry_after is not None else None,
        reason=reason,
    )


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
    if tag == AGENTIC_PROBE_TAG:
        payload = multi_turn_stability_probe_payloads(route.model_id)[0]
    if payload is None:
        return _probe_claim(
            tag=tag,
            status="inconclusive",
            confidence="low",
            checked_at=checked_at,
            evidence="No probe payload for tag",
            reason="no_probe_payload",
            retry_after=3600,
        )
    if not provider.is_configured:
        return _probe_claim(
            tag=tag,
            status="inconclusive",
            confidence="low",
            checked_at=checked_at,
            evidence="Provider missing API key",
            reason="provider_unconfigured",
            retry_after=6 * 3600,
        )
    reason = "verified"
    retry_after: int | None = None
    try:
        async with asyncio.timeout(timeout_seconds):
            if tag == "tool-use":
                profile = await _probe_tool_use_profile(
                    provider,
                    client,
                    route.model_id,
                )
                status, evidence = profile.status, profile.evidence()
            elif tag == AGENTIC_PROBE_TAG:
                status = await _probe_multi_turn_stability(
                    provider,
                    client,
                    route.model_id,
                )
                evidence = f"Multi-turn tool stability probe: {status}"
            else:
                response = await provider.chat_completion(client, payload, route.model_id)
                status = evaluate_probe_response(tag, response.body)
                evidence = f"Probe HTTP {response.status_code}"
    except ProviderRateLimited as exc:
        return _probe_claim(
            tag=tag,
            status="inconclusive",
            confidence="medium",
            checked_at=checked_at,
            evidence=f"Rate limited: {exc}",
            reason="rate_limited",
            retry_after=_retry_after_seconds(exc.headers, default=3600),
        )
    except ProviderError as exc:
        if _provider_error_indicates_unsupported(tag, exc):
            status: CapabilityStatus = "unsupported"
            evidence = str(exc)[:240]
            reason = "provider_rejected"
            retry_after = None
        else:
            status = "inconclusive"
            evidence = str(exc)[:240]
            reason = "provider_error"
            retry_after = 3600
    except (httpx.TimeoutException, TimeoutError):
        return _probe_claim(
            tag=tag,
            status="inconclusive",
            confidence="low",
            checked_at=checked_at,
            evidence="Probe timed out",
            reason="timeout",
            retry_after=3600,
        )
    except Exception as exc:  # noqa: BLE001 — probe boundary
        return _probe_claim(
            tag=tag,
            status="inconclusive",
            confidence="low",
            checked_at=checked_at,
            evidence=str(exc)[:240],
            reason="probe_exception",
            retry_after=3600,
        )

    confidence: Literal["high", "medium", "low"] = (
        "high" if status == "supported" else "medium" if status == "unsupported" else "low"
    )
    if status == "inconclusive" and reason == "verified":
        reason = "inconclusive"
    if status == "inconclusive" and retry_after is None:
        retry_after = 3600
    return _probe_claim(
        tag=tag,
        status=status,
        confidence=confidence,
        checked_at=checked_at,
        evidence=evidence,
        reason=reason if status != "inconclusive" else (reason or "inconclusive"),
        retry_after=retry_after,
    )


async def _probe_tool_use_profile(
    provider: ProviderAdapter,
    client: httpx.AsyncClient,
    model_id: str,
) -> ToolUseProbeProfile:
    payloads = tool_use_probe_payloads(model_id)

    echo_name, echo_payload = payloads[0]
    echo_response = await provider.chat_completion(client, echo_payload, model_id)
    echo_status = evaluate_tool_use_probe_response(
        echo_response.body,
        expected_function=echo_name,
    )
    profile = ToolUseProbeProfile(required_exact_call=echo_status)
    if echo_status != "supported":
        return profile

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
        return profile

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
    return profile


async def _probe_tool_use_variants(
    provider: ProviderAdapter,
    client: httpx.AsyncClient,
    model_id: str,
) -> tuple[CapabilityStatus, str]:
    """Compatibility wrapper retained for callers outside the probe service."""
    profile = await _probe_tool_use_profile(provider, client, model_id)
    return profile.status, profile.evidence()


async def _probe_multi_turn_stability(
    provider: ProviderAdapter,
    client: httpx.AsyncClient,
    model_id: str,
) -> CapabilityStatus:
    first_response = await provider.chat_completion(
        client,
        multi_turn_stability_probe_payloads(model_id)[0],
        model_id,
    )
    first_status = evaluate_multi_turn_stability_response(
        first_response.body,
        expected_function="echo",
    )
    if first_status != "supported":
        return first_status
    first_call = function_tool_calls_from_body(first_response.body)[0]

    second_response = await provider.chat_completion(
        client,
        multi_turn_stability_continuation_payload(model_id, first_call),
        model_id,
    )
    second_status = evaluate_multi_turn_stability_response(
        second_response.body,
        expected_function="add",
    )
    if second_status != "supported":
        return second_status
    second_call = function_tool_calls_from_body(second_response.body)[0]

    final_response = await provider.chat_completion(
        client,
        multi_turn_stability_final_payload(model_id, first_call, second_call),
        model_id,
    )
    final_text = _assistant_text(final_response.body).strip()
    if function_tool_calls_from_body(final_response.body):
        return "unsupported"
    return "supported" if final_text == STABILITY_PROBE_REPLY else "unsupported"


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
        claim = await probe_route_tag(provider, client, route, tag)
        results[tag] = claim
        if tag == "tool-use":
            profile = _profile_from_evidence(claim.evidence)
            if profile is not None:
                for subtag, status in profile.subclaim_statuses().items():
                    results[subtag] = CapabilityClaim(
                        tag=subtag,
                        status=status,
                        source=claim.source,
                        confidence=claim.confidence,
                        checked_at=claim.checked_at,
                        evidence=claim.evidence,
                        last_attempted_at=claim.last_attempted_at,
                        next_probe_at=claim.next_probe_at,
                        reason=claim.reason,
                    )
    return results


def _profile_from_evidence(evidence: str) -> ToolUseProbeProfile | None:
    prefix = "OpenClaw tool profile:"
    if not evidence.startswith(prefix):
        return None
    values: dict[str, CapabilityStatus] = {}
    for part in evidence[len(prefix) :].split(";"):
        key, separator, raw_value = part.strip().partition("=")
        if not separator:
            continue
        if raw_value in {"supported", "unsupported", "inconclusive", "unknown"}:
            values[key] = raw_value  # type: ignore[assignment]
    return ToolUseProbeProfile(
        required_exact_call=values.get("required_exact_call", "inconclusive"),
        auto_selection=values.get("auto_selection", "inconclusive"),
        tool_result_continuation=values.get("tool_result_continuation", "inconclusive"),
        multi_turn_stability=values.get("multi_turn_stability", "inconclusive"),
    )
