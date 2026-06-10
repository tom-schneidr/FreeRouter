from __future__ import annotations

import json
import time
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

# 1x1 red PNG
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

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
                "messages": [{"role": "user", "content": "Call echo with message ping"}],
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
                "messages": [{"role": "user", "content": "Call add with a=2 and b=3"}],
                "tools": [ADD_TOOL],
                "tool_choice": "required",
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
    first = calls[0]
    fn = first.get("function")
    if not isinstance(fn, dict) or fn.get("name") != expected_function:
        return "inconclusive"
    args = parse_function_tool_arguments(fn.get("arguments"))
    if args is None:
        return "inconclusive"
    if expected_function == "echo":
        return "supported" if isinstance(args.get("message"), str) else "inconclusive"
    if expected_function == "add":
        a = args.get("a")
        b = args.get("b")
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return "supported"
        return "inconclusive"
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
        return "supported" if isinstance(parsed, dict) and isinstance(parsed.get("x"), str) else "unsupported"

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
        return any(
            term in haystack
            for term in (
                "tool",
                "function",
                "tools are not",
                "does not support tools",
                "unsupported tool",
            )
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
    except httpx.TimeoutException:
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
    last_status: CapabilityStatus = "unsupported"
    for probe_name, probe_payload in tool_use_probe_payloads(model_id):
        response = await provider.chat_completion(client, probe_payload, model_id)
        status = evaluate_tool_use_probe_response(
            response.body,
            expected_function=probe_name,
        )
        if status != "supported":
            return status, f"{probe_name} probe HTTP {response.status_code}: {status}"
        last_status = status
    return last_status, "echo+add tool probes passed"


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
