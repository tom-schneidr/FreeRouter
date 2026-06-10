from __future__ import annotations

import json
import re
from typing import Any, Literal

from app.request_requirements import (
    _capabilities_from_messages,
    _tool_choice_requires_function_tools,
)

ToolUseOutcome = Literal["supported", "unsupported", "neutral"]

_FAKE_TOOL_TEXT_MARKERS = (
    '"tool_calls"',
    "'tool_calls'",
    "tool_call",
    "<tool",
    "</tool",
    "function_call",
    '"type": "function"',
    '"type":"function"',
    '"name":',
    '{"name":',
    '{"function"',
    "```json",
    "<function=",
    "invoke(",
)

_FAKE_TOOL_TEXT_RE = re.compile(
    r"(\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"arguments\"|\[\s*\{\s*\"type\"\s*:\s*\"function\")",
    re.IGNORECASE,
)


def payload_requires_function_tools(payload: dict[str, Any]) -> bool:
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return False
    return any(isinstance(tool, dict) and tool.get("type") == "function" for tool in tools)


def tool_use_response_mandatory(payload: dict[str, Any]) -> bool:
    """True when tool_choice requires structured tool_calls (not optional auto/none).

    Prior tool or assistant tool_calls in message history only means the session
    can use tools — it does not require every reply to be a tool_call. Agent
    clients (e.g. OpenClaw) always send tool history while still allowing text.
    """
    return _tool_choice_requires_function_tools(payload.get("tool_choice"))


def assistant_text_from_body(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def function_tool_calls_from_body(body: dict[str, Any]) -> list[dict[str, Any]]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return []
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [call for call in tool_calls if isinstance(call, dict)]


def parse_function_tool_arguments(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def function_tool_call_is_valid(call: dict[str, Any]) -> bool:
    if call.get("type") not in {None, "function"}:
        return False
    fn = call.get("function")
    if not isinstance(fn, dict):
        return False
    name = fn.get("name")
    if not isinstance(name, str) or not name.strip():
        return False
    return parse_function_tool_arguments(fn.get("arguments")) is not None


def response_has_valid_function_tool_calls(body: dict[str, Any]) -> bool:
    calls = function_tool_calls_from_body(body)
    return bool(calls) and all(function_tool_call_is_valid(call) for call in calls)


def response_fakes_tool_use_in_text(body: dict[str, Any]) -> bool:
    text = assistant_text_from_body(body).strip()
    if not text or response_has_valid_function_tool_calls(body):
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in _FAKE_TOOL_TEXT_MARKERS):
        return True
    return _FAKE_TOOL_TEXT_RE.search(text) is not None


def evaluate_tool_use_outcome(
    payload: dict[str, Any],
    body: dict[str, Any],
) -> ToolUseOutcome:
    if not payload_requires_function_tools(payload):
        return "neutral"
    if response_has_valid_function_tool_calls(body):
        return "supported"
    if tool_use_response_mandatory(payload) or response_fakes_tool_use_in_text(body):
        return "unsupported"
    return "neutral"


def stream_chunk_commits_for_tool_use(outbound_payload: dict[str, Any]) -> bool:
    """When tool calls are mandatory, do not commit a stream on assistant text alone."""
    return not tool_use_response_mandatory(outbound_payload)


def should_abort_tool_stream_early(
    payload: dict[str, Any],
    *,
    text: str,
    saw_tool_calls: bool,
) -> bool:
    """Stop waiting on a stream that is faking tool_calls in prose when tools are required."""
    if saw_tool_calls or not tool_use_response_mandatory(payload):
        return False
    return response_fakes_tool_use_in_text({"choices": [{"message": {"content": text}}]})


ROUTING_SSE_KEEPALIVE = ": freerouter routing\r\n\r\n"
