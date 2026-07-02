"""Cheap request-size estimates for routing preflight."""

from __future__ import annotations

import json
from typing import Any


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(_content_to_text(item) for item in content)
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        if "content" in content:
            return _content_to_text(content["content"])
    return str(content)


def _serialize_for_estimate(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return str(value)


def estimate_prompt_tokens(payload: dict[str, Any]) -> int:
    """Rough prompt token estimate including tools and tool-call history."""

    text_parts: list[str] = []
    messages = payload.get("messages") or []
    for message in messages:
        if not isinstance(message, dict):
            continue
        text_parts.append(_content_to_text(message.get("content", "")))
        tool_calls = message.get("tool_calls")
        if tool_calls:
            text_parts.append(_serialize_for_estimate(tool_calls))

    for item in payload.get("input") or []:
        text_parts.append(_content_to_text(item))

    for tool in payload.get("tools") or []:
        text_parts.append(_serialize_for_estimate(tool))

    character_count = sum(len(part) for part in text_parts)
    message_overhead = 4 * len(messages)
    tool_overhead = 8 * len(payload.get("tools") or [])
    return max(1, character_count // 4 + message_overhead + tool_overhead)


def estimate_completion_tokens(payload: dict[str, Any]) -> int:
    return int(payload.get("max_completion_tokens") or payload.get("max_tokens") or 0)


def estimate_total_request_tokens(payload: dict[str, Any]) -> tuple[int, int]:
    prompt_tokens = estimate_prompt_tokens(payload)
    return prompt_tokens, prompt_tokens + estimate_completion_tokens(payload)
