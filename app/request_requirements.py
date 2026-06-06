"""Pure chat payload inspection and capability requirement derivation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RequestRequirements:
    required_capabilities: frozenset[str] = field(default_factory=frozenset)


def chat_request_requirements(payload: dict[str, Any]) -> RequestRequirements:
    """Derive required route capabilities from an OpenAI chat-completions payload."""
    caps: set[str] = {"text"}
    caps.update(_capabilities_from_messages(payload.get("messages")))
    caps.update(_capabilities_from_tools(payload))
    caps.update(_capabilities_from_response_format(payload))
    if _has_reasoning_config(payload):
        caps.add("reasoning")
    return RequestRequirements(required_capabilities=frozenset(caps))


def with_extra_capabilities(
    requirements: RequestRequirements,
    *extra: str,
) -> RequestRequirements:
    if not extra:
        return requirements
    return RequestRequirements(
        required_capabilities=requirements.required_capabilities | frozenset(extra)
    )


def _capabilities_from_messages(messages: Any) -> set[str]:
    caps: set[str] = set()
    if not isinstance(messages, list):
        return caps
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "tool":
            caps.add("tool-use")
        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                caps.add("tool-use")
        if _content_has_vision(message.get("content")):
            caps.add("vision")
    return caps


def _content_has_vision(content: Any) -> bool:
    if _part_is_image(content):
        return True
    if isinstance(content, list):
        return any(_part_is_image(part) for part in content)
    return False


def _part_is_image(part: Any) -> bool:
    if not isinstance(part, dict):
        return False
    part_type = part.get("type")
    if part_type in {"image_url", "input_image"}:
        return True
    if "image_url" in part:
        return True
    return False


def _capabilities_from_tools(payload: dict[str, Any]) -> set[str]:
    caps: set[str] = set()
    tools = payload.get("tools")
    if isinstance(tools, list):
        function_tools = [
            tool
            for tool in tools
            if isinstance(tool, dict) and tool.get("type") == "function"
        ]
        if function_tools:
            caps.add("tool-use")
    if _tool_choice_requires_function_tools(payload.get("tool_choice")):
        caps.add("tool-use")
    return caps


def _tool_choice_requires_function_tools(tool_choice: Any) -> bool:
    if tool_choice is None:
        return False
    if tool_choice == "required":
        return True
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            return True
        if "function" in tool_choice:
            return True
    return False


def _capabilities_from_response_format(payload: dict[str, Any]) -> set[str]:
    caps: set[str] = set()
    response_format = payload.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
        caps.add("json-schema")
    return caps


def _has_reasoning_config(payload: dict[str, Any]) -> bool:
    reasoning_effort = payload.get("reasoning_effort")
    if isinstance(reasoning_effort, str) and reasoning_effort.strip():
        return True
    reasoning = payload.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if isinstance(effort, str) and effort.strip():
            return True
    return False
