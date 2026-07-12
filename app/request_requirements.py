"""Pure chat payload inspection and capability requirement derivation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RequestRequirements:
    required_capabilities: frozenset[str] = field(default_factory=frozenset)
    request_class: str = "normal"


def request_requires_tool_use(payload: dict[str, Any]) -> bool:
    """Return true when request shape requires a tool-capable route."""
    tools = payload.get("tools")
    if isinstance(tools, list) and any(_tool_definition_requires_tool_use(tool) for tool in tools):
        return True
    tool_choice = payload.get("tool_choice")
    if _tool_choice_requires_tool_route(tool_choice) and _payload_has_function_tools(payload):
        return True
    return _messages_include_tool_loop(payload.get("messages"))


def chat_request_requirements(payload: dict[str, Any]) -> RequestRequirements:
    """Derive required route capabilities from an OpenAI chat-completions payload."""
    caps: set[str] = {"text"}
    caps.update(_capabilities_from_messages(payload.get("messages")))
    caps.update(_capabilities_from_tools(payload))
    caps.update(_capabilities_from_response_format(payload))
    if _has_reasoning_config(payload):
        caps.add("reasoning")
    request_class = "tool-use" if request_requires_tool_use(payload) else "normal"
    if request_class == "tool-use":
        caps.add("tool-use")
    return RequestRequirements(
        required_capabilities=frozenset(caps),
        request_class=request_class,
    )


def with_extra_capabilities(
    requirements: RequestRequirements,
    *extra: str,
) -> RequestRequirements:
    if not extra:
        return requirements
    request_class = (
        "tool-use"
        if "tool-use" in (requirements.required_capabilities | frozenset(extra))
        else requirements.request_class
    )
    return RequestRequirements(
        required_capabilities=requirements.required_capabilities | frozenset(extra),
        request_class=request_class,
    )


def _capabilities_from_messages(messages: Any) -> set[str]:
    caps: set[str] = set()
    if _messages_include_tool_loop(messages):
        caps.add("tool-use")
    if not isinstance(messages, list):
        return caps
    for message in messages:
        if isinstance(message, dict) and _content_has_vision(message.get("content")):
            caps.add("vision")
    return caps


def _messages_include_tool_loop(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "tool":
            return True
        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                return True
        if _content_has_tool_result(message.get("content")):
            return True
    return False


def _content_has_tool_result(content: Any) -> bool:
    if isinstance(content, dict):
        return content.get("type") == "tool_result"
    if isinstance(content, list):
        return any(_content_has_tool_result(part) for part in content)
    return False


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
    if request_requires_tool_use(payload):
        caps.add("tool-use")
    return caps


def _tool_definition_requires_tool_use(tool: Any) -> bool:
    if not isinstance(tool, dict):
        return False
    return tool.get("type") == "function" and isinstance(tool.get("function"), dict)


def _payload_has_function_tools(payload: dict[str, Any]) -> bool:
    tools = payload.get("tools")
    return isinstance(tools, list) and any(
        _tool_definition_requires_tool_use(tool) for tool in tools
    )


def _tool_choice_requires_tool_route(tool_choice: Any) -> bool:
    if tool_choice in (None, "none"):
        return False
    if isinstance(tool_choice, dict) and tool_choice.get("type") in {
        "web_search_preview",
        "openrouter:web_search",
    }:
        return False
    return True


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
