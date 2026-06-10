from __future__ import annotations

from typing import Any

WEB_SEARCH_TOOL = {"type": "web_search_preview"}


def payload_with_required_web_search(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    prepared = dict(payload)
    tools = prepared.get("tools")
    if not isinstance(tools, list):
        tools = []
    elif not _has_web_search_tool(tools):
        tools = list(tools)
    if not _has_web_search_tool(tools):
        tools.append(dict(WEB_SEARCH_TOOL))
    prepared["tools"] = tools
    prepared["tool_choice"] = dict(WEB_SEARCH_TOOL)
    return prepared


def _has_web_search_tool(tools: list[Any]) -> bool:
    return any(
        isinstance(tool, dict) and tool.get("type") == WEB_SEARCH_TOOL["type"] for tool in tools
    )
