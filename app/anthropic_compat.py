"""Anthropic Messages API wire translation to/from OpenAI Chat Completions."""

from __future__ import annotations

import json
import uuid
from typing import Any

from app.api.gateway_response import resolved_request_model
from app.router import (
    _SSE_DONE,
    _event_block_data_payload,
)

SUPPORTED_TOP_LEVEL_FIELDS = frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "metadata",
        "system",
        "stop_sequences",
        "stream",
        "temperature",
        "top_p",
        "tools",
        "tool_choice",
    }
)

_FINISH_REASON_TO_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


def anthropic_error_body(error_type: str, message: str) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {"type": error_type, "message": message},
    }


def anthropic_stream_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


def messages_payload_to_chat(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anthropic Messages request into an OpenAI chat-completions payload."""
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")

    unsupported = sorted(set(payload) - SUPPORTED_TOP_LEVEL_FIELDS)
    if unsupported:
        raise ValueError(f"Unsupported request fields: {', '.join(unsupported)}")

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Request body must include a non-empty 'messages' array")

    max_tokens = payload.get("max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")

    chat_messages: list[dict[str, Any]] = []
    system = payload.get("system")
    if system is not None:
        chat_messages.append({"role": "system", "content": _system_to_chat_content(system)})

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] must be an object")
        role = message.get("role")
        if role not in {"user", "assistant"}:
            raise ValueError(f"messages[{index}].role must be 'user' or 'assistant'")
        chat_messages.extend(_anthropic_message_to_chat_messages(message, index))

    chat_payload: dict[str, Any] = {
        "model": payload.get("model") or "auto",
        "messages": chat_messages,
        "max_tokens": max_tokens,
    }
    if "temperature" in payload:
        chat_payload["temperature"] = payload["temperature"]
    if "top_p" in payload:
        chat_payload["top_p"] = payload["top_p"]
    if "stop_sequences" in payload:
        chat_payload["stop"] = payload["stop_sequences"]
    if payload.get("stream"):
        chat_payload["stream"] = True
    if "tools" in payload:
        chat_payload["tools"] = _anthropic_tools_to_chat(payload["tools"])
    if "tool_choice" in payload:
        chat_payload["tool_choice"] = _anthropic_tool_choice_to_chat(payload["tool_choice"])
    return chat_payload


def chat_body_to_anthropic_message(
    chat_body: dict[str, Any],
    *,
    requested_model: Any,
    message_id: str | None = None,
) -> dict[str, Any]:
    """Translate an OpenAI chat completion body into an Anthropic message response."""
    choice = _first_choice(chat_body)
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        message = {}

    content = _assistant_message_to_anthropic_content(message)
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    has_tool_use = any(block.get("type") == "tool_use" for block in content)
    stop_reason = (
        "tool_use"
        if has_tool_use
        else _FINISH_REASON_TO_STOP.get(str(finish_reason), "end_turn")
    )

    return {
        "id": message_id or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": resolved_request_model(requested_model),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": _anthropic_usage_from_chat(chat_body.get("usage")),
    }


def _system_to_chat_content(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts: list[str] = []
        for index, block in enumerate(system):
            if not isinstance(block, dict):
                raise ValueError(f"system[{index}] must be an object")
            if block.get("type") != "text":
                raise ValueError(f"Unsupported system block type: {block.get('type')}")
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError(f"system[{index}].text must be a string")
            parts.append(text)
        return "\n".join(parts)
    raise ValueError("system must be a string or an array of text blocks")


def _anthropic_message_to_chat_messages(message: dict[str, Any], index: int) -> list[dict[str, Any]]:
    role = message["role"]
    content = message.get("content")
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if not isinstance(content, list):
        raise ValueError(f"messages[{index}].content must be a string or array")

    if role == "assistant":
        return [_assistant_blocks_to_chat_message(content, index)]
    return _user_blocks_to_chat_messages(content, index)


def _assistant_blocks_to_chat_message(blocks: list[Any], message_index: int) -> dict[str, Any]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block_index, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise ValueError(f"messages[{message_index}].content[{block_index}] must be an object")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"messages[{message_index}].content[{block_index}].text must be a string"
                )
            text_parts.append(text)
            continue
        if block_type == "tool_use":
            tool_id = block.get("id")
            name = block.get("name")
            tool_input = block.get("input")
            if not isinstance(tool_id, str) or not isinstance(name, str):
                raise ValueError(
                    f"messages[{message_index}].content[{block_index}] tool_use requires id and name"
                )
            if not isinstance(tool_input, dict):
                raise ValueError(
                    f"messages[{message_index}].content[{block_index}].input must be an object"
                )
            tool_calls.append(
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(tool_input, separators=(",", ":")),
                    },
                }
            )
            continue
        raise ValueError(f"Unsupported assistant content block type: {block_type}")

    chat_message: dict[str, Any] = {"role": "assistant"}
    if text_parts:
        chat_message["content"] = "\n".join(text_parts)
    else:
        chat_message["content"] = None
    if tool_calls:
        chat_message["tool_calls"] = tool_calls
    if chat_message.get("content") is None and not tool_calls:
        chat_message["content"] = ""
    return chat_message


def _user_blocks_to_chat_messages(blocks: list[Any], message_index: int) -> list[dict[str, Any]]:
    content_parts: list[dict[str, Any]] = []
    tool_messages: list[dict[str, Any]] = []
    for block_index, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise ValueError(f"messages[{message_index}].content[{block_index}] must be an object")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"messages[{message_index}].content[{block_index}].text must be a string"
                )
            content_parts.append({"type": "text", "text": text})
            continue
        if block_type == "image":
            content_parts.append(_image_block_to_openai_part(block, message_index, block_index))
            continue
        if block_type == "tool_result":
            tool_use_id = block.get("tool_use_id")
            if not isinstance(tool_use_id, str):
                raise ValueError(
                    f"messages[{message_index}].content[{block_index}].tool_use_id must be a string"
                )
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": _tool_result_content_to_text(block.get("content")),
                }
            )
            continue
        raise ValueError(f"Unsupported user content block type: {block_type}")

    out: list[dict[str, Any]] = list(tool_messages)
    if content_parts:
        out.append({"role": "user", "content": content_parts})
    if not out:
        raise ValueError(f"messages[{message_index}] must include at least one content block")
    return out


def _image_block_to_openai_part(block: dict[str, Any], message_index: int, block_index: int) -> dict[str, Any]:
    source = block.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"messages[{message_index}].content[{block_index}].source must be an object")
    source_type = source.get("type")
    if source_type == "base64":
        media_type = source.get("media_type")
        data = source.get("data")
        if not isinstance(media_type, str) or not isinstance(data, str):
            raise ValueError(
                f"messages[{message_index}].content[{block_index}] base64 source requires media_type and data"
            )
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{data}"},
        }
    if source_type == "url":
        url = source.get("url")
        if not isinstance(url, str):
            raise ValueError(f"messages[{message_index}].content[{block_index}] url source requires url")
        return {"type": "image_url", "image_url": {"url": url}}
    raise ValueError(f"Unsupported image source type: {source_type}")


def _tool_result_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return "" if content is None else str(content)


def _anthropic_tools_to_chat(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list) or not tools:
        raise ValueError("tools must be a non-empty array when provided")
    chat_tools: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"tools[{index}] must be an object")
        name = tool.get("name")
        if not isinstance(name, str):
            raise ValueError(f"tools[{index}].name must be a string")
        input_schema = tool.get("input_schema")
        if not isinstance(input_schema, dict):
            raise ValueError(f"tools[{index}].input_schema must be an object")
        chat_tool: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": name,
                "parameters": input_schema,
            },
        }
        description = tool.get("description")
        if isinstance(description, str):
            chat_tool["function"]["description"] = description
        chat_tools.append(chat_tool)
    return chat_tools


def _anthropic_tool_choice_to_chat(tool_choice: Any) -> Any:
    if tool_choice in (None, "auto", "any"):
        return "auto" if tool_choice in (None, "auto") else "required"
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type == "auto":
            return "auto"
        if choice_type == "any":
            return "required"
        if choice_type == "tool":
            name = tool_choice.get("name")
            if not isinstance(name, str):
                raise ValueError("tool_choice.name must be a string when type is 'tool'")
            return {"type": "function", "function": {"name": name}}
        if choice_type == "none":
            return "none"
        raise ValueError(f"Unsupported tool_choice type: {choice_type}")
    raise ValueError("tool_choice must be a string or object")


def _first_choice(chat_body: dict[str, Any]) -> dict[str, Any]:
    choices = chat_body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _assistant_message_to_anthropic_content(message: dict[str, Any]) -> list[dict[str, Any]]:
    content_blocks: list[dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        content_blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str) and text:
                    content_blocks.append({"type": "text", "text": text})

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            arguments = function.get("arguments")
            if not isinstance(name, str):
                continue
            parsed_input: Any = {}
            if isinstance(arguments, str) and arguments.strip():
                try:
                    parsed_input = json.loads(arguments)
                except json.JSONDecodeError:
                    parsed_input = {"raw": arguments}
            if not isinstance(parsed_input, dict):
                parsed_input = {"value": parsed_input}
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": call.get("id") if isinstance(call.get("id"), str) else f"toolu_{uuid.uuid4().hex[:12]}",
                    "name": name,
                    "input": parsed_input,
                }
            )
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})
    return content_blocks


def _anthropic_usage_from_chat(usage: Any) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
    }


class AnthropicStreamMapper:
    """Map OpenAI chat completion SSE blocks to Anthropic Messages stream events."""

    def __init__(self, *, message_id: str, model: Any) -> None:
        self.message_id = message_id
        self.model = resolved_request_model(model)
        self.message_started = False
        self.text_block_index: int | None = None
        self.text_started = False
        self.text_stopped = False
        self.tool_blocks: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.usage: dict[str, int] | None = None
        self._next_block_index = 0

    def events_from_openai_sse(self, event_block: str) -> list[str]:
        payload = _event_block_data_payload(event_block)
        if payload is _SSE_DONE:
            return self._completion_events()
        if not isinstance(payload, dict):
            return []

        usage = payload.get("usage")
        if isinstance(usage, dict):
            self.usage = _anthropic_usage_from_chat(usage)

        events: list[str] = []
        choices = payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str):
                self.finish_reason = finish_reason
            delta = choice.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str) and content:
                    events.extend(self._text_delta_events(content))
                tool_calls = delta.get("tool_calls")
                if isinstance(tool_calls, list):
                    events.extend(self._tool_call_delta_events(tool_calls))
        return events

    def _ensure_message_start(self) -> list[str]:
        if self.message_started:
            return []
        self.message_started = True
        return [
            anthropic_stream_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": self.message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self.model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )
        ]

    def _allocate_text_block_index(self) -> int:
        if self.text_block_index is None:
            self.text_block_index = self._next_block_index
            self._next_block_index += 1
        return self.text_block_index

    def _text_delta_events(self, delta: str) -> list[str]:
        events = self._ensure_message_start()
        block_index = self._allocate_text_block_index()
        if not self.text_started:
            self.text_started = True
            events.append(
                anthropic_stream_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            )
        events.append(
            anthropic_stream_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "text_delta", "text": delta},
                },
            )
        )
        return events

    def _tool_call_delta_events(self, tool_calls: list[Any]) -> list[str]:
        events = self._ensure_message_start()
        for chunk in tool_calls:
            if not isinstance(chunk, dict):
                continue
            openai_index = chunk.get("index")
            if not isinstance(openai_index, int):
                openai_index = len(self.tool_blocks)
            state = self.tool_blocks.setdefault(
                openai_index,
                {
                    "anthropic_index": None,
                    "id": "",
                    "name": "",
                    "pending_arguments": [],
                    "started": False,
                    "stopped": False,
                },
            )
            if state["anthropic_index"] is None:
                state["anthropic_index"] = self._next_block_index
                self._next_block_index += 1

            if isinstance(chunk.get("id"), str):
                state["id"] = chunk["id"]
            function = chunk.get("function")
            if isinstance(function, dict):
                if isinstance(function.get("name"), str):
                    state["name"] = function["name"]
                arguments = function.get("arguments")
                if isinstance(arguments, str) and arguments:
                    state["pending_arguments"].append(arguments)
                    if not state["started"] and state["id"] and state["name"]:
                        state["started"] = True
                        events.append(
                            anthropic_stream_event(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": state["anthropic_index"],
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": state["id"] or f"toolu_{uuid.uuid4().hex[:12]}",
                                        "name": state["name"],
                                        "input": {},
                                    },
                                },
                            )
                        )
                        for pending in state["pending_arguments"]:
                            events.append(
                                anthropic_stream_event(
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": state["anthropic_index"],
                                        "delta": {
                                            "type": "input_json_delta",
                                            "partial_json": pending,
                                        },
                                    },
                                )
                            )
                        state["pending_arguments"].clear()
                    elif state["started"]:
                        events.append(
                            anthropic_stream_event(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": state["anthropic_index"],
                                    "delta": {"type": "input_json_delta", "partial_json": arguments},
                                },
                            )
                        )
                        state["pending_arguments"].clear()
                elif not state["started"] and state["id"] and state["name"]:
                    state["started"] = True
                    events.append(
                        anthropic_stream_event(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": state["anthropic_index"],
                                "content_block": {
                                    "type": "tool_use",
                                    "id": state["id"] or f"toolu_{uuid.uuid4().hex[:12]}",
                                    "name": state["name"],
                                    "input": {},
                                },
                            },
                        )
                    )
                    for pending in state["pending_arguments"]:
                        events.append(
                            anthropic_stream_event(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": state["anthropic_index"],
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": pending,
                                    },
                                },
                            )
                        )
                    state["pending_arguments"].clear()
        return events

    def _completion_events(self) -> list[str]:
        events: list[str] = []
        if not self.message_started:
            events.extend(self._ensure_message_start())

        if self.text_started and not self.text_stopped and self.text_block_index is not None:
            self.text_stopped = True
            events.append(
                anthropic_stream_event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": self.text_block_index},
                )
            )

        for state in sorted(self.tool_blocks.values(), key=lambda item: item["anthropic_index"] or 0):
            if state["started"] and not state["stopped"] and state["anthropic_index"] is not None:
                state["stopped"] = True
                events.append(
                    anthropic_stream_event(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": state["anthropic_index"]},
                    )
                )

        stop_reason = (
            "tool_use"
            if any(state["started"] for state in self.tool_blocks.values())
            else _FINISH_REASON_TO_STOP.get(self.finish_reason or "stop", "end_turn")
        )
        usage = self.usage or {"input_tokens": 0, "output_tokens": 0}
        events.append(
            anthropic_stream_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": usage.get("output_tokens", 0)},
                },
            )
        )
        events.append(anthropic_stream_event("message_stop", {"type": "message_stop"}))
        return events
