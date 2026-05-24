from __future__ import annotations

import json
import uuid
from time import time
from typing import Any

from app.router import (
    _SSE_DONE,
    _delta_visible_text_from_chunk,
    _event_block_data_payload,
)


def responses_payload_to_chat(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate the practical subset of OpenAI Responses requests into chat completions."""

    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")

    model = payload.get("model") or "auto"
    messages = _responses_input_to_messages(payload.get("input"))
    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.insert(0, {"role": "system", "content": instructions})

    chat_payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    for key in (
        "temperature",
        "top_p",
        "parallel_tool_calls",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "stop",
        "stream",
        "user",
    ):
        if key in payload:
            chat_payload[key] = payload[key]
    if "max_output_tokens" in payload:
        chat_payload["max_tokens"] = payload["max_output_tokens"]
    elif "max_tokens" in payload:
        chat_payload["max_tokens"] = payload["max_tokens"]
    if "tools" in payload:
        chat_payload["tools"] = _responses_tools_to_chat_tools(payload["tools"])
    if "tool_choice" in payload:
        chat_payload["tool_choice"] = _responses_tool_choice_to_chat_tool_choice(
            payload["tool_choice"]
        )
    return chat_payload


def chat_body_to_response(
    chat_body: dict[str, Any],
    *,
    requested_model: Any,
    response_id: str | None = None,
) -> dict[str, Any]:
    created = int(time())
    text = _assistant_text_from_chat_body(chat_body)
    output = _response_output_from_chat_body(chat_body, text=text)
    usage = _responses_usage_from_chat_usage(chat_body.get("usage"))
    return {
        "id": response_id or f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": created,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": requested_model or chat_body.get("model") or "auto",
        "output": output,
        "output_text": text,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": None,
        "store": False,
        "temperature": None,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": usage,
    }


def response_stream_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def responses_stream_start(*, response_id: str, model: Any) -> str:
    return response_stream_event(
        "response.created",
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": int(time()),
                "status": "in_progress",
                "model": model or "auto",
                "output": [],
            },
        },
    )


class ResponsesStreamMapper:
    def __init__(self, *, response_id: str) -> None:
        self.response_id = response_id
        self.message_id = f"msg_{response_id.removeprefix('resp_')}"
        self.sequence_number = 0
        self.message_started = False
        self.text_parts: list[str] = []
        self.tool_calls: dict[int, dict[str, Any]] = {}

    def events_from_openai_sse(self, event_block: str) -> list[str]:
        payload = _event_block_data_payload(event_block)
        if payload is _SSE_DONE:
            return self._completion_events()
        if not isinstance(payload, dict):
            return []

        events: list[str] = []
        delta = _delta_visible_text_from_chunk(payload)
        if delta:
            events.extend(self._text_delta_events(delta))
        events.extend(self._tool_call_delta_events(payload))
        return events

    def _text_delta_events(self, delta: str) -> list[str]:
        events: list[str] = []
        if not self.message_started:
            self.message_started = True
            events.append(
                self._event(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "id": self.message_id,
                            "type": "message",
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        },
                    },
                )
            )
            events.append(
                self._event(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "item_id": self.message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    },
                )
            )
        self.text_parts.append(delta)
        events.append(
            self._event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": self.message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": delta,
                },
            )
        )
        return events

    def _tool_call_delta_events(self, payload: dict[str, Any]) -> list[str]:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return []
        choice = choices[0]
        if not isinstance(choice, dict):
            return []
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return []
        tool_calls = delta.get("tool_calls")
        if not isinstance(tool_calls, list):
            return []

        for chunk in tool_calls:
            if not isinstance(chunk, dict):
                continue
            index = chunk.get("index")
            if not isinstance(index, int):
                index = len(self.tool_calls)
            state = self.tool_calls.setdefault(
                index,
                {
                    "id": chunk.get("id") if isinstance(chunk.get("id"), str) else "",
                    "name": "",
                    "arguments": "",
                },
            )
            if isinstance(chunk.get("id"), str):
                state["id"] = chunk["id"]
            function = chunk.get("function")
            if isinstance(function, dict):
                if isinstance(function.get("name"), str):
                    state["name"] += function["name"]
                if isinstance(function.get("arguments"), str):
                    state["arguments"] += function["arguments"]
        return []

    def _completion_events(self) -> list[str]:
        events: list[str] = []
        if self.message_started:
            text = "".join(self.text_parts)
            events.append(
                self._event(
                    "response.output_text.done",
                    {
                        "type": "response.output_text.done",
                        "item_id": self.message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "text": text,
                    },
                )
            )
            events.append(
                self._event(
                    "response.content_part.done",
                    {
                        "type": "response.content_part.done",
                        "item_id": self.message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": text, "annotations": []},
                    },
                )
            )
            events.append(
                self._event(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "id": self.message_id,
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": text,
                                    "annotations": [],
                                }
                            ],
                        },
                    },
                )
            )

        base_index = 1 if self.message_started else 0
        for offset, call in enumerate(self.tool_calls.values()):
            item = _function_call_output_item(call)
            output_index = base_index + offset
            events.append(
                self._event(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": {**item, "status": "in_progress"},
                    },
                )
            )
            events.append(
                self._event(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": output_index,
                        "item": item,
                    },
                )
            )

        events.append(
            self._event(
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {
                        "id": self.response_id,
                        "object": "response",
                        "status": "completed",
                    },
                },
            )
        )
        return events

    def _event(self, event: str, payload: dict[str, Any]) -> str:
        self.sequence_number += 1
        payload["sequence_number"] = self.sequence_number
        return response_stream_event(event, payload)


def responses_stream_delta_from_openai_sse(
    event_block: str,
    *,
    response_id: str,
    output_index: int = 0,
    content_index: int = 0,
) -> str | None:
    del output_index, content_index
    events = ResponsesStreamMapper(response_id=response_id).events_from_openai_sse(event_block)
    return events[0] if events else None


def responses_stream_done() -> str:
    return "data: [DONE]\n\n"


def _responses_input_to_messages(input_value: Any) -> list[dict[str, Any]]:
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]
    if not isinstance(input_value, list) or not input_value:
        raise ValueError("Responses payload must include non-empty 'input'")

    messages: list[dict[str, Any]] = []
    for index, item in enumerate(input_value):
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            raise ValueError(f"input[{index}] must be a string or object")
        item_type = item.get("type")
        if item_type == "function_call_output":
            call_id = item.get("call_id")
            if not isinstance(call_id, str):
                raise ValueError(f"input[{index}].call_id must be a string")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _responses_tool_output_to_text(item.get("output")),
                }
            )
            continue
        if item_type == "function_call":
            messages.append(_responses_function_call_to_chat_message(item))
            continue
        role = item.get("role")
        if not isinstance(role, str):
            role = "user"
        content = item.get("content")
        messages.append({"role": role, "content": _responses_content_to_chat_content(content)})
    return messages


def _responses_content_to_chat_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)

    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"input_text", "output_text", "text"}:
            text = part.get("text")
            if isinstance(text, str):
                parts.append({"type": "text", "text": text})
        elif part_type == "input_image":
            image_url = part.get("image_url")
            if isinstance(image_url, str):
                parts.append({"type": "image_url", "image_url": {"url": image_url}})
        else:
            parts.append(dict(part))
    return parts


def _assistant_text_from_chat_body(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
    text = choice.get("text")
    return text if isinstance(text, str) else ""


def _response_output_from_chat_body(chat_body: dict[str, Any], *, text: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if text:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        )

    message = _first_chat_message(chat_body)
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    output.append(_chat_tool_call_to_response_item(tool_call))

    if not output:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [],
            }
        )
    return output


def _first_chat_message(chat_body: dict[str, Any]) -> dict[str, Any] | None:
    choices = chat_body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    message = choice.get("message")
    return message if isinstance(message, dict) else None


def _chat_tool_call_to_response_item(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function")
    function = function if isinstance(function, dict) else {}
    call_id = tool_call.get("id")
    if not isinstance(call_id, str) or not call_id:
        call_id = f"call_{uuid.uuid4().hex}"
    name = function.get("name")
    arguments = function.get("arguments")
    return {
        "id": call_id,
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "name": name if isinstance(name, str) else "",
        "arguments": arguments if isinstance(arguments, str) else "{}",
    }


def _function_call_output_item(call: dict[str, Any]) -> dict[str, Any]:
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id:
        call_id = f"call_{uuid.uuid4().hex}"
    name = call.get("name")
    arguments = call.get("arguments")
    return {
        "id": call_id,
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "name": name if isinstance(name, str) else "",
        "arguments": arguments if isinstance(arguments, str) else "{}",
    }


def _responses_tools_to_chat_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        raise ValueError("tools must be an array")
    chat_tools: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"tools[{index}] must be an object")
        if tool.get("type") != "function":
            chat_tools.append(dict(tool))
            continue
        if isinstance(tool.get("function"), dict):
            chat_tools.append(dict(tool))
            continue
        name = tool.get("name")
        if not isinstance(name, str):
            raise ValueError(f"tools[{index}].name must be a string")
        function: dict[str, Any] = {"name": name}
        for src, dst in (
            ("description", "description"),
            ("parameters", "parameters"),
            ("strict", "strict"),
        ):
            if src in tool:
                function[dst] = tool[src]
        chat_tools.append({"type": "function", "function": function})
    return chat_tools


def _responses_tool_choice_to_chat_tool_choice(tool_choice: Any) -> Any:
    if isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, dict):
        return tool_choice
    if tool_choice.get("type") != "function":
        return tool_choice
    name = tool_choice.get("name")
    if not isinstance(name, str):
        return tool_choice
    return {"type": "function", "function": {"name": name}}


def _responses_tool_output_to_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(output)


def _responses_function_call_to_chat_message(item: dict[str, Any]) -> dict[str, Any]:
    call_id = item.get("call_id") or item.get("id")
    if not isinstance(call_id, str):
        call_id = f"call_{uuid.uuid4().hex}"
    name = item.get("name")
    arguments = item.get("arguments")
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name if isinstance(name, str) else "",
                    "arguments": arguments if isinstance(arguments, str) else "{}",
                },
            }
        ],
    }


def _responses_usage_from_chat_usage(usage: Any) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    total_tokens = usage.get("total_tokens") or input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": total_tokens,
    }
