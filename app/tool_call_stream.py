from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class _StreamToolCall:
    choice_index: int
    call_index: int
    call_id: str = ""
    call_type: str = "function"
    name: str = ""
    arguments: str = ""


class ToolCallStreamAccumulator:
    """Assemble fragmented OpenAI tool-call deltas before a route is committed."""

    def __init__(self) -> None:
        self._calls: dict[tuple[int, int], _StreamToolCall] = {}
        self.finish_reasons: dict[int, str] = {}
        self.metadata: dict[str, Any] = {}
        self.usage: dict[str, Any] | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self._calls)

    def ingest(self, payload: dict[str, Any]) -> None:
        for key in ("id", "model", "created", "system_fingerprint", "service_tier"):
            if key in payload and key not in self.metadata:
                self.metadata[key] = payload[key]
        if isinstance(payload.get("usage"), dict):
            self.usage = dict(payload["usage"])
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return
        for fallback_choice_index, choice in enumerate(choices):
            if not isinstance(choice, dict):
                continue
            raw_choice_index = choice.get("index")
            choice_index = (
                raw_choice_index if isinstance(raw_choice_index, int) else fallback_choice_index
            )
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str) and finish_reason:
                self.finish_reasons[choice_index] = finish_reason
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            chunks = delta.get("tool_calls")
            if not isinstance(chunks, list):
                continue
            for fallback_call_index, chunk in enumerate(chunks):
                if not isinstance(chunk, dict):
                    continue
                call_index = self._resolve_call_index(
                    choice_index=choice_index,
                    fallback_index=fallback_call_index,
                    chunk=chunk,
                    chunk_count=len(chunks),
                )
                key = (choice_index, call_index)
                call = self._calls.setdefault(
                    key,
                    _StreamToolCall(choice_index=choice_index, call_index=call_index),
                )
                chunk_id = chunk.get("id")
                if isinstance(chunk_id, str) and chunk_id:
                    call.call_id = _merge_identifier_fragment(call.call_id, chunk_id)
                chunk_type = chunk.get("type")
                if isinstance(chunk_type, str) and chunk_type:
                    call.call_type = chunk_type
                function = chunk.get("function")
                if not isinstance(function, dict):
                    continue
                name = function.get("name")
                if isinstance(name, str) and name:
                    call.name = _merge_identifier_fragment(call.name, name)
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    call.arguments = _merge_argument_fragment(call.arguments, arguments)
                elif isinstance(arguments, dict):
                    call.arguments += json.dumps(arguments, separators=(",", ":"))

    def to_chat_body(self, *, text: str = "") -> dict[str, Any]:
        by_choice: dict[int, list[_StreamToolCall]] = {}
        for call in self._calls.values():
            by_choice.setdefault(call.choice_index, []).append(call)

        choice_indexes = sorted(set(by_choice) | set(self.finish_reasons) | {0})
        choices: list[dict[str, Any]] = []
        for choice_index in choice_indexes:
            calls = sorted(by_choice.get(choice_index, []), key=lambda call: call.call_index)
            message: dict[str, Any] = {
                "role": "assistant",
                "content": text if text else None,
            }
            if calls:
                message["tool_calls"] = [
                    {
                        "id": call.call_id,
                        "type": call.call_type,
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments,
                        },
                    }
                    for call in calls
                ]
            choices.append(
                {
                    "index": choice_index,
                    "message": message,
                    "finish_reason": self.finish_reasons.get(choice_index),
                }
            )
        body = {**self.metadata, "choices": choices}
        if self.usage is not None:
            body["usage"] = self.usage
        return body

    def _resolve_call_index(
        self,
        *,
        choice_index: int,
        fallback_index: int,
        chunk: dict[str, Any],
        chunk_count: int,
    ) -> int:
        raw_index = chunk.get("index")
        if isinstance(raw_index, int):
            return raw_index
        existing = [call for call in self._calls.values() if call.choice_index == choice_index]
        chunk_id = chunk.get("id")
        if isinstance(chunk_id, str) and chunk_id:
            for call in existing:
                merged = _merge_identifier_fragment(call.call_id, chunk_id)
                if merged in {call.call_id, chunk_id}:
                    return call.call_index
        function = chunk.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name:
            for call in existing:
                merged = _merge_identifier_fragment(call.name, name)
                if merged in {call.name, name}:
                    return call.call_index
        if chunk_count == 1 and len(existing) == 1:
            return existing[0].call_index
        return fallback_index


def _merge_identifier_fragment(current: str, incoming: str) -> str:
    if not current:
        return incoming
    if incoming == current or current.endswith(incoming):
        return current
    if incoming.startswith(current):
        return incoming
    max_overlap = min(len(current), len(incoming))
    for size in range(max_overlap, 0, -1):
        if current.endswith(incoming[:size]):
            return current + incoming[size:]
    return current + incoming


def _merge_argument_fragment(current: str, incoming: str) -> str:
    if current and len(incoming) > len(current) and incoming.startswith(current):
        return incoming
    return current + incoming
