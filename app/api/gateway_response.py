from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from app.router import ChatStreamPart, RouteStreamDiag, _split_sse_event_blocks


def resolved_request_model(requested_model: Any) -> str:
    if isinstance(requested_model, str) and requested_model.strip():
        return requested_model
    return "auto"


def normalize_chat_completion_body(body: Any, requested_model: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        return {}
    normalized = dict(body)
    normalized["model"] = resolved_request_model(requested_model)
    return normalized


def rewrite_sse_block_requested_model(block: str, requested_model: Any) -> str:
    model = resolved_request_model(requested_model)
    lines: list[str] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if line.startswith("data: "):
            inner = line[6:].strip()
            if inner == "[DONE]":
                lines.append(raw_line)
                continue
            try:
                payload = json.loads(inner)
            except json.JSONDecodeError:
                lines.append(raw_line)
                continue
            if isinstance(payload, dict) and "model" in payload:
                rewritten = dict(payload)
                rewritten["model"] = model
                lines.append(f"data: {json.dumps(rewritten, separators=(',', ':'))}")
                continue
        lines.append(raw_line)
    return "\n".join(lines)


async def normalize_openai_sse_stream(
    parts: AsyncIterator[ChatStreamPart],
    requested_model: Any,
) -> AsyncIterator[ChatStreamPart]:
    carry = ""
    async for part in parts:
        if isinstance(part, RouteStreamDiag):
            yield part
            continue
        if not isinstance(part, str):
            yield part
            continue
        carry += part
        blocks, carry = _split_sse_event_blocks(carry)
        for block in blocks:
            yield rewrite_sse_block_requested_model(block, requested_model) + "\n\n"
    if carry:
        yield carry
