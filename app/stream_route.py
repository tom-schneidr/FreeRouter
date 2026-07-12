from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

from app.providers import ProviderError
from app.router import (
    _SSE_DONE,
    NoProviderAvailable,
    RouteStreamDiag,
    UnsupportedCapabilities,
    WaterfallRouter,
    _delta_visible_text_from_chunk,
    _event_block_data_payload,
    _split_sse_event_blocks,
)
from app.tool_call_stream import ToolCallStreamAccumulator


async def stream_route_chat(
    payload: dict[str, Any],
    router: WaterfallRouter,
    *,
    chunk_replay_sleep_seconds: float = 0.0,
    on_emit: Any | None = None,
) -> AsyncGenerator[str, None]:
    """Emit SSE route progress plus assistant deltas (true upstream streaming)."""

    async def emit(data: dict[str, Any]) -> str:
        if on_emit is not None:
            maybe = on_emit(data)
            if asyncio.iscoroutine(maybe):
                await maybe
        return f"data: {json.dumps(data)}\n\n"

    final_provider = ""
    final_model = ""
    final_route = ""
    full_text = ""
    carry = ""
    tool_calls = ToolCallStreamAccumulator()
    stream_failed = False

    try:
        async for part in router.iter_chat_completion_openai_stream(payload):
            if isinstance(part, RouteStreamDiag):
                if part.event_type == "route_skipped":
                    yield await emit(
                        {
                            "type": "route_skip",
                            "provider": part.provider_name,
                            "model_id": part.model_id,
                            "route_id": part.route_id,
                            "reason": part.reason or "route_unavailable",
                        }
                    )
                    continue
                if part.event_type == "route_trying":
                    yield await emit(
                        {
                            "type": "route_trying",
                            "provider": part.provider_name,
                            "model_id": part.model_id,
                            "route_id": part.route_id,
                        }
                    )
                    continue
                if part.event_type == "route_failed":
                    yield await emit(
                        {
                            "type": "route_fail",
                            "provider": part.provider_name,
                            "model_id": part.model_id,
                            "route_id": part.route_id,
                            "reason": part.reason or "provider_error",
                        }
                    )
                    continue
                if part.event_type == "route_flagged":
                    yield await emit(
                        {
                            "type": "route_flagged",
                            "provider": part.provider_name,
                            "model_id": part.model_id,
                            "route_id": part.route_id,
                            "reason": part.reason or "health_flagged",
                        }
                    )
                    continue
                if part.event_type == "route_selected":
                    final_provider = part.provider_name or ""
                    final_model = part.model_id or ""
                    final_route = part.route_id or ""
                    yield await emit(
                        {
                            "type": "route_selected",
                            "provider": part.provider_name,
                            "model_id": part.model_id,
                            "route_id": part.route_id,
                        }
                    )
                    continue
                if part.event_type == "usage_summary":
                    if part.usage:
                        yield await emit({"type": "usage", "usage": part.usage})
                    continue
                continue

            carry += part
            blocks, carry = _split_sse_event_blocks(carry)
            for block in blocks:
                pl = _event_block_data_payload(block)
                if pl is _SSE_DONE:
                    continue
                if isinstance(pl, dict):
                    upstream_error = pl.get("error")
                    if isinstance(upstream_error, dict):
                        stream_failed = True
                        yield await emit(
                            {
                                "type": "error",
                                "code": upstream_error.get("code") or "stream_error",
                                "message": upstream_error.get("message")
                                or "The upstream model stream failed.",
                            }
                        )
                        continue
                    tool_calls.ingest(pl)
                    choices = pl.get("choices")
                    if isinstance(choices, list):
                        for choice in choices:
                            delta_obj = choice.get("delta") if isinstance(choice, dict) else None
                            chunks = (
                                delta_obj.get("tool_calls") if isinstance(delta_obj, dict) else None
                            )
                            if isinstance(chunks, list) and chunks:
                                yield await emit({"type": "tool_call_delta", "tool_calls": chunks})
                    delta = _delta_visible_text_from_chunk(pl)
                    if delta:
                        full_text += delta
                        yield await emit({"type": "content", "text": delta})
                        if chunk_replay_sleep_seconds > 0:
                            await asyncio.sleep(chunk_replay_sleep_seconds)

        if carry.strip():
            for raw_line in carry.splitlines():
                line = raw_line.strip()
                if not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if raw == "[DONE]":
                    continue
                try:
                    pl = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(pl, dict):
                    upstream_error = pl.get("error")
                    if isinstance(upstream_error, dict):
                        stream_failed = True
                        yield await emit(
                            {
                                "type": "error",
                                "code": upstream_error.get("code") or "stream_error",
                                "message": upstream_error.get("message")
                                or "The upstream model stream failed.",
                            }
                        )
                        continue
                    tool_calls.ingest(pl)
                    delta = _delta_visible_text_from_chunk(pl)
                    if delta:
                        full_text += delta
                        yield await emit({"type": "content", "text": delta})
                        if chunk_replay_sleep_seconds > 0:
                            await asyncio.sleep(chunk_replay_sleep_seconds)

        if stream_failed:
            return
        done_payload: dict[str, Any] = {
            "type": "done",
            "content": full_text,
            "provider": final_provider,
            "model_id": final_model,
            "route_id": final_route,
        }
        if tool_calls.has_tool_calls:
            done_payload["tool_calls"] = tool_calls.to_chat_body(text=full_text)["choices"][0][
                "message"
            ]["tool_calls"]
        yield await emit(done_payload)
    except ValueError as exc:
        yield await emit({"type": "error", "message": str(exc)})
    except ProviderError as exc:
        yield await emit({"type": "error", "message": str(exc)})
    except NoProviderAvailable:
        yield await emit(
            {
                "type": "error",
                "message": "All providers exhausted. No model could serve this request.",
            }
        )
    except UnsupportedCapabilities as exc:
        yield await emit(
            {
                "type": "error",
                "message": str(exc),
                "code": "unsupported_capabilities",
                "required_capabilities": sorted(exc.required),
            }
        )
