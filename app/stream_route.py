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
    WaterfallRouter,
    _delta_visible_text_from_chunk,
    _event_block_data_payload,
    _split_sse_event_blocks,
)


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
                continue

            carry += part
            blocks, carry = _split_sse_event_blocks(carry)
            for block in blocks:
                pl = _event_block_data_payload(block)
                if pl is _SSE_DONE:
                    continue
                if isinstance(pl, dict):
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
                    delta = _delta_visible_text_from_chunk(pl)
                    if delta:
                        full_text += delta
                        yield await emit({"type": "content", "text": delta})
                        if chunk_replay_sleep_seconds > 0:
                            await asyncio.sleep(chunk_replay_sleep_seconds)

        yield await emit(
            {
                "type": "done",
                "content": full_text,
                "provider": final_provider,
                "model_id": final_model,
                "route_id": final_route,
            }
        )
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
