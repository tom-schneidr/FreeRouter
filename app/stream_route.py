from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

from app.providers import ProviderError
from app.router import NoProviderAvailable, WaterfallRouter


async def stream_route_chat(
    payload: dict[str, Any],
    router: WaterfallRouter,
    *,
    chunk_replay_sleep_seconds: float = 0.0,
    on_emit: Any | None = None,
) -> AsyncGenerator[str, None]:
    """Emit SSE route progress events using canonical waterfall router events."""

    async def emit(data: dict[str, Any]) -> str:
        if on_emit is not None:
            maybe = on_emit(data)
            if asyncio.iscoroutine(maybe):
                await maybe
        return f"data: {json.dumps(data)}\n\n"

    try:
        async for event in router.iter_route_events(payload):
            if event.event_type == "route_skipped":
                yield await emit(
                    {
                        "type": "route_skip",
                        "provider": event.provider_name,
                        "model_id": event.model_id,
                        "route_id": event.route_id,
                        "reason": event.reason or "route_unavailable",
                    }
                )
                continue

            if event.event_type == "route_trying":
                yield await emit(
                    {
                        "type": "route_trying",
                        "provider": event.provider_name,
                        "model_id": event.model_id,
                        "route_id": event.route_id,
                    }
                )
                continue

            if event.event_type == "route_failed":
                yield await emit(
                    {
                        "type": "route_fail",
                        "provider": event.provider_name,
                        "model_id": event.model_id,
                        "route_id": event.route_id,
                        "reason": event.reason or "provider_error",
                    }
                )
                continue

            if event.event_type == "route_flagged":
                yield await emit(
                    {
                        "type": "route_flagged",
                        "provider": event.provider_name,
                        "model_id": event.model_id,
                        "route_id": event.route_id,
                        "reason": event.reason or "health_flagged",
                    }
                )
                continue

            if event.event_type == "route_selected":
                yield await emit(
                    {
                        "type": "route_selected",
                        "provider": event.provider_name,
                        "model_id": event.model_id,
                        "route_id": event.route_id,
                    }
                )
                response = event.response
                if response is None:
                    yield await emit(
                        {"type": "error", "message": "Selected route missing response payload."}
                    )
                    return

                content = ""
                try:
                    content = response.body["choices"][0]["message"]["content"] or ""
                except (KeyError, IndexError):
                    pass

                chunk_size = 12
                for index in range(0, len(content), chunk_size):
                    yield await emit(
                        {"type": "content", "text": content[index : index + chunk_size]}
                    )
                    if chunk_replay_sleep_seconds > 0:
                        await asyncio.sleep(chunk_replay_sleep_seconds)

                yield await emit(
                    {
                        "type": "done",
                        "content": content,
                        "provider": event.provider_name,
                        "model_id": event.model_id,
                        "route_id": event.route_id,
                    }
                )
                return

            if event.event_type == "routing_exhausted":
                yield await emit(
                    {
                        "type": "error",
                        "message": "All providers exhausted. No model could serve this request.",
                    }
                )
                return
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
