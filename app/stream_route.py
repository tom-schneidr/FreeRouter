from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

from app.providers import ProviderError
from app.router import NoProviderAvailable, WaterfallRouter


async def stream_route_chat(
    payload: dict[str, Any], router: WaterfallRouter
) -> AsyncGenerator[str, None]:
    """Emit SSE route progress events using canonical waterfall router events."""

    def evt(data: dict[str, Any]) -> str:
        return f"data: {json.dumps(data)}\n\n"

    try:
        async for event in router.iter_route_events(payload):
            if event.event_type == "route_skipped":
                yield evt(
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
                yield evt(
                    {
                        "type": "route_trying",
                        "provider": event.provider_name,
                        "model_id": event.model_id,
                        "route_id": event.route_id,
                    }
                )
                await asyncio.sleep(0)
                continue

            if event.event_type == "route_failed":
                yield evt(
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
                yield evt(
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
                yield evt(
                    {
                        "type": "route_selected",
                        "provider": event.provider_name,
                        "model_id": event.model_id,
                        "route_id": event.route_id,
                    }
                )
                response = event.response
                if response is None:
                    yield evt(
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
                    yield evt({"type": "content", "text": content[index : index + chunk_size]})
                    await asyncio.sleep(0.015)

                yield evt(
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
                yield evt(
                    {
                        "type": "error",
                        "message": "All providers exhausted. No model could serve this request.",
                    }
                )
                return
    except ValueError as exc:
        yield evt({"type": "error", "message": str(exc)})
    except ProviderError as exc:
        yield evt({"type": "error", "message": str(exc)})
    except NoProviderAvailable:
        yield evt(
            {
                "type": "error",
                "message": "All providers exhausted. No model could serve this request.",
            }
        )
