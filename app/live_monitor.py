from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import time
from typing import Any

from app.consumer_contracts import safe_receipt


@dataclass(frozen=True)
class LiveRequestEvent:
    event_id: int
    event_type: str
    request_id: str
    timestamp: int
    payload: dict[str, Any]


class APILiveMonitor:
    """In-memory request event bus with an optional content-free receipt sink."""

    def __init__(
        self,
        *,
        max_events: int = 500,
        receipt_sink: Callable[[dict], Awaitable[None]] | None = None,
    ) -> None:
        self._events: deque[LiveRequestEvent] = deque(maxlen=max(10, max_events))
        self._subscribers: set[asyncio.Queue[LiveRequestEvent]] = set()
        self._event_seq = 0
        self._lock = asyncio.Lock()
        self._receipt_sink = receipt_sink
        self._request_context: dict[str, dict[str, Any]] = {}

    async def publish(
        self,
        *,
        event_type: str,
        request_id: str,
        payload: dict[str, Any] | None = None,
    ) -> LiveRequestEvent:
        event_payload = payload or {}
        receipt: dict[str, Any] | None = None
        async with self._lock:
            self._event_seq += 1
            event = LiveRequestEvent(
                event_id=self._event_seq,
                event_type=event_type,
                request_id=request_id,
                timestamp=int(time()),
                payload=event_payload,
            )
            self._events.append(event)
            subscribers = list(self._subscribers)
            if event_type == "request_started":
                self._request_context[request_id] = {
                    "path": event_payload.get("path"),
                    "stream": bool(event_payload.get("stream")),
                    "model": event_payload.get("model"),
                    "required_capabilities": list(
                        event_payload.get("required_capabilities") or []
                    ),
                }
            else:
                context = self._request_context.get(request_id, {})
                receipt = safe_receipt(
                    event_type=event_type,
                    request_id=request_id,
                    timestamp=event.timestamp,
                    context=context,
                    payload=event_payload,
                )
                if receipt is not None:
                    self._request_context.pop(request_id, None)

        for queue in subscribers:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                continue
        if receipt is not None and self._receipt_sink is not None:
            try:
                await self._receipt_sink(receipt)
            except Exception:
                # Observability can never break the gateway request path.
                pass
        return event

    async def snapshot(self) -> list[dict[str, Any]]:
        async with self._lock:
            events = list(self._events)
        return [self._to_payload(event) for event in events]

    async def subscribe(self) -> asyncio.Queue[LiveRequestEvent]:
        queue: asyncio.Queue[LiveRequestEvent] = asyncio.Queue(maxsize=200)
        async with self._lock:
            self._subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[LiveRequestEvent]) -> None:
        async with self._lock:
            self._subscribers.discard(queue)

    @staticmethod
    def _to_payload(event: LiveRequestEvent) -> dict[str, Any]:
        return {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "request_id": event.request_id,
            "timestamp": event.timestamp,
            "payload": event.payload,
        }

    @classmethod
    def event_to_payload(cls, event: LiveRequestEvent) -> dict[str, Any]:
        return cls._to_payload(event)
