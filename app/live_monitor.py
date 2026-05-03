from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from time import time
from typing import Any


@dataclass(frozen=True)
class LiveRequestEvent:
    event_id: int
    event_type: str
    request_id: str
    timestamp: int
    payload: dict[str, Any]


class APILiveMonitor:
    """In-memory request event bus for the local monitoring dashboard."""

    def __init__(self, *, max_events: int = 500) -> None:
        self._events: deque[LiveRequestEvent] = deque(maxlen=max(10, max_events))
        self._subscribers: set[asyncio.Queue[LiveRequestEvent]] = set()
        self._event_seq = 0
        self._lock = asyncio.Lock()

    async def publish(
        self,
        *,
        event_type: str,
        request_id: str,
        payload: dict[str, Any] | None = None,
    ) -> LiveRequestEvent:
        async with self._lock:
            self._event_seq += 1
            event = LiveRequestEvent(
                event_id=self._event_seq,
                event_type=event_type,
                request_id=request_id,
                timestamp=int(time()),
                payload=payload or {},
            )
            self._events.append(event)
            subscribers = list(self._subscribers)

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
