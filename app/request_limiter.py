from __future__ import annotations

import asyncio


class GatewayRequestLimiter:
    """Bound concurrent gateway work and fail fast when a burst waits too long."""

    def __init__(self, max_concurrent_requests: int, queue_timeout_seconds: float) -> None:
        self.max_concurrent_requests = max(1, int(max_concurrent_requests))
        self.queue_timeout_seconds = max(0.0, float(queue_timeout_seconds))
        self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)

    async def acquire(self) -> bool:
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.queue_timeout_seconds,
            )
        except TimeoutError:
            return False
        return True

    def release(self) -> None:
        self._semaphore.release()
