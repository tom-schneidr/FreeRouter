from __future__ import annotations

import asyncio


class GatewayRequestLimiter:
    """Bound concurrent gateway work and fail fast when a burst waits too long."""

    def __init__(
        self,
        max_concurrent_requests: int,
        queue_timeout_seconds: float,
        max_waiting_requests: int | None = None,
    ) -> None:
        self.max_concurrent_requests = max(1, int(max_concurrent_requests))
        self.queue_timeout_seconds = max(0.0, float(queue_timeout_seconds))
        self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self.max_waiting_requests = (
            None if max_waiting_requests is None else max(0, int(max_waiting_requests))
        )
        self._waiting_requests = 0
        self._waiting_lock = asyncio.Lock()

    async def acquire(self) -> bool:
        async with self._waiting_lock:
            if (
                self.max_waiting_requests is not None
                and self._waiting_requests >= self.max_waiting_requests
            ):
                return False
            self._waiting_requests += 1

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.queue_timeout_seconds,
            )
        except TimeoutError:
            return False
        finally:
            async with self._waiting_lock:
                self._waiting_requests = max(0, self._waiting_requests - 1)

        return True

    def release(self) -> None:
        self._semaphore.release()
