from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi.responses import Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from app.request_limiter import GatewayRequestLimiter


class GatewayLimiterLease:
    """Idempotent limiter ownership shared by a stream and its response."""

    def __init__(self, limiter: GatewayRequestLimiter) -> None:
        self._limiter = limiter
        self.held = False

    async def acquire(self) -> bool:
        if self.held:
            return True
        self.held = await self._limiter.acquire()
        return self.held

    def release(self) -> None:
        if not self.held:
            return
        self._limiter.release()
        self.held = False


class LimitedStreamingResponse(StreamingResponse):
    """Acquire streaming capacity only once ASGI starts the response."""

    def __init__(
        self,
        *args,
        lease: GatewayLimiterLease,
        rejected_response: Response,
        on_rejected: Callable[[], Awaitable[None]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lease = lease
        self.rejected_response = rejected_response
        self.on_rejected = on_rejected

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not await self.lease.acquire():
            if self.on_rejected is not None:
                await self.on_rejected()
            await self.rejected_response(scope, receive, send)
            return
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.lease.release()
