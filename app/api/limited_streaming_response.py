from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi.responses import Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from app.api.gateway_headers import GatewayRoutingContext, gateway_route_headers
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


class RoutedLimitedStreamingResponse(LimitedStreamingResponse):
    """Defer HTTP response headers until the selected route is known."""

    def __init__(
        self,
        *args,
        routing: GatewayRoutingContext,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.routing = routing

    def _response_headers(self) -> list[tuple[bytes, bytes]]:
        headers = list(self.raw_headers)
        info = self.routing.info
        if info is not None:
            for name, value in gateway_route_headers(info).items():
                headers.append((name.lower().encode("latin-1"), value.encode("latin-1")))
        return headers

    async def _send_body_chunk(self, send: Send, chunk: str | bytes | memoryview) -> None:
        if not isinstance(chunk, bytes | memoryview):
            chunk = chunk.encode(self.charset)
        await send({"type": "http.response.body", "body": chunk, "more_body": True})

    async def _start_response(self, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self._response_headers(),
            }
        )

    async def stream_response(self, send: Send) -> None:
        buffer: list[str | bytes | memoryview] = []
        started = False

        async for chunk in self.body_iterator:
            if not started:
                buffer.append(chunk)
                if self.routing.ready:
                    await self._start_response(send)
                    started = True
                    for buffered in buffer:
                        await self._send_body_chunk(send, buffered)
                    buffer.clear()
                continue
            await self._send_body_chunk(send, chunk)

        if not started:
            await self._start_response(send)
            for buffered in buffer:
                await self._send_body_chunk(send, buffered)

        await send({"type": "http.response.body", "body": b"", "more_body": False})
