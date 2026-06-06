from __future__ import annotations

from fastapi.responses import JSONResponse

from app.api.limited_streaming_response import GatewayLimiterLease, LimitedStreamingResponse
from app.request_limiter import GatewayRequestLimiter


def _scope() -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/stream",
        "raw_path": b"/stream",
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 123),
        "server": ("testserver", 80),
        "root_path": "",
    }


async def _receive() -> dict:
    return {"type": "http.disconnect"}


def _sender(sent: list[dict]):
    async def send(message: dict) -> None:
        sent.append(message)

    return send


async def test_unstarted_limited_stream_does_not_acquire_capacity():
    limiter = GatewayRequestLimiter(max_concurrent_requests=1, queue_timeout_seconds=0.01)
    lease = GatewayLimiterLease(limiter)

    LimitedStreamingResponse(
        iter(["data: ok\n\n"]),
        lease=lease,
        rejected_response=JSONResponse(status_code=429, content={"error": "busy"}),
    )

    assert lease.held is False
    assert await limiter.acquire() is True
    limiter.release()


async def test_limited_stream_releases_capacity_when_client_disconnects():
    limiter = GatewayRequestLimiter(max_concurrent_requests=1, queue_timeout_seconds=0.01)
    lease = GatewayLimiterLease(limiter)
    sent: list[dict] = []
    response = LimitedStreamingResponse(
        iter(["data: ok\n\n"]),
        lease=lease,
        rejected_response=JSONResponse(status_code=429, content={"error": "busy"}),
    )

    await response(_scope(), _receive, _sender(sent))

    assert lease.held is False
    assert await limiter.acquire() is True
    limiter.release()


async def test_limited_stream_preserves_http_429_when_capacity_is_full():
    limiter = GatewayRequestLimiter(max_concurrent_requests=1, queue_timeout_seconds=0.01)
    assert await limiter.acquire() is True
    lease = GatewayLimiterLease(limiter)
    sent: list[dict] = []
    response = LimitedStreamingResponse(
        iter(["data: should-not-run\n\n"]),
        lease=lease,
        rejected_response=JSONResponse(status_code=429, content={"error": "busy"}),
    )

    await response(_scope(), _receive, _sender(sent))

    start = next(message for message in sent if message["type"] == "http.response.start")
    assert start["status"] == 429
    assert lease.held is False
    limiter.release()
