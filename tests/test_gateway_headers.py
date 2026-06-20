from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.api.gateway_headers import GatewayRouteInfo, GatewayRoutingContext, gateway_route_headers
from app.api.gateway_response import (
    normalize_chat_completion_body,
    rewrite_sse_block_requested_model,
)
from app.api.limited_streaming_response import (
    GatewayLimiterLease,
    RoutedLimitedStreamingResponse,
)
from app.app_services import AppServices, attach_app_services
from app.endpoint_diagnosis import EndpointDiagnosisService
from app.factory import build_core_gateway_stack
from app.live_monitor import APILiveMonitor
from app.main import app
from app.model_catalog import ModelCatalog
from app.providers.base import ProviderResponse
from app.request_limiter import GatewayRequestLimiter
from app.router import WaterfallRouter
from app.settings import get_settings
from app.state import ProviderQuota, StateManager


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


def _header_map(start_message: dict) -> dict[str, str]:
    return {
        name.decode("latin-1").lower(): value.decode("latin-1")
        for name, value in start_message["headers"]
    }


def test_gateway_route_headers_shape():
    headers = gateway_route_headers(
        GatewayRouteInfo(
            provider_name="groq",
            route_id="groq-llama-3-3-70b",
            model_id="llama-3.3-70b-versatile",
        )
    )
    assert headers == {
        "X-Gateway-Provider": "groq",
        "X-Gateway-Route": "groq-llama-3-3-70b",
        "X-Gateway-Model": "llama-3.3-70b-versatile",
    }
    assert "X-Gateway-Attempts" not in headers


def test_normalize_chat_completion_body_uses_requested_model():
    body = normalize_chat_completion_body(
        {
            "id": "chatcmpl-test",
            "model": "provider/model-id",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        },
        "auto",
    )
    assert body["model"] == "auto"
    assert body["choices"][0]["message"]["content"] == "ok"


def test_rewrite_sse_block_requested_model():
    block = (
        'data: {"id":"1","model":"provider/model-id",'
        '"choices":[{"index":0,"delta":{"content":"hi"}}]}'
    )
    rewritten = rewrite_sse_block_requested_model(block, "auto")
    assert '"model":"auto"' in rewritten.replace(" ", "")
    assert "provider/model-id" not in rewritten


def test_gateway_routing_context_ready():
    routing = GatewayRoutingContext()
    assert routing.ready is False
    routing.set("groq", "groq-test", "llama")
    assert routing.ready is True
    assert routing.info is not None
    assert routing.info.route_id == "groq-test"


async def test_routed_stream_includes_gateway_headers_when_route_known():
    limiter = GatewayRequestLimiter(max_concurrent_requests=1, queue_timeout_seconds=0.01)
    lease = GatewayLimiterLease(limiter)
    routing = GatewayRoutingContext()
    routing.set("groq", "groq-test", "llama/model")
    sent: list[dict] = []

    async def body():
        yield "data: ok\n\n"

    response = RoutedLimitedStreamingResponse(
        body(),
        lease=lease,
        routing=routing,
        rejected_response=JSONResponse(status_code=429, content={"error": "busy"}),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
    await response(_scope(), _receive, _sender(sent))

    start = next(message for message in sent if message["type"] == "http.response.start")
    headers = _header_map(start)
    assert headers["x-gateway-provider"] == "groq"
    assert headers["x-gateway-route"] == "groq-test"
    assert headers["x-gateway-model"] == "llama/model"
    assert "x-gateway-attempts" not in headers
    body_chunks = [message["body"] for message in sent if message["type"] == "http.response.body"]
    assert b"".join(body_chunks[:-1]) == b"data: ok\n\n"


async def test_routed_stream_omits_gateway_headers_when_route_unknown():
    limiter = GatewayRequestLimiter(max_concurrent_requests=1, queue_timeout_seconds=0.01)
    lease = GatewayLimiterLease(limiter)
    routing = GatewayRoutingContext()
    sent: list[dict] = []

    async def body():
        yield "data: exhausted\n\n"

    response = RoutedLimitedStreamingResponse(
        body(),
        lease=lease,
        routing=routing,
        rejected_response=JSONResponse(status_code=429, content={"error": "busy"}),
        media_type="text/event-stream",
    )
    await response(_scope(), _receive, _sender(sent))

    start = next(message for message in sent if message["type"] == "http.response.start")
    headers = _header_map(start)
    assert "x-gateway-provider" not in headers
    assert "x-gateway-route" not in headers
    assert "x-gateway-model" not in headers


async def test_routed_stream_starts_immediately_for_client_keepalive():
    limiter = GatewayRequestLimiter(max_concurrent_requests=1, queue_timeout_seconds=0.01)
    lease = GatewayLimiterLease(limiter)
    routing = GatewayRoutingContext()
    sent: list[dict] = []

    async def body():
        yield ": freerouter routing\r\n\r\n"
        routing.set("primary", "primary-test", "primary/model")
        yield "data: first\n\n"

    response = RoutedLimitedStreamingResponse(
        body(),
        lease=lease,
        routing=routing,
        rejected_response=JSONResponse(status_code=429, content={"error": "busy"}),
        media_type="text/event-stream",
    )
    await response(_scope(), _receive, _sender(sent))

    start_index = next(
        index for index, message in enumerate(sent) if message["type"] == "http.response.start"
    )
    first_body_index = next(
        index for index, message in enumerate(sent) if message["type"] == "http.response.body"
    )
    assert start_index < first_body_index
    body_chunks = [
        message["body"]
        for message in sent
        if message["type"] == "http.response.body" and message["body"]
    ]
    assert b"freerouter routing" in b"".join(body_chunks)


async def _services_with_provider(
    tmp_path,
    provider: Any,
    routes: list[dict[str, Any]],
) -> AppServices:
    state = StateManager(
        str(tmp_path / "state.sqlite3"),
        [
            ProviderQuota(
                provider.name,
                tokens_per_day=None,
                requests_per_day=None,
                requests_per_minute=30,
            )
        ],
    )
    catalog = ModelCatalog(str(tmp_path / "models.json"))
    stack = await build_core_gateway_stack(state=state, model_catalog=catalog)
    catalog.replace_routes(routes)
    router = WaterfallRouter(
        [provider],
        catalog,
        state,
        request_timeout_seconds=stack.settings.request_timeout_seconds,
        http_client=stack.http_client,
    )
    settings = stack.settings
    return AppServices(
        gateway_state=state,
        model_catalog=catalog,
        request_limiter=GatewayRequestLimiter(
            settings.max_concurrent_requests,
            settings.request_queue_timeout_seconds,
            settings.request_queue_max_waiting_requests,
        ),
        live_monitor=APILiveMonitor(max_events=1000),
        http_client=stack.http_client,
        waterfall_router=router,
        endpoint_diagnosis=EndpointDiagnosisService(
            [provider],
            catalog,
            state,
            request_timeout_seconds=settings.request_timeout_seconds,
        ),
        background_endpoint_diagnosis=None,
        benchmark_research=None,
    )


class _FakeProvider:
    name = "primary"
    api_key = "test-key"
    max_context_tokens = 8192

    @property
    def is_configured(self) -> bool:
        return True

    async def chat_completion(self, client, payload, target_model=None):
        return ProviderResponse(
            "primary",
            200,
            {},
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hello from gateway"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 6},
            },
        )

    async def chat_completion_stream(self, client, payload, target_model=None):
        chunk = json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": "Hi"}}],
            }
        )
        yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_chat_completions_non_stream_exposes_gateway_headers(tmp_path, monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "gateway.sqlite3"))
    monkeypatch.setenv("MODEL_CATALOG_PATH", str(tmp_path / "model_catalog.json"))
    services = await _services_with_provider(
        tmp_path,
        _FakeProvider(),
        [
            {
                "route_id": "primary-test",
                "provider_name": "primary",
                "model_id": "primary/model",
                "display_name": "Primary",
                "rank": 1,
                "enabled": True,
                "tags": ["text"],
            }
        ],
    )
    probe = TestClient(app)
    attach_app_services(probe.app, services)
    response = probe.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.headers["X-Gateway-Provider"] == "primary"
    assert response.headers["X-Gateway-Route"] == "primary-test"
    assert response.headers["X-Gateway-Model"] == "primary/model"
    assert "X-Gateway-Attempts" not in response.headers
    assert response.json()["model"] == "auto"


@pytest.mark.asyncio
async def test_chat_completions_stream_exposes_gateway_headers(tmp_path, monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "gateway.sqlite3"))
    monkeypatch.setenv("MODEL_CATALOG_PATH", str(tmp_path / "model_catalog.json"))
    services = await _services_with_provider(
        tmp_path,
        _FakeProvider(),
        [
            {
                "route_id": "primary-test",
                "provider_name": "primary",
                "model_id": "primary/model",
                "display_name": "Primary",
                "rank": 1,
                "enabled": True,
                "tags": ["text"],
            }
        ],
    )
    probe = TestClient(app)
    attach_app_services(probe.app, services)
    with probe.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "auto",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as response:
        assert response.status_code == 200
        assert "X-Gateway-Attempts" not in response.headers
        body = "".join(response.iter_text())
    assert "freerouter routing" in body
    assert "data:" in body

class _SlowStreamingProvider(_FakeProvider):
    async def chat_completion(self, client, payload, target_model=None):
        await asyncio.sleep(30)
        return await super().chat_completion(client, payload, target_model)

    async def chat_completion_stream(self, client, payload, target_model=None):
        chunk = json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": "Hi"}}],
            }
        )
        yield f"data: {chunk}\n\n"
        await asyncio.sleep(30)
        yield "data: [DONE]\n\n"



async def _post_then_disconnect(path: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    request_sent = False
    sent_disconnect = False
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal request_sent, sent_disconnect
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        if not sent_disconnect:
            sent_disconnect = True
            return {"type": "http.disconnect"}
        await asyncio.sleep(30)
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
        "client": ("testclient", 123),
        "server": ("testserver", 80),
        "root_path": "",
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=5)
    assert sent_disconnect, "test request did not emit disconnect"
    return sent

async def _post_stream_until_marker_then_disconnect(
    path: str,
    payload: dict[str, Any],
    *,
    marker: bytes,
) -> list[dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    disconnect = asyncio.Event()
    request_sent = False
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)
        if message.get("type") != "http.response.body":
            return
        chunk = bytes(message.get("body") or b"")
        if marker in chunk:
            disconnect.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
        "client": ("testclient", 123),
        "server": ("testserver", 80),
        "root_path": "",
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=5)
    assert disconnect.is_set(), "test stream did not reach the disconnect marker"
    return sent


async def _services_with_slow_stream(tmp_path, monkeypatch) -> AppServices:
    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "gateway.sqlite3"))
    monkeypatch.setenv("MODEL_CATALOG_PATH", str(tmp_path / "model_catalog.json"))
    return await _services_with_provider(
        tmp_path,
        _SlowStreamingProvider(),
        [
            {
                "route_id": "primary-test",
                "provider_name": "primary",
                "model_id": "primary/model",
                "display_name": "Primary",
                "rank": 1,
                "enabled": True,
                "tags": ["text"],
            }
        ],
    )


async def _closed_event_for_path(services: AppServices, path: str) -> dict[str, Any]:
    events = await services.live_monitor.snapshot()
    request_ids = {
        event["request_id"]
        for event in events
        if event["event_type"] == "request_started" and event["payload"].get("path") == path
    }
    closed = [
        event
        for event in events
        if event["event_type"] == "request_closed" and event["request_id"] in request_ids
    ]
    assert closed, events
    assert not [
        event
        for event in events
        if event["event_type"] == "request_completed" and event["request_id"] in request_ids
    ]
    return closed[-1]


@pytest.mark.asyncio
async def test_chat_completions_stream_publishes_closed_when_client_disconnects(
    tmp_path, monkeypatch
):
    services = await _services_with_slow_stream(tmp_path, monkeypatch)
    attach_app_services(app, services)

    sent = await _post_stream_until_marker_then_disconnect(
        "/v1/chat/completions",
        {
            "model": "auto",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
        marker=b"Hi",
    )

    assert any(message.get("type") == "http.response.start" for message in sent)
    closed = await _closed_event_for_path(services, "/v1/chat/completions")
    assert closed["payload"]["status_code"] == 499
    assert closed["payload"]["reason"] == "client_closed_or_stream_interrupted"


@pytest.mark.asyncio
async def test_responses_stream_publishes_closed_when_client_disconnects(tmp_path, monkeypatch):
    services = await _services_with_slow_stream(tmp_path, monkeypatch)
    attach_app_services(app, services)

    sent = await _post_stream_until_marker_then_disconnect(
        "/v1/responses",
        {"model": "auto", "input": "hi", "stream": True},
        marker=b"Hi",
    )

    assert any(message.get("type") == "http.response.start" for message in sent)
    closed = await _closed_event_for_path(services, "/v1/responses")
    assert closed["payload"]["status_code"] == 499
    assert closed["payload"]["reason"] == "client_closed_or_stream_interrupted"

@pytest.mark.asyncio
async def test_anthropic_messages_stream_publishes_closed_when_client_disconnects(
    tmp_path, monkeypatch
):
    services = await _services_with_slow_stream(tmp_path, monkeypatch)
    attach_app_services(app, services)

    sent = await _post_stream_until_marker_then_disconnect(
        "/v1/messages",
        {
            "model": "auto",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
        marker=b"Hi",
    )

    assert any(message.get("type") == "http.response.start" for message in sent)
    closed = await _closed_event_for_path(services, "/v1/messages")
    assert closed["payload"]["status_code"] == 499
    assert closed["payload"]["reason"] == "client_closed_or_stream_interrupted"

@pytest.mark.asyncio
async def test_chat_completions_non_stream_publishes_closed_when_client_disconnects(
    tmp_path, monkeypatch
):
    services = await _services_with_slow_stream(tmp_path, monkeypatch)
    attach_app_services(app, services)

    sent = await _post_then_disconnect(
        "/v1/chat/completions",
        {"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert any(message.get("type") == "http.response.start" for message in sent)
    closed = await _closed_event_for_path(services, "/v1/chat/completions")
    assert closed["payload"]["status_code"] == 499
    assert closed["payload"]["reason"] == "client_closed_or_stream_interrupted"


@pytest.mark.asyncio
async def test_responses_non_stream_publishes_closed_when_client_disconnects(
    tmp_path, monkeypatch
):
    services = await _services_with_slow_stream(tmp_path, monkeypatch)
    attach_app_services(app, services)

    sent = await _post_then_disconnect(
        "/v1/responses",
        {"model": "auto", "input": "hi"},
    )

    assert any(message.get("type") == "http.response.start" for message in sent)
    closed = await _closed_event_for_path(services, "/v1/responses")
    assert closed["payload"]["status_code"] == 499
    assert closed["payload"]["reason"] == "client_closed_or_stream_interrupted"


@pytest.mark.asyncio
async def test_anthropic_messages_non_stream_publishes_closed_when_client_disconnects(
    tmp_path, monkeypatch
):
    services = await _services_with_slow_stream(tmp_path, monkeypatch)
    attach_app_services(app, services)

    sent = await _post_then_disconnect(
        "/v1/messages",
        {
            "model": "auto",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert any(message.get("type") == "http.response.start" for message in sent)
    closed = await _closed_event_for_path(services, "/v1/messages")
    assert closed["payload"]["status_code"] == 499
    assert closed["payload"]["reason"] == "client_closed_or_stream_interrupted"
