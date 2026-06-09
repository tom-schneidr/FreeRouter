from __future__ import annotations

import json
import uuid
from time import perf_counter
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from app.api.chat_handlers import (
    _monitor_trim,
    _payload_with_required_web_search,
    _route_chat_completion_request,
    _route_chat_completion_stream_request,
    _route_responses_stream_request,
)
from app.api.gateway_headers import GatewayRoutingContext
from app.api.limited_streaming_response import GatewayLimiterLease, RoutedLimitedStreamingResponse
from app.app_services import get_app_services
from app.codex_compat import chat_body_to_response, responses_payload_to_chat
from app.request_requirements import chat_request_requirements, with_extra_capabilities
from app.settings import get_settings
from app.stream_route import stream_route_chat

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    try:
        payload: dict[str, Any] = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    if payload.get("stream"):
        return await _route_chat_completion_stream_request(
            request,
            payload=payload,
            path="/v1/chat/completions",
        )
    return await _route_chat_completion_request(
        request,
        payload=payload,
        path="/v1/chat/completions",
    )


@router.post("/v1/responses")
async def responses(request: Request) -> Response:
    try:
        responses_payload: dict[str, Any] = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc
    if not isinstance(responses_payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    try:
        chat_payload = responses_payload_to_chat(responses_payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if chat_payload.get("stream"):
        return await _route_responses_stream_request(
            request,
            payload=chat_payload,
            requested_model=responses_payload.get("model"),
        )

    chat_response = await _route_chat_completion_request(
        request,
        payload=chat_payload,
        path="/v1/responses",
    )
    chat_body = json.loads(chat_response.body.decode("utf-8"))
    if chat_response.status_code >= 400:
        return JSONResponse(status_code=chat_response.status_code, content=chat_body)
    return JSONResponse(
        content=chat_body_to_response(chat_body, requested_model=responses_payload.get("model")),
        headers={
            key: value
            for key, value in chat_response.headers.items()
            if key.lower().startswith("x-gateway-")
        },
    )


@router.post("/v1/chat/completions/web-search")
async def chat_completions_web_search(request: Request) -> Response:
    payload: Any = await request.json()
    try:
        prepared_payload = _payload_with_required_web_search(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    web_search_requirements = with_extra_capabilities(
        chat_request_requirements(prepared_payload),
        "web-search",
    )
    if prepared_payload.get("stream"):
        return await _route_chat_completion_stream_request(
            request,
            payload=prepared_payload,
            path="/v1/chat/completions/web-search",
            requirements=web_search_requirements,
            require_assistant_content=True,
        )
    return await _route_chat_completion_request(
        request,
        payload=prepared_payload,
        path="/v1/chat/completions/web-search",
        requirements=web_search_requirements,
        require_assistant_content=True,
    )


@router.post("/v1/chat/completions/stream-route")
async def chat_completions_stream_route(request: Request) -> Response:
    payload: dict[str, Any] = await request.json()
    services = get_app_services(request)
    router_svc = services.waterfall_router
    limiter = services.request_limiter
    monitor = services.live_monitor
    request_id = uuid.uuid4().hex[:12]
    started_at = perf_counter()
    client_ip = request.client.host if request.client else None
    done_published = False
    await monitor.publish(
        event_type="request_started",
        request_id=request_id,
        payload={
            "path": "/v1/chat/completions/stream-route",
            "stream": True,
            "model": payload.get("model"),
            "client_ip": client_ip,
            "request_payload": _monitor_trim(payload),
        },
    )
    rejected_response = JSONResponse(
        status_code=429,
        content={
            "error": {
                "message": "Gateway is busy; retry shortly",
                "type": "gateway_overloaded",
                "code": "request_queue_timeout",
            }
        },
        headers={"Retry-After": "1"},
    )

    async def on_rejected() -> None:
        await monitor.publish(
            event_type="request_rejected",
            request_id=request_id,
            payload={"status_code": 429, "reason": "request_queue_timeout"},
        )

    stream_settings = get_settings()
    lease = GatewayLimiterLease(limiter)
    routing = GatewayRoutingContext()

    async def on_stream_event(event_payload: dict[str, Any]) -> None:
        nonlocal done_published
        event_type = event_payload.get("type")
        if event_type == "usage":
            usage = event_payload.get("usage")
            if isinstance(usage, dict) and usage:
                await monitor.publish(
                    event_type="usage_update",
                    request_id=request_id,
                    payload={"usage": _monitor_trim(usage)},
                )
            return
        if event_type == "route_selected":
            routing.set(
                str(event_payload.get("provider") or ""),
                str(event_payload.get("route_id") or ""),
                str(event_payload.get("model_id") or ""),
            )
            await monitor.publish(
                event_type="route_selected",
                request_id=request_id,
                payload={
                    "provider_name": event_payload.get("provider"),
                    "route_id": event_payload.get("route_id"),
                    "model_id": event_payload.get("model_id"),
                    "stream_event": _monitor_trim(event_payload),
                },
            )
            if (
                stream_settings.streaming_release_slot_after_route_selected
                and lease.held
            ):
                lease.release()
            return
        if event_type in {"route_trying", "route_skip", "route_fail", "route_flagged"}:
            await monitor.publish(
                event_type="route_attempt",
                request_id=request_id,
                payload={
                    "route_event": {
                        "type": event_type,
                        "provider_name": event_payload.get("provider"),
                        "route_id": event_payload.get("route_id"),
                        "model_id": event_payload.get("model_id"),
                        "reason": event_payload.get("reason"),
                    }
                },
            )
            return
        if event_type == "content":
            await monitor.publish(
                event_type="response_content",
                request_id=request_id,
                payload={
                    "stream_event": _monitor_trim(event_payload),
                    "content": event_payload.get("text") or "",
                },
            )
            return
        if event_type == "error":
            await monitor.publish(
                event_type="request_failed",
                request_id=request_id,
                payload={
                    "status_code": 502,
                    "reason": "stream_error",
                    "message": event_payload.get("message"),
                    "response_body": _monitor_trim(event_payload),
                    "latency_ms": round((perf_counter() - started_at) * 1000),
                },
            )
            done_published = True
            return
        if event_type == "done":
            await monitor.publish(
                event_type="request_completed",
                request_id=request_id,
                payload={
                    "status_code": 200,
                    "provider_name": event_payload.get("provider"),
                    "route_id": event_payload.get("route_id"),
                    "model_id": event_payload.get("model_id"),
                    "route_event": {
                        "type": "success",
                        "provider_name": event_payload.get("provider"),
                        "route_id": event_payload.get("route_id"),
                        "model_id": event_payload.get("model_id"),
                    },
                    "assistant_text": _monitor_trim(
                        event_payload.get("content") or "",
                        max_string=8000,
                    ),
                    "response_body": _monitor_trim(event_payload),
                    "latency_ms": round((perf_counter() - started_at) * 1000),
                },
            )
            done_published = True

    async def limited_stream():
        nonlocal done_published
        try:
            async for chunk in stream_route_chat(
                payload,
                router_svc,
                chunk_replay_sleep_seconds=get_settings().sse_chunk_replay_sleep_seconds,
                on_emit=on_stream_event,
            ):
                yield chunk
        finally:
            lease.release()
            if not done_published:
                await monitor.publish(
                    event_type="request_closed",
                    request_id=request_id,
                    payload={
                        "status_code": 499,
                        "reason": "client_closed_or_stream_interrupted",
                        "latency_ms": round((perf_counter() - started_at) * 1000),
                    },
                )

    return RoutedLimitedStreamingResponse(
        limited_stream(),
        lease=lease,
        routing=routing,
        rejected_response=rejected_response,
        on_rejected=on_rejected,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
