from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from time import perf_counter
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from app.anthropic_compat import (
    AnthropicStreamMapper,
    anthropic_error_body,
    anthropic_stream_event,
    chat_body_to_anthropic_message,
    messages_payload_to_chat,
)
from app.api.chat_handlers import _monitor_trim, _publish_route_stream_diag
from app.api.gateway_headers import GatewayRouteInfo, GatewayRoutingContext, gateway_route_headers
from app.api.gateway_response import normalize_openai_sse_stream
from app.api.limited_streaming_response import (
    GatewayLimiterLease,
    RoutedLimitedStreamingResponse,
)
from app.app_services import get_app_services
from app.providers import ProviderError
from app.request_requirements import chat_request_requirements
from app.router import (
    NoProviderAvailable,
    RouteStreamDiag,
    UnsupportedCapabilities,
    _split_sse_event_blocks,
    validate_chat_completion_payload,
)
from app.settings import get_settings

router = APIRouter()


def _unsupported_capabilities_anthropic_error(exc: UnsupportedCapabilities) -> tuple[int, dict[str, Any]]:
    caps = sorted(cap for cap in exc.required if cap != "text") or sorted(exc.required)
    message = f"No enabled route supports all required capabilities: {', '.join(caps)}"
    return 400, anthropic_error_body("invalid_request_error", message)


def _waterfall_exhausted_anthropic_error() -> tuple[int, dict[str, Any]]:
    return 503, anthropic_error_body(
        "api_error",
        "No configured provider is currently available",
    )


def _provider_error_anthropic_error(exc: ProviderError) -> tuple[int, dict[str, Any]]:
    status = exc.status_code or 502
    if status == 429:
        error_type = "rate_limit_error"
    elif status == 529:
        error_type = "overloaded_error"
    elif status in {401, 403, 404, 413}:
        error_type = {
            401: "authentication_error",
            403: "permission_error",
            404: "not_found_error",
            413: "request_too_large_error",
        }[status]
    elif status >= 500:
        error_type = "api_error"
    else:
        error_type = "api_error"
    return status, anthropic_error_body(error_type, str(exc))


@router.post("/v1/messages")
async def messages(request: Request) -> Response:
    services = get_app_services(request)
    router_svc = services.waterfall_router
    limiter = services.request_limiter
    monitor = services.live_monitor
    request_id = uuid.uuid4().hex[:12]
    started_at = perf_counter()
    client_ip = request.client.host if request.client else None

    try:
        anthropic_payload: dict[str, Any] = await request.json()
    except json.JSONDecodeError:
        body = anthropic_error_body("invalid_request_error", "Request body must be valid JSON")
        return JSONResponse(status_code=400, content=body)

    if not isinstance(anthropic_payload, dict):
        body = anthropic_error_body("invalid_request_error", "Request body must be a JSON object")
        return JSONResponse(status_code=400, content=body)

    try:
        chat_payload = messages_payload_to_chat(anthropic_payload)
        validate_chat_completion_payload(chat_payload)
    except ValueError as exc:
        body = anthropic_error_body("invalid_request_error", str(exc))
        return JSONResponse(status_code=400, content=body)

    requirements = chat_request_requirements(chat_payload)
    requested_model = anthropic_payload.get("model")
    stream = bool(anthropic_payload.get("stream"))

    await monitor.publish(
        event_type="request_started",
        request_id=request_id,
        payload={
            "path": "/v1/messages",
            "stream": stream,
            "model": requested_model,
            "client_ip": client_ip,
            "request_payload": _monitor_trim(anthropic_payload),
        },
    )

    if stream:
        return await _route_anthropic_stream(
            request_id=request_id,
            started_at=started_at,
            router=router_svc,
            limiter=limiter,
            monitor=monitor,
            chat_payload=chat_payload,
            requirements=requirements,
            requested_model=requested_model,
        )

    if not await limiter.acquire():
        await monitor.publish(
            event_type="request_rejected",
            request_id=request_id,
            payload={"status_code": 429, "reason": "request_queue_timeout"},
        )
        body = anthropic_error_body("rate_limit_error", "Gateway is busy; retry shortly")
        return JSONResponse(status_code=429, content=body, headers={"Retry-After": "1"})

    return await _route_anthropic_non_stream(
        request_id=request_id,
        started_at=started_at,
        router=router_svc,
        limiter=limiter,
        monitor=monitor,
        chat_payload=chat_payload,
        requirements=requirements,
        requested_model=requested_model,
    )


async def _route_anthropic_non_stream(
    *,
    request_id: str,
    started_at: float,
    router: Any,
    limiter: Any,
    monitor: Any,
    chat_payload: dict[str, Any],
    requirements: Any,
    requested_model: Any,
) -> JSONResponse:
    try:
        result = await router.route_chat_completion(chat_payload, requirements=requirements)
    except UnsupportedCapabilities as exc:
        status, body = _unsupported_capabilities_anthropic_error(exc)
        await monitor.publish(
            event_type="request_failed",
            request_id=request_id,
            payload={
                "status_code": status,
                "reason": "unsupported_capabilities",
                "required_capabilities": sorted(exc.required),
                "latency_ms": round((perf_counter() - started_at) * 1000),
            },
        )
        return JSONResponse(status_code=status, content=body)
    except ValueError as exc:
        body = anthropic_error_body("invalid_request_error", str(exc))
        await monitor.publish(
            event_type="request_failed",
            request_id=request_id,
            payload={
                "status_code": 400,
                "reason": "validation_error",
                "message": str(exc),
                "latency_ms": round((perf_counter() - started_at) * 1000),
            },
        )
        return JSONResponse(status_code=400, content=body)
    except NoProviderAvailable as exc:
        status, body = _waterfall_exhausted_anthropic_error()
        await monitor.publish(
            event_type="request_failed",
            request_id=request_id,
            payload={
                "status_code": status,
                "reason": "waterfall_exhausted",
                "attempts": len(exc.attempts),
                "attempts_detail": [asdict(attempt) for attempt in exc.attempts],
                "latency_ms": round((perf_counter() - started_at) * 1000),
            },
        )
        return JSONResponse(status_code=status, content=body)
    except ProviderError as exc:
        status, body = _provider_error_anthropic_error(exc)
        await monitor.publish(
            event_type="request_failed",
            request_id=request_id,
            payload={
                "status_code": status,
                "reason": "provider_error",
                "message": str(exc),
                "latency_ms": round((perf_counter() - started_at) * 1000),
            },
        )
        return JSONResponse(status_code=status, content=body)
    finally:
        limiter.release()

    message_id = f"msg_{uuid.uuid4().hex}"
    anthropic_message = chat_body_to_anthropic_message(
        result.body,
        requested_model=requested_model,
        message_id=message_id,
    )
    await monitor.publish(
        event_type="request_completed",
        request_id=request_id,
        payload={
            "status_code": 200,
            "provider_name": result.provider_name,
            "route_id": result.route_id,
            "model_id": result.model_id,
            "attempts": len(result.attempts),
            "usage": _monitor_trim(anthropic_message.get("usage")),
            "latency_ms": round((perf_counter() - started_at) * 1000),
        },
    )
    return JSONResponse(
        content=anthropic_message,
        headers=gateway_route_headers(
            GatewayRouteInfo(
                provider_name=result.provider_name,
                route_id=result.route_id,
                model_id=result.model_id,
            )
        ),
    )


async def _route_anthropic_stream(
    *,
    request_id: str,
    started_at: float,
    router: Any,
    limiter: Any,
    monitor: Any,
    chat_payload: dict[str, Any],
    requirements: Any,
    requested_model: Any,
) -> Response:
    settings = get_settings()
    message_id = f"msg_{uuid.uuid4().hex}"
    lease = GatewayLimiterLease(limiter)
    routing = GatewayRoutingContext()
    selected_provider = ""
    selected_route = ""
    selected_model = ""

    async def anthropic_sse_stream():
        nonlocal selected_provider, selected_route, selected_model
        carry = ""
        completed = False
        mapper = AnthropicStreamMapper(message_id=message_id, model=requested_model)
        try:
            async for part in normalize_openai_sse_stream(
                router.iter_chat_completion_openai_stream(
                    chat_payload,
                    requirements=requirements,
                ),
                requested_model,
            ):
                if isinstance(part, RouteStreamDiag):
                    await _publish_route_stream_diag(monitor, request_id, part)
                    if part.event_type == "route_selected":
                        selected_provider = part.provider_name or ""
                        selected_route = part.route_id or ""
                        selected_model = part.model_id or ""
                        routing.set(selected_provider, selected_route, selected_model)
                        if (
                            settings.streaming_release_slot_after_route_selected
                            and lease.held
                        ):
                            lease.release()
                    continue
                carry += part
                blocks, carry = _split_sse_event_blocks(carry)
                for block in blocks:
                    for event in mapper.events_from_openai_sse(block):
                        yield event
                        if "message_stop" in event:
                            completed = True
            if not completed:
                for event in mapper.events_from_openai_sse("data: [DONE]\n\n"):
                    yield event
                    if "message_stop" in event:
                        completed = True
        except UnsupportedCapabilities as exc:
            _, body = _unsupported_capabilities_anthropic_error(exc)
            await monitor.publish(
                event_type="request_failed",
                request_id=request_id,
                payload={
                    "status_code": 400,
                    "reason": "unsupported_capabilities",
                    "required_capabilities": sorted(exc.required),
                    "latency_ms": round((perf_counter() - started_at) * 1000),
                },
            )
            yield anthropic_stream_event("error", body)
        except NoProviderAvailable as exc:
            _, body = _waterfall_exhausted_anthropic_error()
            await monitor.publish(
                event_type="request_failed",
                request_id=request_id,
                payload={
                    "status_code": 503,
                    "reason": "waterfall_exhausted",
                    "attempts": len(exc.attempts),
                    "latency_ms": round((perf_counter() - started_at) * 1000),
                },
            )
            yield anthropic_stream_event("error", body)
        except ValueError as exc:
            body = anthropic_error_body("invalid_request_error", str(exc))
            await monitor.publish(
                event_type="request_failed",
                request_id=request_id,
                payload={
                    "status_code": 400,
                    "reason": "validation_error",
                    "message": str(exc),
                    "latency_ms": round((perf_counter() - started_at) * 1000),
                },
            )
            yield anthropic_stream_event("error", body)
        except ProviderError as exc:
            _, body = _provider_error_anthropic_error(exc)
            await monitor.publish(
                event_type="request_failed",
                request_id=request_id,
                payload={
                    "status_code": exc.status_code or 502,
                    "reason": "provider_error",
                    "message": str(exc),
                    "latency_ms": round((perf_counter() - started_at) * 1000),
                },
            )
            yield anthropic_stream_event("error", body)
        finally:
            lease.release()
            if completed:
                await monitor.publish(
                    event_type="request_completed",
                    request_id=request_id,
                    payload={
                        "status_code": 200,
                        "provider_name": selected_provider,
                        "route_id": selected_route,
                        "model_id": selected_model,
                        "latency_ms": round((perf_counter() - started_at) * 1000),
                    },
                )

    async def on_rejected() -> None:
        await monitor.publish(
            event_type="request_rejected",
            request_id=request_id,
            payload={"status_code": 429, "reason": "request_queue_timeout"},
        )

    return RoutedLimitedStreamingResponse(
        anthropic_sse_stream(),
        lease=lease,
        routing=routing,
        rejected_response=JSONResponse(
            status_code=429,
            content=anthropic_error_body("rate_limit_error", "Gateway is busy; retry shortly"),
            headers={"Retry-After": "1"},
        ),
        on_rejected=on_rejected,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
