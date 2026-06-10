from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from time import perf_counter
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response

from app.api.gateway_headers import GatewayRouteInfo, GatewayRoutingContext, gateway_route_headers
from app.api.gateway_response import normalize_chat_completion_body, normalize_openai_sse_stream
from app.api.limited_streaming_response import (
    GatewayLimiterLease,
    RoutedLimitedStreamingResponse,
)
from app.api.stream_monitor import StreamMonitorTracker, monitor_live_value
from app.app_services import get_app_services
from app.codex_compat import (
    ResponsesStreamMapper,
    response_stream_event,
    responses_stream_done,
    responses_stream_start,
)
from app.live_monitor import APILiveMonitor
from app.model_catalog import ModelCatalog
from app.providers import ProviderError
from app.request_requirements import RequestRequirements, chat_request_requirements
from app.tool_use_validation import ROUTING_SSE_KEEPALIVE
from app.router import (
    NoProviderAvailable,
    RouteStreamDiag,
    UnsupportedCapabilities,
    _display_capabilities,
    _split_sse_event_blocks,
    unsupported_capabilities_error_body,
    validate_chat_completion_payload,
)
from app.settings import get_settings
from app.state import StateManager
from app.web_search_payload import WEB_SEARCH_TOOL, payload_with_required_web_search


def _live_monitor_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: monitor_live_value(value) for key, value in payload.items()}


def _monitor_trim(
    value: Any, *, max_string: int = 1200, max_items: int = 30, depth: int = 0
) -> Any:
    if depth >= 4:
        return "<truncated-depth>"
    if isinstance(value, str):
        if len(value) <= max_string:
            return value
        return value[:max_string] + f"... <truncated {len(value) - max_string} chars>"
    if isinstance(value, list):
        items = [
            _monitor_trim(item, max_string=max_string, max_items=max_items, depth=depth + 1)
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            items.append(f"<truncated {len(value) - max_items} items>")
        return items
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= max_items:
                out["__truncated__"] = f"{len(value) - max_items} keys omitted"
                break
            out[str(k)] = _monitor_trim(
                v, max_string=max_string, max_items=max_items, depth=depth + 1
            )
        return out
    return value


def _assistant_text_from_response_body(body: Any) -> str:
    if not isinstance(body, dict):
        return ""
    direct_content = body.get("content")
    if isinstance(direct_content, str):
        return direct_content
    message = body.get("message")
    if isinstance(message, dict):
        message_content = message.get("content")
        if isinstance(message_content, str):
            return message_content
        if isinstance(message_content, list):
            return _content_parts_to_text(message_content)
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            choice_message = first_choice.get("message")
            if isinstance(choice_message, dict):
                message_content = choice_message.get("content")
                if isinstance(message_content, str):
                    return message_content
                if isinstance(message_content, list):
                    return _content_parts_to_text(message_content)
            choice_text = first_choice.get("text")
            if isinstance(choice_text, str):
                return choice_text
    return ""


def _content_parts_to_text(parts: list[Any]) -> str:
    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, str):
            text_parts.append(part)
        elif isinstance(part, dict):
            text = part.get("text") or part.get("content")
            if isinstance(text, str):
                text_parts.append(text)
    return "\n".join(text_parts)


def _payload_with_required_web_search(payload: Any) -> dict[str, Any]:
    return payload_with_required_web_search(payload)


async def _catalog_payload_with_health(
    catalog: ModelCatalog, state: StateManager
) -> dict[str, Any]:
    payload = catalog.to_payload()
    route_rows = [
        (
            route_payload["route_id"],
            route_payload["provider_name"],
            route_payload["model_id"],
        )
        for route_payload in payload["data"]
    ]
    health_map = await state.get_route_states_batch(route_rows)
    for route_payload in payload["data"]:
        route_state = health_map[route_payload["route_id"]]
        route_payload["health"] = asdict(route_state)
    return payload


async def _publish_route_stream_diag(
    monitor: APILiveMonitor,
    request_id: str,
    diag: RouteStreamDiag,
) -> None:
    if diag.event_type == "usage_summary":
        if diag.usage:
            await monitor.publish(
                event_type="usage_update",
                request_id=request_id,
                payload={"usage": monitor_live_value(dict(diag.usage))},
            )
        return
    if diag.event_type == "route_trying":
        await monitor.publish(
            event_type="route_attempt",
            request_id=request_id,
            payload={
                "route_event": {
                    "type": "route_trying",
                    "provider_name": diag.provider_name,
                    "route_id": diag.route_id,
                    "model_id": diag.model_id,
                }
            },
        )
        return
    if diag.event_type == "route_skipped":
        await monitor.publish(
            event_type="route_attempt",
            request_id=request_id,
            payload={
                "route_event": {
                    "type": "route_skip",
                    "provider_name": diag.provider_name,
                    "route_id": diag.route_id,
                    "model_id": diag.model_id,
                    "reason": diag.reason,
                }
            },
        )
        return
    if diag.event_type == "route_failed":
        await monitor.publish(
            event_type="route_attempt",
            request_id=request_id,
            payload={
                "route_event": {
                    "type": "route_fail",
                    "provider_name": diag.provider_name,
                    "route_id": diag.route_id,
                    "model_id": diag.model_id,
                    "reason": diag.reason,
                }
            },
        )
        return
    if diag.event_type == "route_flagged":
        await monitor.publish(
            event_type="route_attempt",
            request_id=request_id,
            payload={
                "route_event": {
                    "type": "route_flagged",
                    "provider_name": diag.provider_name,
                    "route_id": diag.route_id,
                    "model_id": diag.model_id,
                    "reason": diag.reason,
                }
            },
        )
        return
    if diag.event_type == "route_selected":
        selected_payload: dict[str, Any] = {
            "provider_name": diag.provider_name,
            "route_id": diag.route_id,
            "model_id": diag.model_id,
        }
        if diag.route_tags:
            selected_payload["route_tags"] = list(diag.route_tags)
        if diag.required_capabilities:
            selected_payload["required_capabilities"] = _display_capabilities(
                diag.required_capabilities
            )
        await monitor.publish(
            event_type="route_selected",
            request_id=request_id,
            payload=selected_payload,
        )
        route_event: dict[str, Any] = {
            "type": "route_selected",
            "provider_name": diag.provider_name,
            "route_id": diag.route_id,
            "model_id": diag.model_id,
        }
        if diag.route_tags:
            route_event["route_tags"] = list(diag.route_tags)
        if diag.required_capabilities:
            route_event["required_capabilities"] = _display_capabilities(
                diag.required_capabilities
            )
        await monitor.publish(
            event_type="route_attempt",
            request_id=request_id,
            payload={"route_event": route_event},
        )
        return


async def _route_chat_completion_stream_request(
    request: Request,
    *,
    payload: dict[str, Any],
    path: str,
    requirements: RequestRequirements | None = None,
    require_assistant_content: bool = False,
) -> Response:
    services = get_app_services(request)
    router = services.waterfall_router
    limiter = services.request_limiter
    monitor = services.live_monitor
    request_id = uuid.uuid4().hex[:12]
    started_at = perf_counter()
    client_ip = request.client.host if request.client else None

    try:
        validate_chat_completion_payload(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    resolved_requirements = requirements or chat_request_requirements(payload)
    await monitor.publish(
        event_type="request_started",
        request_id=request_id,
        payload={
            "path": path,
            "stream": True,
            "model": payload.get("model"),
            "client_ip": client_ip,
            "required_capabilities": _display_capabilities(
                resolved_requirements.required_capabilities
            ),
            "request_payload": monitor_live_value(payload),
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

    settings = get_settings()
    lease = GatewayLimiterLease(limiter)
    routing = GatewayRoutingContext()
    tracker = StreamMonitorTracker(requested_model=payload.get("model"))

    async def openai_sse_stream():
        completed = False
        try:
            yield ROUTING_SSE_KEEPALIVE
            async for part in normalize_openai_sse_stream(
                router.iter_chat_completion_openai_stream(
                    payload,
                    requirements=requirements,
                    require_assistant_content=require_assistant_content,
                ),
                payload.get("model"),
            ):
                if isinstance(part, RouteStreamDiag):
                    tracker.record_diag(part)
                    await _publish_route_stream_diag(monitor, request_id, part)
                    if part.event_type in {
                        "route_trying",
                        "route_failed",
                        "route_skipped",
                        "route_flagged",
                    }:
                        yield ROUTING_SSE_KEEPALIVE
                    if part.event_type == "route_selected":
                        routing.set(
                            part.provider_name or "",
                            part.route_id or "",
                            part.model_id or "",
                        )
                        if (
                            settings.streaming_release_slot_after_route_selected
                            and lease.held
                        ):
                            lease.release()
                else:
                    tracker.record_openai_sse(part)
                    yield part
            completed = True
        except UnsupportedCapabilities as exc:
            error_body = unsupported_capabilities_error_body(exc)
            await monitor.publish(
                event_type="request_failed",
                request_id=request_id,
                payload={
                    "status_code": 400,
                    "reason": "unsupported_capabilities",
                    "required_capabilities": error_body["error"]["required_capabilities"],
                    "latency_ms": round((perf_counter() - started_at) * 1000),
                },
            )
            yield "data: " + json.dumps(error_body) + "\n\n"
            yield "data: [DONE]\n\n"
        except NoProviderAvailable as exc:
            await monitor.publish(
                event_type="request_failed",
                request_id=request_id,
                payload={
                    "status_code": 503,
                    "reason": "waterfall_exhausted",
                    "attempts": len(exc.attempts),
                    "attempts_detail": [asdict(attempt) for attempt in exc.attempts],
                    "latency_ms": round((perf_counter() - started_at) * 1000),
                },
            )
            yield (
                "data: "
                + json.dumps(
                    {
                        "error": {
                            "message": "No configured provider is currently available",
                            "type": "provider_unavailable",
                            "code": "waterfall_exhausted",
                            "attempts": [asdict(attempt) for attempt in exc.attempts],
                        }
                    }
                )
                + "\n\n"
            )
            yield "data: [DONE]\n\n"
        except ValueError as exc:
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
            yield "data: " + json.dumps({"error": {"message": str(exc), "type": "invalid_request"}}) + "\n\n"
        finally:
            lease.release()
            if completed:
                await monitor.publish(
                    event_type="request_completed",
                    request_id=request_id,
                    payload=_live_monitor_payload(
                        tracker.completed_payload(
                            status_code=200,
                            latency_ms=round((perf_counter() - started_at) * 1000),
                        )
                    ),
                )

    return RoutedLimitedStreamingResponse(
        openai_sse_stream(),
        lease=lease,
        routing=routing,
        rejected_response=rejected_response,
        on_rejected=on_rejected,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _route_chat_completion_request(
    request: Request,
    *,
    payload: dict[str, Any],
    path: str,
    requirements: RequestRequirements | None = None,
    require_assistant_content: bool = False,
) -> JSONResponse:
    services = get_app_services(request)
    router = services.waterfall_router
    limiter = services.request_limiter
    monitor = services.live_monitor
    request_id = uuid.uuid4().hex[:12]
    started_at = perf_counter()
    client_ip = request.client.host if request.client else None
    resolved_requirements = requirements or chat_request_requirements(payload)
    await monitor.publish(
        event_type="request_started",
        request_id=request_id,
        payload={
            "path": path,
            "stream": bool(payload.get("stream")),
            "model": payload.get("model"),
            "client_ip": client_ip,
            "required_capabilities": _display_capabilities(
                resolved_requirements.required_capabilities
            ),
            "request_payload": monitor_live_value(payload),
        },
    )

    if not await limiter.acquire():
        await monitor.publish(
            event_type="request_rejected",
            request_id=request_id,
            payload={"status_code": 429, "reason": "request_queue_timeout"},
        )
        return JSONResponse(
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

    try:
        result = await router.route_chat_completion(
            payload,
            requirements=requirements,
            require_assistant_content=require_assistant_content,
        )
    except UnsupportedCapabilities as exc:
        error_body = unsupported_capabilities_error_body(exc)
        await monitor.publish(
            event_type="request_failed",
            request_id=request_id,
            payload={
                "status_code": 400,
                "reason": "unsupported_capabilities",
                "required_capabilities": error_body["error"]["required_capabilities"],
                "latency_ms": round((perf_counter() - started_at) * 1000),
            },
        )
        return JSONResponse(status_code=400, content=error_body)
    except ValueError as exc:
        await monitor.publish(
            event_type="request_failed",
            request_id=request_id,
            payload={
                "status_code": 400,
                "reason": "validation_error",
                "message": str(exc),
                "response_body": {"detail": str(exc)},
            },
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NoProviderAvailable as exc:
        await monitor.publish(
            event_type="request_failed",
            request_id=request_id,
            payload={
                "status_code": 503,
                "reason": "waterfall_exhausted",
                "attempts": len(exc.attempts),
                "attempts_detail": [asdict(attempt) for attempt in exc.attempts],
                "latency_ms": round((perf_counter() - started_at) * 1000),
            },
        )
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": "No configured provider is currently available",
                    "type": "provider_unavailable",
                    "code": "waterfall_exhausted",
                    "attempts": [asdict(attempt) for attempt in exc.attempts],
                }
            },
        )
    except ProviderError as exc:
        await monitor.publish(
            event_type="request_failed",
            request_id=request_id,
            payload={
                "status_code": exc.status_code or 502,
                "reason": "provider_error",
                "message": str(exc),
                "response_body": _monitor_trim(
                    {
                        "error": {
                            "message": str(exc),
                            "status_code": exc.status_code,
                            "body": exc.body,
                        }
                    }
                ),
                "latency_ms": round((perf_counter() - started_at) * 1000),
            },
        )
        return JSONResponse(
            status_code=exc.status_code or 502,
            content={
                "error": {
                    "message": str(exc),
                    "type": "provider_error",
                    "code": exc.status_code,
                    "body": exc.body,
                }
            },
        )
    finally:
        limiter.release()

    response_body = normalize_chat_completion_body(result.body, payload.get("model"))
    await monitor.publish(
        event_type="request_completed",
        request_id=request_id,
        payload=_live_monitor_payload(
            {
                "status_code": 200,
                "model": payload.get("model") or "auto",
                "provider_name": result.provider_name,
                "route_id": result.route_id,
                "model_id": result.model_id,
                "attempts": len(result.attempts),
                "usage": response_body.get("usage")
                if isinstance(response_body, dict)
                else None,
                "attempts_detail": [asdict(attempt) for attempt in result.attempts],
                "assistant_text": _assistant_text_from_response_body(response_body),
                "response_body": response_body,
                "latency_ms": round((perf_counter() - started_at) * 1000),
            }
        ),
    )
    return JSONResponse(
        content=response_body,
        headers=gateway_route_headers(
            GatewayRouteInfo(
                provider_name=result.provider_name,
                route_id=result.route_id,
                model_id=result.model_id,
            )
        ),
    )


async def _route_responses_stream_request(
    request: Request,
    *,
    payload: dict[str, Any],
    requested_model: Any,
) -> Response:
    services = get_app_services(request)
    router = services.waterfall_router
    limiter = services.request_limiter
    monitor = services.live_monitor
    request_id = uuid.uuid4().hex[:12]
    started_at = perf_counter()
    client_ip = request.client.host if request.client else None
    response_id = f"resp_{uuid.uuid4().hex}"

    try:
        validate_chat_completion_payload(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await monitor.publish(
        event_type="request_started",
        request_id=request_id,
        payload={
            "path": "/v1/responses",
            "stream": True,
            "model": requested_model or payload.get("model"),
            "client_ip": client_ip,
            "request_payload": monitor_live_value(payload),
        },
    )

    lease = GatewayLimiterLease(limiter)
    routing = GatewayRoutingContext()
    tracker = StreamMonitorTracker(requested_model=requested_model or payload.get("model"))
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

    async def responses_sse_stream():
        carry = ""
        completed = False
        mapper = ResponsesStreamMapper(response_id=response_id)
        try:
            yield responses_stream_start(response_id=response_id, model=requested_model)
            async for part in normalize_openai_sse_stream(
                router.iter_chat_completion_openai_stream(payload),
                requested_model,
            ):
                if isinstance(part, RouteStreamDiag):
                    tracker.record_diag(part)
                    await _publish_route_stream_diag(monitor, request_id, part)
                    if part.event_type in {
                        "route_trying",
                        "route_failed",
                        "route_skipped",
                        "route_flagged",
                    }:
                        yield ROUTING_SSE_KEEPALIVE
                    if part.event_type == "route_selected":
                        routing.set(
                            part.provider_name or "",
                            part.route_id or "",
                            part.model_id or "",
                        )
                    continue
                tracker.record_openai_sse(part)
                carry += part
                blocks, carry = _split_sse_event_blocks(carry)
                for block in blocks:
                    for event in mapper.events_from_openai_sse(block):
                        yield event
                        if "response.completed" in event:
                            completed = True
            if not completed:
                yield response_stream_event(
                    "response.completed",
                    {
                        "type": "response.completed",
                        "response": {
                            "id": response_id,
                            "object": "response",
                            "status": "completed",
                        },
                    },
                )
            yield responses_stream_done()
        except UnsupportedCapabilities as exc:
            error_body = unsupported_capabilities_error_body(exc)
            yield response_stream_event(
                "response.failed",
                {
                    "type": "response.failed",
                    "sequence_number": mapper.sequence_number + 1,
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "status": "failed",
                        "error": error_body["error"],
                    },
                },
            )
            yield responses_stream_done()
        except NoProviderAvailable as exc:
            yield response_stream_event(
                "response.failed",
                {
                    "type": "response.failed",
                    "sequence_number": mapper.sequence_number + 1,
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "status": "failed",
                        "error": {
                            "message": "No configured provider is currently available",
                            "type": "provider_unavailable",
                            "code": "waterfall_exhausted",
                            "attempts": [asdict(attempt) for attempt in exc.attempts],
                        },
                    },
                },
            )
            yield responses_stream_done()
        except (ProviderError, ValueError) as exc:
            yield response_stream_event(
                "response.failed",
                {
                    "type": "response.failed",
                    "sequence_number": mapper.sequence_number + 1,
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "status": "failed",
                        "error": {
                            "message": str(exc),
                            "type": "provider_error",
                            "code": getattr(exc, "status_code", None),
                        },
                    },
                },
            )
            yield responses_stream_done()
        finally:
            lease.release()
            if completed:
                await monitor.publish(
                    event_type="request_completed",
                    request_id=request_id,
                    payload=_live_monitor_payload(
                        tracker.completed_payload(
                            status_code=200,
                            latency_ms=round((perf_counter() - started_at) * 1000),
                        )
                    ),
                )

    return RoutedLimitedStreamingResponse(
        responses_sse_stream(),
        lease=lease,
        routing=routing,
        rejected_response=rejected_response,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
