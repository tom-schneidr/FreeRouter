from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from time import perf_counter, time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse

from app.codex_compat import (
    ResponsesStreamMapper,
    chat_body_to_response,
    response_stream_event,
    responses_payload_to_chat,
    responses_stream_done,
    responses_stream_start,
)
from app.desktop_api import (
    desktop_capabilities,
    desktop_logs,
    desktop_settings_payload,
    export_desktop_backup,
    import_desktop_backup,
    import_desktop_backup_upload,
    request_desktop_restart,
    save_desktop_settings,
)
from app.endpoint_diagnosis import (
    BackgroundEndpointDiagnosis,
    EndpointDiagnosisService,
    EndpointSupervisor,
)
from app.live_monitor import APILiveMonitor
from app.model_catalog import ModelCatalog
from app.providers import PROVIDER_QUOTAS, ProviderError, build_provider_adapters
from app.request_limiter import GatewayRequestLimiter
from app.react_app import mount_react_app
from app.router import (
    NoProviderAvailable,
    RouteStreamDiag,
    WaterfallRouter,
    _split_sse_event_blocks,
    validate_chat_completion_payload,
)
from app.settings import get_settings
from app.state import StateManager
from app.stream_route import stream_route_chat
from app.legacy_pages import CHAT_HTML, LIVE_API_HTML, ROUTE_HEALTH_HTML, USAGE_STATS_HTML
from app.ui.brand import FAVICON_PATH, LOGO_PATH
from app.ui.docs_page import swagger_docs_html
from app.ui.embed import with_embed_support

WEB_SEARCH_TOOL = {"type": "web_search_preview"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    gateway_http_client: httpx.AsyncClient | None = None
    state = StateManager(
        settings.database_path,
        PROVIDER_QUOTAS,
        busy_timeout_ms=settings.sqlite_busy_timeout_ms,
    )
    await state.initialize()
    model_catalog = ModelCatalog(settings.model_catalog_path)
    model_catalog.initialize()
    providers = build_provider_adapters(settings)

    app.state.gateway_state = state
    app.state.model_catalog = model_catalog
    app.state.request_limiter = GatewayRequestLimiter(
        settings.max_concurrent_requests,
        settings.request_queue_timeout_seconds,
        settings.request_queue_max_waiting_requests,
    )
    app.state.live_monitor = APILiveMonitor(max_events=1000)
    limits = httpx.Limits(
        max_connections=settings.http_max_connections,
        max_keepalive_connections=settings.http_max_keepalive_connections,
        keepalive_expiry=settings.http_keepalive_expiry_seconds,
    )
    gateway_http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.request_timeout_seconds),
        limits=limits,
    )
    app.state.http_client = gateway_http_client
    app.state.waterfall_router = WaterfallRouter(
        providers,
        model_catalog,
        state,
        request_timeout_seconds=settings.request_timeout_seconds,
        http_client=gateway_http_client,
    )
    app.state.endpoint_diagnosis = EndpointDiagnosisService(
        providers,
        model_catalog,
        state,
        request_timeout_seconds=settings.request_timeout_seconds,
        supervisor=EndpointSupervisor(
            enabled=settings.endpoint_diagnosis_supervisor_enabled,
            providers=providers,
            catalog=model_catalog,
            state=state,
            preferred_model=settings.endpoint_diagnosis_supervisor_model,
        ),
    )
    app.state.background_endpoint_diagnosis = None
    if settings.auto_endpoint_diagnosis_enabled:
        background = BackgroundEndpointDiagnosis(
            app.state.endpoint_diagnosis,
            interval_seconds=settings.auto_endpoint_diagnosis_interval_seconds,
            startup_delay_seconds=settings.auto_endpoint_diagnosis_startup_delay_seconds,
            apply_safe_suggestions=settings.auto_endpoint_maintenance_enabled,
        )
        app.state.background_endpoint_diagnosis = background
        background.start()
    try:
        yield
    finally:
        background = app.state.background_endpoint_diagnosis
        if background is not None:
            await background.stop()
        if gateway_http_client is not None:
            await gateway_http_client.aclose()


app = FastAPI(
    title="FreeRouter",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
)
mount_react_app(app)


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


@app.get("/", include_in_schema=False)
async def index() -> RedirectResponse:
    return RedirectResponse(url="/app", status_code=307)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(content=FAVICON_PATH.read_bytes(), media_type="image/png")


@app.get("/brand/favicon.png", include_in_schema=False)
async def brand_favicon() -> Response:
    return Response(content=FAVICON_PATH.read_bytes(), media_type="image/png")


@app.get("/brand/logo.png", include_in_schema=False)
async def brand_logo() -> Response:
    return Response(content=LOGO_PATH.read_bytes(), media_type="image/png")


@app.get("/docs", include_in_schema=False)
async def swagger_docs_page() -> HTMLResponse:
    return swagger_docs_html(openapi_url=app.openapi_url, title=f"{app.title} API")


@app.get("/status", response_class=HTMLResponse, include_in_schema=False)
async def provider_status_page() -> HTMLResponse:
    return HTMLResponse(with_embed_support(USAGE_STATS_HTML))


@app.get("/health", response_class=HTMLResponse, include_in_schema=False)
async def route_health_page() -> str:
    return with_embed_support(ROUTE_HEALTH_HTML)


@app.get("/live", response_class=HTMLResponse, include_in_schema=False)
async def live_api_page() -> HTMLResponse:
    return HTMLResponse(with_embed_support(LIVE_API_HTML))


@app.get("/chat", response_class=HTMLResponse, include_in_schema=False)
async def chat_page() -> HTMLResponse:
    return HTMLResponse(with_embed_support(CHAT_HTML))


@app.get("/v1/desktop/capabilities")
async def desktop_capabilities_endpoint(request: Request) -> dict[str, Any]:
    return desktop_capabilities(request)


@app.post("/v1/desktop/restart")
async def desktop_restart_endpoint(request: Request) -> dict[str, Any]:
    return request_desktop_restart(request)


@app.get("/v1/desktop/settings")
async def desktop_settings_endpoint(request: Request) -> dict[str, Any]:
    return desktop_settings_payload(request)


@app.post("/v1/desktop/settings")
async def save_desktop_settings_endpoint(request: Request) -> dict[str, Any]:
    return await save_desktop_settings(request)


@app.post("/v1/desktop/backups/export")
async def export_desktop_backup_endpoint(request: Request) -> dict[str, Any]:
    return export_desktop_backup(request)


@app.post("/v1/desktop/backups/import")
async def import_desktop_backup_endpoint(request: Request) -> dict[str, Any]:
    return await import_desktop_backup(request)


@app.post("/v1/desktop/backups/import-upload")
async def import_desktop_backup_upload_endpoint(request: Request) -> dict[str, Any]:
    return await import_desktop_backup_upload(request)


@app.get("/v1/desktop/logs")
async def desktop_logs_endpoint(request: Request) -> dict[str, Any]:
    return desktop_logs(request)


@app.get("/v1/gateway/health.json")
async def gateway_health(request: Request) -> dict[str, Any]:
    settings = get_settings()
    catalog: ModelCatalog = request.app.state.model_catalog
    state: StateManager = request.app.state.gateway_state
    router: WaterfallRouter = request.app.state.waterfall_router
    background: BackgroundEndpointDiagnosis | None = request.app.state.background_endpoint_diagnosis

    provider_names = [provider.name for provider in router.providers]
    usage = await state.snapshot_providers_usage(provider_names)
    configured_provider_count = sum(1 for provider in router.providers if provider.is_configured)
    enabled_routes = catalog.enabled_routes()

    return {
        "status": "ok",
        "service": "freerouter",
        "version": app.version,
        "database_path": settings.database_path,
        "model_catalog_path": settings.model_catalog_path,
        "providers": {
            "total": len(router.providers),
            "configured": configured_provider_count,
            "available": sum(1 for _, availability in usage.values() if availability.available),
        },
        "routes": {
            "total": len(catalog.all_routes()),
            "enabled": len(enabled_routes),
        },
        "request_limits": {
            "max_concurrent_requests": settings.max_concurrent_requests,
            "queue_timeout_seconds": settings.request_queue_timeout_seconds,
            "max_waiting_requests": settings.request_queue_max_waiting_requests,
        },
        "background_endpoint_diagnosis": {
            "enabled": settings.auto_endpoint_diagnosis_enabled,
            "running": background is not None,
            "auto_maintenance_enabled": bool(background and background.apply_safe_suggestions),
        },
    }


@app.get("/v1/models")
async def models(request: Request) -> dict[str, Any]:
    settings = get_settings()
    catalog: ModelCatalog = request.app.state.model_catalog
    created = int(time())
    return {
        "object": "list",
        "data": [
            {
                "id": settings.gateway_model_name,
                "object": "model",
                "created": created,
                "owned_by": "freerouter",
            }
        ]
        + [
            {
                "id": route.route_id,
                "object": "model",
                "created": created,
                "owned_by": route.provider_name,
            }
            for route in catalog.enabled_routes()
        ],
    }


@app.get("/v1/gateway/models")
async def gateway_models(request: Request) -> dict[str, Any]:
    catalog: ModelCatalog = request.app.state.model_catalog
    state: StateManager = request.app.state.gateway_state
    return await _catalog_payload_with_health(catalog, state)


@app.put("/v1/gateway/models")
async def update_gateway_models(request: Request) -> dict[str, Any]:
    catalog: ModelCatalog = request.app.state.model_catalog
    payload = await request.json()
    routes = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(routes, list):
        raise HTTPException(status_code=400, detail="Expected a JSON array or { data: [...] }")
    try:
        catalog.replace_routes(routes)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    state: StateManager = request.app.state.gateway_state
    return await _catalog_payload_with_health(catalog, state)


@app.post("/v1/gateway/models/reset")
async def reset_gateway_models(request: Request) -> dict[str, Any]:
    catalog: ModelCatalog = request.app.state.model_catalog
    state: StateManager = request.app.state.gateway_state
    catalog.reset_to_defaults()
    return await _catalog_payload_with_health(catalog, state)


@app.post("/v1/gateway/models/auto-rank")
async def auto_rank_gateway_models(request: Request) -> dict[str, Any]:
    catalog: ModelCatalog = request.app.state.model_catalog
    state: StateManager = request.app.state.gateway_state
    catalog.auto_rank_routes()
    catalog.save()
    return await _catalog_payload_with_health(catalog, state)


@app.post("/v1/gateway/models/{route_id}/disable")
async def disable_gateway_model(route_id: str, request: Request) -> dict[str, Any]:
    catalog: ModelCatalog = request.app.state.model_catalog
    state: StateManager = request.app.state.gateway_state
    try:
        route = catalog.set_route_enabled(route_id, False)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    route_state = await state.get_route_state(route.route_id, route.provider_name, route.model_id)
    return {"data": {**asdict(route), "health": asdict(route_state)}}


@app.post("/v1/gateway/models/{route_id}/enable")
async def enable_gateway_model(route_id: str, request: Request) -> dict[str, Any]:
    catalog: ModelCatalog = request.app.state.model_catalog
    state: StateManager = request.app.state.gateway_state
    try:
        route = catalog.set_route_enabled(route_id, True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    route_state = await state.get_route_state(route.route_id, route.provider_name, route.model_id)
    return {"data": {**asdict(route), "health": asdict(route_state)}}


@app.post("/v1/gateway/models/{route_id}/health/reset")
async def reset_gateway_model_health(route_id: str, request: Request) -> dict[str, Any]:
    catalog: ModelCatalog = request.app.state.model_catalog
    state: StateManager = request.app.state.gateway_state
    route = next((route for route in catalog.all_routes() if route.route_id == route_id), None)
    if route is None:
        raise HTTPException(status_code=404, detail=f"Unknown route_id: {route_id}")
    route_state = await state.clear_route_health(
        route.route_id, route.provider_name, route.model_id
    )
    return {"data": {**asdict(route), "health": asdict(route_state)}}


@app.get("/v1/gateway/endpoint-diagnosis")
async def endpoint_diagnosis_status(request: Request) -> dict[str, Any]:
    service: EndpointDiagnosisService = request.app.state.endpoint_diagnosis
    background: BackgroundEndpointDiagnosis | None = request.app.state.background_endpoint_diagnosis
    report = service.last_report
    return {
        "enabled": get_settings().auto_endpoint_diagnosis_enabled,
        "auto_maintenance_enabled": bool(background and background.apply_safe_suggestions),
        "last_auto_applied": (
            [asdict(suggestion) for suggestion in background.last_auto_applied]
            if background is not None
            else []
        ),
        "last_report": asdict(report) if report is not None else None,
    }


@app.post("/v1/gateway/endpoint-diagnosis/refresh")
async def refresh_endpoint_diagnosis(request: Request) -> dict[str, Any]:
    service: EndpointDiagnosisService = request.app.state.endpoint_diagnosis
    report = await service.run_once()
    return {"data": asdict(report)}


@app.post("/v1/gateway/endpoint-diagnosis/apply")
async def apply_endpoint_diagnosis(request: Request) -> dict[str, Any]:
    service: EndpointDiagnosisService = request.app.state.endpoint_diagnosis
    payload = await request.json()
    suggestion_ids = payload.get("suggestion_ids") if isinstance(payload, dict) else None
    if not isinstance(suggestion_ids, list) or not all(
        isinstance(item, str) for item in suggestion_ids
    ):
        raise HTTPException(status_code=400, detail="Expected { suggestion_ids: [string, ...] }")
    applied = await service.apply_suggestions(suggestion_ids)
    return {"data": [asdict(suggestion) for suggestion in applied]}


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


@app.get("/v1/providers/status")
async def provider_status(request: Request) -> dict[str, Any]:
    router: WaterfallRouter = request.app.state.waterfall_router
    state: StateManager = request.app.state.gateway_state
    catalog: ModelCatalog = request.app.state.model_catalog

    all_routes = catalog.all_routes()
    route_usage = await state.get_route_usage_stats([route.route_id for route in all_routes])
    health_map = await state.get_route_states_batch(
        [(r.route_id, r.provider_name, r.model_id) for r in all_routes]
    )
    providers_snap = await state.snapshot_providers_usage(
        [provider.name for provider in router.providers]
    )
    providers = []
    for provider in router.providers:
        pair = providers_snap.get(provider.name)
        if pair is None:
            provider_state = await state.get_state(provider.name)
            availability = await state.check_available(provider.name)
        else:
            provider_state, availability = pair
        models = []
        for route in all_routes:
            if route.provider_name != provider.name:
                continue
            route_state = health_map[route.route_id]
            route_payload = asdict(route)
            route_payload["health"] = asdict(route_state)
            route_payload["usage"] = route_usage[route.route_id]
            models.append(route_payload)
        providers.append(
            {
                "name": provider.name,
                "configured": provider.is_configured,
                "available": availability.available,
                "unavailable_reason": availability.reason,
                "retry_after_seconds": availability.retry_after_seconds,
                "tokens_used_today": provider_state.tokens_used_today,
                "requests_today": provider_state.requests_today,
                "requests_this_minute": provider_state.requests_this_minute,
                "cooldown_until": provider_state.cooldown_until,
                "max_context_tokens": provider.max_context_tokens,
                "models": models,
            }
        )

    return {"object": "list", "data": providers}


@app.get("/v1/gateway/live/snapshot")
async def live_api_snapshot(request: Request) -> dict[str, Any]:
    monitor: APILiveMonitor = request.app.state.live_monitor
    return {"object": "list", "data": await monitor.snapshot()}


@app.get("/v1/gateway/live/events")
async def live_api_events(request: Request) -> StreamingResponse:
    monitor: APILiveMonitor = request.app.state.live_monitor

    async def event_stream():
        queue = await monitor.subscribe()
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    payload = APILiveMonitor.event_to_payload(event)
                    yield f"data: {json.dumps(payload)}\n\n"
                except TimeoutError:
                    yield ": heartbeat\n\n"
                except asyncio.CancelledError:
                    break
        finally:
            await monitor.unsubscribe(queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _payload_with_required_web_search(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    prepared = dict(payload)
    tools = prepared.get("tools")
    if not isinstance(tools, list):
        tools = []
    elif not _has_web_search_tool(tools):
        tools = list(tools)
    if not _has_web_search_tool(tools):
        tools.append(dict(WEB_SEARCH_TOOL))
    prepared["tools"] = tools
    prepared["tool_choice"] = dict(WEB_SEARCH_TOOL)
    return prepared


def _has_web_search_tool(tools: list[Any]) -> bool:
    return any(
        isinstance(tool, dict) and tool.get("type") == WEB_SEARCH_TOOL["type"] for tool in tools
    )


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
                payload={"usage": _monitor_trim(dict(diag.usage))},
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
        await monitor.publish(
            event_type="route_selected",
            request_id=request_id,
            payload={
                "provider_name": diag.provider_name,
                "route_id": diag.route_id,
                "model_id": diag.model_id,
            },
        )
        return


async def _route_chat_completion_stream_request(
    request: Request,
    *,
    payload: dict[str, Any],
    path: str,
    required_tag: str | None = None,
) -> Response:
    router: WaterfallRouter = request.app.state.waterfall_router
    limiter: GatewayRequestLimiter = request.app.state.request_limiter
    monitor: APILiveMonitor = request.app.state.live_monitor
    request_id = uuid.uuid4().hex[:12]
    started_at = perf_counter()
    client_ip = request.client.host if request.client else None

    try:
        validate_chat_completion_payload(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await monitor.publish(
        event_type="request_started",
        request_id=request_id,
        payload={
            "path": path,
            "stream": True,
            "model": payload.get("model"),
            "client_ip": client_ip,
            "request_payload": _monitor_trim(payload),
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

    settings = get_settings()
    limiter_slot_held = True
    selected_provider = ""
    selected_route = ""
    selected_model = ""

    async def openai_sse_stream():
        nonlocal selected_provider, selected_route, selected_model, limiter_slot_held
        completed = False
        try:
            async for part in router.iter_chat_completion_openai_stream(
                payload,
                required_tag=required_tag,
                require_assistant_content=required_tag == "web-search",
            ):
                if isinstance(part, RouteStreamDiag):
                    await _publish_route_stream_diag(monitor, request_id, part)
                    if part.event_type == "route_selected":
                        selected_provider = part.provider_name or ""
                        selected_route = part.route_id or ""
                        selected_model = part.model_id or ""
                        if (
                            settings.streaming_release_slot_after_route_selected
                            and limiter_slot_held
                        ):
                            limiter.release()
                            limiter_slot_held = False
                else:
                    yield part
            completed = True
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
            if limiter_slot_held:
                limiter.release()
                limiter_slot_held = False
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

    return StreamingResponse(
        openai_sse_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _route_chat_completion_request(
    request: Request,
    *,
    payload: dict[str, Any],
    path: str,
    required_tag: str | None = None,
) -> JSONResponse:
    router: WaterfallRouter = request.app.state.waterfall_router
    limiter: GatewayRequestLimiter = request.app.state.request_limiter
    monitor: APILiveMonitor = request.app.state.live_monitor
    request_id = uuid.uuid4().hex[:12]
    started_at = perf_counter()
    client_ip = request.client.host if request.client else None
    await monitor.publish(
        event_type="request_started",
        request_id=request_id,
        payload={
            "path": path,
            "stream": bool(payload.get("stream")),
            "model": payload.get("model"),
            "client_ip": client_ip,
            "request_payload": _monitor_trim(payload),
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
            required_tag=required_tag,
            require_assistant_content=required_tag == "web-search",
        )
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

    await monitor.publish(
        event_type="request_completed",
        request_id=request_id,
        payload={
            "status_code": 200,
            "provider_name": result.provider_name,
            "route_id": result.route_id,
            "model_id": result.model_id,
            "attempts": len(result.attempts),
            "usage": _monitor_trim(
                result.body.get("usage") if isinstance(result.body, dict) else None
            ),
            "attempts_detail": [asdict(attempt) for attempt in result.attempts],
            "assistant_text": _monitor_trim(
                _assistant_text_from_response_body(result.body),
                max_string=8000,
            ),
            "response_body": _monitor_trim(result.body),
            "latency_ms": round((perf_counter() - started_at) * 1000),
        },
    )
    return JSONResponse(
        content=result.body,
        headers={
            "X-Gateway-Provider": result.provider_name,
            "X-Gateway-Route": result.route_id,
            "X-Gateway-Model": result.model_id,
            "X-Gateway-Attempts": json.dumps([asdict(attempt) for attempt in result.attempts]),
        },
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    payload: dict[str, Any] = await request.json()
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


@app.post("/v1/responses")
async def responses(request: Request) -> Response:
    responses_payload: dict[str, Any] = await request.json()
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


async def _route_responses_stream_request(
    request: Request,
    *,
    payload: dict[str, Any],
    requested_model: Any,
) -> Response:
    router: WaterfallRouter = request.app.state.waterfall_router
    limiter: GatewayRequestLimiter = request.app.state.request_limiter
    response_id = f"resp_{uuid.uuid4().hex}"

    try:
        validate_chat_completion_payload(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not await limiter.acquire():
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

    async def responses_sse_stream():
        carry = ""
        completed = False
        mapper = ResponsesStreamMapper(response_id=response_id)
        try:
            yield responses_stream_start(response_id=response_id, model=requested_model)
            async for part in router.iter_chat_completion_openai_stream(payload):
                if isinstance(part, RouteStreamDiag):
                    continue
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
            limiter.release()

    return StreamingResponse(
        responses_sse_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/v1/chat/completions/web-search")
async def chat_completions_web_search(request: Request) -> Response:
    payload: Any = await request.json()
    try:
        prepared_payload = _payload_with_required_web_search(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if prepared_payload.get("stream"):
        return await _route_chat_completion_stream_request(
            request,
            payload=prepared_payload,
            path="/v1/chat/completions/web-search",
            required_tag="web-search",
        )
    return await _route_chat_completion_request(
        request,
        payload=prepared_payload,
        path="/v1/chat/completions/web-search",
        required_tag="web-search",
    )


@app.post("/v1/chat/completions/stream-route")
async def chat_completions_stream_route(request: Request) -> Response:
    payload: dict[str, Any] = await request.json()
    router: WaterfallRouter = request.app.state.waterfall_router
    limiter: GatewayRequestLimiter = request.app.state.request_limiter
    monitor: APILiveMonitor = request.app.state.live_monitor
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

    stream_settings = get_settings()
    limiter_slot_held = True

    async def on_stream_event(event_payload: dict[str, Any]) -> None:
        nonlocal done_published, limiter_slot_held
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
                and limiter_slot_held
            ):
                limiter.release()
                limiter_slot_held = False
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
        nonlocal done_published, limiter_slot_held
        try:
            async for chunk in stream_route_chat(
                payload,
                router,
                chunk_replay_sleep_seconds=get_settings().sse_chunk_replay_sleep_seconds,
                on_emit=on_stream_event,
            ):
                yield chunk
        finally:
            if limiter_slot_held:
                limiter.release()
                limiter_slot_held = False
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

    return StreamingResponse(
        limited_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


