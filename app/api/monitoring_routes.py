from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.app_services import get_app_services
from app.live_monitor import APILiveMonitor
from app.settings import get_settings

router = APIRouter()


@router.get("/v1/gateway/health.json")
async def gateway_health(request: Request) -> dict[str, Any]:
    settings = get_settings()
    services = get_app_services(request)
    catalog = services.model_catalog
    state = services.gateway_state
    router_svc = services.waterfall_router
    background = services.background_endpoint_diagnosis

    provider_names = [provider.name for provider in router_svc.providers]
    usage = await state.snapshot_providers_usage(provider_names)
    configured_provider_count = sum(1 for provider in router_svc.providers if provider.is_configured)
    enabled_routes = catalog.enabled_routes()

    return {
        "status": "ok",
        "service": "freerouter",
        "version": request.app.version,
        "database_path": settings.database_path,
        "model_catalog_path": settings.model_catalog_path,
        "providers": {
            "total": len(router_svc.providers),
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


@router.get("/v1/providers/status")
async def provider_status(request: Request) -> dict[str, Any]:
    services = get_app_services(request)
    router_svc = services.waterfall_router
    state = services.gateway_state
    catalog = services.model_catalog

    all_routes = catalog.all_routes()
    route_usage = await state.get_route_usage_stats([route.route_id for route in all_routes])
    health_map = await state.get_route_states_batch(
        [(r.route_id, r.provider_name, r.model_id) for r in all_routes]
    )
    providers_snap = await state.snapshot_providers_usage(
        [provider.name for provider in router_svc.providers]
    )
    providers = []
    for provider in router_svc.providers:
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


@router.get("/v1/gateway/live/snapshot")
async def live_api_snapshot(request: Request) -> dict[str, Any]:
    monitor = get_app_services(request).live_monitor
    return {"object": "list", "data": await monitor.snapshot()}


@router.get("/v1/gateway/live/events")
async def live_api_events(request: Request) -> StreamingResponse:
    monitor = get_app_services(request).live_monitor

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
