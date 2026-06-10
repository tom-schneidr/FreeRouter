from __future__ import annotations

from time import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.api.chat_handlers import _catalog_payload_with_health
from app.app_services import get_app_services
from app.capability_probes import PROBE_TAGS, probe_route_capabilities
from app.capability_tags import capability_claim_to_dict
from app.model_catalog import _route_to_dict
from app.settings import get_settings

router = APIRouter()


@router.get("/v1/models")
async def models(request: Request) -> dict[str, Any]:
    settings = get_settings()
    services = get_app_services(request)
    catalog = services.model_catalog
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


@router.get("/v1/gateway/models")
async def gateway_models(request: Request) -> dict[str, Any]:
    services = get_app_services(request)
    return await _catalog_payload_with_health(services.model_catalog, services.gateway_state)


@router.put("/v1/gateway/models")
async def update_gateway_models(request: Request) -> dict[str, Any]:
    services = get_app_services(request)
    catalog = services.model_catalog
    payload = await request.json()
    routes = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(routes, list):
        raise HTTPException(status_code=400, detail="Expected a JSON array or { data: [...] }")
    try:
        catalog.replace_routes(routes)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return await _catalog_payload_with_health(catalog, services.gateway_state)


@router.post("/v1/gateway/models/reset")
async def reset_gateway_models(request: Request) -> dict[str, Any]:
    services = get_app_services(request)
    catalog = services.model_catalog
    catalog.reset_to_defaults()
    return await _catalog_payload_with_health(catalog, services.gateway_state)


@router.post("/v1/gateway/models/auto-rank")
async def auto_rank_gateway_models(request: Request) -> dict[str, Any]:
    services = get_app_services(request)
    catalog = services.model_catalog
    catalog.auto_rank_routes()
    catalog.save()
    return await _catalog_payload_with_health(catalog, services.gateway_state)


@router.post("/v1/gateway/models/{route_id}/disable")
async def disable_gateway_model(route_id: str, request: Request) -> dict[str, Any]:
    services = get_app_services(request)
    catalog = services.model_catalog
    state = services.gateway_state
    try:
        route = catalog.set_route_enabled(route_id, False)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    route_state = await state.get_route_state(route.route_id, route.provider_name, route.model_id)
    return {"data": {**_route_to_dict(route), "health": route_state.__dict__}}


@router.post("/v1/gateway/models/{route_id}/enable")
async def enable_gateway_model(route_id: str, request: Request) -> dict[str, Any]:
    services = get_app_services(request)
    catalog = services.model_catalog
    state = services.gateway_state
    try:
        route = catalog.set_route_enabled(route_id, True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    route_state = await state.get_route_state(route.route_id, route.provider_name, route.model_id)
    return {"data": {**_route_to_dict(route), "health": route_state.__dict__}}


@router.post("/v1/gateway/models/{route_id}/health/reset")
async def reset_gateway_model_health(route_id: str, request: Request) -> dict[str, Any]:
    services = get_app_services(request)
    catalog = services.model_catalog
    state = services.gateway_state
    route = next((route for route in catalog.all_routes() if route.route_id == route_id), None)
    if route is None:
        raise HTTPException(status_code=404, detail=f"Unknown route_id: {route_id}")
    route_state = await state.clear_route_health(
        route.route_id, route.provider_name, route.model_id
    )
    return {"data": {**_route_to_dict(route), "health": route_state.__dict__}}


@router.post("/v1/gateway/models/{route_id}/probe-capabilities")
async def probe_gateway_model_capabilities(route_id: str, request: Request) -> dict[str, Any]:
    services = get_app_services(request)
    catalog = services.model_catalog
    router_svc = services.waterfall_router
    route = next((item for item in catalog.all_routes() if item.route_id == route_id), None)
    if route is None:
        raise HTTPException(status_code=404, detail=f"Unknown route_id: {route_id}")

    provider = router_svc.provider_by_name.get(route.provider_name)
    if provider is None or not provider.is_configured:
        raise HTTPException(
            status_code=400,
            detail=f"Provider {route.provider_name} is not configured for probing",
        )

    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        body = {}
    tags = body.get("tags") if isinstance(body, dict) else None
    probe_tags = tuple(str(tag) for tag in tags) if isinstance(tags, list) and tags else PROBE_TAGS

    claims = await probe_route_capabilities(
        provider,
        services.http_client,
        route,
        tags=probe_tags,
    )
    updated = catalog.apply_probe_claims_to_route(route_id, claims)
    route_state = await services.gateway_state.get_route_state(
        updated.route_id, updated.provider_name, updated.model_id
    )
    return {
        "data": {
            **_route_to_dict(updated),
            "health": route_state.__dict__,
            "probe_results": {
                tag: capability_claim_to_dict(claim) for tag, claim in claims.items()
            },
        }
    }
