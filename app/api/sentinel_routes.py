from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.app_services import get_app_services

router = APIRouter()


def _sentinel(request: Request):
    service = get_app_services(request).sentinel
    if service is None:
        raise HTTPException(status_code=503, detail="Sentinel service is unavailable")
    return service


@router.get("/v1/gateway/sentinel")
async def sentinel_snapshot(request: Request):
    base_url = str(request.base_url).rstrip("/")
    return await _sentinel(request).snapshot(base_url=base_url)


@router.get("/v1/gateway/sentinel/doctor")
async def sentinel_doctor(request: Request, profile: str = "safe-coding"):
    try:
        return await _sentinel(request).doctor(profile)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/v1/gateway/sentinel/routes/{route_id}/evaluate")
async def sentinel_evaluate_route(route_id: str, request: Request):
    try:
        evaluation = await _sentinel(request).evaluate_route(route_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"data": evaluation.to_dict()}
